"""Teacher embedding cache.

Encodes corpus documents and queries with a teacher model once and caches
L2-normalized float32 embeddings under data/emb/<dataset>/<teacher_key>/.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

E5_QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query" + chr(10) + "Query: "

# MTEB / E5-mistral task instructions (the ones the Qwen3-Embedding and harrier leaderboard runs use); keys = our dataset names
TASK_PROMPTS = {
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
    "arguana": "Given a claim, find documents that refute the claim",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question",
    "webfaq-cs": "Given a question, retrieve relevant documents that answer the question",
    "miracl-fi": "Given a question, retrieve Wikipedia passages that answer the question",
}


def task_query_prompt(dataset: str) -> str:
    return "Instruct: " + TASK_PROMPTS[dataset] + chr(10) + "Query: "


TEACHERS = {
    # key: (hf id, query prompt, doc prompt, max_seq_len, notes)
    "e5-mistral-7b": dict(hf="intfloat/e5-mistral-7b-instruct", q_prompt=E5_QUERY_PROMPT, d_prompt="", max_len=512,
                          notes="Mistral-7B decoder, last-token pooling, 4096d - the big NON-Qwen control"),
    "bge-m3": dict(hf="BAAI/bge-m3", q_prompt="", d_prompt="", max_len=512, pooling="cls",
                   notes="XLM-R large encoder, CLS pooling, 1024d"),
    "qwen3-0.6b": dict(hf="Qwen/Qwen3-Embedding-0.6B",
                       q_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",  # official ST prompt: NO trailing space (config_sentence_transformers.json); the fp32 cache uses it, so must the runtime (lab log 2026-09-09, E7)
                       d_prompt="", max_len=512,
                       notes="Qwen3 decoder, last-token pooling, 1024d"),
    "qwen3-4b": dict(hf="Qwen/Qwen3-Embedding-4B",
                     q_prompt=E5_QUERY_PROMPT.rstrip(" "),  # official Qwen3-Embedding prompt ends with 'Query:' (no space); overridden by the model's own ST prompt at load time
                     d_prompt="", max_len=512, notes="Qwen3-4B decoder embedding model, last-token pooling, 2560d"),
    "me5-small": dict(hf="intfloat/multilingual-e5-small", q_prompt="query: ", d_prompt="passage: ", max_len=512, pooling="mean",
                      notes="small baseline encoder, 384d"),
    # harrier-oss-v1 = the bf16 teachers BitEmbed itself distilled from (same backbones as bitnet-embedding-*)
    "harrier-270m": dict(hf="microsoft/harrier-oss-v1-270m", q_prompt=E5_QUERY_PROMPT,
                         d_prompt="", max_len=512, notes="Gemma3-270M decoder, last-token pooling, 640d, bf16"),
    "harrier-0.6b": dict(hf="microsoft/harrier-oss-v1-0.6b", q_prompt=E5_QUERY_PROMPT,
                         d_prompt="", max_len=512, notes="Qwen3-0.6B decoder, last-token pooling, 1024d, bf16"),
    # "-tp" variants: per-dataset MTEB task instruction for queries (documents identical -> reused from the base key)
    "harrier-0.6b-tp": dict(hf="microsoft/harrier-oss-v1-0.6b", q_prompt=E5_QUERY_PROMPT, d_prompt="", max_len=512,
                            task_prompt=True, docs_from="harrier-0.6b", notes="harrier-0.6b with MTEB task instructions"),
    "qwen3-4b-tp": dict(hf="Qwen/Qwen3-Embedding-4B", q_prompt=E5_QUERY_PROMPT, d_prompt="", max_len=512,
                        task_prompt=True, docs_from="qwen3-4b", notes="Qwen3-Embedding-4B with MTEB task instructions"),
    # jina-embeddings-v5-text-small, retrieval task: the MERGED per-task weights (base Qwen3-0.6B + LoRA r=32 folded in),
    # a plain Qwen3Model (model_type qwen3, no custom code, rope_theta 3.5e6, tokenizer appends NO eos).  Prompts are the
    # short jina prefixes, NOT the E5 instruction (task_prompt stays off: the MTEB task instructions are an E5/Qwen3-Embedding
    # convention).  1024-d last-token pooling, matryoshka 32..1024, supports 32k context (we cap at 512).  Licence CC BY-NC 4.0.
    # Official f16 GGUF (general.architecture = qwen3): models/gguf/jina-v5-small-f16.gguf.
    "jina-v5-small": dict(hf="jinaai/jina-embeddings-v5-text-small-retrieval", q_prompt="Query: ", d_prompt="Document: ",
                          max_len=512, pooling="last", dim=1024, gguf_f16="models/gguf/jina-v5-small-f16.gguf",
                          notes="jina-embeddings-v5-text-small (Qwen3-0.6B-Base fine-tune), retrieval adapter merged, "
                                "last-token pooling, 1024d, prompts 'Query: ' / 'Document: ', no EOS, CC BY-NC 4.0"),
}


def query_prompt(spec_or_key, dataset: str | None = None) -> str:
    """The query prefix a teacher key uses: the per-dataset MTEB task instruction for the '-tp' keys, otherwise the
    teacher's own q_prompt (E5 instruction for harrier/qwen3-embedding, 'Query: ' for jina).  Byte-identical to the old
    `task_query_prompt(d) if spec.get("task_prompt") else E5_QUERY_PROMPT` for every pre-existing key."""
    spec = TEACHERS[spec_or_key] if isinstance(spec_or_key, str) else spec_or_key
    if spec.get("task_prompt"):
        if dataset is None:
            raise ValueError("task_prompt teacher needs the dataset name")
        return task_query_prompt(dataset)
    return spec.get("q_prompt", E5_QUERY_PROMPT)


def doc_prompt(spec_or_key) -> str:
    spec = TEACHERS[spec_or_key] if isinstance(spec_or_key, str) else spec_or_key
    return spec.get("d_prompt", "")


@dataclass
class TeacherEmbeddings:
    dataset: str
    teacher: str
    doc_ids: list[str]
    D: np.ndarray  # (N, d) float32, unit norm
    query_ids: list[str]
    Q: np.ndarray  # (M, d) float32, unit norm
    meta: dict

    def doc_index(self) -> dict[str, int]:
        return {d: i for i, d in enumerate(self.doc_ids)}

    def query_index(self) -> dict[str, int]:
        return {q: i for i, q in enumerate(self.query_ids)}


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def emb_dir(dataset: str, teacher: str, root: str = "data") -> Path:
    return Path(root) / "emb" / dataset / teacher


def load_teacher_embeddings(dataset: str, teacher: str, root: str = "data") -> TeacherEmbeddings:
    p = emb_dir(dataset, teacher, root)
    ids = json.loads((p / "ids.json").read_text(encoding="utf-8"))
    D = np.load(p / "corpus.npy")
    Q = np.load(p / "queries.npy")
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    return TeacherEmbeddings(dataset, teacher, ids["doc_ids"], D, ids["query_ids"], Q, meta)


def _pick_device(max_vram_gb: float | None):
    import torch
    if os.environ.get("EQ_FORCE_CPU") == "1" or not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    if max_vram_gb is not None and free / 2**30 < max_vram_gb:
        return "cpu"
    return "cuda"


def encode_dataset(ds, teacher: str, root: str = "data", batch_size: int = 32, max_vram_gb: float | None = 2.5,
                   threads: int | None = None, force: bool = False, doc_limit: int | None = None) -> TeacherEmbeddings:
    """Encode ds (RetrievalDataset) with teacher; cache to disk. Idempotent."""
    import torch
    from sentence_transformers import SentenceTransformer

    p = emb_dir(ds.name, teacher, root)
    imported = None  # meta of an index imported from the customer's own vectors (scripts/import_corpus.py)
    if not force and (p / "corpus.npy").exists() and (p / "queries.npy").exists():
        cached = load_teacher_embeddings(ds.name, teacher, root)
        if len(cached.query_ids) or not ds.queries:
            return cached
        # an imported index without query vectors: keep the customer's corpus.npy, encode only the queries below
        imported = cached.meta.get("imported")
    p.mkdir(parents=True, exist_ok=True)
    spec = TEACHERS[teacher]
    device = _pick_device(max_vram_gb)
    if threads:
        torch.set_num_threads(threads)
    t0 = time.time()
    # bf16 on GPU: Gemma3-based teachers (harrier-270m) overflow to NaN in fp16
    model_kwargs = {"torch_dtype": torch.bfloat16} if device == "cuda" else {}
    local = Path("models/hf") / spec["hf"].split("/")[-1]
    src = str(local) if (local / "config.json").exists() else spec["hf"]
    model = SentenceTransformer(src, device=device, model_kwargs=model_kwargs, trust_remote_code=False)
    model.max_seq_length = spec["max_len"]
    load_s = time.time() - t0
    # prefer the model's OFFICIAL query prompt (config_sentence_transformers.json) over our hardcoded one
    q_prompt = spec["q_prompt"]
    official = getattr(model, "prompts", None) or {}
    for name in ("web_search_query", "query"):
        if name in official and official[name]:
            q_prompt = official[name]
            break
    if spec.get("task_prompt"):
        q_prompt = task_query_prompt(ds.name)  # MTEB task instruction instead of the generic web-search one
    d_prompt = spec["d_prompt"] if not official.get("document") else official["document"]

    doc_ids = sorted(ds.corpus.keys())
    if doc_limit:
        doc_ids = doc_ids[:doc_limit]
    base = emb_dir(ds.name, spec["docs_from"], root) if spec.get("docs_from") else (p if imported is not None else None)
    if base is not None and (base / "corpus.npy").exists() and not doc_limit:
        base_ids = json.loads((base / "ids.json").read_text(encoding="utf-8"))["doc_ids"]
        assert base_ids == doc_ids, "doc id order differs from the base cache"
        D = np.load(base / "corpus.npy"); doc_s = 0.0  # documents carry no instruction -> identical embeddings
    else:
        docs = []
        for did in doc_ids:
            r = ds.corpus[did]
            title = (r.get("title") or "").strip()
            text = (r.get("text") or "").strip()
            docs.append(d_prompt + (f"{title}\n{text}" if title else text))
        # sort by length for throughput
        order = np.argsort([len(x) for x in docs])[::-1]
        t1 = time.time()
        D_sorted = model.encode([docs[i] for i in order], batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=True)
        D = np.empty_like(D_sorted)
        D[order] = D_sorted
        doc_s = time.time() - t1

    query_ids = sorted(ds.queries.keys())
    qs = [q_prompt + ds.queries[q] for q in query_ids]
    t2 = time.time()
    Q = model.encode(qs, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=True)
    q_s = time.time() - t2

    D = _l2n(D)
    Q = _l2n(Q)
    np.save(p / "corpus.npy", D)
    np.save(p / "queries.npy", Q)
    (p / "ids.json").write_text(json.dumps({"doc_ids": doc_ids, "query_ids": query_ids}), encoding="utf-8")
    meta = dict(teacher=teacher, hf=spec["hf"], dim=int(D.shape[1]), n_docs=len(doc_ids), n_queries=len(query_ids),
                device=device, batch_size=batch_size, max_len=spec["max_len"], load_s=load_s, doc_encode_s=doc_s,
                query_encode_s=q_s, docs_per_s=len(doc_ids) / max(doc_s, 1e-9),
                torch_threads=torch.get_num_threads(), notes=spec["notes"], q_prompt=q_prompt, d_prompt=d_prompt,
                dtype="bf16" if device == "cuda" else "fp32", nan_docs=int(np.isnan(D).any(1).sum()),
                nan_queries=int(np.isnan(Q).any(1).sum()))
    if imported is not None:
        meta["imported"] = imported  # provenance of the document vectors (customer index), the queries are ours
    (p / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return TeacherEmbeddings(ds.name, teacher, doc_ids, D, query_ids, Q, meta)
