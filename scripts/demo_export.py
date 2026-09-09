"""Export the data files of the public browser demo (demo/) for ONE corpus: the fp32 document index of the original
model (stored as fp16, or as row-scaled int8 when that provably does not change nDCG@10), the document texts, the test
queries with their qrels, the reference numbers, and the client GGUF in byte chunks.

Multi-corpus layout (the page reads demo/data/index.json to list the corpora and demo/data/<ds>/meta.json for one):
  demo/data/index.json              {"default": "scifact", "corpora": [{id, label, n_docs, n_queries, model_file, ...}]}
  demo/data/<ds>/meta.json          model / file / prompt / reference numbers / index + docs shard lists / date
  demo/data/<ds>/corpus.f16.NNN     raw little-endian float16 rows (L2-normalised), row-major, sharded by row ranges
       or  corpus.i8.NNN + corpus.scale.f32   int8 codes (row-wise symmetric: x ~= q * scale[row]) + fp32 scale per row
  demo/data/<ds>/docs.NNN.json      [{id, title, text}] in index order, sharded by index ranges (Pages: 25 MiB per file)
  demo/data/<ds>/test_queries.json  [{qid, text, rel: {doc_id: grade}}] in ds.splits['test'] order (= native per-query
                                    order); text WITHOUT the instruction prefix -- the page prepends meta.json "prompt"
  demo/model/<ds>/<file>.chunks.json + <file>.chunkNNN   the GGUF as raw byte ranges of at most --chunk_mib (24) MiB
Why bytes and not `llama-gguf-split`: the Q2_K token table (token_embd.weight) alone is 48.7 MiB, a GGUF shard cannot
split a tensor, and Cloudflare Pages caps every asset at 25 MiB.

int8 index (--index_dtype auto, the default): rows are quantised as q = round(x / s), s = max|x| / 127 per row, and the
exporter scores the corpus' test queries (fp32 query cache of the same model) against the fp32 and the dequantised
index; int8 is kept only if |delta nDCG@10| < --int8_tol (0.001), otherwise fp16 shards are written.  The decision and
both numbers go to meta.json ("index.check").

The prompt is the teacher's own query prefix (eq.teacher.query_prompt: the generic E5 web-search instruction for
harrier-0.6b, "Query: " for jina-v5-small): the protocol of the native reference numbers and of the fp32 query cache --
NOT the MTEB task instruction.  --teacher (default harrier-0.6b) also selects the model name / base model / licence
notice of the page (TEACHER_INFO) and the file prefix <teacher>-gptq-; the corpus notice comes from DATASETS[ds].

Usage:
  .venv/Scripts/python.exe scripts/demo_export.py                                   # scifact, fp16 index (as deployed)
  .venv/Scripts/python.exe scripts/demo_export.py --dataset legal-cs --teacher jina-v5-small --client models/gguf/<file>.gguf
  .venv/Scripts/python.exe scripts/demo_export.py --dataset scidocs --client models/gguf/<file>.gguf
  options: --no_model (keep the existing chunks, only refresh data + meta), --index_dtype auto|f16|int8, --chunk_mib 24
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eq import graph_metrics as G  # noqa: E402
from eq.data import load_dataset  # noqa: E402
from eq.teacher import TEACHERS, query_prompt  # noqa: E402
from eq.verify import rel_matrix, self_mask, mask_self  # noqa: E402

# what the page says about the query encoder, per teacher key (the numbers come from the measurements, not from here)
TEACHER_INFO = {
    "qwen3-0.6b": dict(
        model_name="Qwen3-Embedding-0.6B (Qwen/Qwen3-Embedding-0.6B), query side, llama-quantize Q4_K_M + q4_0 token table", model_short="Qwen3-Embedding-0.6B",
        base_model="Qwen/Qwen3-Embedding-0.6B", base_model_url="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        base_model_license="Apache-2.0 (per the Hugging Face model card)",
        license_notice="Quantized query-side client released under Apache-2.0 (github.com/rosecky/embedding-quantization-public); base model "
                       "Qwen/Qwen3-Embedding-0.6B Apache-2.0, runtime llama.cpp/wllama MIT, corpus {corpus}.",
        prompt_kind="the model's official web-search instruction, no trailing space (protocol of the fp32 index)", prompt_short="Qwen3 instruction",
        fp16_size_mib=1142, native_latency_ms=None, native_peak_rss_mib=None, native_latency_note="not measured natively for this file"),
    "harrier-0.6b": dict(
        model_name="harrier-0.6b (microsoft/harrier-oss-v1-0.6b), query side, GPTQ on the llama.cpp Q2_K grid", model_short="harrier-0.6b",
        base_model="microsoft/harrier-oss-v1-0.6b", base_model_url="https://huggingface.co/microsoft/harrier-oss-v1-0.6b",
        base_model_license="MIT (per the Hugging Face model card)",
        license_notice="Quantized query-side client released under MIT (github.com/rosecky/embedding-quantization-public); base model "
                       "microsoft/harrier-oss-v1-0.6b MIT, Qwen3 base Apache-2.0, runtime llama.cpp/wllama MIT, corpus {corpus}.",
        prompt_kind="generic E5 web-search instruction (protocol of the native reference)", prompt_short="E5 instruction",
        fp16_size_mib=1143, native_latency_ms=151, native_peak_rss_mib=928,
        native_latency_note="llama-embedding, 300 SciFact test queries one by one, laptop (Core Ultra 7 155H), 8 threads, ctx 512, machine idle"),
    "jina-v5-small": dict(
        model_name="jinaai/jina-embeddings-v5-text-small (retrieval), query side, GPTQ on the llama.cpp Q2_K grid", model_short="jina-embeddings-v5-text-small",
        base_model="Qwen3-0.6B-Base (jina-embeddings-v5-text-small fine-tune, retrieval adapter merged)",
        base_model_url="https://huggingface.co/jinaai/jina-embeddings-v5-text-small", base_model_license="CC BY-NC 4.0 (jina-embeddings-v5-text-small model card)",
        license_notice="Evaluation only. Model weights are not licensed for redistribution; jinaai/jina-embeddings-v5-text-small is CC BY-NC 4.0 "
                       "(non-commercial), base Qwen3-0.6B-Base Apache-2.0, runtime llama.cpp/wllama MIT, corpus {corpus}.",
        prompt_kind="jina retrieval query prefix 'Query: ' (protocol of the native reference)", prompt_short="'Query: ' prefix",
        fp16_size_mib=None, native_latency_ms=None, native_peak_rss_mib=None, native_latency_note=None),
}

PAGES_FILE_LIMIT = 25 * 2**20  # Cloudflare Pages: 25 MiB per asset
SHARD_BYTES = 24 * 2**20       # our shard size for index / docs files
DOCS_PER_SHARD = 5000
DEFAULT_CLIENT = {
    "scifact": "models/gguf/harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf",
    "scidocs": "models/gguf/harrier-0.6b-gptq-Q2_K-scidocs_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf",
    "legal-cs": "models/gguf/jina-v5-small-gptq-Q2_K-legal-cs_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf",
}
# corpus-specific texts shown by the page (everything else comes from the measurements)
DATASETS = {
    "scifact": dict(label="SciFact", long="SciFact (scientific claim verification, 5 183 abstracts)",
                    corpus_description="scientific abstracts (SciFact)", unit="abstracts",
                    query_hint='Ask a scientific claim, e.g. "vitamin D supplementation reduces fracture risk"',
                    query_kind="a scientific claim", dataset_license="SciFact: CC BY-NC 2.0 (non-commercial)",
                    dataset_url="https://github.com/allenai/scifact", dataset_name="SciFact",
                    task="claim -> supporting/refuting abstract",
                    corpus_notice="SciFact corpus (AllenAI, Wadden et al. 2020), CC BY-NC 2.0: research use only"),
    "scidocs": dict(label="SciDocs", long="SciDocs (BEIR, scientific papers, 25 656 title+abstract docs)",
                    corpus_description="scientific paper title+abstract documents (SciDocs, BEIR)", unit="papers",
                    query_hint='Type a paper title or topic, e.g. "graph neural networks for citation recommendation"',
                    query_kind="a paper title or topic", dataset_license="SciDocs: CC BY 4.0",
                    dataset_url="https://github.com/allenai/scidocs", dataset_name="SciDocs",
                    task="paper title -> cited papers (citation prediction)",
                    corpus_notice="SciDocs corpus (AllenAI, BEIR), CC BY 4.0"),
    "legal-cs": dict(label="Legal CS", long="Czech supreme-court reasoning segments (customer corpus, {n} segments)",
                     corpus_description="Czech supreme-court reasoning segments (court_argument, civil law, 2020+)", unit="segments",
                     query_hint='Ask a legal question in Czech, e.g. "promlčení nároku na náhradu škody"',
                     query_kind="a legal question in Czech", dataset_license="Czech supreme-court decisions (public documents); index provided by the owner",
                     dataset_url=None, dataset_name="Czech supreme-court decisions (customer index)",
                     task="Czech legal question -> court reasoning segment (synthetic doc2query test set, source segment = relevant)",
                     corpus_notice="Czech supreme-court decisions (public documents); index provided by the owner"),
}
# result rows of the native evaluation (llama-embedding, query side quantised, fp32 index, test split) per dataset
NATIVE_ROWS = {"scifact": ["results/raw/gptq_export/results.jsonl"],
               "scidocs": ["results/raw/scidocs_local/results.jsonl", "results/raw/release_local/results.jsonl"],
               "legal-cs": ["results/raw/legal_local/results.jsonl"]}
NATIVE_PERQ = {"scifact": ["results/raw/gptq_export/perq"], "scidocs": ["results/raw/scidocs_local/perq", "results/raw/release_local/perq"], "legal-cs": ["results/raw/legal_local/perq"]}
BROWSER_REF = {"scifact": "results/raw/browser_local/browser_load_mt.json", "scidocs": "results/raw/scidocs_local/smoke_scidocs.json",
               "legal-cs": "results/raw/legal_local/smoke_legal-cs.json"}
# fallbacks for scifact when the result files are not present (values of 2026-09-07)
REF_FALLBACK = {"scifact": dict(native_ndcg10=0.7440, fp16_ndcg10=0.7559, browser_ndcg10=0.7425, native_cos_fp=0.8261,
                                browser_memory_after_load_bytes=1202863635, browser_memory_api="measureUserAgentSpecificMemory")}


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def chunk_model(model_path: Path, out_dir: Path, chunk_mib: int) -> dict:
    """Cut the GGUF into raw byte chunks (<= chunk_mib MiB each) + manifest; returns the manifest."""
    if out_dir.exists():
        for f in out_dir.glob(f"{model_path.name}.chunk*"):
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytes = chunk_mib * 2**20
    size = model_path.stat().st_size
    n = (size + chunk_bytes - 1) // chunk_bytes
    chunks, h = [], hashlib.sha256()
    with open(model_path, "rb") as f:
        for i in range(n):
            data = f.read(chunk_bytes)
            h.update(data)
            name = f"{model_path.name}.chunk{i:03d}"
            (out_dir / name).write_bytes(data)
            chunks.append({"url": name, "bytes": len(data)})
    manifest = {"file": model_path.name, "size_bytes": size, "chunk_bytes": chunk_bytes, "sha256": h.hexdigest(), "chunks": chunks,
                "note": "raw byte ranges of one GGUF file in order; concatenate to rebuild the file"}
    (out_dir / f"{model_path.name}.chunks.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    assert sum(c["bytes"] for c in chunks) == size
    print(f"model {model_path.name}: {size / 2**20:.1f} MiB -> {n} chunks of <= {chunk_mib} MiB in {out_dir} (sha256 {manifest['sha256'][:12]}...)")
    return manifest


def reference_numbers(gguf_name: str, dataset: str, prompt_short: str = "E5 instruction") -> dict:
    ref = dict(REF_FALLBACK.get(dataset, {}))
    for rel in NATIVE_ROWS.get(dataset, []):
        rows = ROOT / rel
        if not rows.exists():
            continue
        for line in rows.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("gguf") == gguf_name and d.get("dataset") == dataset and d.get("splits") == "test" and not d.get("both_sides"):
                ref["native_ndcg10"] = round(float(d["gt_ndcg10"]), 4)
                ref["fp16_ndcg10"] = round(float(d["fp_ndcg10"]), 4)
                ref["native_cos_fp"] = round(float(d["q_cos_fp"]), 4)
                ref["native_n_queries"] = int(d.get("n_test", 0))
                ref["native_source"] = f"{rel} (llama-embedding, pooling last, 8 threads, {prompt_short}, fp32 index)"
    br = ROOT / BROWSER_REF.get(dataset, "")
    if br.is_file():
        b = json.loads(br.read_text(encoding="utf-8"))
        if b.get("ok") and b.get("prompt_kind", "generic") == "generic" and "ndcg_mean" in b:        # browser_run.py result
            ref["browser_ndcg10"] = round(float(b["ndcg_mean"]), 4)
            ref["browser_n_queries"] = int(b.get("n", 0) or 0)
            mem = b.get("mem_after_load") or {}
            if mem.get("bytes"):
                ref["browser_memory_after_load_bytes"] = int(mem["bytes"]); ref["browser_memory_api"] = mem.get("api")
            ref["browser_source"] = f"{BROWSER_REF[dataset]} (wllama {b.get('wllama_version')}, {b.get('threads_used')} threads, headless Chrome, machine under load)"
        elif b.get("ok") and (b.get("verify") or {}).get("n"):                                      # demo_smoke.py result
            v = b["verify"]
            ref["browser_ndcg10"] = round(float(v["ndcg10_mean"]), 4)
            ref["browser_n_queries"] = int(v["n"])
            ref["browser_native_same_n"] = round(float(v["ndcg10_native_same_n"]), 4) if v.get("ndcg10_native_same_n") is not None else None
            mem = b.get("memory_after_load") or {}
            if mem.get("bytes"):
                ref["browser_memory_after_load_bytes"] = int(mem["bytes"]); ref["browser_memory_api"] = mem.get("api")
            ref["browser_source"] = f"{BROWSER_REF[dataset]} (demo smoke test, wllama {v.get('wllama_version')}, {v.get('threads')} threads, headless Chrome; first {v['n']} test queries)"
    for rel in NATIVE_PERQ.get(dataset, []):
        perq = ROOT / rel / f"{dataset}_{Path(gguf_name).stem}.npz"
        if perq.exists():
            nd = np.load(perq)["ndcg"].astype(np.float64)
            ref["native_ndcg10_per_query"] = [None if np.isnan(v) else round(float(v), 6) for v in nd]
            ref["native_perq_source"] = str(perq.relative_to(ROOT)).replace("\\", "/")
    return ref


def write_shards(arr: np.ndarray, out: Path, stem: str, ext: str) -> list[dict]:
    """Row-range shards of a 2-D array (< SHARD_BYTES each); returns [{url, rows, bytes}]."""
    for f in out.glob(f"{stem}.{ext}.*"):
        f.unlink()
    row_bytes = arr.shape[1] * arr.dtype.itemsize
    rows_per = max(1, SHARD_BYTES // row_bytes)
    shards = []
    for i, s in enumerate(range(0, arr.shape[0], rows_per)):
        part = np.ascontiguousarray(arr[s:s + rows_per])
        name = f"{stem}.{ext}.{i:03d}"
        part.tofile(out / name)
        shards.append({"url": name, "rows": int(part.shape[0]), "bytes": int(part.nbytes)})
    return shards


def ndcg_test(ds, Q: np.ndarray, qids: list[str], D: np.ndarray, doc_ids: list[str], test_q: list[str]) -> float:
    tdi = {x: i for i, x in enumerate(doc_ids)}; tqi = {q: i for i, q in enumerate(qids)}
    rel = rel_matrix(test_q, tdi, ds.qrels, len(doc_ids)); msk = self_mask(test_q, tdi, len(doc_ids))
    S = mask_self(Q[[tqi[q] for q in test_q]] @ D.T, msk)
    return float(np.nanmean(G.ndcg_at_k(S, rel, 10)))


def update_registry(out_root: Path, entry: dict, default: str = "scifact") -> None:
    p = out_root / "index.json"
    reg = {"default": default, "corpora": []}
    if p.exists():
        try:
            reg = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    reg["corpora"] = [c for c in reg.get("corpora", []) if c.get("id") != entry["id"]] + [entry]
    order = {k: i for i, k in enumerate(DATASETS)}
    reg["corpora"].sort(key=lambda c: order.get(c["id"], 99))
    reg.setdefault("default", default)
    reg["note"] = "corpora of the demo; the page reads data/<id>/meta.json, URL parameter ds=<id> selects one"
    p.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def calib_description(tag: str, prompt_short: str = "E5 query format") -> str:
    m = re.match(r"Q\d_K-([\w-]+?_(corpus_only|synth_only|corpus_synth))-ps(\d+)-t(\d+)k-ao-tab", tag)
    if not m:
        return tag
    calib, kind, ps, tk = m.group(1), m.group(2), m.group(3), m.group(4)
    what = {"corpus_only": "corpus documents (512-token windows)", "synth_only": f"synthetic queries (doc2query, Qwen3-1.7B) in the {prompt_short}",
            "corpus_synth": "corpus documents + synthetic queries"}[kind]
    return f"{calib}: {what}, per-sample, act-order, token budget {tk}k"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=sorted(DATASETS))
    ap.add_argument("--teacher", default="harrier-0.6b", help="eq.teacher.TEACHERS key of the fp32 index and the client (prompt, model texts, file prefix)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--id", default=None, help="corpus id in the demo (directory demo/data/<id>, demo/model/<id>, registry entry); default = dataset name. Use e.g. scidocs-qwen3 for a second teacher over the same dataset (its own fp32 index)")
    ap.add_argument("--client", "--model", dest="client", default=None, help="the GGUF the demo loads for this corpus (default: DEFAULT_CLIENT[dataset])")
    ap.add_argument("--chunk_mib", type=int, default=24, help="byte chunk size for demo/model/<ds>/ (Cloudflare Pages: 25 MiB per file)")
    ap.add_argument("--no_model", action="store_true", help="skip cutting the GGUF (keep the chunks already in demo/model/<ds>/)")
    ap.add_argument("--index_dtype", default="auto", choices=["auto", "f16", "int8"], help="auto = int8 if the nDCG@10 check passes, else f16")
    ap.add_argument("--int8_tol", type=float, default=0.001)
    ap.add_argument("--out_root", type=Path, default=ROOT / "demo/data")
    ap.add_argument("--model_root", type=Path, default=ROOT / "demo/model")
    args = ap.parse_args()
    dsname = args.dataset
    info = dict(DATASETS[dsname])
    spec = TEACHERS[args.teacher]
    model_key = spec.get("docs_from", args.teacher)   # "-tp" keys share the files of their base key (gptq_export_gguf.py)
    tinfo = TEACHER_INFO.get(model_key) or dict(
        model_name=f"{model_key} ({spec['hf']}), query side, GPTQ on the llama.cpp Q2_K grid", base_model=spec["hf"],
        base_model_url=f"https://huggingface.co/{spec['hf']}", base_model_license="see the model card",
        license_notice="Evaluation only. Model weights are not licensed for redistribution; base model " + spec["hf"] + " (see its model card), "
                       "runtime llama.cpp/wllama MIT, corpus {corpus}.",
        prompt_kind="the teacher's own query prefix (protocol of the native reference)", prompt_short="teacher query prefix",
        fp16_size_mib=None, native_latency_ms=None, native_peak_rss_mib=None, native_latency_note=None)
    q_prompt = query_prompt(spec, dsname)  # E5 instruction for harrier, "Query: " for jina: what the page prepends to the query
    cid = args.id or dsname
    out = args.out_root / cid; out.mkdir(parents=True, exist_ok=True)
    model_out = args.model_root / cid
    client = ROOT / (args.client or DEFAULT_CLIENT[dsname])

    emb_dir = ROOT / "data/emb" / dsname / args.teacher
    ids = json.loads((emb_dir / "ids.json").read_text(encoding="utf-8"))
    emb_meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    assert emb_meta.get("q_prompt", q_prompt) == q_prompt, f"the fp32 query cache used a different instruction: {emb_meta.get('q_prompt')!r} != {q_prompt!r}"
    D = l2n(np.load(emb_dir / "corpus.npy").astype(np.float32))
    Q = l2n(np.load(emb_dir / "queries.npy").astype(np.float32))
    doc_ids = ids["doc_ids"]
    assert D.shape[0] == len(doc_ids) and Q.shape[0] == len(ids["query_ids"]), "index / ids mismatch"

    ds = load_dataset(dsname)
    qi = {q: i for i, q in enumerate(ids["query_ids"])}
    test_q = [q for q in ds.splits[args.split] if q in qi]  # same filter and order as quant_eval_queries.py
    nd_fp32 = ndcg_test(ds, Q, ids["query_ids"], D, doc_ids, test_q)

    # ---- index: fp16 or row-scaled int8, decided by the nDCG@10 check on this corpus' test queries ----
    D16 = D.astype("<f2"); back16 = D16.astype(np.float32)
    nd_f16 = ndcg_test(ds, Q, ids["query_ids"], back16, doc_ids, test_q)
    scale = (np.abs(D).max(axis=1) / 127.0).astype(np.float32)
    q8 = np.clip(np.round(D / np.maximum(scale, 1e-12)[:, None]), -127, 127).astype(np.int8)
    back8 = q8.astype(np.float32) * scale[:, None]
    nd_i8 = ndcg_test(ds, Q, ids["query_ids"], back8, doc_ids, test_q)
    cos8 = float(np.mean(np.sum(back8 * D, 1) / np.linalg.norm(back8, axis=1)))
    check = {"n_test_queries": len(test_q), "ndcg10_fp32": round(nd_fp32, 6), "ndcg10_f16": round(nd_f16, 6), "ndcg10_int8": round(nd_i8, 6),
             "delta_int8_vs_fp32": round(nd_i8 - nd_fp32, 6), "delta_f16_vs_fp32": round(nd_f16 - nd_fp32, 6),
             "int8_mean_cos_to_fp32_row": round(cos8, 6), "int8_max_abs_err": float(np.abs(back8 - D).max()),
             "f16_max_abs_err": float(np.abs(back16 - D).max()), "tolerance": args.int8_tol,
             "protocol": "fp32 query embeddings of the same model (data/emb cache) vs the fp32 / fp16 / int8-dequantised index, test split, nDCG@10"}
    use_int8 = args.index_dtype == "int8" or (args.index_dtype == "auto" and abs(nd_i8 - nd_fp32) < args.int8_tol)
    check["decision"] = "int8_rowscale" if use_int8 else "float16"
    print(f"index {D.shape}: nDCG@10 test fp32 {nd_fp32:.6f}  f16 {nd_f16:.6f} ({nd_f16 - nd_fp32:+.6f})  int8 {nd_i8:.6f} ({nd_i8 - nd_fp32:+.6f}) -> {check['decision']}")
    for f in list(out.glob("corpus.*")):
        f.unlink()
    if use_int8:
        shards = write_shards(q8, out, "corpus", "i8")
        scale.astype("<f4").tofile(out / "corpus.scale.f32")
        index = {"dtype_on_disk": "int8_rowscale", "files": shards, "scale_file": "corpus.scale.f32",
                 "note": "the unchanged fp32 document index of the original model, stored as row-wise symmetric int8: x ~= q * scale[row]"}
    else:
        shards = write_shards(D16, out, "corpus", "f16")
        index = {"dtype_on_disk": "float16", "files": shards, "note": "the unchanged fp32 document index of the original model, stored as fp16"}
    index.update({"source_model": args.teacher, "dim": int(D.shape[1]), "n_docs": int(D.shape[0]), "rows_l2_normalised": True, "check": check})

    # ---- documents (sharded) and test queries ----
    for f in list(out.glob("docs.*.json")) + [out / "docs.json"]:
        if f.exists():
            f.unlink()
    docs_files = []
    for i, s in enumerate(range(0, len(doc_ids), DOCS_PER_SHARD)):
        part = [{"id": d, "title": ds.corpus[d]["title"], "text": ds.corpus[d]["text"]} for d in doc_ids[s:s + DOCS_PER_SHARD]]
        name = f"docs.{i:03d}.json"
        (out / name).write_text(json.dumps(part, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        docs_files.append({"url": name, "rows": len(part), "first_index": s})
    queries = [{"qid": q, "text": ds.queries[q], "rel": {d: int(g) for d, g in ds.qrels.get(q, {}).items() if g > 0}} for q in test_q]
    (out / "test_queries.json").write_text(json.dumps(queries, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ---- model chunks ----
    size = client.stat().st_size if client.exists() else None
    if size is None:
        print(f"WARNING: {client} not found; size_bytes left null")
    gguf_name = client.name
    manifest = None
    if size is not None and not args.no_model:
        manifest = chunk_model(client, model_out, args.chunk_mib)
    elif (model_out / f"{gguf_name}.chunks.json").exists():
        manifest = json.loads((model_out / f"{gguf_name}.chunks.json").read_text(encoding="utf-8"))
    tag = gguf_name.replace(f"{model_key}-gptq-", "").replace(".gguf", "")
    ref = reference_numbers(gguf_name, dsname, tinfo["prompt_short"])
    info["long"] = info["long"].replace("{n}", f"{D.shape[0]:,}".replace(",", " "))
    fp16_mib = tinfo.get("fp16_size_mib")
    if fp16_mib is None and spec.get("gguf_f16") and (ROOT / spec["gguf_f16"]).exists():
        fp16_mib = round((ROOT / spec["gguf_f16"]).stat().st_size / 2**20)
    meta = {
        "title": "Thinletter demo", "dataset": dsname, "label": info["label"], "dataset_long": info["long"],
        "corpus_description": info["corpus_description"], "unit": info["unit"], "query_hint": info["query_hint"], "query_kind": info["query_kind"],
        "dataset_license": info["dataset_license"], "dataset_url": info["dataset_url"], "dataset_name": info["dataset_name"], "task": info["task"],
        "corpus_notice": info.get("corpus_notice"), "teacher": args.teacher,
        "model_name": tinfo["model_name"], "model_short": tinfo.get("model_short", model_key),
        "base_model": tinfo["base_model"], "base_model_url": tinfo.get("base_model_url"), "base_model_license": tinfo["base_model_license"],
        "file": gguf_name, "size_bytes": size, "size_mib": round(size / 2**20, 1) if size else None,
        "model_url": f"/model/{cid}/{gguf_name}.chunks.json",
        "model_url_note": "same-origin byte-chunk manifest (see demo/model/<ds>/); ?model=<url>.gguf or ?model=<url>.chunks.json overrides it",
        "model_sha256": manifest["sha256"] if manifest else None, "model_chunks": len(manifest["chunks"]) if manifest else None,
        "license_notice": tinfo["license_notice"].replace("{corpus}", info["dataset_license"]),
        "quantization": {"type": "Q2_K", "method": "GPTQ (sequential, act-order) onto the llama.cpp K-quant grid",
                         "calibration": calib_description(tag, "E5 query format" if model_key == "harrier-0.6b" else tinfo["prompt_short"]),
                         "token_table": "Q2_K", "tag": tag},
        "runtime": {"engine": "llama.cpp compiled to WebAssembly (wllama 3.6.1)", "pooling": "last", "n_ctx": 512, "n_batch": 512,
                    "embeddings_normalised": True, "default_threads": 8},
        "prompt": q_prompt, "prompt_kind": tinfo["prompt_kind"],
        "split": args.split, "index": index, "docs_files": docs_files, "n_queries": len(queries),
        "reference": {**ref, "fp16_size_mib": fp16_mib, "native_latency_ms": tinfo.get("native_latency_ms"),
                      "native_latency_note": tinfo.get("native_latency_note"), "native_peak_rss_mib": tinfo.get("native_peak_rss_mib")},
        "date": dt.date.today().isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    label = info["label"] if cid == dsname else f"{info['label']} · {tinfo.get('model_short', model_key)}"
    update_registry(args.out_root, {"id": cid, "label": label, "long": info["long"] + ("" if cid == dsname else f" — index of {model_key}"), "n_docs": int(D.shape[0]), "n_queries": len(queries),
                                    "model_file": gguf_name, "size_bytes": size, "index_dtype": index["dtype_on_disk"]})

    ok = True
    files = sorted(out.iterdir()) + (sorted(model_out.iterdir()) if model_out.exists() else [])
    total = 0
    for f in files:
        s = f.stat().st_size; total += s
        flag = "" if s < PAGES_FILE_LIMIT else "  <-- OVER the 25 MiB Pages limit"
        ok &= s < PAGES_FILE_LIMIT
        print(f"  {f.name:70s} {s / 2**20:7.2f} MiB{flag}")
    n_rel = sum(len(q["rel"]) for q in queries)
    print(f"wrote {len(queries)} queries ({n_rel} qrels), {len(doc_ids)} docs in {len(docs_files)} shards, {D.shape[0]}x{D.shape[1]} index "
          f"({index['dtype_on_disk']}, {len(index['files'])} shards) -> {out}; total {total / 2**20:.1f} MiB; "
          f"native {ref.get('native_ndcg10')}, browser {ref.get('browser_ndcg10')}, fp {ref.get('fp16_ndcg10')}")
    if not ok:
        raise SystemExit("a file exceeds the Cloudflare Pages 25 MiB per-file limit")


if __name__ == "__main__":
    main()
