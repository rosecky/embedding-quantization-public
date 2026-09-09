"""Import a customer-provided corpus (and, optionally, its existing document index) as a dataset of this repo.

Writes the five files eq.data.load_dataset expects under data/<name>/ and, when the customer's own document vectors are
given, the teacher cache data/emb/<name>/<teacher>/{corpus.npy, ids.json, meta.json, queries.npy} directly from those
vectors, so the index is never re-encoded (the fp32 document index the client is measured against IS the customer's
index).  Queries, if any, are written as the "test" split; the query vectors are encoded later by
scripts/build_teacher_cache.py (eq.teacher.encode_dataset reuses the imported corpus.npy and encodes only the queries).

Relevance judgements: the customer has none, so qrels.tsv is written EMPTY (header only) unless --query_emb is given, in
which case the top-K documents of every query under the customer's own fp32 vectors are written as pseudo-relevance
(grade 1, K = --pseudo_k).  With empty qrels the nDCG/recall columns of the evaluation scripts are NaN; the qrels-free
readouts (cos(q_client, q_fp32), top-10 overlap with the fp32 ranking, and the paired contrasts in
scripts/quant_eval_queries.py) are what a corpus without labels can support.  Pseudo-qrels from the fp32 model's top-10
turn "agreement with the fp32 ranking" into the usual nDCG@10 number; they are NOT ground truth and must be labelled as
pseudo-relevance wherever they are reported.  A later scripts/build_teacher_cache.py run can also produce them once
query vectors exist (see --pseudo_k).

Inputs
  --docs     jsonl, one {"id": ..., "title": ..., "text": ...} per line ("_id" accepted; title optional)
  --doc_emb  document vectors: .npy (N, d) or raw little-endian .f16 / .f32 (N*d values; d = --dim or inferred from N)
  --doc_ids  json list of N ids in the row order of --doc_emb (default: the order of --docs)
  --queries  jsonl, one {"id": ..., "text": ...} per line (optional)
  --query_emb / --query_ids  same as for documents (optional, only for pseudo-qrels + a complete cache)

Usage
  python scripts/import_corpus.py --name legal-cs --docs corpus.jsonl --doc_emb index.f32 --doc_ids ids.json \
         --queries queries.jsonl --teacher jina-v5-small --language cs
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq.data import RetrievalDataset, save_dataset, load_dataset  # noqa: E402
from eq.teacher import TEACHERS, emb_dir, load_teacher_embeddings, query_prompt, doc_prompt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl(path: Path, text_key: str = "text") -> list[dict]:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r.get("id", r.get("_id"))
            if rid is None:
                raise SystemExit(f"{path}:{ln}: no 'id' / '_id' field")
            rows.append({"id": str(rid), "title": (r.get("title") or "").strip(), "text": (r.get(text_key) or "").strip()})
    return rows


def load_vectors(path: Path, n: int, dim: int | None) -> np.ndarray:
    """(n, dim) float32 from .npy or a raw .f16/.f32 little-endian dump."""
    if path.suffix == ".npy":
        X = np.load(path)
    else:
        dt = {".f16": "<f2", ".f32": "<f4"}.get(path.suffix)
        if dt is None:
            raise SystemExit(f"{path}: expected .npy, .f16 or .f32")
        raw = np.fromfile(path, dtype=dt)
        if dim is None:
            if raw.size % n:
                raise SystemExit(f"{path}: {raw.size} values are not a multiple of {n} rows; pass --dim")
            dim = raw.size // n
        X = raw.reshape(-1, dim)
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != n:
        raise SystemExit(f"{path}: shape {X.shape}, expected ({n}, d)")
    return X


def l2n(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(nrm, 1e-12)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="dataset name, e.g. legal-cs -> data/legal-cs/")
    ap.add_argument("--docs", required=True, type=Path)
    ap.add_argument("--doc_emb", type=Path, default=None)
    ap.add_argument("--doc_ids", type=Path, default=None)
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--query_emb", type=Path, default=None)
    ap.add_argument("--query_ids", type=Path, default=None)
    ap.add_argument("--dim", type=int, default=None, help="vector width of raw .f16/.f32 dumps (default: inferred from the row count)")
    ap.add_argument("--teacher", default="jina-v5-small", help="eq.teacher.TEACHERS key the customer's vectors come from")
    ap.add_argument("--pseudo_k", type=int, default=10, help="pseudo-relevance depth when both --doc_emb and --query_emb are given")
    ap.add_argument("--language", default="cs")
    ap.add_argument("--source", default="customer corpus (private)")
    ap.add_argument("--license", default="customer data, not redistributable")
    ap.add_argument("--root", default="data")
    ap.add_argument("--force", action="store_true", help="overwrite an existing data/<name>/ and data/emb/<name>/<teacher>/")
    args = ap.parse_args()
    spec = TEACHERS[args.teacher]
    root = ROOT / args.root
    ds_dir = root / args.name
    if ds_dir.exists() and not args.force:
        raise SystemExit(f"{ds_dir} exists; pass --force to overwrite")

    # ---- texts
    docs = read_jsonl(args.docs)
    ids = [d["id"] for d in docs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{args.docs}: {len(ids) - len(set(ids))} duplicate document ids")
    corpus = {d["id"]: {"title": d["title"], "text": d["text"]} for d in docs}
    empty = [i for i, d in corpus.items() if not (d["title"] or d["text"])]
    if empty:
        print(f"WARNING: {len(empty)} documents have neither title nor text (kept; they will embed as the bare prompt)")
    queries: dict[str, str] = {}
    if args.queries:
        qrows = read_jsonl(args.queries)
        queries = {q["id"]: q["text"] for q in qrows}
        if len(queries) != len(qrows):
            raise SystemExit(f"{args.queries}: duplicate query ids")
        overlap = set(queries) & set(corpus)
        if overlap:
            raise SystemExit(f"{len(overlap)} ids are both a query and a document id; make them distinct")
    doc_order = sorted(corpus)  # eq.teacher.encode_dataset order: the cache rows follow sorted doc ids
    q_order = sorted(queries)

    # ---- vectors
    D = Q = None
    if args.doc_emb:
        emb_ids = json.loads(args.doc_ids.read_text(encoding="utf-8")) if args.doc_ids else ids
        emb_ids = [str(x) for x in emb_ids]
        X = load_vectors(args.doc_emb, len(emb_ids), args.dim)
        pos = {x: i for i, x in enumerate(emb_ids)}
        missing = [i for i in doc_order if i not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} documents have no vector (first: {missing[:5]})")
        extra = len(pos) - len(doc_order)
        if extra:
            print(f"WARNING: {extra} vectors belong to no document in --docs and are dropped")
        D = X[[pos[i] for i in doc_order]]
        want = spec.get("dim")
        if want and D.shape[1] != want:
            raise SystemExit(f"vectors are {D.shape[1]}-d but teacher {args.teacher} is {want}-d: a matryoshka-truncated index "
                             f"needs the same truncation on the query side, which the evaluation scripts do not do yet")
        nrm = np.linalg.norm(D, axis=1)
        print(f"document vectors: {D.shape}, row norms {nrm.min():.4f}..{nrm.max():.4f} -> L2-normalised; "
              f"{int(np.isnan(D).any(1).sum())} NaN rows, {int((nrm < 1e-6).sum())} zero rows")
        D = l2n(D)
    if args.query_emb:
        if not queries:
            raise SystemExit("--query_emb needs --queries")
        qe_ids = json.loads(args.query_ids.read_text(encoding="utf-8")) if args.query_ids else [q["id"] for q in qrows]
        qe_ids = [str(x) for x in qe_ids]
        X = load_vectors(args.query_emb, len(qe_ids), D.shape[1] if D is not None else args.dim)
        pos = {x: i for i, x in enumerate(qe_ids)}
        missing = [q for q in q_order if q not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} queries have no vector (first: {missing[:5]})")
        Q = l2n(X[[pos[q] for q in q_order]])

    # ---- qrels: none (customer has no labels) or pseudo-relevance from the customer's own fp32 vectors
    qrels: dict[str, dict[str, int]] = {}
    qrels_kind = "none"
    if D is not None and Q is not None and args.pseudo_k > 0:
        k = min(args.pseudo_k, len(doc_order))
        for qi, q in enumerate(q_order):
            s = D @ Q[qi]
            top = np.argpartition(-s, k - 1)[:k]
            qrels[q] = {doc_order[j]: 1 for j in top}
        qrels_kind = f"pseudo: top-{k} of the imported fp32 vectors (teacher {args.teacher}); NOT human judgements"
    splits = {"train": [], "dev": [], "test": q_order}
    meta = {
        "language": args.language, "source": args.source, "license": args.license,
        "notes": [f"imported by scripts/import_corpus.py from {args.docs.name}" + (f" + {args.queries.name}" if args.queries else ""),
                  f"qrels: {qrels_kind}",
                  "all real queries are in the test split (no train/dev: nothing is fitted on them; the ridge 'linfix' rung "
                  "of gptq_export_gguf.py needs train+test and is therefore skipped)"],
        "num_docs": len(corpus), "num_queries": {s: len(v) for s, v in splits.items()}, "num_queries_total": len(queries),
        "avg_rel_per_query": (float(np.mean([len(v) for v in qrels.values()])) if qrels else 0.0), "graded": False,
        "split_source": "imported: all queries -> test", "qrels_kind": qrels_kind,
        "dedup_exact_removed": 0, "qrels_dangling_doc_refs_dropped": 0, "orphaned_queries_dropped": 0,
        "near_dup_group_count": 0, "near_dup_groups": [], "query_equals_doc_count": 0, "query_equals_doc_ids": [],
        "test_queries_with_train_dup_relevant": [], "subsample": {"applied": False},
        "imported_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    ds = RetrievalDataset(name=args.name, corpus=corpus, queries=queries, qrels=qrels, splits=splits, meta=meta)
    out = save_dataset(ds, root=str(root))
    print(f"dataset -> {out}: {len(corpus)} docs, {len(queries)} queries (test), qrels: {qrels_kind}")

    # ---- teacher cache from the customer's vectors
    if D is not None:
        p = emb_dir(args.name, args.teacher, str(root))
        if p.exists() and not args.force and (p / "corpus.npy").exists():
            raise SystemExit(f"{p} exists; pass --force to overwrite")
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "corpus.npy", D)
        np.save(p / "queries.npy", Q if Q is not None else np.zeros((0, D.shape[1]), np.float32))
        (p / "ids.json").write_text(json.dumps({"doc_ids": doc_order, "query_ids": q_order if Q is not None else []}), encoding="utf-8")
        imported = dict(doc_emb=str(args.doc_emb), doc_ids=str(args.doc_ids) if args.doc_ids else None,
                        query_emb=str(args.query_emb) if args.query_emb else None, at=meta["imported_at"],
                        note="document vectors are the customer's own index (their jina deployment), L2-normalised, rows in sorted doc-id order")
        emeta = dict(teacher=args.teacher, hf=spec["hf"], dim=int(D.shape[1]), n_docs=len(doc_order),
                     n_queries=int(Q.shape[0]) if Q is not None else 0, device="imported", batch_size=None, max_len=spec["max_len"],
                     load_s=0.0, doc_encode_s=0.0, query_encode_s=0.0, docs_per_s=0.0, torch_threads=None, notes=spec["notes"],
                     q_prompt=query_prompt(spec, args.name), d_prompt=doc_prompt(spec), dtype="imported",
                     nan_docs=int(np.isnan(D).any(1).sum()), nan_queries=int(np.isnan(Q).any(1).sum()) if Q is not None else 0,
                     imported=imported)
        (p / "meta.json").write_text(json.dumps(emeta, indent=1), encoding="utf-8")
        print(f"teacher cache -> {p}: corpus.npy {D.shape}, queries.npy {(Q.shape if Q is not None else (0, D.shape[1]))}"
              + ("" if Q is not None else f"  (query vectors: run scripts/build_teacher_cache.py --datasets {args.name} --teacher {args.teacher})"))

    # ---- prove the files load
    chk = load_dataset(args.name, root=str(root))
    assert sorted(chk.corpus) == doc_order and sorted(chk.queries) == q_order and chk.splits["test"] == q_order
    if D is not None:
        te = load_teacher_embeddings(args.name, args.teacher, str(root))
        assert te.doc_ids == doc_order and te.D.shape == D.shape and np.allclose(te.D, D)
        assert te.query_ids == (q_order if Q is not None else []) and te.Q.shape[1] == D.shape[1]
    print("load_dataset + load_teacher_embeddings: OK")


if __name__ == "__main__":
    main()
