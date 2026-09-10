"""Retrieval quality of a QUANTIZED query encoder (llama.cpp GGUF) against the full-precision document index of the same
model (E10b: corpus-fitted post-training quantization of the server model as the client).

For each GGUF: encode the test queries with llama-embedding (pooling last, E5 instruction, L2-normalised), score
against the cached fp32 document embeddings of the teacher key (default harrier-0.6b), and report GT nDCG@10 / R@100,
cosine to the fp query embeddings and top-10 overlap with the fp model.  ArguAna self-document is ignored (MTEB).
Optionally (--both_sides) the documents are encoded with the same GGUF too (pure "quantized model on both sides").

Usage: python scripts/quant_eval_queries.py --teacher harrier-0.6b --datasets scifact nfcorpus \
           --ggufs models/gguf/harrier-0.6b-IQ2_M-corpus.gguf models/gguf/harrier-0.6b-IQ2_M-generic.gguf --threads 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eq import graph_metrics as G  # noqa: E402
from eq.data import load_dataset  # noqa: E402
from eq.teacher import E5_QUERY_PROMPT, TEACHERS, task_query_prompt, query_prompt, doc_prompt  # noqa: E402
from eq.verify import load_emb, rel_matrix, self_mask, mask_self, l2n, encode_runtime  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def truncate_512(texts, teacher, max_len=512):
    """Cut every text to the teacher's context (the fp document cache was built the same way)."""
    from transformers import AutoTokenizer
    spec = TEACHERS[teacher]
    local = ROOT / "models/hf" / spec["hf"].split("/")[-1]
    tok = AutoTokenizer.from_pretrained(str(local) if (local / "config.json").exists() else spec["hf"])
    out = []
    for i in range(0, len(texts), 256):
        enc = tok(texts[i:i + 256], truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
        out += tok.batch_decode(enc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="harrier-0.6b")
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--ggufs", nargs="+", required=True)
    ap.add_argument("--bin", type=Path, default=ROOT / "third_party/BitNet/build/bin/llama-embedding")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--both_sides", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["test"], help="query splits to evaluate (calibration never uses real queries except the real_queries control)")
    ap.add_argument("--out", default="results/raw/quant_pt")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fout = open(out_dir / "results.jsonl", "a", encoding="utf-8")
    spec = TEACHERS[args.teacher]
    for d in args.datasets:
        ds = load_dataset(d)
        tdoc, D_T, tq, Q_T = load_emb(d, args.teacher)
        tdi = {x: i for i, x in enumerate(tdoc)}; tqi = {q: i for i, q in enumerate(tq)}
        te = [q for sp in args.splits for q in ds.splits.get(sp, []) if q in tqi]
        rel_te = rel_matrix(te, tdi, ds.qrels, len(tdoc)); msk = self_mask(te, tdi, len(tdoc))
        Qte = Q_T[[tqi[q] for q in te]]
        S0 = mask_self(Qte @ D_T.T, msk)
        qp = query_prompt(spec, d)  # teacher-specific prefix (E5 instruction for harrier, "Query: " for jina, MTEB task instruction for -tp keys)
        prompts = [qp + ds.queries[q] for q in te]
        nd0 = float(np.nanmean(G.ndcg_at_k(S0, rel_te, 10)))
        print(f"[{d}] fp teacher {args.teacher}: nDCG@10={nd0:.4f}", flush=True)
        for g in args.ggufs:
            g = Path(g); tag = g.stem
            sp_tag = "" if args.splits == ["test"] else "_" + "+".join(args.splits)  # chunk cache is keyed by the query set
            work = ROOT / "data/emb" / d / f"quant_{tag}{sp_tag}" / "chunks"
            t0 = time.time()
            pooling = spec.get("pooling", "last")  # last-token for the Qwen3 decoders, cls for bge-m3, mean for multilingual-e5
            Qq, stats = encode_runtime(args.bin, g, prompts, work, "q", args.threads, args.workers, 400, args.ctx, pooling)
            Qq = Qq.astype(np.float32)
            rot_map = g.with_suffix(".rot.npy")  # fused Hadamard rotation: the file's embeddings live in a rotated frame
            A = np.load(rot_map).astype(np.float32) if rot_map.exists() else None
            if A is not None:
                Qq = Qq @ A
                print(f"    [rot] applied output map {rot_map.name} {A.shape}", flush=True)
            Qq = l2n(Qq)
            if Qq.shape[1] != D_T.shape[1]:
                raise SystemExit(f"dim mismatch {Qq.shape} vs docs {D_T.shape}")
            Dq = D_T
            if args.both_sides:
                docs = [doc_prompt(spec) + ((ds.corpus[x].get("title") or "") + "\n" + (ds.corpus[x].get("text") or "")).strip() for x in tdoc]
                Dq, _ = encode_runtime(args.bin, g, docs, ROOT / "data/emb" / d / f"quant_{tag}" / "chunks_docs", "d", args.threads, args.workers, 400, args.ctx, pooling)
                Dq = Dq.astype(np.float32)
                if A is not None:
                    Dq = Dq @ A
                Dq = l2n(Dq)
            S = mask_self(Qq @ Dq.T, msk)
            row = dict(dataset=d, teacher=args.teacher, splits="+".join(args.splits), rotated=bool(A is not None), gguf=g.name, gguf_mib=g.stat().st_size / 2**20, both_sides=args.both_sides,
                       gt_ndcg10=float(np.nanmean(G.ndcg_at_k(S, rel_te, 10))), gt_recall100=float(np.nanmean(G.recall_at_k(S, rel_te, 100))),
                       fp_ndcg10=nd0, q_cos_fp=float(np.mean(np.sum(Qq * Qte, 1))), fp_top10_overlap=float(G.topk_overlap(S0, S, 10).mean()),
                       n_test=len(te), encode_s=time.time() - t0, attribution="OUR_MEASUREMENT")
            perq = out_dir / "perq"; perq.mkdir(exist_ok=True)  # per-query nDCG for paired bootstrap between variants
            np.savez(perq / f"{d}_{tag}{sp_tag}{'_both' if args.both_sides else ''}.npz", ndcg=G.ndcg_at_k(S, rel_te, 10), ndcg_fp=G.ndcg_at_k(S0, rel_te, 10))
            fout.write(json.dumps(row) + "\n"); fout.flush()
            print(f"[{d}] {g.name:45s} {row['gguf_mib']:.0f} MiB nDCG@10={row['gt_ndcg10']:.4f} ({row['gt_ndcg10']/nd0*100:.1f} % fp) cos={row['q_cos_fp']:.4f} top10ov={row['fp_top10_overlap']:.3f}", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
