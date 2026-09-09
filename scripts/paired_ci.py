"""Paired bootstrap over queries between two per-query nDCG files (results/raw/<dir>/perq/<dataset>_<gguf stem>.npz).

Usage: python scripts/paired_ci.py A.npz B.npz [--n 10000 --seed 0]     -> mean(A-B) and the 95 % percentile CI
The two files must come from the same query set in the same order (quant_eval_queries.py writes them that way).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from eq.graph_metrics import paired_bootstrap  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--n", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
a, b = np.load(args.a)["ndcg"], np.load(args.b)["ndcg"]
assert a.shape == b.shape, (a.shape, b.shape)
d, lo, hi = paired_bootstrap(a, b, n=args.n, seed=args.seed)
print(f"n={len(a)}  A={np.nanmean(a):.4f}  B={np.nanmean(b):.4f}  A-B={d:+.4f} [{lo:+.4f}; {hi:+.4f}]  {'CI excludes 0' if lo > 0 or hi < 0 else 'n.s.'}")
