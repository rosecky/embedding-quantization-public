"""Build (or refresh) the cached teacher embeddings for a dataset: data/emb/<dataset>/<teacher>/{corpus,queries}.npy.

Everything downstream (quantization evaluation, navigator, ablations) reads this cache, so this is the one GPU step
a new dataset needs before it can be used.

Usage: python scripts/build_teacher_cache.py --datasets webfaq-cs --teacher harrier-0.6b [--batch_size 32]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq.data import load_dataset  # noqa: E402
from eq.teacher import encode_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--teacher", default="harrier-0.6b")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_vram_gb", type=float, default=8.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for d in args.datasets:
        t0 = time.time()
        ds = load_dataset(d)
        print(f"[{d}] corpus {len(ds.corpus)} docs, {len(ds.queries)} queries, splits " +
              ", ".join(f"{k}={len(v)}" for k, v in ds.splits.items()), flush=True)
        te = encode_dataset(ds, args.teacher, batch_size=args.batch_size, max_vram_gb=args.max_vram_gb, force=args.force)
        print(f"[{d}] {args.teacher}: docs {te.D.shape} queries {te.Q.shape} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
