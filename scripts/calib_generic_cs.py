"""Czech GENERIC calibration set (data/calib/generic_cs.txt): encyclopaedic Czech prose, the language-matched control for
the "does corpus / synthetic-query calibration add anything over generic text" question on legal-cs (jina-v5-small).

Source: HF dataset wikimedia/wikipedia, config 20231101.cs, parquet shard train-00000-of-00004 (fetched with
huggingface_hub.hf_hub_download into the HF cache, read with pyarrow; no `datasets` streaming needed).
Sample rule (deterministic, seed 0): rng(0) permutation of the shard's articles; from each article at most ONE
paragraph, chosen by the same rng among its qualifying paragraphs; stop at --n_paras.  A paragraph = one non-empty
line of the article text with 100-400 whitespace words, ending in sentence punctuation, without wiki-table / list
markers ('|', '{{', leading '*', '-', '#', ':'), with < 10 % digit characters and without "may refer to"-style
disambiguation.  Output format = one paragraph per block, blank-line separated (same as generic_wikitext.txt).

Usage: .venv/Scripts/python.exe scripts/calib_generic_cs.py [--n_paras 3000] [--seed 0]
Prints the paragraph count, the jina (Qwen3) token count (raw and capped at 512 per paragraph, the per-sample cut of
scripts/gptq_export_gguf.py) and how many paragraphs fit the 300k-token budget.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
REPO, CFG, SHARD = "wikimedia/wikipedia", "20231101.cs", "train-00000-of-00004.parquet"
BAD_START = ("*", "-", "#", ":", "|", "{", "!", "=", "•")
END_OK = tuple(".!?\"“”»)")


def qualifies(line: str, lo: int, hi: int) -> bool:
    s = line.strip()
    if not s or s.startswith(BAD_START) or not s.endswith(END_OK):
        return False
    if "|" in s or "{{" in s or "}}" in s or "[[" in s or "\t" in s:
        return False
    w = s.split()
    if not (lo <= len(w) <= hi):
        return False
    if sum(c.isdigit() for c in s) > 0.10 * len(s):
        return False
    if "může znamenat" in s[:80] or "může být" in s[:40]:  # disambiguation-style leads
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_paras", type=int, default=3000)
    ap.add_argument("--min_words", type=int, default=100)
    ap.add_argument("--max_words", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/calib/generic_cs.txt")
    ap.add_argument("--tokenizer", default=str(ROOT / "models/hf/jina-embeddings-v5-text-small-retrieval"))
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(REPO, f"{CFG}/{SHARD}", repo_type="dataset")
    tbl = pq.read_table(path, columns=["id", "title", "text"])
    titles = tbl.column("title").to_pylist(); texts = tbl.column("text").to_pylist()
    print(f"[src] {REPO} {CFG}/{SHARD}: {len(texts)} articles ({Path(path).stat().st_size / 2**20:.0f} MiB)")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(texts))
    paras, used = [], 0
    for i in order:
        cand = [ln.strip() for ln in texts[i].split("\n") if qualifies(ln, args.min_words, args.max_words)]
        if not cand:
            continue
        paras.append(re.sub(r"\s+", " ", cand[int(rng.integers(len(cand)))]))
        used += 1
        if len(paras) >= args.n_paras:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(paras) + "\n", encoding="utf-8")
    words = [len(p.split()) for p in paras]
    print(f"[out] {args.out}: {len(paras)} paragraphs from {used} articles (articles scanned: {int(np.where(order == i)[0][0]) + 1}), "
          f"{args.out.stat().st_size / 2**20:.2f} MB, words/para mean {np.mean(words):.0f} (min {min(words)}, max {max(words)})")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    L = [len(x) for x in tok(paras, add_special_tokens=False)["input_ids"]]
    cap = [min(l, 512) for l in L]
    tot, n_fit = 0, 0
    for k in cap:  # the export's per-sample budget walk (scripts/gptq_harrier.py calib_batches with max_tokens)
        if tot + k > 300_000:
            break
        tot += k; n_fit += 1
    print(f"[tok] jina tokenizer: {sum(L):,} tokens raw, {sum(cap):,} capped at 512/para; tokens/para mean {np.mean(L):.0f}, "
          f"{sum(l > 512 for l in L)} paragraphs > 512 tokens; 300k budget = first {n_fit} paragraphs ({tot:,} tokens)".replace(",", " "))


if __name__ == "__main__":
    main()
