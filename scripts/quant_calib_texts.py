"""Calibration texts for corpus-fitted post-training quantization (llama.cpp importance matrix).

Writes plain-text files (one document / query per paragraph, blank-line separated):
  data/calib/<dataset>_corpus_synth.txt   corpus documents (title + text, sampled) + synthetic queries with the E5 prompt
  data/calib/generic_wikitext.txt         generic text (wikitext-2-raw-v1, train split, first N chars) as the control

Usage: python scripts/quant_calib_texts.py --datasets scifact nfcorpus --max_docs 2000 --max_synth 3000
       python scripts/quant_calib_texts.py --datasets legal-cs --teacher jina-v5-small   # "Query: " / "Document: " prefixes
--teacher selects the prompt prefixes the calibration texts carry (default harrier-0.6b: E5 instruction on queries, none on
documents -> byte-identical to the historical files); --tag appends a suffix to the set names when two teachers share a dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq.data import load_dataset  # noqa: E402
from eq.teacher import E5_QUERY_PROMPT, TEACHERS, query_prompt, doc_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--max_docs", type=int, default=2000)
    ap.add_argument("--max_synth", type=int, default=3000)
    ap.add_argument("--max_doc_chars", type=int, default=1500)
    ap.add_argument("--generic_chars", type=int, default=3_000_000)
    ap.add_argument("--force_generic", action="store_true", help="rebuild generic_wikitext.txt even if a full one exists (default: never truncate an existing one)")
    ap.add_argument("--teacher", default="harrier-0.6b", help="eq.teacher.TEACHERS key whose query/document prefixes the texts carry")
    ap.add_argument("--tag", default="", help="suffix for the set names, e.g. _jina -> <dataset>_corpus_only_jina.txt")
    args = ap.parse_args()
    spec = TEACHERS[args.teacher]
    dp = doc_prompt(spec)
    out = ROOT / "data/calib"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for d in args.datasets:
        ds = load_dataset(d)
        ids = sorted(ds.corpus); ids = [ids[i] for i in sorted(rng.choice(len(ids), min(args.max_docs, len(ids)), replace=False))]
        docs = [dp + ((ds.corpus[i].get("title") or "") + "\n" + (ds.corpus[i].get("text") or "")).strip()[: args.max_doc_chars] for i in ids]
        qp = query_prompt(spec, d)
        sp = ROOT / "data/synth" / d / "queries.jsonl"
        synth = [json.loads(l)["query"] for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()] if sp.exists() else []
        if synth:
            synth = [synth[i] for i in sorted(rng.choice(len(synth), min(args.max_synth, len(synth)), replace=False))]
        else:
            print(f"[{d}] no synthetic queries at {sp}: writing the corpus-only calibration set")
        qparts = [qp + q for q in synth]
        sets = [(f"{d}_corpus_only{args.tag}", list(docs))] + ([(f"{d}_corpus_synth{args.tag}", docs + qparts), (f"{d}_synth_only{args.tag}", list(qparts))] if qparts else [])
        for name, parts in sets:
            parts = list(parts); rng.shuffle(parts)
            p = out / f"{name}.txt"
            p.write_text("\n\n".join(x.replace("\r", "") for x in parts) + "\n", encoding="utf-8")
            print(f"[{d}] {name}: {len(parts)} parts -> {p} ({p.stat().st_size/2**20:.1f} MB)")
    short_path = out / "generic_short.txt"
    gen_path = out / "generic_wikitext.txt"
    if gen_path.exists() and gen_path.stat().st_size > 1_000_000 and not short_path.exists():
        # generic CONTENT in query SHAPE: isolates "which text" from "how long the sequences are"
        import re as _re
        txt = gen_path.read_text(encoding="utf-8")
        sents = [x.strip() for x in _re.split(r"(?<=[.!?])\s+", txt) if 5 <= len(x.split()) <= 30]
        sents = [sents[i] for i in sorted(rng.choice(len(sents), min(args.max_synth, len(sents)), replace=False))]
        short_path.write_text("\n\n".join(query_prompt(spec) + x for x in sents) + "\n", encoding="utf-8")
        print(f"[generic] query-shaped control: {len(sents)} sentences -> {short_path} ({short_path.stat().st_size/2**20:.1f} MB)")
    if gen_path.exists() and gen_path.stat().st_size > 1_000_000 and not args.force_generic:
        print(f"[generic] keeping existing {gen_path} ({gen_path.stat().st_size/2**20:.1f} MB); use --force_generic to rebuild")
        return
    try:
        from datasets import load_dataset as hf_load
        try:
            wt = hf_load("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        except Exception:  # noqa: BLE001
            wt = hf_load("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in wt["text"] if t.strip())[: args.generic_chars]
        p = out / "generic_wikitext.txt"; p.write_text(text, encoding="utf-8")
        print(f"[generic] wikitext-2 train -> {p} ({p.stat().st_size/2**20:.1f} MB)")
    except Exception as e:  # noqa: BLE001
        print(f"[generic] wikitext unavailable: {e}")


if __name__ == "__main__":
    main()
