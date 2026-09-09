"""doc2query: generate synthetic search queries for every document of a dataset with a small local LLM
(default Qwen/Qwen3-1.7B, bf16, HF generate, batched). Output: data/synth/<dataset>/queries.jsonl with
{"doc_id":..., "query":..., "k": i}. Used as extra query-side supervision for small client query towers.

Usage: python scripts/doc2query.py --datasets scifact nfcorpus arguana --n_per_doc 3 --max_docs 0
       python scripts/doc2query.py --datasets legal-cs --lang Czech      # queries in the corpus language (Qwen3-1.7B handles Czech)
Teacher-independent: reads only data/<dataset>/corpus.jsonl.gz (an imported customer corpus works unchanged).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq.data import load_dataset  # noqa: E402

PROMPT = ("You write short search queries. Given the passage below, write {n} different natural-language search queries "
          "(questions or keyword phrases, max 12 words each) that a user could type into a search engine and for which "
          "this passage would be a relevant result. Output exactly {n} lines, one query per line, no numbering.\n\n"
          "Passage:\n{doc}\n\nQueries:\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "arguana"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n_per_doc", type=int, default=3)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max_doc_chars", type=int, default=1500)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--lang", default=None, help="language the queries must be written in (default: unchanged prompt, i.e. the passage's language is not enforced)")
    ap.add_argument("--doc_ids", type=Path, default=None, help="json list of document ids: only these documents get queries (default: all documents of the dataset)")
    args = ap.parse_args()
    only_ids = None
    if args.doc_ids is not None:
        only_ids = {str(x) for x in json.loads(args.doc_ids.read_text(encoding="utf-8"))}
    prompt_tpl = PROMPT if not args.lang else PROMPT.replace("no numbering.\n\n", f"no numbering. Write the queries in {args.lang}.\n\n")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for d in args.datasets:
        ds = load_dataset(d)
        out_dir = ROOT / "data/synth" / d; out_dir.mkdir(parents=True, exist_ok=True)
        out_f = out_dir / "queries.jsonl"
        done = set()
        if out_f.exists():
            for line in out_f.read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(line)["doc_id"])
                except Exception:  # noqa: BLE001
                    pass
        doc_ids = [x for x in sorted(ds.corpus) if x not in done]
        if only_ids is not None:
            missing = only_ids - set(ds.corpus)
            if missing:
                print(f"[{d}] WARNING: {len(missing)} ids of --doc_ids are not in the corpus (first: {sorted(missing)[:3]})", flush=True)
            doc_ids = [x for x in doc_ids if x in only_ids]
            print(f"[{d}] restricted to {len(only_ids)} ids from {args.doc_ids}", flush=True)
        if args.max_docs:
            doc_ids = doc_ids[: args.max_docs]
        print(f"[{d}] {len(doc_ids)} docs to do ({len(done)} done)", flush=True)
        f = open(out_f, "a", encoding="utf-8")
        t0 = time.time()
        for s in range(0, len(doc_ids), args.batch):
            ids = doc_ids[s: s + args.batch]
            prompts = []
            for did in ids:
                r = ds.corpus[did]
                text = ((r.get("title") or "").strip() + "\n" + (r.get("text") or "").strip()).strip()[: args.max_doc_chars]
                msgs = [{"role": "user", "content": prompt_tpl.format(n=args.n_per_doc, doc=text)}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for did, t in zip(ids, texts):
                qs = [q.strip(" -•*0123456789.)\t") for q in t.strip().split("\n")]
                qs = [q for q in qs if 3 <= len(q) <= 200][: args.n_per_doc]
                for i, q in enumerate(qs):
                    f.write(json.dumps({"doc_id": did, "query": q, "k": i}, ensure_ascii=False) + "\n")
            f.flush()
            if (s // args.batch) % 20 == 0:
                print(f"[{d}] {s + len(ids)}/{len(doc_ids)} docs, {time.time() - t0:.0f}s", flush=True)
        f.close()
        print(f"[{d}] done -> {out_f}", flush=True)


if __name__ == "__main__":
    main()
