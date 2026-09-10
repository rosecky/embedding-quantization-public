"""Prefix KV in full precision for the quantised query encoder (pre-registered 2026-09-10, second probe of the softmax brainstorm).

The query prompt (E5 instruction for harrier, 'Query: ' for jina) is a constant text in front of every query.  Its keys and
values in every block can therefore be computed once by the fp model and shipped with the client (harrier: 28 blocks x 2 x
~20 tokens x 1024 x f16 ~ 2.3 MiB); the quantised model then only processes the user's tokens and attends to an exact prefix.
Mechanism this targets: the peer's decoder finding that 2-bit models fail to form the attention sink on some first tokens
(lab log 2026-09-09); here the first tokens are always the prefix.  Side effect: fewer tokens through the quantised model.

`encode_queries_prefix_kv(model_q, model_fp, ...)` is the simulation: fp prefix cache from model_fp, the rest of the tokens
through model_q.  With model_q = model_fp it must reproduce `gptq_harrier.encode_queries` (self-check in __main__).

Tokenisation: the prompt ends with a space that the BPE merges into the first query token, so the prefix is defined at the
TOKEN level as the longest common token prefix of all (prompt + query) encodings, not as tok(prompt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def split_prefix(tok, texts, max_len=512, min_prefix=2):
    """-> (prefix_ids: list[int], rest_ids: list[list[int]]) with tok(text) == prefix_ids + rest for every text."""
    ids = [tok(t, truncation=True, max_length=max_len)["input_ids"] for t in texts]
    p = 0
    while all(len(x) > p for x in ids) and len({x[p] for x in ids}) == 1:
        p += 1
    assert p >= min_prefix, f"no common token prefix (length {p}) -- is the prompt in front of every text?"
    return ids[0][:p], [x[p:] for x in ids]


@torch.no_grad()
def encode_queries_prefix_kv(model_q, model_fp, tok, texts, device, batch=32, max_len=512, A=None, n_fp=None):
    """Last-token pooled, L2-normalised embeddings: prefix KV from model_fp, the query tokens through model_q.
    Right padding of the query part; the pooled token is the last real one (it never attends to the pads after it).
    n_fp: how many prefix tokens come from the fp model (None = the whole common prefix; 1 = the sink token only, the
    remaining prefix tokens go through model_q with the query -- the control that separates the sink from token count)."""
    prefix, rest = split_prefix(tok, texts, max_len)
    if n_fp is not None and 0 < n_fp < len(prefix):
        rest = [prefix[n_fp:] + r for r in rest]; prefix = prefix[:n_fp]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    P = len(prefix)
    out = np.zeros((len(texts), model_q.config.hidden_size), np.float32)
    order = sorted(range(len(texts)), key=lambda i: -len(rest[i]))
    pre = torch.tensor(prefix, device=device)[None, :]
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]; b = len(idx)
        T = max(len(rest[i]) for i in idx)
        x = torch.full((b, T), pad_id, dtype=torch.long, device=device); m = torch.zeros((b, T), dtype=torch.long, device=device)
        for r, i in enumerate(idx):
            x[r, :len(rest[i])] = torch.tensor(rest[i], device=device); m[r, :len(rest[i])] = 1
        cache = model_fp(input_ids=pre.expand(b, -1), use_cache=True).past_key_values
        full_mask = torch.cat([torch.ones((b, P), dtype=torch.long, device=device), m], 1)
        hs = model_q(input_ids=x, attention_mask=full_mask, past_key_values=cache, use_cache=True).last_hidden_state
        last = m.sum(1) - 1
        pooled = hs[torch.arange(b, device=device), last].float()
        if A is not None:
            pooled = pooled @ A
        out[idx] = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
    return out, P


def main():
    ap = argparse.ArgumentParser(description="self-check: fp model with its own prefix cache must reproduce plain encoding")
    ap.add_argument("--teacher", default="harrier-0.6b")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    from eq.data import load_dataset
    from eq.teacher import TEACHERS, query_prompt
    from gptq_harrier import encode_queries
    spec = TEACHERS[args.teacher]
    local = ROOT / "models/hf" / spec["hf"].split("/")[-1]
    src = str(local) if (local / "config.json").exists() else spec["hf"]
    tok = AutoTokenizer.from_pretrained(src); tok.padding_side = "left"
    dev = args.device
    model = AutoModel.from_pretrained(src, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    ds = load_dataset(args.dataset)
    qp = query_prompt(spec, args.dataset)
    texts = [qp + ds.queries[q] for q in list(ds.queries)[: args.n]]
    ref = encode_queries(model, tok, texts, dev, max_len=spec["max_len"])
    print(f"prefix = {tok.decode(split_prefix(tok, texts)[0])!r}")
    for n_fp in (None, 1):
        pkv, P = encode_queries_prefix_kv(model, model, tok, texts, dev, max_len=spec["max_len"], n_fp=n_fp)
        cos = np.sum(ref * pkv, 1)
        print(f"n_fp={n_fp} ({P} prefix tokens from fp): cos(plain, prefix-kv) over {len(texts)} queries: min {cos.min():.6f} mean {cos.mean():.6f}")
        assert cos.min() > 0.999, "prefix-KV path does not reproduce the plain encoding"
    print("prefix_kv self-check ok")


if __name__ == "__main__":
    main()
