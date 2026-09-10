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


def common_prefix(id_lists, min_prefix: int = 1) -> list[int]:
    """Longest common token prefix of already tokenised sequences (the container stores it as prefix.ids)."""
    p = 0
    while all(len(x) > p for x in id_lists) and len({x[p] for x in id_lists}) == 1:
        p += 1
    assert p >= min_prefix, f"no common token prefix (length {p})"
    return list(id_lists[0][:p])


@torch.no_grad()
def prefix_cache_tensors(model_fp, prefix_ids, device):
    """K / V of the prefix tokens from the fp model as numpy f16 arrays of shape (layers, P, kv_heads * head_dim) -- the layout
    of the runtime's k / v buffers ([token, kv_head, head_dim]).  Keys are what the HF cache holds: after k_norm and RoPE at
    positions 0..P-1 (M6 of the VQ runtime: the container ships them, the attention kernel starts from them)."""
    ids = torch.tensor(prefix_ids, device=device)[None, :]
    cache = model_fp(input_ids=ids, use_cache=True).past_key_values
    K, V = [], []
    for li in range(model_fp.config.num_hidden_layers):
        k, v = cache_layer(cache, li)                       # (1, n_kv, P, hd)
        K.append(k[0].transpose(0, 1).reshape(k.shape[2], -1).float().cpu().numpy().astype(np.float16))
        V.append(v[0].transpose(0, 1).reshape(v.shape[2], -1).float().cpu().numpy().astype(np.float16))
    return np.stack(K, 0), np.stack(V, 0)


def cache_layer(cache, li):
    """(keys, values) of layer li from a transformers cache object (4.x tuple API or 5.x .layers[])."""
    if hasattr(cache, "layers"):
        return cache.layers[li].keys, cache.layers[li].values
    return cache[li][0], cache[li][1]


def cache_from_tensors(K, V, n_kv, head_dim, device, dtype):
    """DynamicCache holding the container's prefix K / V (layers, P, n_kv * hd) -> what the quantised model attends to."""
    from transformers import DynamicCache
    c = DynamicCache()
    for li in range(K.shape[0]):
        k = torch.from_numpy(np.asarray(K[li], dtype=np.float32)).to(device=device, dtype=dtype).reshape(1, -1, n_kv, head_dim).transpose(1, 2)
        v = torch.from_numpy(np.asarray(V[li], dtype=np.float32)).to(device=device, dtype=dtype).reshape(1, -1, n_kv, head_dim).transpose(1, 2)
        c.update(k.contiguous(), v.contiguous(), li)
    return c


@torch.no_grad()
def encode_ids_prefix_kv(model_q, id_lists, prefix_ids, K, V, device, pad_id, batch=32, max_len=512):
    """vocab_trim.encode_ids with the stored prefix K / V: sequences that start with prefix_ids are cut and run through model_q
    on top of the cache (right padding, last real token pooled); the others go through plainly (fallback = the runtime's)."""
    n_kv, hd = model_q.config.num_key_value_heads, getattr(model_q.config, "head_dim", model_q.config.hidden_size // model_q.config.num_attention_heads)
    dtype = next(model_q.parameters()).dtype
    P = len(prefix_ids)
    out = np.zeros((len(id_lists), model_q.config.hidden_size), np.float32)
    use = [len(x) > P and list(x[:P]) == list(prefix_ids) for x in id_lists]
    plain = [i for i, u in enumerate(use) if not u]
    if plain:
        from vocab_trim import encode_ids
        out[plain] = encode_ids(model_q, [id_lists[i] for i in plain], device, pad_id, batch=batch, max_len=max_len)
    idx_all = [i for i, u in enumerate(use) if u]
    order = sorted(idx_all, key=lambda i: -len(id_lists[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]; b = len(idx)
        rest = [id_lists[i][P:max_len] for i in idx]
        T = max(len(r) for r in rest)
        x = torch.full((b, T), pad_id, dtype=torch.long, device=device); m = torch.zeros((b, T), dtype=torch.long, device=device)
        for r, seq in enumerate(rest):
            x[r, :len(seq)] = torch.tensor(seq, device=device); m[r, :len(seq)] = 1
        cache = cache_from_tensors(K, V, n_kv, hd, device, dtype)
        for li in range(K.shape[0]):   # batch the cache: repeat along the batch axis

            kk, vv = cache_layer(cache, li)
            if hasattr(cache, "layers"):
                cache.layers[li].keys = kk.expand(b, -1, -1, -1).contiguous(); cache.layers[li].values = vv.expand(b, -1, -1, -1).contiguous()
            else:
                cache.key_cache[li] = kk.expand(b, -1, -1, -1).contiguous(); cache.value_cache[li] = vv.expand(b, -1, -1, -1).contiguous()
        full_mask = torch.cat([torch.ones((b, P), dtype=torch.long, device=device), m], 1)
        hs = model_q(input_ids=x, attention_mask=full_mask, past_key_values=cache, use_cache=True).last_hidden_state
        pooled = hs[torch.arange(b, device=device), m.sum(1) - 1].float()
        out[idx] = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
    return out, sum(use)


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
    # the container path: prefix K/V as stored f16 tensors -> DynamicCache -> encode_ids_prefix_kv on token-id lists
    from vocab_trim import encode_ids
    ids = [tok(t, truncation=True, max_length=spec["max_len"])["input_ids"] for t in texts]
    pre = common_prefix(ids)
    K, V = prefix_cache_tensors(model, pre, dev)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    e_plain = encode_ids(model, ids, dev, pad_id, max_len=spec["max_len"])
    e_pkv, n_used = encode_ids_prefix_kv(model, ids, pre, K, V, dev, pad_id, max_len=spec["max_len"])
    cos = np.sum(e_plain * e_pkv, 1)
    print(f"stored-tensor path: prefix {len(pre)} tokens, K/V {K.shape} f16 ({(K.nbytes + V.nbytes) / 2**20:.2f} MiB), used on {n_used}/{len(ids)}; cos(plain, prefix-kv) min {cos.min():.6f} mean {cos.mean():.6f}")
    assert cos.min() > 0.999, "stored-tensor prefix path does not reproduce the plain encoding"
    print("prefix_kv self-check ok")


if __name__ == "__main__":
    main()
