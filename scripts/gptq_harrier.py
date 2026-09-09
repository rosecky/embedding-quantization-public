"""E10e: GPTQ (Hessian-based, sequential, group-wise asymmetric) quantization of the server embedding model
(harrier-0.6b, Qwen3 architecture) with CALIBRATION DATA AS THE VARIABLE -- generic text vs corpus documents +
synthetic queries vs real queries -- evaluated as a QUERY-SIDE encoder against the full-precision document index.

Unlike llama.cpp's importance matrix (per-channel error weights), GPTQ solves a least-squares problem on the
calibration activations (OBS/OBQ error compensation), so the calibration distribution should matter much more.

Pipeline (standard GPTQ, Frantar et al. 2022):
  * calibration batch: N sequences x T tokens from the calibration text (stream cut into T-token chunks, like
    llama-imatrix) or, with --per_sample, one query per sequence (right-padded, masked) so queries are seen exactly as
    at inference;
  * layer by layer: collect H = sum x x^T for the inputs of every nn.Linear in the block (q/k/v/o, gate/up/down) from
    the OUTPUTS OF THE ALREADY QUANTIZED previous blocks; quantize each weight matrix column-block-wise with error
    propagation (dampening 1 %, block 128), asymmetric min/max per group of --group input columns; replace the weight
    with its dequantized value; propagate the quantized block's outputs to the next block.
  * evaluation: last-token pooled, L2-normalised query embeddings of the dequantized model vs the fp document cache
    (GT nDCG@10, R@100, cosine to fp query, top-10 overlap; ArguAna self-doc masked).  Token table and norms untouched.
Effective bits per weight: bits + (16 + bits) / group (fp16 scale + integer zero per group).

Usage: python scripts/gptq_harrier.py --bits 3 2 --calib generic_wikitext scifact_corpus_synth --datasets scifact --splits train dev test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("EQ_THREADS", "4"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eq import graph_metrics as G  # noqa: E402
from eq.data import load_dataset  # noqa: E402
from eq.teacher import TEACHERS, E5_QUERY_PROMPT, task_query_prompt, query_prompt  # noqa: E402
from eq.verify import load_emb, rel_matrix, self_mask, mask_self, l2n  # noqa: E402
try:  # the structured-rotation research code is not part of the public export; --rotate none works without it
    from rotate_qwen3 import prepare as rotate_prepare, incoherence, dense_rotation  # noqa: E402
except ImportError:
    rotate_prepare = incoherence = dense_rotation = None

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def find_blocks(model):
    """the transformer block list, whatever the family calls it (Qwen/Llama: layers; BERT/XLM-R: encoder.layer)"""
    for path in ("layers", "encoder.layer", "encoder.layers", "transformer.h"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False; break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.ModuleList) and len(obj):
            return obj
    raise RuntimeError("no transformer block list found on this model")


def find_linears(block):
    """every nn.Linear inside one block, by dotted name — architecture agnostic"""
    return [n for n, m in block.named_modules() if isinstance(m, nn.Linear)]


def get_module(root, path):
    m = root
    for p in path.split("."):
        m = getattr(m, p)
    return m


# ------------------------------------------------------------------ GPTQ core
@torch.no_grad()
def _group_params(g, maxq, hw=None, clip_search=False):
    """Asymmetric min/max group parameters, optionally shrunk to the Hessian-weighted MSE optimum.
    A Hadamard rotation makes the weights Gaussian, and min/max over a Gaussian group wastes levels on the extremes,
    so the shrink search is what lets a rotated model use a scalar grid at all."""
    wmin = torch.minimum(g.min(1).values, torch.zeros(1, device=g.device)); wmax = torch.maximum(g.max(1).values, torch.zeros(1, device=g.device))
    if not clip_search:
        s = ((wmax - wmin) / maxq).clamp(min=1e-8)
        return s, torch.round(-wmin / s)
    best_s = best_z = None; best_e = None
    w = torch.ones(g.shape[1], device=g.device) if hw is None else hw.clamp_min(1e-12)
    for f in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        s = ((wmax - wmin) * f / maxq).clamp(min=1e-8); z = torch.round(-wmin * f / s)
        q = torch.clamp(torch.round(g / s.unsqueeze(1)) + z.unsqueeze(1), 0, maxq)
        e = (((g - s.unsqueeze(1) * (q - z.unsqueeze(1))) ** 2) * w.unsqueeze(0)).sum(1)
        if best_e is None:
            best_e, best_s, best_z = e, s, z
        else:
            better = e < best_e
            best_e = torch.where(better, e, best_e); best_s = torch.where(better, s, best_s); best_z = torch.where(better, z, best_z)
    return best_s, best_z


def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, group: int, blocksize: int = 128, percdamp: float = 0.01, act_order: bool = False, frozen_scales: bool = False, clip_search: bool = False, levels: int | None = None):
    """levels: override the number of representable values (3 = ternary, i.e. 1.585 bits/weight)."""
    """W (out, in) fp32, H (in, in) fp32 -> dequantized W_q (out, in) fp32. Asymmetric per-(row, group) min/max.
    act_order: quantize columns in order of decreasing diag(H) (GPTQ "desc_act"), groups defined in that order."""
    W = W.clone().float(); H = H.clone()
    n_in = W.shape[1]
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]; H = H[perm][:, perm]
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    H = 0.5 * (H + H.T)                       # kill any asymmetry from fp accumulation
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0; W[:, dead] = 0.0
    base = float(torch.mean(torch.diag(H)))
    Hinv = None
    for k in range(8):                        # escalating dampening instead of crashing on a non-PSD Hessian
        try:
            Hd = H + torch.eye(n_in, device=H.device) * (percdamp * (10 ** k) * base)
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except Exception:  # noqa: BLE001
            continue
    if Hinv is None:                          # last resort: ignore the off-diagonal structure
        Hinv = torch.diag(torch.rsqrt(torch.diag(H) + percdamp * base))
    maxq = (levels - 1) if levels else 2 ** bits - 1
    Q = torch.zeros_like(W)
    scale = zero = None
    frozen = None
    if frozen_scales:  # group parameters from the ORIGINAL weights (GPTQ static groups): compensation inflates them by 15-19 % at 2 bits
        frozen = {}
        hd_all = torch.diag(H)
        for c0 in range(0, n_in, group):
            frozen[c0] = _group_params(W[:, c0:c0 + group], maxq, hd_all[c0:c0 + group], clip_search)
    for i1 in range(0, n_in, blocksize):
        i2 = min(i1 + blocksize, n_in); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if col % group == 0 and frozen is not None:
                scale, zero = frozen[col]
            elif col % group == 0:  # new group: scale/zero from the CURRENT (error-updated) weights of the group
                scale, zero = _group_params(W[:, col:col + group], maxq, torch.diag(H)[col:col + group], clip_search)
            w = W1[:, i]; d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
            q = scale * (q - zero)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), device=perm.device)
        Q = Q[:, inv]
    return Q


# ------------------------------------------------------------------ calibration data
def calib_batches(tok, text_path: Path, n_seq: int, seq_len: int, per_sample: bool, device, max_tokens: int | None = None):
    """max_tokens: stop adding sequences once this many (truncated) tokens are collected, so arms can be budget-matched."""
    text = Path(text_path).read_text(encoding="utf-8")
    if per_sample:
        parts = [p for p in text.split("\n\n") if p.strip()][:n_seq]
        if len(parts) >= max(8, n_seq // 8):  # natural units (documents / queries)
            if max_tokens:
                keep, tot = [], 0
                for x in parts:
                    k = min(len(tok(x, add_special_tokens=False)["input_ids"]), seq_len)
                    if tot + k > max_tokens:
                        break
                    keep.append(x); tot += k
                parts = keep
                print(f"[calib] {Path(text_path).name}: {len(parts)} sequences = {tot} tokens (budget {max_tokens})", flush=True)
            enc = tok(parts, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
            return enc["input_ids"].to(device), enc["attention_mask"].to(device)
        print(f"[calib] {Path(text_path).name}: only {len(parts)} paragraph(s) -> falling back to {seq_len}-token windows", flush=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = min(n_seq, ids.numel() // seq_len, (max_tokens // seq_len) if max_tokens else 10**9)
    ids = ids[: n * seq_len].view(n, seq_len)
    return ids.to(device), torch.ones_like(ids)


@torch.no_grad()
def run_gptq(model, input_ids, attn, bits, group, device, log, act_order=False, quantizer=None, frozen_scales=False, clip_search=False, levels=None):
    """Sequential GPTQ over model.layers; modifies the model's linear weights in place (dequantized values).
    quantizer: optional callable (W, H, ctx) -> (W_deq, extra) replacing gptq_quantize; extras are returned as {(layer, name): extra}."""
    extras = {}
    layers = find_blocks(model)
    names_all = find_linears(layers[0])
    # capture the inputs of layer 0 (hidden states + kwargs such as attention mask / position embeddings)
    cache = {"inps": [], "kws": []}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, hs, *a, **kw):
            # drop the KV cache: replaying a layer with a cache object that has already grown makes the mask shapes
            # disagree on the second pass (seen on Qwen3-Embedding-4B: mask [b,1,512,512] vs expected [b,H,512,1024])
            kw = {k: v for k, v in kw.items() if k not in ("past_key_value", "past_key_values", "cache_position")}
            kw["use_cache"] = False
            cache["inps"].append(hs.detach().cpu()); cache["kws"].append({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in kw.items()}); raise ValueError("catch")
    layers[0] = Catcher(layers[0])
    bs = 8
    for s in range(0, input_ids.shape[0], bs):
        try:
            model(input_ids=input_ids[s:s + bs], attention_mask=attn[s:s + bs])
        except ValueError:
            pass
    layers[0] = layers[0].m
    kws = cache["kws"]  # per-batch kwargs (4-D attention masks depend on the padding of each batch)
    inps = cache["inps"]  # list of (bs, T, d)
    masks = [attn[s:s + bs].cpu() for s in range(0, input_ids.shape[0], bs)]
    for li, layer in enumerate(layers):
        t0 = time.time()
        H = {name: None for name in names_all}; cnt = {name: 0 for name in names_all}
        cur_mask = {"m": None}
        hooks = []
        for name in names_all:
            mod = get_module(layer, name)
            def mk(name):
                def hook(m, inp, out):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    mm = cur_mask["m"].reshape(-1).bool()
                    if mm.numel() == x.shape[0]:
                        x = x[mm]
                    h = x.t() @ x
                    H[name] = h if H[name] is None else H[name] + h
                    cnt[name] += x.shape[0]
                return hook
            hooks.append(mod.register_forward_hook(mk(name)))
        for x, m, kw in zip(inps, masks, kws):
            cur_mask["m"] = m.to(device)
            layer(x.to(device), **{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()})
        for h in hooks:
            h.remove()
        for name in names_all:
            mod = get_module(layer, name)
            if H[name] is None:
                raise RuntimeError(f"no activations captured for block {li} {name}: inps={len(inps)} shapes={[tuple(x.shape) for x in inps[:2]]} kw_keys={list(kws[0].keys()) if kws else None}")
            Hn = H[name] / max(cnt[name], 1)
            if quantizer is None:
                Wq = gptq_quantize(mod.weight.data.float(), Hn, bits, group, act_order=act_order, frozen_scales=frozen_scales, clip_search=clip_search, levels=levels)
            else:
                Wq, extras[(li, name)] = quantizer(mod.weight.data.float(), Hn, dict(li=li, name=name, layer=layer, inps=inps, masks=masks, kws=kws, names=names_all))
            mod.weight.data = Wq.to(mod.weight.dtype)
        # propagate quantized outputs
        outs = []
        for x, m, kw in zip(inps, masks, kws):
            o = layer(x.to(device), **{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()})
            outs.append((o[0] if isinstance(o, tuple) else o).detach().cpu())
        inps = outs
        del outs
        torch.cuda.empty_cache() if device != "cpu" else None
        log(f"  block {li+1}/{len(layers)} quantized [{time.time()-t0:.0f}s]")
    return extras


@torch.no_grad()
def encode_queries(model, tok, texts, device, batch=32, max_len=512, A=None, pooling="last"):
    """A: optional (hidden, hidden) map applied to the pooled vector before normalization (undo of a fused rotation)."""
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        hs = model(**enc).last_hidden_state
        if pooling == "cls":
            pooled = hs[:, 0, :].float()
        else:
            last = enc["attention_mask"].shape[1] - 1 if tok.padding_side == "left" else enc["attention_mask"].sum(1) - 1
            pooled = hs[torch.arange(len(idx), device=device), last].float()
        if A is not None:
            pooled = pooled @ A
        out[idx] = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="harrier-0.6b")
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 2])
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--calib", nargs="+", default=["generic_wikitext", "scifact_corpus_synth"])
    ap.add_argument("--n_seq", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--per_sample", action="store_true", help="one calibration text per sequence (queries seen as at inference)")
    ap.add_argument("--act_order", action="store_true", help="GPTQ desc_act column ordering")
    ap.add_argument("--linfix", action="store_true", help="also measure a fixed linear map from quantized to fp embedding space (fitted on held-out texts)")
    ap.add_argument("--linfix_n", type=int, default=4000)
    ap.add_argument("--ternary", action="store_true", help="3 levels per group (1.585 bits/weight) instead of 2**bits; the PTQ counterpart of ternary QAT models")
    ap.add_argument("--calib_tokens", type=int, default=0, help="cut the calibration set to this many tokens (matched budget across arms)")
    ap.add_argument("--clip_search", action="store_true", help="pick each group scale by Hessian-weighted MSE over shrink factors instead of plain min/max")
    ap.add_argument("--frozen_scales", action="store_true", help="group scale/zero from the ORIGINAL weights instead of the error-updated ones (GPTQ static groups)")
    ap.add_argument("--doc_mode", default="fp", choices=["fp", "same", "doccal"], help="documents encoded by: fp cache | the same quantized model | a second copy calibrated on --doc_calib")
    ap.add_argument("--doc_calib", default=None, help="calibration set for the document-side quantized copy (doc_mode=doccal)")
    ap.add_argument("--datasets", nargs="+", default=["scifact"])
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--rotate", default="none", choices=["none", "fold", "r1", "r2", "r1r2"], help="fold the RMSNorm scales (control) and/or fuse randomized Hadamard rotations before quantizing (QuaRot-style, absorbed into the weights)")
    ap.add_argument("--rot_seed", type=int, default=0)
    ap.add_argument("--rotate_matrix", action="store_true",
                    help="two-sided rotation per matrix (quantize A W B, keep the codes). DEPLOYABLE: at inference "
                         "y = A^T (W'_q (B^T x)), i.e. the rotations move onto the activations and cost O(n log n) "
                         "(scripts/fast_rotation.py). No RMSNorm folding, unlike the fused --rotate variants.")
    ap.add_argument("--rotate_side", default="both", choices=["both", "in", "out"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/raw/gptq")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoTokenizer
    spec = TEACHERS[args.teacher]
    local = ROOT / "models/hf" / spec["hf"].split("/")[-1]
    src = str(local) if (local / "config.json").exists() else spec["hf"]
    tok = AutoTokenizer.from_pretrained(src); tok.padding_side = "left"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fout = open(out_dir / "results.jsonl", "a", encoding="utf-8")
    log = lambda s: print(s, flush=True)  # noqa: E731
    # datasets: queries + fp docs
    data = {}
    for d in args.datasets:
        ds = load_dataset(d)
        tdoc, D_T, tq, Q_T = load_emb(d, args.teacher)
        tdi = {x: i for i, x in enumerate(tdoc)}; tqi = {q: i for i, q in enumerate(tq)}
        te = [q for sp in args.splits for q in ds.splits.get(sp, []) if q in tqi]
        qp = query_prompt(spec, d)  # teacher-specific prefix (E5 instruction for harrier, "Query: " for jina, MTEB task instruction for -tp keys)
        data[d] = dict(D=D_T, rel=rel_matrix(te, tdi, ds.qrels, len(tdoc)), msk=self_mask(te, tdi, len(tdoc)),
                       Qfp=Q_T[[tqi[q] for q in te]], texts=[qp + ds.queries[q] for q in te], n=len(te),
                       doc_texts=[((ds.corpus[x].get("title") or "") + "\n" + (ds.corpus[x].get("text") or "")).strip() for x in tdoc])
        data[d]["n_test"] = sum(1 for q in ds.splits.get("test", []) if q in tqi)
        data[d]["S0"] = mask_self(data[d]["Qfp"] @ D_T.T, data[d]["msk"])
        data[d]["nd0"] = float(np.nanmean(G.ndcg_at_k(data[d]["S0"], data[d]["rel"], 10)))
        log(f"[{d}] fp teacher nDCG@10={data[d]['nd0']:.4f} on {len(te)} queries ({'+'.join(args.splits)})")
    def build_model():
        m = AutoModel.from_pretrained(src, torch_dtype=torch.float16).to(dev).eval()
        if args.rotate == "none":
            return m, None
        m = m.float()  # fuse in fp32, then back to fp16 (rotation mixes channels; fp16 accumulation would lose ~1e-3)
        A, info = rotate_prepare(m, seed=args.rot_seed, r1=args.rotate in ("r1", "r1r2"), use_r2=args.rotate in ("r2", "r1r2"))  # "fold" => neither, i.e. norm folding only
        log(f"[rotate] {args.rotate} seed={args.rot_seed} {info}")
        return m.half().eval(), A.float().to(dev)

    fit_texts, fit_fp = {}, {}
    if args.linfix:
        fp_model = AutoModel.from_pretrained(src, torch_dtype=torch.float16).to(dev).eval()
        for d, dd in data.items():
            f = ROOT / "data/calib" / f"{d}_synth_big.txt"
            f = f if f.exists() else ROOT / "data/calib" / f"{d}_synth_only.txt"
            if not f.exists():
                fit_fp[d] = None; continue
            parts = [x for x in f.read_text(encoding="utf-8").split(chr(10)*2) if x.strip()][: args.linfix_n]
            fit_texts[d] = parts
            fit_fp[d] = encode_queries(fp_model, tok, parts, dev)
            log(f"[linfix] {d}: {len(parts)} fit texts encoded with the fp model")
        del fp_model; torch.cuda.empty_cache()

    # two-sided per-matrix rotation: quantize A W B on the same grid, then map back for EVALUATION only.
    # Deployment stores the rotated codes and re-associates (y = A^T (W'_q (B^T x))); scripts/rot_deploy.py checks
    # that the two paths give the same embeddings (cosine 0.999996), so this measures a shippable configuration.
    rot_cache = {}

    def _rot(n, side):
        key = (n, side)
        if key not in rot_cache:
            rot_cache[key] = dense_rotation(n, seed=args.rot_seed + (0 if side == "out" else 977)).float().to(dev)
        return rot_cache[key]

    def make_quantizer(bits):
        if not args.rotate_matrix:
            return None

        def q(W, H, ctx):
            A = _rot(W.shape[0], "out") if args.rotate_side in ("both", "out") else None
            B = _rot(W.shape[1], "in") if args.rotate_side in ("both", "in") else None
            Wp = W if A is None else A @ W
            Wp = Wp if B is None else Wp @ B
            Hp = H if B is None else B.t() @ H @ B
            Wq = gptq_quantize(Wp, Hp, bits, args.group, act_order=args.act_order, frozen_scales=args.frozen_scales,
                               clip_search=args.clip_search, levels=(3 if args.ternary else None))
            Wq = Wq if A is None else A.t() @ Wq
            return (Wq if B is None else Wq @ B.t()), None
        return q

    for bits in args.bits:
        # document-side encoder for this bit width: fp cache (default), or a SECOND quantized copy calibrated on --doc_calib
        doc_model = None
        if args.doc_mode == "doccal":
            doc_model, A_doc = build_model()
            ids_d, attn_d = calib_batches(tok, ROOT / "data/calib" / f"{args.doc_calib}.txt", args.n_seq, args.seq_len, False, dev, args.calib_tokens or None)
            log(f"[gptq] DOC model: calib={args.doc_calib} bits={bits} group={args.group} n_seq={ids_d.shape[0]}")
            run_gptq(doc_model, ids_d, attn_d, bits, args.group, dev, log, act_order=args.act_order, frozen_scales=args.frozen_scales, clip_search=args.clip_search, levels=(3 if args.ternary else None))
            doc_embs = {d: encode_queries(doc_model, tok, dd["doc_texts"], dev, batch=16, A=A_doc) for d, dd in data.items()}
            del doc_model; torch.cuda.empty_cache()
        for cal in args.calib:
            t0 = time.time()
            model, A = build_model()
            if args.rotate != "none":  # incoherence of the weights the rotation was supposed to flatten (block 0)
                log("[rotate] mu = max|w|/rms(w) block 0: " + " ".join(f"{n.split('.')[-1]}={incoherence(get_module(model.layers[0], n).weight):.2f}" for n in LINEARS))
            ids, attn = calib_batches(tok, ROOT / "data/calib" / f"{cal}.txt", args.n_seq, args.seq_len, args.per_sample, dev, args.calib_tokens or None)
            log(f"[gptq] calib={cal} bits={bits} group={args.group} n_seq={ids.shape[0]} seq_len={ids.shape[1]} per_sample={args.per_sample} doc_mode={args.doc_mode}")
            run_gptq(model, ids, attn, bits, args.group, dev, log, act_order=args.act_order, quantizer=make_quantizer(bits),
                     frozen_scales=args.frozen_scales, clip_search=args.clip_search, levels=(3 if args.ternary else None))
            n_w = sum(get_module(l, n).weight.numel() for l in model.layers for n in LINEARS)
            bpw = (math.log2(3) if args.ternary else bits) + (16 + (2 if args.ternary else bits)) / args.group
            size_blocks_mib = n_w * bpw / 8 / 2**20
            for d, dd in data.items():
                Qq = encode_queries(model, tok, dd["texts"], dev, A=A)
                if args.doc_mode == "same":      # quantized model on BOTH sides (single calibration)
                    Dq = encode_queries(model, tok, dd["doc_texts"], dev, batch=16, A=A)
                elif args.doc_mode == "doccal":  # dual calibration: doc-calibrated copy encodes documents
                    Dq = doc_embs[d]
                else:
                    Dq = dd["D"]
                S = mask_self(Qq @ Dq.T, dd["msk"])
                row = dict(dataset=d, teacher=args.teacher, method="gptq", bits=bits, group=args.group, calib=cal, per_sample=args.per_sample, act_order=args.act_order,
                           doc_mode=args.doc_mode, doc_calib=(args.doc_calib if args.doc_mode == "doccal" else ("fp" if args.doc_mode == "fp" else cal)),
                           doc_cos_fp=(float(np.mean(np.sum(Dq * dd["D"], 1))) if args.doc_mode != "fp" else 1.0),
                           n_seq=int(ids.shape[0]), seq_len=int(ids.shape[1]), splits="+".join(args.splits), n_queries=dd["n"],
                           gt_ndcg10=float(np.nanmean(G.ndcg_at_k(S, dd["rel"], 10))), gt_recall100=float(np.nanmean(G.recall_at_k(S, dd["rel"], 100))),
                           fp_ndcg10=dd["nd0"], q_cos_fp=float(np.mean(np.sum(Qq * dd["Qfp"], 1))), fp_top10_overlap=float(G.topk_overlap(dd["S0"], S, 10).mean()),
                           rotate=args.rotate, rot_seed=args.rot_seed, rotate_matrix=args.rotate_matrix, rotate_side=(args.rotate_side if args.rotate_matrix else None), frozen_scales=args.frozen_scales, clip_search=args.clip_search, calib_tokens=args.calib_tokens, ternary=args.ternary,
                           eff_bpw=bpw, size_blocks_mib=size_blocks_mib, size_total_mib_q5table=size_blocks_mib + 102.0 + 1,  # Q5_0 table = 151936*1024*22/32 B = 102 MiB (earlier estimate ~53 divided by 16 instead of 8)
                           quant_s=time.time() - t0, attribution="OUR_MEASUREMENT")
                # cheapest rung of "quantization + fitting": one fixed linear map from quantized to fp embedding space.
                # Fitted on THOUSANDS of held-out texts (teacher targets are free), never on the evaluation queries;
                # ridge strength picked on a 20 % slice of the fit set; orthogonal (Procrustes) variant as the
                # overfitting-proof control. At deployment it is 2 MiB on the client or folds into the index.
                nt = dd.get("n_test", 0)
                if nt and fit_fp.get(d) is not None and args.doc_mode == "fp":
                    Fq = torch.from_numpy(encode_queries(model, tok, fit_texts[d], dev, A=A)).float()
                    Ft = torch.from_numpy(fit_fp[d]).float()
                    n_hold = max(200, Fq.shape[0] // 5)
                    Xtr, Ytr, Xho, Yho = Fq[:-n_hold], Ft[:-n_hold], Fq[-n_hold:], Ft[-n_hold:]
                    C = Xtr.T @ Xtr; dm = float(torch.diagonal(C).mean()); I = torch.eye(C.shape[0])
                    best = (None, -1.0, None)
                    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
                        Wl = torch.linalg.solve(C + lam * dm * I, Xtr.T @ Ytr)
                        cos = float(torch.nn.functional.cosine_similarity(Xho @ Wl, Yho, dim=-1).mean())
                        if cos > best[1]:
                            best = (Wl, cos, lam)
                    U, _, Vh = torch.linalg.svd(Xtr.T @ Ytr, full_matrices=False)   # Procrustes: nearest rotation
                    Wo = U @ Vh
                    mk = dd["msk"][-nt:] if getattr(dd["msk"], "size", 0) else dd["msk"]
                    Qt = torch.from_numpy(Qq[-nt:]).float()
                    def nd_of(M):
                        Z = Qt if M is None else torch.nn.functional.normalize(Qt @ M, dim=-1)
                        return float(np.nanmean(G.ndcg_at_k(mask_self(Z.numpy() @ Dq.T, mk), dd["rel"][-nt:], 10)))
                    nd_q, nd_r, nd_o = nd_of(None), nd_of(best[0]), nd_of(Wo)
                    nd_f = float(np.nanmean(G.ndcg_at_k(mask_self(dd["Qfp"][-nt:] @ Dq.T, mk), dd["rel"][-nt:], 10)))
                    gap = max(nd_f - nd_q, 1e-9)
                    row.update(test_ndcg10=nd_q, test_ndcg10_linfix=nd_r, test_ndcg10_procrustes=nd_o, test_fp_ndcg10=nd_f,
                               linfix_lambda=best[2], linfix_fit_n=int(Fq.shape[0]), linfix_holdout_cos=best[1])
                    log(f"    [linfix] fit on {Fq.shape[0]} texts (lambda {best[2]}, held-out cos {best[1]:.4f}): "
                        f"test {nd_q:.4f} -> ridge {nd_r:.4f} ({(nd_r-nd_q)/gap*100:+.0f} % of the gap), "
                        f"rotation {nd_o:.4f} ({(nd_o-nd_q)/gap*100:+.0f} %); fp {nd_f:.4f}")
                perq = out_dir / "perq"; perq.mkdir(exist_ok=True)  # per-query nDCG for paired bootstrap between calibrations
                rot_tag = ("" if args.rotate == "none" else f"_rot{args.rotate}") + ("_fs" if args.frozen_scales else "") + ("_cs" if args.clip_search else "") + (f"_t{args.calib_tokens//1000}k" if args.calib_tokens else "") + ("_tern" if args.ternary else "")
                np.savez(perq / f"{d}_{bits}b_g{args.group}_{cal}_ps{int(args.per_sample)}_ao{int(args.act_order)}_doc{args.doc_mode}{rot_tag}.npz",
                         ndcg=G.ndcg_at_k(S, dd["rel"], 10), ndcg_fp=G.ndcg_at_k(dd["S0"], dd["rel"], 10))
                fout.write(json.dumps(row) + "\n"); fout.flush()
                log(f"[{d}] gptq {'ternary' if args.ternary else str(bits)+'b'} g{args.group}{'' if args.rotate=='none' else ' rot='+args.rotate} calib={cal:24s} nDCG@10={row['gt_ndcg10']:.4f} ({row['gt_ndcg10']/dd['nd0']*100:.1f} % fp) "
                    f"cos={row['q_cos_fp']:.4f} ov={row['fp_top10_overlap']:.3f} blocks {size_blocks_mib:.0f} MiB (+Q5_0 table 102 / Q3_K table 64 / Q2_K table 49)")
            del model; torch.cuda.empty_cache()
    fout.close()


if __name__ == "__main__":
    main()
