"""Step 2 of the quantization branch: a REAL runtime file for the GPTQ client.

GPTQ (our sequential implementation, calibration as chosen) rounds directly onto the llama.cpp K-quant grid
(scripts/kquant_gptq.py: Q3_K = 3-bit codes, 6-bit sub-block scales, fp16 super-block scale; Q2_K = 2-bit codes, 4-bit
sub-block scales and mins, fp16 d/dmin), the codes and parameters are packed bit-exactly into ggml blocks and written into
a GGUF whose other tensors (norms, token table -> Q5_0 by default) are copied from the f16 GGUF.  The file runs unchanged
in bitnet.cpp/llama.cpp (`llama-embedding`); there is no second rounding, the runtime sees exactly the torch-side weights
(up to the fp16 storage of the torch model used for the torch-side evaluation).

Outputs: models/gguf/<teacher>-gptq-<TYPE>-<calib>[-ps<n>][-ao][-sg].gguf (harrier-0.6b by default; --teacher jina-v5-small
reads models/hf/jina-embeddings-v5-text-small-retrieval + models/gguf/jina-v5-small-f16.gguf and prompts with "Query: "), plus a jsonl row with
  * torch-side nDCG@10 of the GPTQ model on the requested datasets/splits,
  * pack check: max |gguf-dequant - torch fp16 weight| (expected ~fp16 rounding) and vs our own dequant (expected 0),
  * file size.  Runtime quality is then measured with scripts/quant_eval_queries.py --ggufs <file>.

Usage (on the box):
  PYTHONPATH=/tmp/llama.cpp/gguf-py python scripts/gptq_export_gguf.py --type Q3_K \
      --calib scifact_synth_only --per_sample --n_seq 3000 --datasets scifact nfcorpus
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("EQ_THREADS", "8"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gguf  # noqa: E402  (PYTHONPATH must point to a gguf-py consistent with the f16 GGUF's converter)
import gguf.quants  # noqa: E402
from eq import graph_metrics as G  # noqa: E402
from eq.data import load_dataset  # noqa: E402
from eq.teacher import TEACHERS, E5_QUERY_PROMPT, task_query_prompt, query_prompt  # noqa: E402
from eq.verify import load_emb, rel_matrix, self_mask, mask_self  # noqa: E402
from gptq_harrier import run_gptq, calib_batches, encode_queries, get_module, find_blocks  # noqa: E402
from kquant_gptq import FORMATS, gptq_quantize_kq, cd_refine_kq, pack_any, dequant_from_pack, rtn_kq  # noqa: E402
try:  # the structured-rotation research code is not part of the public export; --rotate none works without it
    from rotate_qwen3 import prepare as rotate_prepare, incoherence, dense_rotation  # noqa: E402
except ImportError:
    rotate_prepare = incoherence = dense_rotation = None
try:  # the peer's coordinate-descent post-pass (module / global scope) is not part of the public export; without it
    from module_scope import module_G, ENDPOINT  # noqa: E402
    from global_scope import global_G, truncate_blocks, last_token_pool, cls_pool, MLP_NAMES  # noqa: E402
except ImportError:  # --cd_scope module/global and --layers are unavailable, everything else works
    module_G = global_G = truncate_blocks = last_token_pool = cls_pool = None
    ENDPOINT = frozenset(); MLP_NAMES = frozenset()

HF2GGUF = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
           "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
GGUF2HF = {v: k for k, v in HF2GGUF.items()}


def byte_shape(np_shape, qtype):  # gguf-py add_tensor_info takes the BYTE shape of the packed array when raw_dtype is given
    bs, ts = gguf.GGML_QUANT_SIZES[qtype]
    return list(np_shape[:-1]) + [np_shape[-1] // bs * ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="harrier-0.6b", help="eq.teacher.TEACHERS key: selects the HF weights, the query prompt, the source f16 GGUF "
                                                             "(spec gguf_f16 or models/gguf/<teacher>-f16.gguf) and the output prefix <teacher>-gptq-")
    ap.add_argument("--src_gguf", default=None, help="f16 GGUF whose non-block tensors and metadata are copied (default: from --teacher)")
    ap.add_argument("--type", default="Q3_K", choices=sorted(FORMATS), help="llama.cpp K-quant type for the block linears")
    ap.add_argument("--table_type", default="Q5_0", help="token_embd type (F16 keeps the original)")
    ap.add_argument("--hi_type", default=None, choices=[None, "Q2_K", "Q3_K"], help="higher-precision type for selected matrices (mixed precision inside one file)")
    ap.add_argument("--hi_names", nargs="*", default=["mlp.down_proj"], help="HF linear names that get --hi_type")
    ap.add_argument("--hi_last", type=int, default=0, help="also give the last N blocks --hi_type (all their linears)")
    ap.add_argument("--calib", default="scifact_synth_only")
    ap.add_argument("--n_seq", type=int, default=3000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--per_sample", action="store_true")
    ap.add_argument("--calib_tokens", type=int, default=0, help="cut the calibration set to this many tokens (matched budget across arms)")
    ap.add_argument("--act_order", action="store_true", help="GPTQ act-order (implies --static_groups)")
    ap.add_argument("--static_groups", action="store_true", help="K-quant parameters from the original weights up-front")
    ap.add_argument("--rotate", default="none", choices=["none", "fold", "r1", "r2", "r1r2"], help="fuse randomized Hadamard rotations into the weights before quantizing (stays a standard Qwen3)")
    ap.add_argument("--rot_seed", type=int, default=0)
    ap.add_argument("--rot_rounds", type=int, default=2, help="mixing rounds inside the structured rotation; runtime cost is linear in this")
    ap.add_argument("--rotate_side", default="both", choices=["both", "in", "out"], help="which side of the diagnostic rotation to apply (in = input/column space, fusable for q/k/v/gate/up; out = output/row space)")
    ap.add_argument("--rotate_matrix", action="store_true",
                    help="DIAGNOSTIC rotation per matrix (W -> A W B, H -> B^T H B, quantize, map back). This is the "
                         "transform the published incoherence results use; a fused deployment cannot express it, and the "
                         "mapped-back weights are no longer on the grid, so use with --no_gguf.")
    ap.add_argument("--cd_sweeps", type=int, default=0, help="coordinate-descent post-pass over the integer codes (same file format, offline only)")
    ap.add_argument("--cd_scope", default="layer", choices=["layer", "module", "global"],
                    help="CD objective: the linear's own output (layer), the enclosing attention/MLP block output (module, Rademacher probes), "
                         "or the final pooled L2-normalised embedding (global, Rademacher probes in embedding space; all seven linears get a dense G)")
    ap.add_argument("--cd_probes", type=int, default=2)
    ap.add_argument("--cd_pre", type=int, default=0, help="layer-local CD sweeps (G=None) run BEFORE the module-scope sweeps; the peer's tested recipe is --cd_pre 6 --cd_sweeps 6")
    ap.add_argument("--cd_batches", type=int, default=4)
    ap.add_argument("--cd_mlp_only", action="store_true", help="control arm: the CD post-pass refines only gate/up/down; the attention linears keep their GPTQ solution untouched")
    ap.add_argument("--cd_strict", action="store_true", help="dense-G CD: halve the accepted row set until the exact joint change is negative and reject the column otherwise "
                                                            "(monotone dense objective; tag letter s). Default off = the peer's port, which applies the set left after 6 halvings unchecked")
    ap.add_argument("--cd_gdamp", type=float, default=1.0, help="trust-region damping of the dense G (module/global): G += cd_gdamp * mean(diag G) * I; tag suffix d<value> when not 1.0 (0.3 -> d03)")
    ap.add_argument("--cd_chunk", type=int, default=0, help="global scope: sequences per probe forward/backward (0 = whole calibration batch; 1-2 on a 6 GB GPU)")
    ap.add_argument("--layers", type=int, default=0, help="DEVELOPMENT: keep only the first N transformer blocks (CPU tests); the GGUF gets block_count N and the tag -L<N>")
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_gguf", action="store_true", help="only quantize on the K-quant grid and evaluate in torch (runtime matches to 0.002); skip writing the file")
    ap.add_argument("--out", default="results/raw/gptq_export")
    args = ap.parse_args()
    if args.act_order:
        args.static_groups = True
    fmt = FORMATS[args.type]

    def type_for(li, name, n_layers):
        if args.hi_type and (name in args.hi_names or (args.hi_last and li >= n_layers - args.hi_last)):
            return args.hi_type
        return args.type

    dev = args.device if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoTokenizer
    spec = TEACHERS[args.teacher]
    local = ROOT / "models/hf" / spec["hf"].split("/")[-1]
    src = str(local) if (local / "config.json").exists() else spec["hf"]
    pooling = spec.get("pooling", "last")
    model_key = spec.get("docs_from", args.teacher)  # "-tp" keys are the same model with another prompt: same file names
    if args.src_gguf is None:
        args.src_gguf = spec.get("gguf_f16", f"models/gguf/{model_key}-f16.gguf")
    tok = AutoTokenizer.from_pretrained(src); tok.padding_side = "right" if pooling == "cls" else "left"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    # tag letter of the CD scope: '' layer, 'm' module, 'g' global; MLP-only control appends 'm' ('lm' for layer scope so it cannot read as module)
    cd_letter = {"layer": "", "module": "m", "global": "g"}[args.cd_scope] + (("m" if args.cd_scope != "layer" else "lm") if args.cd_mlp_only else "") + ("s" if args.cd_strict else "")
    gdamp_tag = "" if args.cd_gdamp == 1.0 else "d" + f"{args.cd_gdamp:g}".replace(".", "")  # 0.3 -> d03, 0.1 -> d01, 2 -> d2
    tag = f"{args.type}{('-' + args.hi_type + ('L%d' % args.hi_last if args.hi_last else '') + ''.join('-' + n.split('.')[-1] for n in (args.hi_names or []))) if args.hi_type else ''}-{args.calib}" + (f"-ps{args.n_seq}" if args.per_sample else "") + (f"-t{args.calib_tokens//1000}k" if args.calib_tokens else "") + ("-ao" if args.act_order else "") + ("-sg" if args.static_groups and not args.act_order else "") + (f"-tab{args.table_type}" if args.table_type.upper() != "Q5_0" else "") + ("" if args.rotate == "none" else f"-{args.rotate}") + (f"-2{args.rotate_side}" + (f"r{args.rot_rounds}" if args.rot_rounds != 2 else "") + (f"s{args.rot_seed}" if args.rot_seed else "") if args.rotate_matrix else "") + (f"-cd{args.cd_sweeps}{cd_letter}{('p%d' % args.cd_pre) if args.cd_pre else ''}{gdamp_tag}" if args.cd_sweeps else "") + (f"-L{args.layers}" if args.layers else "")
    out_gguf = ROOT / "models/gguf" / f"{model_key}-gptq-{tag}.gguf"

    # ---- GPTQ on the K-quant grid ----
    t0 = time.time()
    model = AutoModel.from_pretrained(src, torch_dtype=torch.float16).to(dev).eval()
    if args.layers:
        if truncate_blocks is None:
            raise SystemExit("--layers needs global_scope.py (not in the public export)")
        print(f"[dev] model truncated to {truncate_blocks(model, args.layers)} blocks", flush=True)
    pool_fn = (cls_pool if pooling == "cls" else last_token_pool(tok.padding_side)) if last_token_pool else None  # the readout global_G differentiates (as encode_queries pools)
    A = None
    if args.rotate != "none":  # fuse the rotation in fp32, then back to fp16 (the frame the quantizer and the runtime see)
        model = model.float()
        if rotate_prepare is None:
            raise SystemExit("--rotate needs rotate_qwen3.py / fast_rotation.py, which are not in the public export")
        A, rinfo = rotate_prepare(model, seed=args.rot_seed, r1=args.rotate in ("r1", "r1r2"), use_r2=args.rotate in ("r2", "r1r2"))  # "fold" = norm folding only, the control that isolates its cost from the rotation
        model = model.half().eval(); A = A.float()
        print(f"[rotate] {args.rotate} seed={args.rot_seed} {rinfo}; mu=max|w|/rms block 0: "
              + " ".join(f"{n.split('.')[-1]}={incoherence(get_module(model.layers[0], n).weight):.2f}" for n in
                         ["self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.down_proj"]), flush=True)
    ids, attn = calib_batches(tok, ROOT / "data/calib" / f"{args.calib}.txt", args.n_seq, args.seq_len, args.per_sample, dev, args.calib_tokens or None)
    print(f"[gptq] {args.type} grid: calib={args.calib} n_seq={ids.shape[0]} per_sample={args.per_sample} act_order={args.act_order} static_groups={args.static_groups}", flush=True)
    cd_stats = []

    G_cache = {}
    G_fresh = {}   # an independent probe draw (seed + 1000, same budget, same damping): the final codes are scored on it too, so that
                   # "objective went down on the decision G" can be separated from "objective went down on an independent estimate"
    FRESH_SEED = args.rot_seed + 1000

    rot_cache = {}

    def _rot(n, tag):
        key = (n, tag)
        if key not in rot_cache:
            rot_cache[key] = dense_rotation(n, seed=args.rot_seed + (0 if tag == "out" else 977), rounds=args.rot_rounds, device="cpu").float().to(dev)
        return rot_cache[key]

    def quantizer(W, H, ctx):
        t = type_for(ctx["li"], ctx["name"], len(find_blocks(model)))
        if args.rotate_matrix:   # diagnostic: quantize in a rotated basis, then map back
            A = _rot(W.shape[0], "out") if args.rotate_side in ("both", "out") else None
            B = _rot(W.shape[1], "in") if args.rotate_side in ("both", "in") else None
            Wp = W if A is None else A @ W
            Wp = Wp if B is None else Wp @ B
            Hp = H if B is None else B.t() @ H @ B
            Wq_p, pack = gptq_quantize_kq(Wp, Hp, t, act_order=args.act_order, static_groups=args.static_groups)
            Wq = Wq_p if A is None else A.t() @ Wq_p
            return (Wq if B is None else Wq @ B.t()), pack
        Wq, pack = gptq_quantize_kq(W, H, t, act_order=args.act_order, static_groups=args.static_groups)
        if args.cd_sweeps:  # refine WHICH integers are stored; scales, format and file size are untouched
            if args.cd_mlp_only and ctx["name"] not in MLP_NAMES:
                return Wq, pack   # control arm: attention linears keep the GPTQ solution
            G = None; Gf = None
            if args.cd_scope == "module" and ctx["name"] in ENDPOINT:
                key = ctx["li"]
                if key not in G_cache:
                    G_cache.clear(); G_fresh.clear()
                    G_cache[key] = module_G(ctx["layer"], ctx["names"], ctx["inps"], ctx["masks"], ctx["kws"],
                                            n_probe=args.cd_probes, max_batches=args.cd_batches, damp_rel=args.cd_gdamp, seed=args.rot_seed)
                    G_fresh[key] = module_G(ctx["layer"], ctx["names"], ctx["inps"], ctx["masks"], ctx["kws"],
                                            n_probe=args.cd_probes, max_batches=args.cd_batches, damp_rel=args.cd_gdamp, seed=FRESH_SEED)
                G = G_cache[key].get(ctx["name"]); Gf = G_fresh[key].get(ctx["name"])
            elif args.cd_scope == "global":
                key = ctx["li"]
                if key not in G_cache:  # once per block: probes through blocks li..N-1 + final norm + pooling + L2 norm
                    G_cache.clear(); G_fresh.clear()
                    t_g = time.time()
                    want = [n for n in ctx["names"] if (n in MLP_NAMES or not args.cd_mlp_only)]
                    G_cache[key] = global_G(model, ctx["li"], want, ctx["inps"], ctx["masks"], ctx["kws"], pool_fn, n_probe=args.cd_probes,
                                            max_batches=args.cd_batches, damp_rel=args.cd_gdamp, seed=args.rot_seed, chunk=args.cd_chunk)
                    G_fresh[key] = global_G(model, ctx["li"], want, ctx["inps"], ctx["masks"], ctx["kws"], pool_fn, n_probe=args.cd_probes,
                                            max_batches=args.cd_batches, damp_rel=args.cd_gdamp, seed=FRESH_SEED, chunk=args.cd_chunk)
                    print(f"  [global G] block {ctx['li']}: {len(want)} metrics from {args.cd_probes} probes x {args.cd_batches} batches, decision + fresh draw [{time.time() - t_g:.0f}s]", flush=True)
                G = G_cache[key].get(ctx["name"]); Gf = G_fresh[key].get(ctx["name"])
            if args.cd_pre and G is not None:   # peer's recipe: layer-local sweeps first (pack['codes'] updated in place), then under dense G
                cd_refine_kq(W, H, pack, sweeps=args.cd_pre, G=None)
            if G is not None:  # the FULL dense objective tr(D H D^T G) at the GPTQ start (cd_refine_kq's obj_* use diag(G) only)
                Hd = H.float() + torch.eye(H.shape[0], device=H.device) * (0.01 * torch.diag(H.float()).mean())
                D0 = Wq.float() - W.float(); D0H = D0 @ Hd
                obj0_G = float(((G @ D0) * D0H).sum()); obj0_Gf = float(((Gf @ D0) * D0H).sum()); obj0_I = float((D0 * D0H).sum())
            Wq, st = cd_refine_kq(W, H, pack, sweeps=args.cd_sweeps, G=G, strict=args.cd_strict)
            st["scope"] = args.cd_scope if G is not None else "layer"
            if G is not None:
                D1 = Wq.float() - W.float(); D1H = D1 @ Hd
                st.update(obj_before_G=obj0_G, obj_after_G=float(((G @ D1) * D1H).sum()), obj_before_G_fresh=obj0_Gf, obj_after_G_fresh=float(((Gf @ D1) * D1H).sum()),
                          obj_before_I=obj0_I, obj_after_I=float((D1 * D1H).sum()))
            st.update(li=ctx["li"], name=ctx["name"])
            cd_stats.append(st)
        return Wq, pack

    packs = run_gptq(model, ids, attn, fmt["bits"], fmt["group"], dev, lambda s: print(s, flush=True), act_order=args.act_order, quantizer=quantizer)
    quant_s = time.time() - t0
    if cd_stats:
        rel = float(np.mean([c["obj_after"] / max(c["obj_before"], 1e-30) for c in cd_stats]))
        print(f"[cd] {args.cd_sweeps} sweeps over {len(cd_stats)} matrices: layer objective -> {rel*100:.1f} % of GPTQ, "
              f"{np.mean([c['frac_moved'] for c in cd_stats])*100:.1f} % of codes moved", flush=True)
        dense = [c for c in cd_stats if "obj_after_G" in c]
        if dense:  # under a dense G the line above is a diag(G) proxy; this is the objective the sweeps minimise, plus the layer objective it trades
            rg = [c["obj_after_G"] / max(c["obj_before_G"], 1e-30) for c in dense]; ri = [c["obj_after_I"] / max(c["obj_before_I"], 1e-30) for c in dense]
            rf = [c["obj_after_G_fresh"] / max(c["obj_before_G_fresh"], 1e-30) for c in dense]
            print(f"[cd] dense-G matrices ({len(dense)}, scope {args.cd_scope}{', strict' if args.cd_strict else ''}, gdamp {args.cd_gdamp:g}): FULL objective tr(D H D^T G) -> mean {np.mean(rg)*100:.1f} % of GPTQ "
                  f"on the decision G (max {np.max(rg)*100:.1f} %, {sum(r > 1 for r in rg)} worse), {np.mean(rf)*100:.1f} % on a FRESH draw (seed {FRESH_SEED}; max {np.max(rf)*100:.1f} %, {sum(r > 1 for r in rf)} worse), "
                  f"layer objective -> {np.mean(ri)*100:.1f} %, codes moved {np.mean([c['frac_moved'] for c in dense])*100:.2f} %, "
                  f"backoff {int(np.sum([c['backoff'] for c in dense]))}, rejected columns {int(np.sum([c.get('rejected', 0) for c in dense]))}", flush=True)
        cd_dir = out_dir / "cd"; cd_dir.mkdir(exist_ok=True)   # per-matrix record of this arm (objective ratios on both G draws, moves, backoffs)
        with open(cd_dir / f"{tag}.json", "w", encoding="utf-8") as f:
            json.dump([{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in c.items()} for c in cd_stats], f)

    def dense_G_keys():  # jsonl keys present only when a dense G was used: mean full-objective ratios on the decision G and on the fresh draw
        dense = [c for c in cd_stats if "obj_after_G" in c]
        if not dense:
            return {}
        return dict(cd_gdamp=args.cd_gdamp, cd_obj_ratio_G=float(np.mean([c["obj_after_G"] / max(c["obj_before_G"], 1e-30) for c in dense])),
                    cd_obj_ratio_G_fresh=float(np.mean([c["obj_after_G_fresh"] / max(c["obj_before_G_fresh"], 1e-30) for c in dense])),
                    cd_obj_ratio_I=float(np.mean([c["obj_after_I"] / max(c["obj_before_I"], 1e-30) for c in dense])),
                    cd_frac_moved_G=float(np.mean([c["frac_moved"] for c in dense])), cd_fresh_seed=FRESH_SEED)

    # ---- torch-side quality of the quantized model ----
    perq_dir = out_dir / "perq"; perq_dir.mkdir(exist_ok=True)
    torch_rows = {}
    for d in args.datasets:
        ds = load_dataset(d)
        tdoc, D_T, tq, Q_T = load_emb(d, args.teacher)
        tdi = {x: i for i, x in enumerate(tdoc)}; tqi = {q: i for i, q in enumerate(tq)}
        te = [q for sp in args.splits for q in ds.splits.get(sp, []) if q in tqi]
        rel = rel_matrix(te, tdi, ds.qrels, len(tdoc)); msk = self_mask(te, tdi, len(tdoc))
        qp = query_prompt(spec, d)  # teacher-specific prefix (E5 instruction for harrier, "Query: " for jina, MTEB task instruction for -tp keys)
        Qq = encode_queries(model, tok, [qp + ds.queries[q] for q in te], dev, A=(A.to(dev) if A is not None else None), pooling=pooling)
        S = mask_self(Qq @ D_T.T, msk); S0 = mask_self(Q_T[[tqi[q] for q in te]] @ D_T.T, msk)
        nd = G.ndcg_at_k(S, rel, 10)
        Qfp_te = Q_T[[tqi[q] for q in te]]
        torch_rows[d] = dict(ndcg10=float(np.nanmean(nd)), fp_ndcg10=float(np.nanmean(G.ndcg_at_k(S0, rel, 10))), n=len(te),
                             q_cos_fp=float(np.mean(np.sum(Qq * Qfp_te, 1))),          # continuous readout: resolves effects nDCG cannot
                             emb_mse=float(np.mean(np.sum((Qq - Qfp_te) ** 2, 1))))
        # cheapest rung of "quantization + fitting": one ridge-fitted linear map that undoes the SYSTEMATIC part of the
        # quantization distortion. Fitted on train+dev queries, scored on test only; at deployment it folds into the
        # document index (d -> d W^T), so the client file and its latency are untouched.
        n_test = len(ds.splits.get("test", []))
        if set(args.splits) >= {"train", "test"} and n_test and len(te) > n_test + 64:
            Qf = torch.from_numpy(Qq[:-n_test]).float(); Tf = torch.from_numpy(Q_T[[tqi[q] for q in te[:-n_test]]]).float()
            A_ = Qf.T @ Qf + 1e-3 * torch.eye(Qf.shape[1]) * float(torch.diagonal(Qf.T @ Qf).mean())
            Wc = torch.linalg.solve(A_, Qf.T @ Tf)
            Qc = torch.nn.functional.normalize(torch.from_numpy(Qq[-n_test:]).float() @ Wc, dim=-1).numpy()
            rel_t = rel[-n_test:]; msk_t = msk[-n_test:] if getattr(msk, "size", 0) else msk
            S_c = mask_self(Qc @ D_T.T, msk_t); S_q = mask_self(Qq[-n_test:] @ D_T.T, msk_t)
            torch_rows[d].update(test_ndcg10=float(np.nanmean(G.ndcg_at_k(S_q, rel_t, 10))),
                                 test_ndcg10_linfix=float(np.nanmean(G.ndcg_at_k(S_c, rel_t, 10))))
            print(f"[linfix] {d}: test {torch_rows[d]['test_ndcg10']:.4f} -> {torch_rows[d]['test_ndcg10_linfix']:.4f} "
                  f"({torch_rows[d]['test_ndcg10_linfix']-torch_rows[d]['test_ndcg10']:+.4f}) with a linear correction folded into the index", flush=True)
        np.savez(perq_dir / f"{d}_{tag}_{'+'.join(args.splits)}.npz", ndcg=nd, ndcg_fp=G.ndcg_at_k(S0, rel, 10))
        print(f"[torch] {d}: GPTQ-{args.type} nDCG@10={torch_rows[d]['ndcg10']:.4f} (fp {torch_rows[d]['fp_ndcg10']:.4f})  "
              f"cos(q, fp)={torch_rows[d]['q_cos_fp']:.6f}  emb MSE={torch_rows[d]['emb_mse']:.6f}", flush=True)

    # ---- pack into the GGUF ----
    if args.no_gguf:
        row = dict(gguf=None, tag=tag, type=args.type, rotate=args.rotate, rotate_matrix=args.rotate_matrix, rot_rounds=args.rot_rounds, rotate_side=(args.rotate_side if args.rotate_matrix else None), cd_sweeps=args.cd_sweeps, calib=args.calib, n_seq=int(ids.shape[0]),
                   per_sample=args.per_sample, act_order=args.act_order, static_groups=args.static_groups, hi_type=args.hi_type,
                   splits="+".join(args.splits), torch_ndcg=torch_rows, quant_s=quant_s, attribution="OUR_MEASUREMENT",
                   **({"cd_scope": args.cd_scope} if args.cd_sweeps else {}), **({"cd_mlp_only": True} if (args.cd_sweeps and args.cd_mlp_only) else {}), **({"cd_strict": True} if (args.cd_sweeps and args.cd_strict) else {}),
                   **dense_G_keys(), **({"layers": args.layers} if args.layers else {}))
        with open(out_dir / "exports.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[grid] {tag}: " + "  ".join(f"{d} {v['ndcg10']:.4f} ({v['ndcg10']/v['fp_ndcg10']*100:.1f} % fp)" for d, v in torch_rows.items()), flush=True)
        return
    qt = getattr(gguf.GGMLQuantizationType, args.type)
    tt = getattr(gguf.GGMLQuantizationType, args.table_type.upper())
    reader = gguf.GGUFReader(str(ROOT / args.src_gguf))
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(None, arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]; sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == f"{arch}.pooling_type":
            value = int(gguf.PoolingType.LAST)
        if args.layers and field.name == f"{arch}.block_count":
            value = int(args.layers)  # development file: only the kept blocks are written
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
    # HF layout == GGUF layout for Qwen3 (verified on the f16 file: max|diff| 3e-8); GGUF dims are reversed (ne0 = inner)
    extra = {}
    if args.rotate != "none":
        extra["token_embd.weight"] = model.embed_tokens.weight.detach().float().cpu().numpy()
        extra["output_norm.weight"] = model.norm.weight.detach().float().cpu().numpy()
        for li, layer in enumerate(model.layers):
            extra[f"blk.{li}.attn_norm.weight"] = layer.input_layernorm.weight.detach().float().cpu().numpy()
            extra[f"blk.{li}.ffn_norm.weight"] = layer.post_attention_layernorm.weight.detach().float().cpu().numpy()
    planned = []; dev_fp16 = []; dev_own = []; nbytes_blocks = 0; nbytes_table = 0
    t1 = time.time()
    blk_re = re.compile(r"^blk\.(\d+)\.(\w+)\.weight$")
    for t in reader.tensors:
        np_shape = tuple(reversed([int(x) for x in t.shape]))
        m = blk_re.match(t.name)
        if m and args.layers and int(m.group(1)) >= args.layers:
            continue  # truncated development model: blocks beyond --layers are not written
        if m and m.group(2) in GGUF2HF:
            li, hname = int(m.group(1)), GGUF2HF[m.group(2)]
            pack = packs[(li, hname)]
            qt = getattr(gguf.GGMLQuantizationType, pack["fmt"])  # mixed precision: the pack knows its own type
            packed = pack_any(pack)
            assert packed.shape[0] == np_shape[0] and pack["codes"].shape == np_shape, (t.name, packed.shape, np_shape)
            deq = gguf.quants.dequantize(packed, qt).reshape(np_shape)
            w16 = get_module(model.layers[li], hname).weight.detach().float().cpu().numpy()
            dev_fp16.append(float(np.abs(deq - w16).max() / (np.abs(w16).max() + 1e-12)))
            dev_own.append(float(np.abs(deq - dequant_from_pack(pack)).max()))
            arr = np.ascontiguousarray(packed).view(np.uint8)
            writer.add_tensor_info(t.name, byte_shape(np_shape, qt), np.uint8, arr.nbytes, raw_dtype=qt); nbytes_blocks += arr.nbytes
        elif t.name == "token_embd.weight" and tt != gguf.GGMLQuantizationType.F16:
            x = (extra[t.name] if t.name in extra else np.array(t.data).astype(np.float32)).reshape(np_shape)
            if args.table_type.upper() in FORMATS:  # K-quant table: RTN on the exact grid with our packer (gguf-py cannot quantize K-quants)
                packed = pack_any(rtn_kq(torch.from_numpy(np.ascontiguousarray(x)).to(dev), args.table_type.upper()))
            else:
                packed = gguf.quants.quantize(np.ascontiguousarray(x), tt)
            arr = np.ascontiguousarray(packed).view(np.uint8)
            writer.add_tensor_info(t.name, byte_shape(np_shape, tt), np.uint8, arr.nbytes, raw_dtype=tt); nbytes_table = arr.nbytes
        elif t.name in extra:  # rotated embedding table kept in the source dtype, or a folded (now all-ones) norm
            src_t = gguf.GGMLQuantizationType(t.tensor_type)
            dt = {gguf.GGMLQuantizationType.F32: np.float32, gguf.GGMLQuantizationType.F16: np.float16}[src_t]
            arr = np.ascontiguousarray(extra[t.name].reshape(np_shape).astype(dt)).view(np.uint8).reshape(-1)
            writer.add_tensor_info(t.name, byte_shape(np_shape, src_t), np.uint8, arr.nbytes, raw_dtype=src_t)
            if t.name == "token_embd.weight":
                nbytes_table = arr.nbytes
        else:
            src_t = gguf.GGMLQuantizationType(t.tensor_type)
            arr = np.ascontiguousarray(np.array(t.data)).view(np.uint8).reshape(-1)
            writer.add_tensor_info(t.name, byte_shape(np_shape, src_t), np.uint8, arr.nbytes, raw_dtype=src_t)
            if t.name == "token_embd.weight":
                nbytes_table = arr.nbytes
        planned.append(arr)
    assert len(dev_own) == len(packs), (len(dev_own), len(packs))
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    writer.open_output_file(out_gguf); writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
    for arr in planned:
        writer.write_tensor_data(arr, tensor_endianess=reader.endianess)
    writer.close()
    n_w = sum(p["codes"].size for p in packs.values())
    if A is not None:
        np.save(out_gguf.with_suffix(".rot.npy"), A.cpu().numpy())
        print(f"[rotate] output map saved: {out_gguf.with_suffix('.rot.npy').name} (apply to the runtime embedding, then normalize)", flush=True)
    row = dict(gguf=out_gguf.name, type=args.type, rotate=args.rotate, rot_seed=args.rot_seed, cd_sweeps=args.cd_sweeps,
               cd_obj_ratio=(float(np.mean([c["obj_after"] / max(c["obj_before"], 1e-30) for c in cd_stats])) if cd_stats else None),
               cd_frac_moved=(float(np.mean([c["frac_moved"] for c in cd_stats])) if cd_stats else None),
               cd_scope=(args.cd_scope if args.cd_sweeps else None), table_type=args.table_type, bits=fmt["bits"], group=fmt["group"], calib=args.calib, n_seq=int(ids.shape[0]),
               per_sample=args.per_sample, act_order=args.act_order, static_groups=args.static_groups, splits="+".join(args.splits), torch_ndcg=torch_rows,
               pack_max_rel_dev_vs_torch_fp16=float(max(dev_fp16)), pack_max_abs_dev_vs_own_dequant=float(max(dev_own)),
               hi_type=args.hi_type, hi_names=(args.hi_names if args.hi_type else None), hi_last=(args.hi_last if args.hi_type else 0),
               eff_bpw_blocks=nbytes_blocks * 8 / n_w, file_mib=out_gguf.stat().st_size / 2**20, blocks_mib=nbytes_blocks / 2**20, table_mib=nbytes_table / 2**20,
               quant_s=quant_s, pack_s=time.time() - t1, attribution="OUR_MEASUREMENT",
               **({"cd_mlp_only": True} if (args.cd_sweeps and args.cd_mlp_only) else {}), **({"cd_strict": True} if (args.cd_sweeps and args.cd_strict) else {}),
               **dense_G_keys(), **({"layers": args.layers} if args.layers else {}))
    with open(out_dir / "exports.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[export] {out_gguf.name}: {row['file_mib']:.0f} MiB (blocks {row['blocks_mib']:.0f} @ {row['eff_bpw_blocks']:.3f} bpw, table {row['table_mib']:.0f}); "
          f"pack check: max rel dev vs torch fp16 {row['pack_max_rel_dev_vs_torch_fp16']:.2e}, vs own dequant {row['pack_max_abs_dev_vs_own_dequant']:.1e}; "
          f"quant {quant_s:.0f}s pack {row['pack_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
