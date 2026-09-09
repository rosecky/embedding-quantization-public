"""K-quant-grid GPTQ + exact Q2_K / Q3_K packers (step 2 of the quantization branch: a real runtime file).

llama.cpp K-quants store 256-weight super-blocks:
  Q2_K: 16 sub-blocks x 16 weights, 2-bit codes q in [0,3], 4-bit sub-block scale sc and min m, fp16 super-block d, dmin:
        w = d*sc*q - dmin*m                      (84 B per 256 weights = 2.625 bpw)
  Q3_K: 3-bit codes q in [-4,3], 6-bit signed sub-block scale sc, fp16 d:
        w = d*sc*q                               (110 B per 256 weights = 3.4375 bpw)
gguf-py can dequantize these but not quantize them, and llama-quantize's own quantizer would re-round GPTQ's output on its
own grid.  Here GPTQ itself rounds onto the exact K-quant grid: when the quantizer reaches a super-block it fixes (d, dmin)
and the integer sub-block (sc, m) from the current, error-updated weights (or from the original weights with
static_groups, which allows act-order), then every column is rounded to the representable values with the usual OBS
error propagation; the codes + parameters are packed bit-exactly (verified against gguf.quants.dequantize).  No second
rounding, so the GGUF reproduces the torch-side weights up to the fp16 storage of the torch model itself.

Self-test (needs gguf-py on PYTHONPATH):  python scripts/kquant_gptq.py --selftest
"""
from __future__ import annotations

import numpy as np
import torch

QK = 256
SUB = 16
FORMATS = {"Q2_K": dict(bits=2, group=16, bytes=84), "Q3_K": dict(bits=3, group=16, bytes=110)}


def _f16(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()


def _ls_scale_min(g, q):
    """least-squares (s, ml) for w ~ s*q - ml over the last axis; returns s, ml (may be invalid where q is constant)."""
    n = g.shape[-1]
    sq = q.sum(-1); sq2 = (q * q).sum(-1); sw = g.sum(-1); swq = (g * q).sum(-1)
    den = n * sq2 - sq * sq
    s = (n * swq - sq * sw) / torch.where(den == 0, torch.ones_like(den), den)
    ml = -(sw - s * sq) / n
    return s, ml, den != 0


FACTORS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 1.05)  # a rotated (Gaussian) weight group needs a much tighter clip than min/max


def params_q2k(Wsb: torch.Tensor, hw: torch.Tensor | None = None):
    """Wsb (rows, 256) -> dict(d, dmin (rows,), sc, m (rows,16) int, s_eff, ml_eff (rows,16)); w ~ s_eff*q - ml_eff, q in [0,3].
    hw: optional per-column weight (diag of the Hessian) so the clip is chosen in the GPTQ objective, not plain MSE."""
    rows = Wsb.shape[0]
    g = Wsb.reshape(rows, QK // SUB, SUB)
    wt = None if hw is None else hw.reshape(1, QK // SUB, SUB).clamp_min(1e-12)
    wmin = torch.clamp(g.min(2).values, max=0.0); wmax = torch.clamp(g.max(2).values, min=0.0)
    best_s = ((wmax - wmin) / 3).clamp(min=1e-8); best_ml = -wmin; best_err = None
    for f in FACTORS:
        s = ((wmax - wmin) / 3 * f).clamp(min=1e-8); ml = -wmin * f
        for _ in range(3):
            q = torch.clamp(torch.round((g + ml.unsqueeze(2)) / s.unsqueeze(2)), 0, 3)
            s2, ml2, ok = _ls_scale_min(g, q)
            s = torch.where(ok & (s2 > 0), s2, s); ml = torch.where(ok, ml2.clamp(min=0.0), ml)
        q = torch.clamp(torch.round((g + ml.unsqueeze(2)) / s.unsqueeze(2)), 0, 3)
        e2 = (g - (s.unsqueeze(2) * q - ml.unsqueeze(2))) ** 2
        err = (e2 if wt is None else e2 * wt).sum(2)
        if best_err is None:
            best_err = err; best_s = s; best_ml = ml
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err); best_s = torch.where(better, s, best_s); best_ml = torch.where(better, ml, best_ml)
    d = _f16(best_s.max(1).values / 15); dmin = _f16(best_ml.max(1).values / 15)
    d_safe = torch.where(d > 0, d, torch.ones_like(d)); dmin_safe = torch.where(dmin > 0, dmin, torch.ones_like(dmin))
    sc = torch.clamp(torch.round(best_s / d_safe.unsqueeze(1)), 0, 15); m = torch.clamp(torch.round(best_ml / dmin_safe.unsqueeze(1)), 0, 15)
    s_eff = d.unsqueeze(1) * sc; ml_eff = dmin.unsqueeze(1) * m
    return dict(d=d, dmin=dmin, sc=sc.to(torch.int64), m=m.to(torch.int64), s_eff=s_eff, ml_eff=ml_eff)


def params_q3k(Wsb: torch.Tensor, hw: torch.Tensor | None = None):
    """Wsb (rows, 256) -> dict(d (rows,), sc (rows,16) int in [-32,31], s_eff (rows,16)); w ~ s_eff*q, q in [-4,3]."""
    rows = Wsb.shape[0]
    g = Wsb.reshape(rows, QK // SUB, SUB)
    wt = None if hw is None else hw.reshape(1, QK // SUB, SUB).clamp_min(1e-12)
    amax_idx = g.abs().argmax(2, keepdim=True)
    wm = torch.gather(g, 2, amax_idx).squeeze(2)  # signed value of the largest-magnitude weight
    best_s = None; best_err = None
    for base in (-wm / 4, wm / 3):
        for f in FACTORS:
            s = base * f
            s = torch.where(s.abs() < 1e-8, torch.full_like(s, 1e-8), s)
            for _ in range(2):
                q = torch.clamp(torch.round(g / s.unsqueeze(2)), -4, 3)
                sq2 = (q * q).sum(2); swq = (g * q).sum(2)
                s2 = swq / torch.where(sq2 == 0, torch.ones_like(sq2), sq2)
                s = torch.where((sq2 > 0) & (s2.abs() > 1e-8), s2, s)
            q = torch.clamp(torch.round(g / s.unsqueeze(2)), -4, 3)
            e2 = (g - s.unsqueeze(2) * q) ** 2
            err = (e2 if wt is None else e2 * wt).sum(2)
            if best_err is None:
                best_err = err; best_s = s
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err); best_s = torch.where(better, s, best_s)
    smax_idx = best_s.abs().argmax(1, keepdim=True)
    s_at = torch.gather(best_s, 1, smax_idx).squeeze(1)
    d = _f16(-s_at / 32)  # sign chosen so the largest |scale| maps to exactly -32
    d_safe = torch.where(d != 0, d, torch.ones_like(d))
    sc = torch.clamp(torch.round(best_s / d_safe.unsqueeze(1)), -32, 31)
    s_eff = d.unsqueeze(1) * sc
    return dict(d=d, sc=sc.to(torch.int64), s_eff=s_eff)


def gptq_quantize_kq(W: torch.Tensor, H: torch.Tensor, fmt: str, blocksize: int = 256, percdamp: float = 0.01,
                     act_order: bool = False, static_groups: bool = False):
    """GPTQ with the exact llama.cpp K-quant grid. W (out, in) fp32, H (in, in). Returns (W_deq (out,in) fp32, pack dict).
    Dynamic super-block parameters (default) need the original column order (no act-order); static_groups fixes all
    parameters from the original weights up-front and then allows act-order processing (GPTQ 'static groups')."""
    assert fmt in FORMATS, fmt
    if act_order and not static_groups:
        raise ValueError("act_order requires static_groups for K-quant packing (groups must stay in the original column order)")
    W = W.clone().float(); H = H.clone()
    rows, n_in = W.shape
    assert n_in % QK == 0, n_in
    n_sb = n_in // QK
    par_fn = params_q2k if fmt == "Q2_K" else params_q3k
    hdiag0 = torch.diag(H).clone()  # in the ORIGINAL column order (parameters are always fitted there)
    params = [None] * n_sb
    if static_groups:
        for sb in range(n_sb):
            params[sb] = par_fn(W[:, sb * QK:(sb + 1) * QK], hdiag0[sb * QK:(sb + 1) * QK])
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]; H = H[perm][:, perm]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0; W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(n_in, device=H.device) * damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    codes = torch.zeros(rows, n_in, dtype=torch.int8, device=W.device)  # in ORIGINAL column order
    lo, hi = (0, 3) if fmt == "Q2_K" else (-4, 3)
    for i1 in range(0, n_in, blocksize):
        i2 = min(i1 + blocksize, n_in); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            oc = int(perm[col]) if perm is not None else col
            sb, j = oc // QK, (oc % QK) // SUB
            if params[sb] is None:  # dynamic: parameters from the current (error-updated) weights of the super-block
                assert perm is None and oc % QK == 0 and i == 0, "dynamic K-quant parameters need blocksize aligned to 256"
                params[sb] = par_fn(W[:, oc:oc + QK], hdiag0[oc:oc + QK])
            p = params[sb]
            s = p["s_eff"][:, j]; s_safe = torch.where(s.abs() > 0, s, torch.full_like(s, 1e-12))
            w = W1[:, i]; dd = Hinv1[i, i]
            if fmt == "Q2_K":
                ml = p["ml_eff"][:, j]
                q = torch.clamp(torch.round((w + ml) / s_safe), lo, hi)
                deq = s * q - ml
            else:
                q = torch.clamp(torch.round(w / s_safe), lo, hi)
                deq = s * q
            codes[:, oc] = q.to(torch.int8)
            Q1[:, i] = deq
            err = (w - deq) / dd
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), device=perm.device)
        Q = Q[:, inv]
    pack = dict(fmt=fmt, codes=codes.cpu().numpy(),
                d=torch.stack([p["d"] for p in params], 1).cpu().numpy().astype(np.float32),
                sc=torch.stack([p["sc"] for p in params], 1).cpu().numpy())
    if fmt == "Q2_K":
        pack["dmin"] = torch.stack([p["dmin"] for p in params], 1).cpu().numpy().astype(np.float32)
        pack["m"] = torch.stack([p["m"] for p in params], 1).cpu().numpy()
    return Q, pack


def rtn_kq(W: torch.Tensor, fmt: str, chunk_rows: int = 8192):
    """Round-to-nearest on the exact K-quant grid (no Hessian; e.g. the token table): W (rows, in) -> pack dict."""
    assert fmt in FORMATS and W.shape[1] % QK == 0
    par_fn = params_q2k if fmt == "Q2_K" else params_q3k
    lo, hi = (0, 3) if fmt == "Q2_K" else (-4, 3)
    parts = []
    for r0 in range(0, W.shape[0], chunk_rows):
        Wc = W[r0:r0 + chunk_rows].float(); rows = Wc.shape[0]; n_sb = Wc.shape[1] // QK
        codes = torch.zeros(rows, Wc.shape[1], dtype=torch.int8, device=W.device); ps = []
        for sb in range(n_sb):
            p = par_fn(Wc[:, sb * QK:(sb + 1) * QK]); ps.append(p)  # RTN table: no Hessian
            g = Wc[:, sb * QK:(sb + 1) * QK].reshape(rows, QK // SUB, SUB)
            s = p["s_eff"].unsqueeze(2); s_safe = torch.where(s.abs() > 0, s, torch.full_like(s, 1e-12))
            if fmt == "Q2_K":
                q = torch.clamp(torch.round((g + p["ml_eff"].unsqueeze(2)) / s_safe), lo, hi)
            else:
                q = torch.clamp(torch.round(g / s_safe), lo, hi)
            codes[:, sb * QK:(sb + 1) * QK] = q.reshape(rows, QK).to(torch.int8)
        part = dict(fmt=fmt, codes=codes.cpu().numpy(), d=torch.stack([p["d"] for p in ps], 1).cpu().numpy().astype(np.float32),
                    sc=torch.stack([p["sc"] for p in ps], 1).cpu().numpy())
        if fmt == "Q2_K":
            part["dmin"] = torch.stack([p["dmin"] for p in ps], 1).cpu().numpy().astype(np.float32)
            part["m"] = torch.stack([p["m"] for p in ps], 1).cpu().numpy()
        parts.append(part)
    out = dict(fmt=fmt)
    for k in parts[0]:
        if k != "fmt":
            out[k] = np.concatenate([q[k] for q in parts], 0)
    return out



# ------------------------------------------------------------------------------- coordinate-descent post-pass
def scales_from_pack(pack, device=None):
    """Per-entry effective scale S and offset ML (rows, in) implied by a pack: What = S*codes - ML."""
    d = torch.as_tensor(pack["d"], device=device).float()      # (rows, n_sb)
    sc = torch.as_tensor(pack["sc"], device=device).float()    # (rows, n_sb, 16)
    S = (d.unsqueeze(2) * sc).repeat_interleave(SUB, dim=2).reshape(d.shape[0], -1)
    if pack["fmt"] == "Q2_K":
        dm = torch.as_tensor(pack["dmin"], device=device).float()
        m = torch.as_tensor(pack["m"], device=device).float()
        ML = (dm.unsqueeze(2) * m).repeat_interleave(SUB, dim=2).reshape(d.shape[0], -1)
    else:
        ML = torch.zeros_like(S)
    return S, ML


def cd_refine_kq(W: torch.Tensor, H: torch.Tensor, pack, sweeps: int = 2, block: int = 128, percdamp: float = 0.01,
                 G: torch.Tensor | None = None, backoff_steps: int = 6, strict: bool = False):
    """Coordinate descent over the integer codes with the scales frozen: minimizes tr(D H D^T) (G=None, rows are
    independent) or tr(D H D^T G) with a dense output metric G (rows coupled -> exact-change check with halving
    backoff, as in the peer repo's joint solver).  Storage format is untouched: only which integers are stored changes.
    strict=True (dense G): keep halving the accepted row set until the exact joint change is negative (down to one
    row, whose exact change equals its predicted gain) and reject the column otherwise, so the dense objective is
    monotone.  The default (strict=False) applies the row set left after backoff_steps halvings UNCHECKED, which
    can increase the dense objective when G is strongly coupled (measured 2026-09-08: global-endpoint G x65-x100,
    module G on q_proj x2.9); it is kept as the default only so that running queues stay bit-identical.
    Returns (What, stats); pack['codes'] is updated in place."""
    dev = W.device
    W = W.float()
    S, ML = scales_from_pack(pack, device=dev)
    C = torch.as_tensor(pack["codes"], device=dev).float()
    lo, hi = (0.0, 3.0) if pack["fmt"] == "Q2_K" else (-4.0, 3.0)
    H = H.float().clone()
    H += torch.eye(H.shape[0], device=dev) * (percdamp * torch.mean(torch.diag(H)))
    D = (S * C - ML) - W                      # current error
    hdiag = torch.diag(H).clamp_min(1e-12)
    gd = torch.ones(W.shape[0], device=dev) if G is None else torch.diag(G.float()).clamp_min(1e-12)
    j0 = float(torch.einsum("ri,ij,rj->", D, H, D) if G is None else torch.einsum("ri,ij,rj,rr->", D, H, D, G.float()))
    M = D @ H if G is None else (G.float() @ D) @ H   # M[r,j] = (G D H)[r,j]; dJ = 2 dd M + G_rr H_jj dd^2
    moved = 0; n_backoff = 0; n_rejected = 0
    for _ in range(sweeps):
        for j in range(W.shape[1]):
            s = S[:, j]
            live = s.abs() > 0
            if not bool(live.any()):
                continue
            s_safe = torch.where(live, s, torch.ones_like(s))
            step = -M[:, j] / (gd * hdiag[j] * s_safe)
            c_new = torch.clamp(torch.round(C[:, j] + step), lo, hi)
            dd = torch.where(live, (c_new - C[:, j]) * s, torch.zeros_like(s))
            gain = 2.0 * dd * M[:, j] + gd * hdiag[j] * dd * dd
            keep = (dd != 0) & torch.isfinite(dd) & (gain < 0)
            if G is not None and bool(keep.any()):   # rows are coupled through G: check the exact joint change, halve until it drops
                Gf = G.float()
                exact = -1.0
                for _b in range(10**6 if strict else backoff_steps):  # strict: terminates at one row (its exact change = its gain < 0)
                    d_try = torch.where(keep, dd, torch.zeros_like(dd))
                    exact = float(2.0 * (d_try * M[:, j]).sum() + hdiag[j] * (d_try * (Gf @ d_try)).sum())
                    if exact < 0.0:
                        break
                    n_backoff += 1
                    if int(keep.sum()) <= 1:
                        keep = torch.zeros_like(keep); break
                    keep = keep & (gain <= torch.quantile(gain[keep].float(), 0.5))
                if strict and exact >= 0.0 and bool(keep.any()):
                    keep = torch.zeros_like(keep); n_rejected += 1
            dd = torch.where(keep, dd, torch.zeros_like(dd))
            if not bool((dd != 0).any()):
                continue
            C[:, j] = torch.where(keep, c_new, C[:, j])
            D[:, j] += dd
            M += (dd if G is None else G.float() @ dd).unsqueeze(1) * H[j, :].unsqueeze(0)
            moved += int((dd != 0).sum())
    pack["codes"] = C.to(torch.int8).cpu().numpy()
    j1 = float(torch.einsum("ri,ij,rj->", D, H, D) if G is None else torch.einsum("ri,ij,rj,rr->", D, H, D, G.float()))
    return (S * C - ML), dict(obj_before=j0, obj_after=j1, moved=moved, frac_moved=moved / C.numel(), backoff=n_backoff, rejected=n_rejected)

# ----------------------------------------------------------------------------------------------------- packers
def _f16_bytes(x: np.ndarray) -> np.ndarray:  # (...,) fp32 (fp16-exact) -> (..., 2) uint8 little-endian
    return np.ascontiguousarray(x.astype("<f2")).view(np.uint8).reshape(x.shape + (2,))


def pack_q2k(pack) -> np.ndarray:
    """-> uint8 (rows, n_sb*84) in ggml block_q2_K layout: scales[16] (sc | m<<4), qs[64], d, dmin."""
    codes = pack["codes"].astype(np.uint8); rows, n = codes.shape; n_sb = n // QK
    scales = (pack["sc"].astype(np.uint8) | (pack["m"].astype(np.uint8) << 4)).reshape(rows, n_sb, 16)
    c = codes.reshape(rows, n_sb, 2, 4, 32)
    qs = np.zeros((rows, n_sb, 2, 32), np.uint8)
    for s in range(4):
        qs |= (c[:, :, :, s, :] << (2 * s)).astype(np.uint8)
    out = np.concatenate([scales, qs.reshape(rows, n_sb, 64), _f16_bytes(pack["d"]), _f16_bytes(pack["dmin"])], axis=2)
    assert out.shape[2] == 84
    return np.ascontiguousarray(out.reshape(rows, n_sb * 84))


def pack_q3k(pack) -> np.ndarray:
    """-> uint8 (rows, n_sb*110) in ggml block_q3_K layout: hmask[32], qs[64], scales[12] (6-bit packed), d."""
    codes = pack["codes"].astype(np.int16); rows, n = codes.shape; n_sb = n // QK
    flag = (codes >= 0).astype(np.uint8).reshape(rows, n_sb, 8, 32)  # hmask bit 1 <=> no -4 offset (q >= 0)
    hmask = np.zeros((rows, n_sb, 32), np.uint8)
    for b in range(8):
        hmask |= (flag[:, :, b, :] << b).astype(np.uint8)
    ql = (codes & 3).astype(np.uint8).reshape(rows, n_sb, 2, 4, 32)
    qs = np.zeros((rows, n_sb, 2, 32), np.uint8)
    for s in range(4):
        qs |= (ql[:, :, :, s, :] << (2 * s)).astype(np.uint8)
    v = (pack["sc"].astype(np.int16) + 32).astype(np.uint8).reshape(rows, n_sb, 16)  # 0..63
    lo = v & 0x0F; hi = (v >> 4) & 0x03
    sb_lo = (lo[:, :, :8] | (lo[:, :, 8:] << 4)).astype(np.uint8)  # bytes 0..7
    sb_hi = np.zeros((rows, n_sb, 4), np.uint8)  # bytes 8..11: scale 4i+k -> byte k, bits 2i
    for i in range(4):
        sb_hi |= (hi[:, :, 4 * i:4 * i + 4] << (2 * i)).astype(np.uint8)
    out = np.concatenate([hmask, qs.reshape(rows, n_sb, 64), sb_lo, sb_hi, _f16_bytes(pack["d"])], axis=2)
    assert out.shape[2] == 110
    return np.ascontiguousarray(out.reshape(rows, n_sb * 110))


def pack_any(pack) -> np.ndarray:
    return pack_q2k(pack) if pack["fmt"] == "Q2_K" else pack_q3k(pack)


def dequant_from_pack(pack) -> np.ndarray:
    """reference dequantization of our own pack (float32), for checks against gguf-py."""
    codes = pack["codes"].astype(np.float32); rows, n = codes.shape; n_sb = n // QK
    c = codes.reshape(rows, n_sb, 16, 16)
    d = pack["d"].reshape(rows, n_sb, 1, 1).astype(np.float32); sc = pack["sc"].reshape(rows, n_sb, 16, 1).astype(np.float32)
    if pack["fmt"] == "Q2_K":
        dmin = pack["dmin"].reshape(rows, n_sb, 1, 1).astype(np.float32); m = pack["m"].reshape(rows, n_sb, 16, 1).astype(np.float32)
        return ((d * sc) * c - dmin * m).reshape(rows, n)
    return ((d * sc) * c).reshape(rows, n)


def selftest():
    import gguf
    import gguf.quants
    torch.manual_seed(0)
    for fmt in ("Q2_K", "Q3_K"):
        for ao, sg in ((False, False), (False, True), (True, True)):
            rows, n_in = 64, 512
            W = torch.randn(rows, n_in) * 0.05
            X = torch.randn(4096, n_in) * (1 + torch.rand(n_in) * 3)
            H = X.t() @ X / X.shape[0]
            Q, pack = gptq_quantize_kq(W, H, fmt, act_order=ao, static_groups=sg)
            packed = pack_any(pack)
            qt = getattr(gguf.GGMLQuantizationType, fmt)
            deq = gguf.quants.dequantize(packed, qt).reshape(rows, n_in)
            ours = dequant_from_pack(pack)
            e_gguf = np.abs(deq - Q.numpy()).max(); e_ours = np.abs(ours - Q.numpy()).max()
            rel = float(torch.norm(Q - W) / torch.norm(W))
            bpw = packed.nbytes * 8 / (rows * n_in)
            print(f"[{fmt} ao={int(ao)} sg={int(sg)}] max|gguf-dequant - GPTQ| = {e_gguf:.3e}  max|own-dequant - GPTQ| = {e_ours:.3e}  "
                  f"rel err vs W {rel:.3f}  bpw {bpw:.4f}  codes range [{pack['codes'].min()}, {pack['codes'].max()}]")
            assert e_gguf < 1e-6, "pack/dequant mismatch"
    for fmt in ("Q2_K", "Q3_K"):
        W = torch.randn(300, 1024) * 0.02
        pack = rtn_kq(W, fmt, chunk_rows=128)
        packed = pack_any(pack)
        deq = gguf.quants.dequantize(packed, getattr(gguf.GGMLQuantizationType, fmt)).reshape(W.shape)
        e = np.abs(deq - dequant_from_pack(pack)).max(); rel = float(np.linalg.norm(deq - W.numpy()) / np.linalg.norm(W.numpy()))
        print(f"[rtn {fmt}] max|gguf-dequant - own| = {e:.1e}  rel err vs W {rel:.3f}  bpw {packed.nbytes*8/W.numel():.4f}")
        assert e < 1e-6
    print("selftest OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest:
        selftest()
