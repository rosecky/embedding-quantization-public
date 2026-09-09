"""Retrieval-relevant agreement metrics between a teacher score matrix and a student score matrix.

All functions take dense score matrices S_T, S_S of shape (n_queries, n_docs) (float32) and are
row-aligned. Row-wise results are returned so paired bootstrap over queries is possible.

Two relevance definitions are supported:
  (1) global threshold: teacher positives = {S_T >= tau}; student positives = {S_S >= theta}, theta calibrated
      on train queries (so the student does NOT need to reproduce the teacher's numeric cosine);
  (2) query-adaptive top-k boundary: teacher positives = top-k(S_T); student positives = top-k(S_S).
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def topk_idx(S: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores per row, sorted descending. Shape (n, k)."""
    k = min(k, S.shape[1])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    order = np.argsort(-S[rows, part], axis=1)
    return part[rows, order]


def kth_score(S: np.ndarray, k: int) -> np.ndarray:
    """Score of the k-th best doc per query (query-adaptive boundary)."""
    k = min(k, S.shape[1])
    return -np.partition(-S, k - 1, axis=1)[:, k - 1]


# ----------------------------------------------------------------------------------------------
# top-k neighborhood agreement
# ----------------------------------------------------------------------------------------------

def topk_overlap(S_T: np.ndarray, S_S: np.ndarray, k: int) -> np.ndarray:
    """Per-query |topk_T & topk_S| / k."""
    t = topk_idx(S_T, k)
    s = topk_idx(S_S, k)
    out = np.empty(S_T.shape[0], dtype=np.float32)
    for i in range(S_T.shape[0]):
        out[i] = len(set(t[i].tolist()) & set(s[i].tolist())) / k
    return out


def topk_recall_at(S_T: np.ndarray, S_S: np.ndarray, k: int, k_cand: int) -> np.ndarray:
    """Fraction of teacher top-k found within student top-k_cand (candidate-generation recall)."""
    t = topk_idx(S_T, k)
    s = topk_idx(S_S, k_cand)
    out = np.empty(S_T.shape[0], dtype=np.float32)
    for i in range(S_T.shape[0]):
        out[i] = len(set(t[i].tolist()) & set(s[i].tolist())) / k
    return out


def relevant_set_order_agreement(S_T: np.ndarray, S_S: np.ndarray, k: int, min_gap: float = 0.0) -> np.ndarray:
    """Pairwise order agreement (Kendall-tau-like, in [0,1]) restricted to the teacher top-k set.

    For each query, over pairs (i,j) in teacher top-k with s_T[i]-s_T[j] > min_gap, fraction of pairs where
    the student agrees on the order. 1.0 = perfect local ranking; 0.5 = random.
    """
    t = topk_idx(S_T, k)
    out = np.full(S_T.shape[0], np.nan, dtype=np.float32)
    for i in range(S_T.shape[0]):
        idx = t[i]
        st = S_T[i, idx]
        ss = S_S[i, idx]
        dt = st[:, None] - st[None, :]
        ds = ss[:, None] - ss[None, :]
        mask = dt > min_gap
        n = mask.sum()
        if n == 0:
            continue
        out[i] = float((ds[mask] > 0).sum()) / n
    return out


# ----------------------------------------------------------------------------------------------
# global threshold agreement
# ----------------------------------------------------------------------------------------------

def calibrate_theta(S_T_train: np.ndarray, S_S_train: np.ndarray, tau: float, mode: str = "f1",
                    n_grid: int = 400) -> float:
    """Pick the student threshold theta on TRAIN queries.

    mode='f1'   : maximize micro F1 of {S_S>=theta} against {S_T>=tau}
    mode='rate' : match the teacher's positive rate (quantile matching)
    """
    pos = S_T_train >= tau
    if mode == "rate":
        rate = pos.mean()
        return float(np.quantile(S_S_train, 1.0 - rate)) if rate > 0 else float(S_S_train.max())
    qs = np.quantile(S_S_train, np.linspace(0.5, 0.99999, n_grid))
    best, best_t = -1.0, float(qs[-1])
    P = pos.sum()
    for th in qs:
        pred = S_S_train >= th
        tp = np.logical_and(pred, pos).sum()
        fp = pred.sum() - tp
        fn = P - tp
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, best_t = f1, float(th)
    return best_t


def threshold_confusion(S_T: np.ndarray, S_S: np.ndarray, tau: float, theta: float, guard: float = 0.0) -> dict:
    """Micro-averaged and per-query threshold-crossing statistics.

    guard: half-width of the teacher guard band. Pairs with |s_T - tau| <= guard are the 'uncertain' class and
    are reported separately (not counted as FP/FN).
    """
    pos = S_T >= tau + guard
    neg = S_T < tau - guard
    band = ~(pos | neg)
    pred = S_S >= theta
    tp = np.logical_and(pred, pos)
    fp = np.logical_and(pred, neg)
    fn = np.logical_and(~pred, pos)
    n_pos = pos.sum(1)
    n_neg = neg.sum(1)
    n_pred = pred.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_q_recall = np.where(n_pos > 0, tp.sum(1) / np.maximum(n_pos, 1), np.nan)
        per_q_precision = np.where(n_pred > 0, tp.sum(1) / np.maximum(n_pred, 1), np.nan)
        per_q_fp_rate = fp.sum(1) / np.maximum(n_neg, 1)
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(
        micro_precision=float(prec), micro_recall=float(rec), micro_f1=float(f1),
        fp_crossings=int(FP), fn_crossings=int(FN), tp=int(TP),
        n_teacher_pos=int(pos.sum()), n_band=int(band.sum()), n_student_pos=int(pred.sum()),
        fp_per_query=float(fp.sum(1).mean()), fn_per_query=float(fn.sum(1).mean()),
        band_pred_pos_rate=float(pred[band].mean()) if band.any() else float("nan"),
        per_query_recall=per_q_recall.astype(np.float32),
        per_query_precision=per_q_precision.astype(np.float32),
        per_query_fp_rate=per_q_fp_rate.astype(np.float32),
        queries_with_pos=int((n_pos > 0).sum()),
    )


# ----------------------------------------------------------------------------------------------
# ground truth
# ----------------------------------------------------------------------------------------------

def ndcg_at_k(S: np.ndarray, rel: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-query nDCG@k with graded relevance matrix rel (n_q, n_docs) (0 = non-relevant). Linear gains."""
    top = topk_idx(S, k)
    rows = np.arange(S.shape[0])[:, None]
    g = rel[rows, top].astype(np.float64)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (g * disc).sum(1)
    ideal = -np.sort(-rel.astype(np.float64), axis=1)[:, :k]
    idcg = (ideal * disc[: ideal.shape[1]]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(idcg > 0, dcg / idcg, np.nan)
    return out.astype(np.float32)


def recall_at_k(S: np.ndarray, rel: np.ndarray, k: int) -> np.ndarray:
    top = topk_idx(S, k)
    rows = np.arange(S.shape[0])[:, None]
    hit = (rel[rows, top] > 0).sum(1)
    n_rel = (rel > 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n_rel > 0, hit / np.maximum(n_rel, 1), np.nan)
    return out.astype(np.float32)


def mrr_at_k(S: np.ndarray, rel: np.ndarray, k: int = 10) -> np.ndarray:
    top = topk_idx(S, k)
    rows = np.arange(S.shape[0])[:, None]
    hits = rel[rows, top] > 0
    out = np.zeros(S.shape[0], dtype=np.float32)
    for i in range(S.shape[0]):
        w = np.where(hits[i])[0]
        out[i] = 1.0 / (w[0] + 1) if len(w) else 0.0
    n_rel = (rel > 0).sum(1)
    return np.where(n_rel > 0, out, np.nan).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------------------------

def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean difference a-b with percentile CI over queries (paired)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    dm = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(dm, alpha / 2)), float(np.quantile(dm, 1 - alpha / 2))


# ----------------------------------------------------------------------------------------------
# global-threshold calibration diagnostic (is a single tau meaningful across queries?)
# ----------------------------------------------------------------------------------------------

def global_threshold_diagnostic(S_T: np.ndarray, rel: np.ndarray | None, k: int = 10, n_grid: int = 200) -> dict:
    """How well does one global cosine threshold reproduce per-query top-k sets / ground truth?"""
    kth = kth_score(S_T, k)
    out = dict(k=k, kth_score_mean=float(kth.mean()), kth_score_std=float(kth.std()),
               kth_score_p10=float(np.quantile(kth, 0.1)), kth_score_p90=float(np.quantile(kth, 0.9)),
               top1_score_mean=float(S_T.max(1).mean()), top1_score_std=float(S_T.max(1).std()))
    topk_mask = np.zeros_like(S_T, dtype=bool)
    rows = np.arange(S_T.shape[0])[:, None]
    topk_mask[rows, topk_idx(S_T, k)] = True
    grid = np.quantile(S_T, np.linspace(0.9, 0.99999, n_grid))
    best = (-1.0, None)
    for t in grid:
        pred = S_T >= t
        tp = (pred & topk_mask).sum()
        f1 = 2 * tp / max(pred.sum() + topk_mask.sum(), 1)
        if f1 > best[0]:
            best = (f1, float(t))
    out["best_tau_for_topk"] = best[1]
    out["best_f1_tau_vs_topk"] = float(best[0])
    pred = S_T >= best[1]
    out["set_size_at_best_tau_mean"] = float(pred.sum(1).mean())
    out["set_size_at_best_tau_p10_p90"] = [float(np.quantile(pred.sum(1), 0.1)), float(np.quantile(pred.sum(1), 0.9))]
    out["queries_with_empty_set_at_best_tau"] = float((pred.sum(1) == 0).mean())
    if rel is not None:
        gt = rel > 0
        best = (-1.0, None)
        for t in grid:
            pred = S_T >= t
            tp = (pred & gt).sum()
            f1 = 2 * tp / max(pred.sum() + gt.sum(), 1)
            if f1 > best[0]:
                best = (f1, float(t))
        out["best_tau_for_groundtruth"] = best[1]
        out["best_f1_tau_vs_groundtruth"] = float(best[0])
        f1s = []
        for i in range(S_T.shape[0]):
            if gt[i].sum() == 0:
                continue
            order = np.argsort(-S_T[i])
            hits = np.cumsum(gt[i][order])
            n = np.arange(1, len(order) + 1)
            f1 = 2 * hits / (n + gt[i].sum())
            f1s.append(f1.max())
        out["per_query_oracle_tau_f1_vs_groundtruth"] = float(np.mean(f1s)) if f1s else float("nan")
    return out
