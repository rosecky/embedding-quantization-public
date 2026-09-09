"""Retrieval evaluation metrics.

``evaluate_run`` wraps pytrec_eval for nDCG@k / Recall@k / MAP@k, computes
MRR@10 by hand (pytrec_eval's recip_rank is not cutoff-aware), and returns
both corpus-level means and per-query values so that ``paired_bootstrap`` can
compare two systems on the same query set.
"""

from __future__ import annotations

import argparse
import random
import sys

import pytrec_eval


def _mrr_at_k(run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    """Per-query MRR@k: reciprocal rank of the first grade>=1 doc among the top k."""
    out: dict[str, float] = {}
    for qid, doc_scores in run.items():
        rel_docs = {d for d, g in qrels.get(qid, {}).items() if g >= 1}
        if not rel_docs:
            out[qid] = 0.0
            continue
        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        rr = 0.0
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            if doc_id in rel_docs:
                rr = 1.0 / rank
                break
        out[qid] = rr
    return out


def evaluate_run(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = (10, 20, 100),
) -> dict:
    """Evaluate a retrieval run against qrels.

    Args:
        run: query_id -> {doc_id: score}, higher score = ranked first.
        qrels: query_id -> {doc_id: grade}, grade >= 1 is relevant.
        k_values: cutoffs used for nDCG/Recall (MAP is always @100, MRR always @10,
            matching the metric names below regardless of what's in ``k_values``).

    Returns:
        {
          "mean": {"ndcg@10": ..., "recall@10": ..., ..., "mrr@10": ..., "map@100": ...},
          "per_query": {"ndcg@10": {qid: val, ...}, ...},
        }
    """
    qrels_str = {qid: {did: int(g) for did, g in docs.items()} for qid, docs in qrels.items()}
    # pytrec_eval requires every query in the run to also appear in qrels.
    run = {qid: scores for qid, scores in run.items() if qid in qrels_str}

    measures = {f"ndcg_cut.{k}" for k in k_values} | {f"recall.{k}" for k in k_values} | {"map_cut.100"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_str, measures)
    raw = evaluator.evaluate(run)  # qid -> {measure_name: value}

    per_query: dict[str, dict[str, float]] = {}
    for k in k_values:
        per_query[f"ndcg@{k}"] = {qid: raw[qid][f"ndcg_cut_{k}"] for qid in raw}
        per_query[f"recall@{k}"] = {qid: raw[qid][f"recall_{k}"] for qid in raw}
    per_query["map@100"] = {qid: raw[qid]["map_cut_100"] for qid in raw}
    per_query["mrr@10"] = _mrr_at_k(run, qrels_str, k=10)

    mean = {
        name: (sum(vals.values()) / len(vals) if vals else 0.0) for name, vals in per_query.items()
    }
    return {"mean": mean, "per_query": per_query}


def paired_bootstrap(
    metric_per_query_a: dict[str, float],
    metric_per_query_b: dict[str, float],
    n: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for the mean difference (A - B) of a per-query metric.

    Resamples query ids (with replacement) ``n`` times from the queries common
    to both inputs and returns the observed mean difference plus a 95% CI.
    """
    common = sorted(set(metric_per_query_a) & set(metric_per_query_b))
    if not common:
        raise ValueError("metric_per_query_a and metric_per_query_b share no query ids")

    diffs = [metric_per_query_a[q] - metric_per_query_b[q] for q in common]
    observed_mean_diff = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n_q = len(diffs)
    boot_means = []
    for _ in range(n):
        resample = [diffs[rng.randrange(n_q)] for _ in range(n_q)]
        boot_means.append(sum(resample) / n_q)
    boot_means.sort()

    lo_idx = int(0.025 * n)
    hi_idx = min(int(0.975 * n), n - 1)
    return {
        "mean_diff": observed_mean_diff,
        "ci95_low": boot_means[lo_idx],
        "ci95_high": boot_means[hi_idx],
        "n_queries": n_q,
        "n_bootstrap": n,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Self-test: a hand-computed toy example
# ---------------------------------------------------------------------------


def _selftest() -> bool:
    # Two queries, 4 docs each judged with grades 0/1/2 (only >=1 stored, per
    # the RetrievalDataset qrels convention -- unjudged docs are implicitly 0).
    qrels = {
        "q1": {"d1": 2, "d2": 1},  # d3, d4 unjudged/non-relevant
        "q2": {"d5": 1},
    }
    # q1: perfect ranking puts d1 (grade 2) first, d2 (grade 1) second.
    run = {
        "q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0, "d4": 0.5},
        # q2: relevant doc d5 ranked 2nd out of 2.
        "q2": {"d6": 1.0, "d5": 0.5},
    }

    result = evaluate_run(run, qrels, k_values=(10,))
    mean = result["mean"]

    ok = True

    def check(name: str, got: float, expected: float, tol: float = 1e-6) -> None:
        nonlocal ok
        status = "OK" if abs(got - expected) < tol else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {name}: got={got:.6f} expected={expected:.6f}")

    # q1 is perfectly ranked (ideal DCG order) -> ndcg@10 = 1.0 for q1.
    # q2 has 1 relevant doc out of 2, ranked at position 2 -> recip rank 0.5.
    # Hand-computed nDCG for q2: DCG = 1/log2(2+1) = 1/log2(3) = 0.6309..., IDCG (d5 first) = 1/log2(2)=1.0
    # -> ndcg@10(q2) = 0.630930
    import math

    ndcg_q2 = (1.0 / math.log2(2 + 1)) / (1.0 / math.log2(1 + 1))
    check("ndcg@10 (q1, perfect ranking)", result["per_query"]["ndcg@10"]["q1"], 1.0)
    check("ndcg@10 (q2, rel doc at rank 2)", result["per_query"]["ndcg@10"]["q2"], ndcg_q2)
    check("mean ndcg@10", mean["ndcg@10"], (1.0 + ndcg_q2) / 2)

    # recall@10: q1 has 2 relevant docs, both retrieved among top 10 -> 1.0. q2: 1/1 -> 1.0.
    check("recall@10 (q1)", result["per_query"]["recall@10"]["q1"], 1.0)
    check("recall@10 (q2)", result["per_query"]["recall@10"]["q2"], 1.0)

    # mrr@10: q1 first relevant at rank 1 -> 1.0. q2 first relevant at rank 2 -> 0.5.
    check("mrr@10 (q1)", result["per_query"]["mrr@10"]["q1"], 1.0)
    check("mrr@10 (q2)", result["per_query"]["mrr@10"]["q2"], 0.5)
    check("mean mrr@10", mean["mrr@10"], 0.75)

    # map@100: q1 both relevant docs at ranks 1,2 -> AP = (1/1 + 2/2)/2 = 1.0.
    # q2: 1 relevant doc at rank 2 -> AP = (1/2)/1 = 0.5.
    check("map@100 (q1)", result["per_query"]["map@100"]["q1"], 1.0)
    check("map@100 (q2)", result["per_query"]["map@100"]["q2"], 0.5)

    # paired_bootstrap sanity: comparing a metric to itself must give mean_diff == 0
    # and a CI that contains 0.
    boot = paired_bootstrap(mean_a := result["per_query"]["ndcg@10"], mean_a, n=200, seed=0)
    check("paired_bootstrap self-diff mean", boot["mean_diff"], 0.0)
    ok = ok and (boot["ci95_low"] <= 0.0 <= boot["ci95_high"])
    print(f"[{'OK' if boot['ci95_low'] <= 0.0 <= boot['ci95_high'] else 'FAIL'}] "
          f"paired_bootstrap self-diff CI contains 0: {boot['ci95_low']:.4f}..{boot['ci95_high']:.4f}")

    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="eq.metrics utilities")
    parser.add_argument("--selftest", action="store_true", help="run the hand-computed self-test")
    args = parser.parse_args()
    if args.selftest:
        ok = _selftest()
        sys.exit(0 if ok else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
