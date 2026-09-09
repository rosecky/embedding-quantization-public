"""Uniform retrieval-dataset interface for the embedding-quantization research project.

This module implements a single normalized on-disk format (jsonl.gz + tsv + json)
under ``data/<name>/`` and a loader (:func:`load_dataset`) that reads it back into a
:class:`RetrievalDataset`. A private/internal corpus can later be plugged in by
writing the same five files (``corpus.jsonl.gz``, ``queries.jsonl.gz``,
``qrels.tsv``, ``splits.json``, ``meta.json``) without touching any downstream
experiment code.

Building a dataset from its public source (Hugging Face) is a separate step,
performed by :func:`build_and_save` (driven by ``scripts/prepare_data.py``), which
downloads, normalizes, deduplicates, splits, and finally writes the on-disk
format via :func:`save_dataset`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------


@dataclass
class RetrievalDataset:
    name: str
    corpus: dict[str, dict]  # doc_id -> {"title": str, "text": str}
    queries: dict[str, str]  # query_id -> text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: grade}, grade >= 1 relevant
    splits: dict[str, list[str]]  # {"train": [qids], "dev": [qids], "test": [qids]}
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm_text(s: str | None) -> str:
    """Whitespace-collapsed, lowercased text used for hashing/comparison."""
    return _WS_RE.sub(" ", (s or "").strip().lower())


def _doc_full_text(doc: dict) -> str:
    title = doc.get("title") or ""
    text = doc.get("text") or ""
    return f"{title} {text}".strip()


def _content_hash(s: str) -> str:
    return hashlib.sha256(_norm_text(s).encode("utf-8")).hexdigest()


def _word_shingles(s: str, k: int = 5, max_chars: int = 2000) -> set[str]:
    """Word k-shingles used as the MinHash input set for near-dup detection.

    ``max_chars`` truncates very long documents before shingling purely for
    runtime (shingling scales with document length); this is a deliberate
    approximation, documented in research/datasets.md.
    """
    words = _norm_text(s)[:max_chars].split()
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _median_token_len(texts: list[str]) -> float:
    lens = [len((t or "").split()) for t in texts]
    return float(statistics.median(lens)) if lens else 0.0


# ---------------------------------------------------------------------------
# Leakage-guard primitives
# ---------------------------------------------------------------------------


def exact_dedup(corpus: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str], int]:
    """Collapse exact-duplicate documents (normalized whitespace/lowercase hash).

    The first-seen id (in the corpus dict's iteration order, which is
    deterministic given a deterministic source) becomes canonical.

    Returns ``(deduped_corpus, remap, num_removed)`` where ``remap`` maps every
    original doc id (including canonical ones, mapped to themselves) to its
    canonical id.
    """
    seen_hash_to_id: dict[str, str] = {}
    remap: dict[str, str] = {}
    deduped: dict[str, dict] = {}
    for doc_id, doc in corpus.items():
        h = _content_hash(_doc_full_text(doc))
        canonical = seen_hash_to_id.get(h)
        if canonical is not None:
            remap[doc_id] = canonical
        else:
            seen_hash_to_id[h] = doc_id
            remap[doc_id] = doc_id
            deduped[doc_id] = doc
    return deduped, remap, len(corpus) - len(deduped)


def remap_qrels(qrels: dict[str, dict[str, int]], remap: dict[str, str]) -> dict[str, dict[str, int]]:
    """Rewrite doc ids in qrels through a dedup remap, keeping max grade on collision."""
    out: dict[str, dict[str, int]] = {}
    for qid, docs in qrels.items():
        merged: dict[str, int] = {}
        for did, grade in docs.items():
            canon = remap.get(did, did)
            if canon in merged:
                merged[canon] = max(merged[canon], grade)
            else:
                merged[canon] = grade
        out[qid] = merged
    return out


def _minhash_params(n_docs: int) -> tuple[int, int]:
    """(num_perm, max_chars) tuned down for large corpora to bound runtime."""
    if n_docs > 50_000:
        return 32, 800
    return 64, 2000


def near_duplicate_groups(
    corpus: dict[str, dict],
    threshold: float = 0.9,
    k: int = 5,
    num_perm: int | None = None,
    max_chars: int | None = None,
) -> list[list[str]]:
    """MinHash-LSH near-duplicate groups (Jaccard >= threshold over word k-shingles).

    Returns a sorted list of sorted doc-id groups (each of size >= 2). Documents
    are never removed by this check -- callers only record the groups.
    """
    from datasketch import MinHash, MinHashLSH

    n_perm, n_chars = _minhash_params(len(corpus))
    if num_perm is not None:
        n_perm = num_perm
    if max_chars is not None:
        n_chars = max_chars

    lsh = MinHashLSH(threshold=threshold, num_perm=n_perm)
    minhashes: dict[str, "MinHash"] = {}
    for doc_id, doc in corpus.items():
        shingles = _word_shingles(_doc_full_text(doc), k=k, max_chars=n_chars)
        if not shingles:
            continue
        mh = MinHash(num_perm=n_perm)
        mh.update_batch([s.encode("utf-8") for s in shingles])
        minhashes[doc_id] = mh
        lsh.insert(doc_id, mh)

    parent: dict[str, str] = {doc_id: doc_id for doc_id in minhashes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for doc_id, mh in minhashes.items():
        for neighbor in lsh.query(mh):
            if neighbor != doc_id:
                union(doc_id, neighbor)

    groups: dict[str, list[str]] = {}
    for doc_id in minhashes:
        groups.setdefault(find(doc_id), []).append(doc_id)

    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def query_equals_doc(queries: dict[str, str], corpus: dict[str, dict]) -> list[str]:
    """Query ids whose text is an exact (normalized) match of some document's text.

    ArguAna-style: the query set is drawn from the same argument pool as the
    corpus, so some queries are byte-for-byte (post-normalization) documents.
    """
    doc_hashes: set[str] = {_content_hash(_doc_full_text(doc)) for doc in corpus.values()}
    return sorted(qid for qid, text in queries.items() if _content_hash(text) in doc_hashes)


def flag_test_queries_with_train_dup_relevant(
    qrels: dict[str, dict[str, int]],
    splits: dict[str, list[str]],
    near_dup_groups: list[list[str]],
) -> list[str]:
    """Test queries whose relevant doc is a near-duplicate of a train query's relevant doc."""
    doc_to_group: dict[str, int] = {}
    for gi, group in enumerate(near_dup_groups):
        for d in group:
            doc_to_group[d] = gi

    def relevant_group_ids(qid: str) -> set[int]:
        return {
            doc_to_group[did]
            for did, grade in qrels.get(qid, {}).items()
            if grade >= 1 and did in doc_to_group
        }

    train_group_to_qids: dict[int, set[str]] = {}
    for qid in splits.get("train", []):
        for gid in relevant_group_ids(qid):
            train_group_to_qids.setdefault(gid, set()).add(qid)

    flagged = []
    for qid in splits.get("test", []):
        if relevant_group_ids(qid) & train_group_to_qids.keys():
            flagged.append(qid)
    return sorted(flagged)


def subsample_corpus(
    corpus: dict[str, dict],
    qrels: dict[str, dict[str, int]],
    max_docs: int,
    seed: int = 0,
) -> tuple[dict[str, dict], dict]:
    """Deterministically shrink ``corpus`` to <= ``max_docs`` docs.

    Every doc referenced by ``qrels`` is kept; the remainder is filled with a
    seeded random sample of "negative" (unjudged) documents. No-op if the
    corpus is already small enough.
    """
    if len(corpus) <= max_docs:
        return corpus, {"applied": False}

    keep_ids = {did for docs in qrels.values() for did in docs} & corpus.keys()
    rng = random.Random(seed)
    candidates = [d for d in corpus if d not in keep_ids]
    rng.shuffle(candidates)
    remaining = max(max_docs - len(keep_ids), 0)
    extra = candidates[:remaining]
    final_ids = keep_ids | set(extra)
    new_corpus = {d: corpus[d] for d in corpus if d in final_ids}
    stats = {
        "applied": True,
        "seed": seed,
        "raw_num_docs": len(corpus),
        "kept_qrels_docs": len(keep_ids),
        "kept_random_negatives": len(extra),
        "final_num_docs": len(new_corpus),
    }
    return new_corpus, stats


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------


def ensure_splits(
    query_pools: dict[str, list[str]],
    seed: int = 0,
    dev_frac_from_train: float = 0.10,
    test_frac_from_train: float = 0.30,
) -> tuple[dict[str, list[str]], str]:
    """Normalize whatever native splits a source provides into train/dev/test.

    ``query_pools`` keys must already be normalized to a subset of
    {"train", "dev", "test"}. Missing splits are synthesized deterministically
    (seed 0); a query id is never placed in two splits. Returns
    ``(splits, split_source_tag)``.
    """
    pools = {k: list(v) for k, v in query_pools.items() if v}
    have_train, have_dev, have_test = "train" in pools, "dev" in pools, "test" in pools
    rng = random.Random(seed)

    if have_train and have_dev and have_test:
        return pools, "native"

    if have_train and have_test and not have_dev:
        train_ids = pools["train"][:]
        rng.shuffle(train_ids)
        n_dev = max(1, round(len(train_ids) * dev_frac_from_train))
        return (
            {
                "train": sorted(train_ids[n_dev:]),
                "dev": sorted(train_ids[:n_dev]),
                "test": sorted(pools["test"]),
            },
            "dev_from_train",
        )

    if have_train and have_dev and not have_test:
        train_ids = pools["train"][:]
        rng.shuffle(train_ids)
        n_test = max(1, round(len(train_ids) * test_frac_from_train))
        return (
            {
                "train": sorted(train_ids[n_test:]),
                "dev": sorted(pools["dev"]),
                "test": sorted(train_ids[:n_test]),
            },
            "test_from_train",
        )

    if have_test and not have_train and not have_dev:
        test_ids = pools["test"][:]
        rng.shuffle(test_ids)
        n = len(test_ids)
        n_train = round(n * 0.40)
        n_dev = round(n * 0.10)
        return (
            {
                "train": sorted(test_ids[:n_train]),
                "dev": sorted(test_ids[n_train : n_train + n_dev]),
                "test": sorted(test_ids[n_train + n_dev :]),
            },
            "synthetic_from_test",
        )

    if have_train and not have_dev and not have_test:
        train_ids = pools["train"][:]
        rng.shuffle(train_ids)
        n = len(train_ids)
        n_tr = round(n * 0.70)
        n_dev = round(n * 0.10)
        return (
            {
                "train": sorted(train_ids[:n_tr]),
                "dev": sorted(train_ids[n_tr : n_tr + n_dev]),
                "test": sorted(train_ids[n_tr + n_dev :]),
            },
            "synthetic_from_train",
        )

    if have_dev and not have_train and not have_test:
        dev_ids = pools["dev"][:]
        rng.shuffle(dev_ids)
        n = len(dev_ids)
        n_tr = round(n * 0.40)
        n_dv = round(n * 0.10)
        return (
            {
                "train": sorted(dev_ids[:n_tr]),
                "dev": sorted(dev_ids[n_tr : n_tr + n_dv]),
                "test": sorted(dev_ids[n_tr + n_dv :]),
            },
            "synthetic_from_dev",
        )

    raise ValueError(f"Cannot construct train/dev/test splits from pools: {sorted(pools)}")


def assert_disjoint_splits(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name, ids in splits.items():
        for qid in ids:
            if qid in seen:
                raise AssertionError(
                    f"query id {qid!r} appears in both split {seen[qid]!r} and {split_name!r}"
                )
            seen[qid] = split_name


# ---------------------------------------------------------------------------
# Source fetchers (raw, un-normalized) -- one per public dataset
# ---------------------------------------------------------------------------
#
# Each fetcher returns: (corpus, queries, qrels_by_split, meta_extra, max_docs)
#   corpus:          doc_id -> {"title", "text"}
#   queries:         query_id -> text   (superset across all native splits)
#   qrels_by_split:  normalized_split_name -> {query_id: {doc_id: grade}}
#                    normalized_split_name in {"train", "dev", "test"}
#   meta_extra:      dict merged into the final meta.json (language/source/license/notes)
#   max_docs:        int or None -- passed through to subsample_corpus in _build_dataset


def _load_beir_raw(hf_name: str) -> tuple[dict, dict, dict]:
    from datasets import load_dataset

    corpus_ds = load_dataset(f"BeIR/{hf_name}", "corpus", split="corpus")
    corpus = {
        str(row["_id"]): {"title": row.get("title") or "", "text": row.get("text") or ""}
        for row in corpus_ds
    }

    queries_ds = load_dataset(f"BeIR/{hf_name}", "queries", split="queries")
    queries = {str(row["_id"]): row.get("text") or "" for row in queries_ds}

    qrels_dd = load_dataset(f"BeIR/{hf_name}-qrels")
    split_map = {"train": "train", "validation": "dev", "dev": "dev", "test": "test"}
    qrels_by_split: dict[str, dict[str, dict[str, int]]] = {}
    for hf_split in qrels_dd.keys():
        norm_split = split_map.get(hf_split, hf_split)
        bucket = qrels_by_split.setdefault(norm_split, {})
        for row in qrels_dd[hf_split]:
            qid, did = str(row["query-id"]), str(row["corpus-id"])
            grade = max(int(round(row["score"])), 0)
            bucket.setdefault(qid, {})[did] = grade

    return corpus, queries, qrels_by_split


def _fetch_scifact():
    corpus, queries, qrels_by_split = _load_beir_raw("scifact")
    meta = {
        "language": "en",
        "source": "BeIR/scifact + BeIR/scifact-qrels (Hugging Face)",
        "license": (
            "Ambiguous across sources: original SciFact paper states CC BY-NC 2.0; "
            "the BeIR/scifact HF dataset card tags cc-by-sa-4.0 for the repackaging. "
            "Treat as research/non-commercial use."
        ),
        "notes": ["Scientific claim verification recast as claim -> supporting-abstract retrieval."],
    }
    return corpus, queries, qrels_by_split, meta, None


def _fetch_nfcorpus():
    corpus, queries, qrels_by_split = _load_beir_raw("nfcorpus")
    meta = {
        "language": "en",
        "source": "BeIR/nfcorpus + BeIR/nfcorpus-qrels (Hugging Face)",
        "license": (
            "Not formally reported by the NFCorpus authors; the BeIR/nfcorpus HF dataset card "
            "tags cc-by-sa-4.0 for the repackaging."
        ),
        "notes": [
            "Graded relevance (1=partially relevant, 2=highly relevant); native train/dev/test "
            "splits from NutritionFacts.org queries.",
        ],
    }
    return corpus, queries, qrels_by_split, meta, None


def _fetch_fiqa():
    corpus, queries, qrels_by_split = _load_beir_raw("fiqa")
    meta = {
        "language": "en",
        "source": "BeIR/fiqa + BeIR/fiqa-qrels (Hugging Face)",
        "license": (
            "Not formally reported (FiQA-2018 challenge data); the BeIR/fiqa HF dataset card "
            "tags cc-by-sa-4.0 for the repackaging."
        ),
        "notes": ["Financial opinion QA; posts/answers from StackExchange-style finance forums."],
    }
    return corpus, queries, qrels_by_split, meta, None


def _fetch_arguana():
    corpus, queries, qrels_by_split = _load_beir_raw("arguana")
    meta = {
        "language": "en",
        "source": "BeIR/arguana + BeIR/arguana-qrels (Hugging Face)",
        "license": (
            "CC BY 4.0 per the original ArguAna corpus; the BeIR/arguana HF dataset card tags "
            "cc-by-sa-4.0 for the repackaging."
        ),
        "notes": [
            "Test-only (synthetic 40/10/50 split). Queries are counter-arguments drawn from the "
            "same debate corpus as the documents; the task description flags this as a query== "
            "doc leakage risk ('ArguAna-style'), but empirically query_equals_doc_count is 0 for "
            "this BEIR repackaging -- queries and their paired documents are topically related, "
            "not byte-identical after normalization. The guard is still run (see "
            "meta['query_equals_doc_count']).",
            "5 of 1406 raw qrels rows reference a corpus id absent from BeIR/arguana's corpus "
            "split (upstream data defect); those rows are dropped and any query left with zero "
            "relevant docs is removed -- see meta['qrels_dangling_doc_refs_dropped'] / "
            "meta['orphaned_queries_dropped'].",
        ],
    }
    return corpus, queries, qrels_by_split, meta, None


def _fetch_scidocs():
    corpus, queries, qrels_by_split = _load_beir_raw("scidocs")
    meta = {
        "language": "en",
        "source": "BeIR/scidocs + BeIR/scidocs-qrels (Hugging Face)",
        "license": (
            "CC BY 4.0 per the original SciDocs release (allenai/scidocs, Cohan et al. 2020) and the BEIR paper's "
            "dataset table; the BeIR/scidocs HF dataset card tags cc-by-sa-4.0 for the repackaging. Open licence: "
            "usable for the public demo corpus (unlike SciFact, CC BY-NC)."
        ),
        "notes": [
            "Scientific paper title+abstract corpus (25 657 docs); the BEIR task is citation prediction: the query is a "
            "paper title, relevant documents are papers it cites (5 per query, plus 25 explicit non-relevant rows with "
            "score 0 in the qrels, kept with grade 0 -- only grade >= 1 counts as relevant).",
            "Test-only in BEIR (1 000 queries); the registry's synthetic 40/10/50 split (seed 0) gives 400/100/500 "
            "train/dev/test query ids. Nothing is fitted on train/dev for the client work (calibration uses corpus "
            "documents and synthetic queries only).",
            "Many abstracts are empty in the BEIR release (title-only documents); they are kept as-is.",
        ],
    }
    return corpus, queries, qrels_by_split, meta, None


def _fetch_trec_covid():
    corpus, queries, qrels_by_split = _load_beir_raw("trec-covid")
    meta = {
        "language": "en",
        "source": "BeIR/trec-covid + BeIR/trec-covid-qrels (Hugging Face)",
        "license": (
            "TREC-COVID 'Dataset License Agreement' (research use only; derived from the CORD-19 "
            "collection). The BeIR/trec-covid HF dataset card tags cc-by-sa-4.0 for the repackaging."
        ),
        "notes": [
            "Large (171k docs) -- only built when --include-large is passed to prepare_data.py.",
            "Graded 0/1/2; qrels also contained 2 rows with score=-1 (TREC 'not judged' sentinel) "
            "which are clipped to 0 during loading.",
        ],
    }
    return corpus, queries, qrels_by_split, meta, None


def _fetch_webfaq_cs():
    from datasets import load_dataset

    corpus_dd = load_dataset("PaDaS-Lab/webfaq-retrieval", "ces-corpus")
    corpus = {
        str(row["_id"]): {"title": row.get("title") or "", "text": row.get("text") or ""}
        for row in corpus_dd["corpus"]
    }

    queries_dd = load_dataset("PaDaS-Lab/webfaq-retrieval", "ces-queries")
    queries: dict[str, str] = {}
    for split in queries_dd:
        for row in queries_dd[split]:
            queries[str(row["_id"])] = row.get("text") or ""

    qrels_dd = load_dataset("PaDaS-Lab/webfaq-retrieval", "ces-qrels")
    qrels_by_split: dict[str, dict[str, dict[str, int]]] = {}
    for split in qrels_dd:
        bucket = qrels_by_split.setdefault(split, {})
        for row in qrels_dd[split]:
            qid, did = str(row["query-id"]), str(row["corpus-id"])
            bucket.setdefault(qid, {})[did] = max(int(round(row["score"])), 0)

    meta = {
        "language": "cs",
        "source": "PaDaS-Lab/webfaq-retrieval, config 'ces-{corpus,queries,qrels}' (Hugging Face)",
        "license": "CC BY 4.0",
        "notes": [
            "FAQ question -> answer-passage retrieval mined from Common Crawl (2022-2024).",
            "Native splits are train/test only; dev is carved from train (10%, seed 0).",
            "Binary relevance (score=1.0), essentially one gold passage per question.",
        ],
    }
    return corpus, queries, qrels_by_split, meta, None


def _fetch_miracl_fi(max_docs: int = 100_000, seed: int = 0):
    """MIRACL Finnish, fetched via raw file download.

    ``datasets.load_dataset("miracl/miracl", ...)`` no longer works: both
    miracl/miracl and miracl/miracl-corpus ship a legacy Python loading script,
    and dataset-scripts support was removed in `datasets` 4+. We instead pull
    the underlying TSV (topics/qrels) and gzipped JSONL (corpus) files directly
    via huggingface_hub, which is unaffected by that removal.

    The raw Finnish corpus has ~1.88M passages; we never materialize it fully.
    A single streaming pass keeps every qrels-referenced doc and reservoir-
    samples negatives down to ``max_docs`` (seed 0), so peak memory is
    O(max_docs) rather than O(corpus size).
    """
    from huggingface_hub import HfApi, hf_hub_download

    lang = "fi"
    topics_repo = "miracl/miracl"
    corpus_repo = "miracl/miracl-corpus"

    def _read_topics(split: str) -> dict[str, str]:
        path = hf_hub_download(
            topics_repo, f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{split}.tsv", repo_type="dataset"
        )
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                qid, text = line.split("\t", 1)
                out[qid] = text
        return out

    def _read_qrels(split: str) -> dict[str, dict[str, int]]:
        path = hf_hub_download(
            topics_repo, f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv", repo_type="dataset"
        )
        bucket: dict[str, dict[str, int]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                qid, _q0, did, grade = line.split("\t")
                bucket.setdefault(qid, {})[did] = max(int(grade), 0)
        return bucket

    queries: dict[str, str] = {}
    queries.update(_read_topics("train"))
    queries.update(_read_topics("dev"))

    qrels_by_split = {"train": _read_qrels("train"), "dev": _read_qrels("dev")}
    keep_ids = {did for bucket in qrels_by_split.values() for docs in bucket.values() for did in docs}

    api = HfApi()
    files = api.list_repo_files(corpus_repo, repo_type="dataset")
    shard_files = sorted(
        (f for f in files if f.startswith(f"miracl-corpus-v1.0-{lang}/docs-")),
        key=lambda f: int(f.split("docs-")[1].split(".")[0]),
    )

    target_negatives = max(max_docs - len(keep_ids), 0)
    rng = random.Random(seed)
    reservoir: list[tuple[str, dict]] = []
    n_seen_negatives = 0
    corpus: dict[str, dict] = {}
    raw_total = 0

    for shard in shard_files:
        path = hf_hub_download(corpus_repo, shard, repo_type="dataset")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                raw_total += 1
                docid = row["docid"]
                doc = {"title": row.get("title") or "", "text": row.get("text") or ""}
                if docid in keep_ids:
                    corpus[docid] = doc
                    continue
                n_seen_negatives += 1
                if len(reservoir) < target_negatives:
                    reservoir.append((docid, doc))
                else:
                    j = rng.randint(0, n_seen_negatives - 1)
                    if j < target_negatives:
                        reservoir[j] = (docid, doc)

    for docid, doc in reservoir:
        corpus[docid] = doc

    meta = {
        "language": "fi",
        "source": (
            "miracl/miracl (topics+qrels) + miracl/miracl-corpus (Wikipedia passages), Hugging "
            "Face, fetched via huggingface_hub raw file download because the datasets-library "
            "loading script for miracl/miracl is no longer supported by datasets>=4"
        ),
        "license": "Apache-2.0",
        "notes": [
            "Chosen over German MIRACL: 'de' ships only a dev split (no train), while 'fi' ships "
            "train+dev, letting us build a genuine train/dev split before synthesizing test.",
            "MIRACL never published public test qrels (test-a/test-b are leaderboard-only), so "
            "test is carved out of train.",
            f"Raw Finnish corpus has {raw_total:,} passages; reservoir-sampled (seed {seed}) down "
            f"to <= {max_docs:,} while keeping all {len(keep_ids):,} qrels-referenced docs.",
        ],
        "miracl_raw_corpus_docs": raw_total,
    }
    return corpus, queries, qrels_by_split, meta, max_docs


DATASET_REGISTRY: dict[str, Callable[[], tuple]] = {
    "scifact": _fetch_scifact,
    "nfcorpus": _fetch_nfcorpus,
    "fiqa": _fetch_fiqa,
    "arguana": _fetch_arguana,
    "scidocs": _fetch_scidocs,
    "trec-covid": _fetch_trec_covid,
    "webfaq-cs": _fetch_webfaq_cs,
    "miracl-fi": _fetch_miracl_fi,
}

LARGE_DATASETS = {"trec-covid"}


# ---------------------------------------------------------------------------
# Generic build pipeline (fetch -> subsample -> dedup -> split -> leakage guards)
# ---------------------------------------------------------------------------


def _build_dataset(
    name: str,
    corpus: dict[str, dict],
    queries: dict[str, str],
    qrels_by_split: dict[str, dict[str, dict[str, int]]],
    meta_extra: dict,
    max_docs: int | None,
    seed: int = 0,
    near_dup_threshold: float = 0.9,
) -> RetrievalDataset:
    flat_qrels: dict[str, dict[str, int]] = {}
    for bucket in qrels_by_split.values():
        for qid, docs in bucket.items():
            flat_qrels.setdefault(qid, {}).update(docs)

    subsample_stats: dict = {"applied": False}
    if max_docs is not None:
        corpus, subsample_stats = subsample_corpus(corpus, flat_qrels, max_docs, seed=seed)

    corpus, remap, dedup_removed = exact_dedup(corpus)
    flat_qrels = remap_qrels(flat_qrels, remap)

    near_dup_groups = near_duplicate_groups(corpus, threshold=near_dup_threshold)
    query_dup_ids = query_equals_doc(queries, corpus)

    pools = {split: list(bucket.keys()) for split, bucket in qrels_by_split.items() if bucket}
    splits, split_source = ensure_splits(pools, seed=seed)
    assert_disjoint_splits(splits)

    all_split_qids = {qid for ids in splits.values() for qid in ids}
    queries = {qid: text for qid, text in queries.items() if qid in all_split_qids}
    flat_qrels = {qid: docs for qid, docs in flat_qrels.items() if qid in all_split_qids}

    # Drop dangling doc references that don't resolve in the (deduped, possibly
    # subsampled) corpus. Observed as a genuine upstream data-quality issue --
    # e.g. 5/1406 BeIR/arguana-qrels test rows point at a corpus id that was
    # never present in BeIR/arguana's corpus split -- not a bug in this
    # pipeline, but the RetrievalDataset contract requires every qrels doc id
    # to exist in `corpus`, so we clean it here and record what was dropped.
    dangling_doc_refs = 0
    for qid, docs in flat_qrels.items():
        kept = {}
        for did, grade in docs.items():
            if did in corpus:
                kept[did] = grade
            else:
                dangling_doc_refs += 1
        flat_qrels[qid] = kept

    # A query that loses every one of its relevant docs to the cleanup above
    # carries no retrieval signal; drop it entirely rather than leave a
    # zero-relevant-doc query in the splits.
    orphaned_qids = sorted(qid for qid, docs in flat_qrels.items() if not docs)
    if orphaned_qids:
        orphan_set = set(orphaned_qids)
        flat_qrels = {qid: docs for qid, docs in flat_qrels.items() if qid not in orphan_set}
        queries = {qid: text for qid, text in queries.items() if qid not in orphan_set}
        splits = {split: [qid for qid in ids if qid not in orphan_set] for split, ids in splits.items()}

    test_dup_flags = flag_test_queries_with_train_dup_relevant(flat_qrels, splits, near_dup_groups)

    rel_counts = [sum(1 for g in docs.values() if g >= 1) for docs in flat_qrels.values()]
    graded = any(g > 1 for docs in flat_qrels.values() for g in docs.values())

    meta = {
        **meta_extra,
        "num_docs": len(corpus),
        "num_queries": {split: len(ids) for split, ids in splits.items()},
        "num_queries_total": len(all_split_qids),
        "avg_rel_per_query": (sum(rel_counts) / len(rel_counts)) if rel_counts else 0.0,
        "graded": graded,
        "split_source": split_source,
        "dedup_exact_removed": dedup_removed,
        "qrels_dangling_doc_refs_dropped": dangling_doc_refs,
        "orphaned_queries_dropped": len(orphaned_qids),
        "near_dup_group_count": len(near_dup_groups),
        "near_dup_groups": near_dup_groups,
        "query_equals_doc_count": len(query_dup_ids),
        "query_equals_doc_ids": query_dup_ids,
        "test_queries_with_train_dup_relevant": test_dup_flags,
        "subsample": subsample_stats,
        "median_doc_tokens": _median_token_len([_doc_full_text(d) for d in corpus.values()]),
        "median_query_tokens": _median_token_len(list(queries.values())),
    }

    return RetrievalDataset(name=name, corpus=corpus, queries=queries, qrels=flat_qrels, splits=splits, meta=meta)


def build_and_save(name: str, root: str = "data", seed: int = 0) -> RetrievalDataset:
    """Fetch ``name`` from its public source, normalize it, and cache it under ``root``."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Known datasets: {sorted(DATASET_REGISTRY)}")
    fetch_fn = DATASET_REGISTRY[name]
    corpus, queries, qrels_by_split, meta_extra, max_docs = fetch_fn()
    ds = _build_dataset(name, corpus, queries, qrels_by_split, meta_extra, max_docs=max_docs, seed=seed)
    save_dataset(ds, root=root)
    return ds


# ---------------------------------------------------------------------------
# On-disk format: save / load
# ---------------------------------------------------------------------------


def save_dataset(ds: RetrievalDataset, root: str = "data") -> Path:
    out_dir = Path(root) / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(out_dir / "corpus.jsonl.gz", "wt", encoding="utf-8") as f:
        for doc_id, doc in ds.corpus.items():
            f.write(
                json.dumps({"_id": doc_id, "title": doc.get("title", ""), "text": doc.get("text", "")}, ensure_ascii=False)
                + "\n"
            )

    with gzip.open(out_dir / "queries.jsonl.gz", "wt", encoding="utf-8") as f:
        for qid, text in ds.queries.items():
            f.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")

    with open(out_dir / "qrels.tsv", "w", encoding="utf-8", newline="") as f:
        f.write("query_id\tdoc_id\tgrade\n")
        for qid, docs in ds.qrels.items():
            for did, grade in docs.items():
                f.write(f"{qid}\t{did}\t{grade}\n")

    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(ds.splits, f, ensure_ascii=False, indent=2)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(ds.meta, f, ensure_ascii=False, indent=2)

    return out_dir


def load_dataset(name: str, root: str = "data") -> RetrievalDataset:
    """Load a dataset previously written by :func:`save_dataset` / ``prepare_data.py``."""
    base = Path(root) / name
    if not base.exists():
        raise FileNotFoundError(
            f"No cached dataset at {base!s}. Run: python scripts/prepare_data.py --datasets {name}"
        )

    corpus: dict[str, dict] = {}
    with gzip.open(base / "corpus.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row.get("title", ""), "text": row.get("text", "")}

    queries: dict[str, str] = {}
    with gzip.open(base / "queries.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row.get("text", "")

    qrels: dict[str, dict[str, int]] = {}
    with open(base / "qrels.tsv", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, did, grade = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(grade)

    with open(base / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)

    with open(base / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    return RetrievalDataset(name=name, corpus=corpus, queries=queries, qrels=qrels, splits=splits, meta=meta)
