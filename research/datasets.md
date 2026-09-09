# Retrieval datasets

All datasets below are loaded from public Hugging Face sources via `scripts/prepare_data.py`
into the uniform `RetrievalDataset` interface (`src/eq/data.py`) and cached under
`data/<name>/{corpus.jsonl.gz, queries.jsonl.gz, qrels.tsv, splits.json, meta.json}`.
`load_dataset(name)` reads only from this cache, so a private corpus can later be dropped in
by writing the same five files without touching any experiment code.

## Summary

Produced by `python scripts/prepare_data.py --datasets scifact nfcorpus arguana webfaq-cs miracl-fi fiqa` (seed 0 throughout). `trec-covid` is implemented but not built here -- see its section below.

| name | lang | #docs | queries train/dev/test | avg rel/query | graded? | dedup removed | near-dup groups | median doc tokens | median query tokens |
|---|---|---|---|---|---|---|---|---|---|
| scifact | en | 5,183 | 728/81/300 | 1.13 | no | 0 | 0 | 204 | 12 |
| nfcorpus | en | 3,593 | 2590/324/323 | 41.32 | yes | 40 | 1 | 237 | 2 |
| arguana | en | 8,626 | 560/141/700 | 1.00 | no | 48 | 78 | 150 | 174 |
| webfaq-cs | cs | 71,529 | 58569/6508/7231 | 1.00 | no | 779 | 73 | 31 | 7 |
| miracl-fi | fi | 99,991 | 2028/1271/869 | 1.77 | no | 9 | 37 | 42 | 4 |
| fiqa | en | 57,599 | 5500/500/648 | 2.57 | no | 39 | 54 | 91 | 10 |

("#docs" is post-dedup, post-subsampling -- i.e. what's actually cached under `data/<name>/corpus.jsonl.gz`. "dedup removed" is exact duplicates only; near-dup groups are recorded, not removed.)

## Multilingual pick: which two, and why

The task asked to investigate four multilingual/Czech candidates and implement the best two.
Findings (all verified empirically against the live Hugging Face Hub, not from
documentation alone):

| Candidate | Verdict | Why |
|---|---|---|
| **PaDaS-Lab/webfaq-retrieval, `ces` config** | **Implemented** (`webfaq-cs`) | Czech config confirmed as `ces` (not `cs`). Loads cleanly via `datasets.load_dataset`. Configs `ces-corpus` (72,308 docs, single `corpus` split), `ces-queries` (`train`=65,077 / `test`=7,231), `ces-qrels` (binary, one gold passage per question). CC BY 4.0. |
| **Seznam/DaReCzech** | **Excluded** | Not distributed through the `datasets` library or any open HF repo at all: per `github.com/seznam/DaReCzech`, access requires reading a disclaimer and emailing `srch.vyzkum@firma.seznam.cz` to receive a download link. That manual, per-user gating is incompatible with a reproducible `datasets`-based pipeline, so it was dropped despite being a strong graded Czech relevance dataset in principle. |
| **MTEB `BelebeleRetrieval`, `ces_Latn`** | **Excluded** | `huggingface.co/api/datasets/mteb/BelebeleRetrieval` returns HTTP 401 (gated, requires accepting terms / auth token), and `datasets.get_dataset_config_names` fails the same way. Also confirmed "tiny" as the task description warned (~500 docs / ~900 queries per language pair in comparable MTEB retrieval configs), so the payoff for handling a gated dependency was low. |
| **`miracl/miracl`, non-English language** | **Implemented** (`miracl-fi`, Finnish) | See below. |

**Why Finnish over German for the MIRACL control:** both are valid non-Czech MIRACL
languages, but `miracl/miracl`'s file listing shows German (`de`) ships **only a `dev`
split** (`miracl-v1.0-de/{qrels,topics}/*-dev.tsv`, `*-test-b.tsv` with no public qrels,
and no `train` file at all) -- it is one of MIRACL's "surprise languages" with no training
data. Finnish ships `train` (2,897 topics) **and** `dev` (1,271 topics), which lets the
pipeline build a genuine native train/dev split and only needs to synthesize `test`
(see below), rather than synthesizing two of the three splits. Apache-2.0 licensed.

**MIRACL loading caveat:** `datasets.load_dataset("miracl/miracl", ...)` and
`("miracl/miracl-corpus", ...)` both fail with `RuntimeError: Dataset scripts are no longer
supported` -- both repos ship a legacy Python loading script, and script-based datasets were
removed in `datasets` 4+. `_fetch_miracl_fi` in `src/eq/data.py` works around this by pulling
the raw files directly via `huggingface_hub.hf_hub_download` (topics/qrels as TSV, corpus as
gzipped JSONL shards), which is unaffected by the removal. The raw Finnish corpus is
1,883,509 Wikipedia passages across 4 shards (~270MB compressed) -- far more than the 100k
cap, so it is never fully materialized: a single streaming pass keeps every qrels-referenced
doc and reservoir-samples (seed 0, Algorithm R) negatives down to the cap, bounding peak
memory to O(max_docs) instead of O(corpus size).

## Per-dataset notes

### scifact
- Source: `BeIR/scifact` (corpus/queries) + `BeIR/scifact-qrels` (train/test).
- Schema: corpus/queries rows are `{_id, title, text}`; qrels rows are `{query-id, corpus-id, score}` (score always 1 -- binary).
- Splits: native `train`/`test` only -> `dev` is carved from `train` (10%, seed 0), `split_source="dev_from_train"`.
- License: ambiguous across sources -- the original SciFact paper states CC BY-NC 2.0; the `BeIR/scifact` HF dataset card tags `cc-by-sa-4.0` for the BEIR repackaging. Treat as research/non-commercial use.
- Caveat: claim-verification task recast as claim -> supporting-abstract retrieval; avg ~1.1 relevant docs/query.

### nfcorpus
- Source: `BeIR/nfcorpus` + `BeIR/nfcorpus-qrels` (native `train`/`validation`/`test`, `validation` normalized to `dev`).
- Schema: same corpus/queries/qrels shape as scifact.
- **Graded**: scores are 1 (partially relevant) or 2 (highly relevant); train split in the raw source is *entirely* grade-1 rows -- graded judgments (2) only start appearing in dev/test.
- License: not formally reported by the NFCorpus authors; HF tag `cc-by-sa-4.0`.
- Caveat: very high avg relevant-docs/query (~40+) because judgments are pooled/exhaustive per NutritionFacts.org query, unlike the ~1/query norm in scifact/arguana.

### fiqa
- Source: `BeIR/fiqa` + `BeIR/fiqa-qrels` (native `train`/`validation`/`test`).
- Schema: same as above; 57.6k docs (financial-forum posts), no subsampling needed (under the 100k cap only applied to the multilingual sets).
- License: not formally reported (FiQA-2018 challenge data); HF tag `cc-by-sa-4.0`.

### arguana
- Source: `BeIR/arguana` + `BeIR/arguana-qrels` (test only) -> synthetic 40/10/50 split, `split_source="synthetic_from_test"`.
- **Data-quality finding**: 5 of 1,406 raw qrels rows reference a corpus id (`test-*-conNNb`/`proNNb` style) that does not exist anywhere in `BeIR/arguana`'s corpus split -- a genuine defect in the upstream BEIR repackaging, not a bug in this pipeline. `_build_dataset` now defensively drops any qrels doc reference that doesn't resolve in the final corpus and removes any query left with zero relevant docs as a result (`meta["qrels_dangling_doc_refs_dropped"]`, `meta["orphaned_queries_dropped"]`); 1,401 of 1,406 queries survive.
- **ArguAna-style leakage check**: the task singled out "queries identical to a document text (ArguAna-style)" as a pattern to guard against. `query_equals_doc()` is implemented generically and was run against all datasets; empirically it found **0** exact matches in this specific BEIR repackaging of ArguAna (queries and their counter-argument documents are topically paired but not byte-identical after normalization). The guard is retained as-is since it is a real risk for other/private corpora built the same way.
- High near-dup group count relative to corpus size (many near-duplicate debate arguments on the same topics) -- combined with the synthetic test-from-test split, this produces the largest `test_queries_with_train_dup_relevant` flag count of the English datasets: a genuine leakage risk from the synthetic split, correctly surfaced by the guard.

### trec-covid (large, opt-in)
- Source: `BeIR/trec-covid` + `BeIR/trec-covid-qrels` (test only, 171,332 docs) -> only built when `--include-large` is passed.
- **Graded** 0/1/2; qrels also contained 2 rows with `score=-1` (TREC "not judged" sentinel), clipped to 0 during loading.
- License: TREC-COVID "Dataset License Agreement" (research use only, derived from CORD-19); HF tag `cc-by-sa-4.0`.
- Not built in this run -- registered and implemented (`DATASET_REGISTRY["trec-covid"]`) but skipped by default per the task's "large" designation; run `python scripts/prepare_data.py --datasets trec-covid --include-large` to build it.

### webfaq-cs (Czech)
- Source: `PaDaS-Lab/webfaq-retrieval`, configs `ces-corpus` / `ces-queries` / `ces-qrels`.
- Schema: corpus rows `{_id, title, text}` (title empty in practice); queries rows `{_id, text}`; qrels rows `{query-id, corpus-id, score}`, score always 1.0.
- Splits: native `train`/`test` only (no `dev`) -> `dev` carved from `train` (10%, seed 0), `split_source="dev_from_train"`.
- License: CC BY 4.0 (Common Crawl 2022-2024 derived FAQ pairs).
- Caveat: 779 exact-duplicate corpus docs found and collapsed (verbatim FAQ answers repeated across near-identical pages) plus 73 near-dup groups -- a real illustration of why the leakage guards matter for web-mined corpora.

### miracl-fi (Finnish, multilingual control)
- Source: `miracl/miracl` (topics/qrels) + `miracl/miracl-corpus` (Wikipedia passages), fetched via raw `huggingface_hub` file download (see loading caveat above).
- Schema: topics are `qid\ttext` TSV; qrels are TREC-format `qid Q0 docid relevance` TSV (relevance 0 or 1, and -- unlike the BEIR qrels files -- explicit non-relevant (0) judgments are present, not just positives); corpus rows are `{docid, title, text}` JSONL, `docid` shaped like `<wikipedia_page_id>#<passage_index>`.
- Splits: native `train`/`dev`; MIRACL never publishes public test qrels (`test-a`/`test-b` are leaderboard-only), so `test` is carved out of `train` (30%, seed 0), `split_source="test_from_train"`.
- **Subsampling**: raw corpus has 1,883,509 docs; deterministically reservoir-sampled (seed 0) down to <=100,000 while keeping every qrels-referenced doc. This happens inside the fetcher itself (streaming, since the raw corpus is too large to materialize in memory) rather than the generic post-fetch `subsample_corpus` step, so `meta["subsample"]["applied"]` is correctly `False` (nothing left to trim by the time the generic pipeline sees it) -- the real numbers are in `meta["miracl_raw_corpus_docs"]` (1,883,509) and `meta["notes"]`.
- License: Apache-2.0.

## Leakage guards implemented (`src/eq/data.py`)

1. **Exact duplicates** -- SHA-256 hash of whitespace-collapsed, lowercased `title + text`; first-seen id is canonical, qrels doc ids are remapped (max grade kept on collision). Count in `meta["dedup_exact_removed"]`.
2. **Near duplicates** -- MinHash-LSH (`datasketch`) over word 5-shingles, Jaccard >= 0.9. Groups are recorded (never merged/removed) in `meta["near_dup_groups"]` / `meta["near_dup_group_count"]`. For corpora over 50k docs, `num_perm` is reduced 64->32 and shingling is truncated to the first 800 (vs 2000) characters per doc purely to bound runtime -- a documented approximation, not expected to materially change which near-duplicate clusters are found since most duplicate content in these corpora concentrates in the opening paragraphs.
3. **Query == document text** -- exact-hash match between query text and any corpus document; recorded in `meta["query_equals_doc_count"]` / `meta["query_equals_doc_ids"]`.
4. **Cross-split leakage from near-dups** -- for every near-dup group, if a train query's relevant doc and a test query's relevant doc fall in the same group, the test query id is flagged in `meta["test_queries_with_train_dup_relevant"]`.
5. **Dangling qrels doc references** (found empirically, not originally scoped, but required to satisfy the "qrels ids all exist in corpus" contract) -- doc ids referenced by qrels but absent from the final corpus are dropped, and any query left with zero relevant docs as a result is removed entirely from queries/qrels/splits. Counts in `meta["qrels_dangling_doc_refs_dropped"]` / `meta["orphaned_queries_dropped"]`.

## Split policy

- Native train/dev/test splits are used as-is when all three exist (`split_source="native"`).
- Train+test only -> `dev` = 10% of train, seed 0 (`"dev_from_train"`).
- Train+dev only (no public test, e.g. MIRACL) -> `test` = 30% of train, seed 0 (`"test_from_train"`).
- Test only (e.g. ArguAna, TREC-COVID) -> synthetic 40/10/50 split of test queries into train/dev/test, seed 0 (`"synthetic_from_test"`).
- A query id is never placed in two splits (`assert_disjoint_splits`, exercised in `tests/test_data.py`).
