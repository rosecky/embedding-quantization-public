---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-0.6B
language:
  - en
  - cs
  - multilingual
library_name: gguf
pipeline_tag: feature-extraction
tags:
  - embeddings
  - retrieval
  - quantization
  - gguf
  - llama.cpp
  - query-encoder
  - asymmetric-retrieval
---

# Qwen3-Embedding-0.6B query-side clients (GGUF)

**Status:** draft model card. Files are listed with their release condition; a file is published only when the
condition is met by the native measurement (`results/tables/release_cs.md`, `results/tables/release_en.md`). Numbers in
brackets are pending.

Quantized **query encoders** for `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0; Qwen3-0.6B decoder, last-token pooling,
1024-d, 100+ languages) for use **against an existing document index computed by the unquantized model**. The
document side is untouched. They run in llama.cpp (`llama-embedding`) and in the browser (wllama).

## Compatibility

Compatible with document vectors from `Qwen/Qwen3-Embedding-0.6B` produced **without** an instruction on the document
side, last-token pooling, L2 normalisation, 1024 dimensions (Matryoshka truncation to fewer dimensions works the same
way for the client's output as for the original: truncate, then renormalise). Queries use the model's instruction
format `Instruct: <task>\nQuery: <text>`; the released files were evaluated with the generic web-search instruction.
**Not compatible** with indexes of `jina-embeddings-v5-text-small`, `harrier-oss-v1-0.6b` or any other model, even
though the architecture and the dimension are the same. Switching a deployed system to this base means re-indexing.

## Planned files and release conditions

| id | file | grid | calibration text | evaluated on | release condition | status |
|---|---|---|---|---|---|---|
| Q1 | `qwen3-0.6b-imx-Q4_K_M-generic_wikitext-tabq4_0.gguf` (**339.9 MiB**) — **general query client**, `llama-quantize` Q4_K_M with an English imatrix, token table q4_0 | Q4_K_M blocks, q4_0 table | generic English (wikitext, 585 × 512 tokens) | SciFact / NFCorpus / ArguAna / SciDocs vs own fp32: **100.0 / 99.4 / 100.1 / 99.3 %**, cosine 0.981 / 0.973 / 0.980 / 0.983, overlap 0.898 / 0.869 / 0.937 / 0.898 | ≥ 95 % of own fp32 nDCG@10, cosine ≥ 0.94, top-10 overlap ≥ 0.75 on ≥ 3 of 4 corpora → **met on 4 of 4** | **released** (same recipe as Q2'; the Q2_K-table variant, 305 MiB, scores 98.7 / 98.2 / 100.8 / 100.2 % at cosine 0.95–0.97 and is not published) |
| Q1' | GPTQ Q3_K generic-EN (235 MiB) | Q3_K, Q2_K table | same | 96.6 / 96.6 / 100.0 / 95.1 % nDCG but cosine 0.933 / 0.905 / 0.945 / 0.936 and overlap 0.741 on NFCorpus | fails the cosine part on 3 of 4 | **not released**; `llama-quantize` Q3_K pure (96.1 / 94.4 / 98.0 / 92.3 %) and IQ3_XXS (95.0 / 89.6 / 93.9 / 91.5 %) also fail |
| Q2 | `qwen3-0.6b-gptq-Q3_K-generic_cs-…-tabQ2_K.gguf` (235.1 MiB) — **quantization calibrated on Czech text** | Q3_K, Q2_K table | Czech Wikipedia paragraphs (300k tokens) | Czech supreme-court segments (55 071 docs, 1 137 synthetic queries) vs own fp32 0.3186 | same criterion on the Czech index | E1 native: 0.2868 (**90.0 %**), cosine 0.881, overlap 0.547 — **fails**; not released in this form |
| Q3 | Q3_K calibrated on synthetic Czech legal queries | Q3_K | doc2query from 1 500 held-out legal segments | same | only if it beats Q2 by ≥ +0.01 nDCG@10 with a CI excluding zero | E4 pending |
| Q2' | `qwen3-0.6b-imx-Q4_K_M-generic_cs-tabq4_0.gguf` (**339.9 MiB**) — `llama-quantize` Q4_K_M with a Czech imatrix, token table q4_0 — **quantization calibrated on Czech text** | Q4_K_M blocks, q4_0 table | Czech Wikipedia imatrix (585 × 512 tokens) | same | same criterion | **0.3145 (98.7 %)**, cosine 0.966, overlap 0.775, Δ vs fp32 −0.004 [−0.011; +0.002] (prompt-corrected run) — meets the criterion; **released**. Variant with q8_0 table: 414.0 MiB, 0.3172 (99.5 %), cosine 0.969, overlap 0.788 (not published) |
| Q4 | `llama-quantize` Q3_K / IQ3_XXS / Q4_K_M with the Q2_K token table | K-quant / IQ | Czech imatrix | same | – | Q3_K pure 80.2 %, IQ3_XXS 69.8 %, Q4_K_M 94.7 % (overlap 0.649), Q3_K mixture 86.2 %; Q3_K mixture with q8_0 table 92.8 % — **none released**. On this model the Q2_K token table costs 4.6–6.6 points on Czech and the 3-bit blocks cost the rest |

If Q1 and Q2 turn out to give the same result on both the English and the Czech evaluation (differences < 0.01), a
single recommended file is published with both evaluations.

**Naming rule.** "Calibrated on Czech text" describes the calibration data of a post-training quantization. It is not a
fine-tune, not a Czech model and not a legal-domain model. The base model's quality on Czech is the base model's; on the
Czech legal evaluation its fp32 nDCG@10 is 0.3186 against 0.3190 for jina-embeddings-v5-text-small on the same
synthetic queries (each against its own index; no human relevance judgements).

sha256 of the candidates: `f76f1112bc43a1cd35c7c55c163e047caa3cde7b21b05b1e6891033db8dd29f4` (Q4_K_M + q4_0 table,
339.9 MiB), `b18ef672d8186430bef6c07792444c43d1fbd4a32c25f517460f25850b274c85` (Q4_K_M + q8_0 table, 414.0 MiB).
Recipe: `llama-imatrix -m qwen3-embedding-0.6b-f16.gguf -f generic_cs.txt -c 512 --chunks 585 -o imatrix.gguf` then
`llama-quantize --imatrix imatrix.gguf --token-embedding-type q4_0 qwen3-embedding-0.6b-f16.gguf out.gguf Q4_K_M`.

## What we measure and report per file

nDCG@10 and recall@100 against the unchanged fp32 index, mean cosine to the fp32 query vector, mean top-10 overlap
with the fp32 ranking, paired bootstrap CI over queries, file size, and (for the recommended file) native latency at
context 512 and browser latency/memory. Differences under 0.01 nDCG are ties (calibration-draw variance ~0.01).

## Licences and provenance

Weights derive from `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0) and are released under Apache-2.0 with attribution.
Calibration texts: wikitext (CC BY-SA), Czech Wikipedia (CC BY-SA; the sampling script is published, not the text),
synthetic queries generated with Qwen3-1.7B from public court decisions. Evaluation code:
`https://github.com/rosecky/embedding-quantization-public` (Apache-2.0).
