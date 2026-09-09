# Keep the index, shrink the query encoder

Query-side quantized clients for an **unchanged** embedding index: a reproducible pipeline that compiles the query
encoder of a 0.6B embedding model into a 190–240 MiB file, verifies it against the existing fp32/int8 index
(nDCG@10, cosine to the fp32 query vector, top-10 overlap, paired bootstrap over queries) and ships it to llama.cpp or
a browser tab (WebGPU / WebAssembly through wllama). Documents are never re-encoded.

Live demo: **https://thinletter.io** (search 25 656 scientific abstracts in your browser with the released clients).
The technical report is in [`docs/release/technical_report.md`](docs/release/technical_report.md); the released
files are described in [`docs/release/model_card_harrier-0.6b-query-clients.md`](docs/release/model_card_harrier-0.6b-query-clients.md).
Every number below has a generated table under `results/tables/`; the research log behind them (Czech) is not part of the public repository.

**What is in this repository and what is not.** This repository contains the scalar pipeline (GPTQ onto llama.cpp's
K-quant grids, `llama-quantize --imatrix` as the first-arm baseline), the verification harness, the browser benchmark of
the scalar files, the demo, the tables and the research log. It does **not** contain our vector-quantised container
format, its compiler or its WebGPU runtime; those are a separate, non-open component (see "Vector-quantised clients"
below). Every scalar number in the report can be reproduced from this repository plus llama.cpp; the VQ numbers are
reported but cannot be reproduced from it.

## What the numbers say

Quantized query side against the fp32 document index of the same model, test split, native llama.cpp
(`llama-embedding`), nDCG@10 in % of the fp32 model. Files are complete GGUFs (token table Q2_K).

| model / corpus | fp32 nDCG@10 | Q3_K, 235 MiB | Q2_K, 192 MiB | what calibration text |
|---|---|---|---|---|
| harrier-0.6b / SciFact (EN) | 0.7559 | 98.9 % (generic EN); 99.7 % (288 MiB, Q5_0 table) | **98.4 %** (0.7440) corpus; 96.4 % generic EN | corpus / synthetic queries |
| harrier-0.6b / NFCorpus (EN) | 0.3808 | 99.3–99.8 % | 94.5 % | generic EN / synthetic |
| harrier-0.6b / ArguAna (EN) | 0.6665 | 101 % (generic EN = synthetic) | 96–99 % | – |
| harrier-0.6b / SciDocs (EN) | 0.2269 | 99.2 % (generic EN) | 90 % generic EN, 93–94.5 % corpus / synthetic | – |
| jina-v5-small / Czech supreme-court segments (55 071 docs, customer index) | 0.3190 | **96.9 %** generic **Czech** text; 94.0 % generic English | 79.5 % generic Czech, 86.2 % synthetic queries, **50.9 % generic English** | language first |
| Qwen3-Embedding-0.6B / Czech supreme-court segments (own fp32 index) | 0.3186 | **90.0 %** generic Czech text (cos 0.881) — fails our release rule; `llama-quantize` Q3_K 80–86 %; first files to pass: **Q4_K_M with a q4_0 token table (340 MiB): 98.3 %**, with q8_0 table (414 MiB) 99.3 % | – | same recipe, different model |

Three things we want a reader to take away:

1. **The language of the calibration text matters more than its domain.** On the Czech legal index at 2.6 bits, Czech
   Wikipedia instead of English Wikipedia is +0.091 nDCG@10 [+0.076; +0.106]; synthetic queries from the corpus add a
   further +0.022 [+0.011; +0.032]. At 3.4 bits generic Czech text already holds 96.9 % and corpus calibration adds nothing
   measurable (−0.002 [−0.009; +0.004]). The second half does **not** transfer to Qwen3-Embedding-0.6B: the same
   recipe holds only 90.0 % there, so "generic text is enough at 3.4 bits" is a one-model finding. Tables:
   `results/tables/calib_question.md`, `release_cs.md`.
2. **Our GPTQ export is not a better file than `llama-quantize --imatrix` at ~200 MiB.** Same model, same calibration
   text and budget, one runtime: in-domain tie (SciFact +0.002 [−0.015; +0.019] for IQ2_M), out-of-domain worse
   (NFCorpus −0.016 [−0.026; −0.007]), and IQ2_M is 1.5× faster (103 vs 160 ms per query). So the pipeline treats
   `llama-quantize` as the first arm and GPTQ as the second. Table: `results/tables/imatrix_compare.md`.
   Czech branch (jina, Q3_K): llama.cpp's K-quant mixture with a Czech imatrix ties our export exactly (0.3092 vs
   0.3092) at 258 vs 235 MiB; in our exact format (`--pure`) it is −0.010 [−0.018; −0.002], the size of the
   calibration-draw noise. Table: `results/tables/release_cs.md`.
3. **The browser client is real and measured idle.** The 192 MiB file in Chrome (wllama 3.6.1): WebGPU on an Intel
   integrated GPU p50 405 ms (530 MiB in the tab), WASM 8 threads p50 1 669 ms (1 147 MiB); nDCG within ±0.003 of
   native. File size is not memory: 192 MiB of weights need 0.93 GB RSS at context 512 and 4 GB at the model's default
   context, so the context must be capped to the query length. Table: `results/tables/browser_client.md`.

Two limits to read the table with: a paired CI over queries does not contain the variance of the calibration draw
(two equally good quantizations differ by ~0.01 nDCG with CIs excluding zero) nor between-machine variation (±0.006);
differences under 0.01 should be read as ties and the cosine to fp32 is the more stable readout. And a statistically
indistinguishable result is not proven equivalence.

## Released and planned query clients

Every client is a **query encoder only** and is compatible with exactly one document encoder and its settings; the
document vectors stay as they are.

| client | base model (licence) | compatible document index | status |
|---|---|---|---|
| harrier-0.6b Q3_K generic-EN (235 MiB), Q2_K SciDocs-synthetic and Q2_K generic-EN (192 MiB) | microsoft/harrier-oss-v1-0.6b (MIT) | harrier-0.6b, no document prompt, last-token pooling, L2 | released with this report (MIT) |
| Qwen3-Embedding-0.6B Q4_K_M + q4_0 token table, generic-EN (340 MiB, `llama-quantize`, "general query client") | Qwen/Qwen3-Embedding-0.6B (Apache-2.0) | Qwen3-Embedding-0.6B, documents without instruction, queries with the `Instruct: …\nQuery:` prefix (no trailing space), last-token, L2 | meets the rule on 4 of 4 English corpora (100.0 / 99.4 / 100.1 / 99.3 %, cosine 0.97–0.98); the 3-bit files of this model fail on cosine and are not released |
| Qwen3-Embedding-0.6B Q4_K_M + q4_0 token table, calibrated on Czech text (340 MiB, `llama-quantize`) | Qwen/Qwen3-Embedding-0.6B (Apache-2.0) | same as above | meets the rule on the Czech index (98.7 %, cosine 0.966, overlap 0.775); released. Every ≤ 3-bit variant failed (80–93 %) |
| jina-v5-small Q3_K (Czech legal case study) | jinaai/jina-embeddings-v5-text-small (CC BY-NC 4.0) | the customer's jina index | numbers only, weights not distributed |

"Calibrated on Czech text" means exactly that: the same post-training quantization with Czech Wikipedia paragraphs as
calibration data. It is not a fine-tuned model and not a legal-domain model; a legal-domain-calibrated file is added only
if it measurably beats the generic Czech calibration.

## Verify a client against your own index (the recipe)

Everything runs from the repository root in the project's virtual environment (Python ≥ 3.10, torch, transformers,
numpy; `gguf-py` from upstream llama.cpp on `PYTHONPATH`; `llama-embedding`, `llama-imatrix`, `llama-quantize` from a
llama.cpp build, revision b81c99b or later).

```
# 1. import your corpus and your existing document vectors (never re-encoded) as a dataset
python scripts/import_corpus.py --name mycorpus --docs corpus.jsonl --doc_emb index.f32 --doc_ids ids.json \
       --teacher qwen3-0.6b --language cs
# 2. synthetic queries for held-out documents (test set) and for calibration (no real queries are used)
python scripts/doc2query.py --datasets mycorpus --lang Czech --n_per_doc 2        # GPU, Qwen3-1.7B
python scripts/build_teacher_cache.py --datasets mycorpus --teacher qwen3-0.6b --max_vram_gb 3   # fp32 query vectors
# 3a. first arm: llama.cpp's own quantizer with an importance matrix from text IN THE CORPUS LANGUAGE
llama-imatrix -m model-f16.gguf -f data/calib/generic_cs.txt -c 512 --chunks 585 -o imatrix.gguf
llama-quantize --imatrix imatrix.gguf --token-embedding-type q2_k model-f16.gguf client-Q3_K.gguf Q3_K
# 3b. second arm: GPTQ rounded directly onto the same K-quant grid (GPU, ~5–10 min for a 0.6B model)
PYTHONPATH=third_party/llama.cpp/gguf-py python scripts/gptq_export_gguf.py --teacher qwen3-0.6b --type Q3_K \
       --calib generic_cs --per_sample --n_seq 20000 --calib_tokens 300000 --act_order --table_type Q2_K --datasets mycorpus
# 4. verify: nDCG@10 / recall / cosine / top-10 overlap against the unchanged index, per-query files for paired CIs
python scripts/quant_eval_queries.py --teacher qwen3-0.6b --datasets mycorpus --ctx 512 \
       --bin llama-embedding --ggufs client-Q3_K.gguf models/gguf/qwen3-0.6b-gptq-Q3_K-generic_cs-*.gguf
python scripts/paired_ci.py results/raw/quant_pt/perq/mycorpus_client-Q3_K.npz results/raw/quant_pt/perq/mycorpus_qwen3-*.npz
```

Without relevance judgements the harness still reports cosine to the fp32 query vector and top-10 overlap with the fp32
ranking; pseudo-relevance from the fp32 model's top-k is available and is labelled as such.

Decision rule we use: compile Q2_K and Q3_K (and `llama-quantize` Q4_K_M when neither passes), keep the smallest file
whose nDCG@10 is ≥ 95 % of fp32 with the 95 % CI of the difference above −0.02, cosine ≥ 0.94 and top-10 overlap
≥ 0.75. On harrier/SciFact that is Q2_K; on jina/Czech legal it is Q3_K.

## Ship it to a browser (scalar files)

`demo/` is a static page (no build) that loads a GGUF in 24 MiB chunks into OPFS, runs it with wllama 3.6.1 (WebGPU
where `shader-f16` is available, WebAssembly otherwise), scores an int8/fp16 copy of the index in JavaScript and shows
the same test queries with their native reference values. `python demo/serve.py` serves it locally;
`scripts/demo_export.py --dataset scidocs` builds the data and model chunks; `scripts/browser_run.py` reproduces the
benchmark rows in headless Chrome. See `demo/README.md`. Nothing in this path is our runtime: it is llama.cpp compiled
to WebAssembly, and the measurements are the contribution.

## Vector-quantised clients (not part of this repository)

Below IQ2_XS llama.cpp has no format. Our vector-quantised container (4-d codebooks per 256-column block, 1.8–2.1 bits
per weight) with its own WebGPU runtime reaches nDCG@10 0.7375 in the browser at 119.5 MiB on SciFact (torch simulation
0.7380; llama.cpp's IQ2_XS: 177.5 MiB, 0.7097). On the same idle laptop and integrated GPU it answers in 103 ms p50 /
113 ms p95 against 444 / 754 ms for llama.cpp's WebGPU path in wllama (Q3_K file), loads in 2.4 s vs 4.2 s and peaks
at 1 592 vs 2 080 MiB of process memory (all Chrome processes), at −0.011 nDCG@10 (report §3.4b; one device, one model, one corpus). The same
format on the Czech index holds only 83 % (jina) and 64 % (Qwen3-Embedding) at 2.1 bpw, so a ~3-bit variant is needed there. The quantizer, the container compiler and the runtime are a separate
component under a separate licence and are being prepared as an SDK for small embedding clients; the report states which
results depend on it. If you have an index and want a client for it, write to the address below.
All measured VQ numbers (size × quality, the idle runtime comparison, the other models, the rotation study) are
collected in [`docs/release/vq_results.md`](docs/release/vq_results.md).

## Reproduce the tables

The public repository contains the verification recipe, the GPTQ exporter that made the harrier files
(`scripts/gptq_export_gguf.py`, its `--rotate` option needs code that is not included), the `llama-quantize` recipes of the released Qwen3 files
(`scripts/imatrix_cs_qwen*.sh`, `scripts/qwen_q4_q40_en.sh`), the fair-comparison scripts and the cited tables. What
it does not contain: the raw per-run results, the structured-rotation code, the vector-quantization quantizer / container / runtime, and the customer data.

## What is ours and what is not

Asymmetric retrieval (small query encoder, frozen document index) is not a new task (Query Encoder Distillation via
Embedding Alignment; KALE). GPTQ, importance matrices, Hadamard rotations (QuIP#, QuaRot, SpinQuant) and codebook
vector quantization (GPTVQ, AQLM) are prior work. Browser inference of the scalar files is wllama / llama.cpp;
Transformers.js (ONNX Runtime Web) is the other established path and has not been compared yet. What this repository
adds is the measured, reproducible chain — compile, verify against the unchanged index, export, browser — with the
negative results kept in, and the calibration map (language before domain, effect growing with quantizer damage).
Details and citations: `research/prior_art.md`.

## Licences

Code in this repository: Apache-2.0 (`LICENSE`). Released weights derive from `microsoft/harrier-oss-v1-0.6b` (MIT)
and are MIT; planned Qwen3-Embedding-0.6B clients will be Apache-2.0. The Czech-legal jina clients derive from
`jinaai/jina-embeddings-v5-text-small` (CC BY-NC 4.0) and are **not** distributed. SciDocs is CC BY 4.0; SciFact is
non-commercial, so the public demo uses SciDocs. The vector-quantised runtime and compiler are not covered by this
licence.

## Contact

Jan Rosecký — honza.rosecky@gmail.com. If you run the recipe on your own index, please open an issue with the table;
that is the most useful thing you can send. If you want a client for your index and your web application, ask for the
verification step: we measure your index first, the numbers are yours either way.
