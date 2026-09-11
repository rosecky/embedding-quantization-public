# Keep the index, shrink the query encoder: what holds at 2–3.5 bits for the query side of a 0.6B embedding model

Jan Rosecký (independent), September 2026. Draft for release; all numbers are our own measurements unless marked
EXTERNAL. Repository: `github.com/rosecky/embedding-quantization-public`. Slots marked `[pending]` are filled by runs that were
still in progress when this draft was written (see §8).

## Abstract

Retrieval systems that already hold an embedding index have a cheap path to on-device or in-browser search: quantize only
the query encoder and keep the document index unchanged. We compile the query side of two Qwen3-0.6B-based embedding
models (microsoft/harrier-oss-v1-0.6b, jinaai/jina-embeddings-v5-text-small) onto llama.cpp's K-quant grids with GPTQ,
export bit-exact GGUF files (192 MiB at 2.6 bits per weight, 235 MiB at 3.4) and verify them against the unchanged fp32
index with nDCG@10, cosine to the fp32 query vector, top-10 overlap and paired bootstrap confidence intervals over queries.
On four English BEIR corpora and on a Czech legal index of 55 071 supreme-court reasoning segments, the 3.4-bit files hold
96–100 % of fp32 nDCG@10; the 2.6-bit files hold 98 % on SciFact but only 93–94 % on SciDocs and 79–86 % on the Czech
index, so the size/quality point must be chosen per deployment. The calibration text decides in a specific order: its
language first (Czech instead of English Wikipedia: +0.091 nDCG@10 at 2.6 bits on the Czech index), its domain second
(synthetic queries from the corpus: +0.01 to +0.02, only at 2.6 bits; nothing measurable at 3.4 bits). Compared fairly
with `llama-quantize --imatrix` on the same text and budget, our GPTQ export is statistically indistinguishable in-domain,
worse out-of-domain (−0.016) and 1.5× slower, so the pipeline uses llama.cpp's own quantizer as the first arm. The same
file runs in a browser tab: WebGPU on an integrated GPU 405 ms per query, WebAssembly 1.7 s, within ±0.003 nDCG of native.
Everything is reproducible from the repository; the customer data of the Czech index is not included.

## 1. Setting and claims

**Task.** A deployed retrieval system has a document index computed by an embedding model $f$ (fp32 or int8 vectors). We
want a small query encoder $\hat f$ that runs on the client, such that ranking documents by $\langle \hat f(q), f(d)\rangle$
is as close as possible to ranking by $\langle f(q), f(d)\rangle$. Nothing on the document side changes. This is the
asymmetric setting of Query Encoder Distillation via Embedding Alignment and KALE [prior art, §7]; we do not train, we
quantize the query side of the same model post hoc and measure end-to-end.

**Claims we make** (each with the table that carries it):

- C1. The 3.4-bit K-quant file of a 0.6B query encoder holds ≥ 96 % of fp32 nDCG@10 against the unchanged index on
  every corpus we measured (4 English, 1 Czech, 2 models); the 2.6-bit file does not generalise (98 → 93 → 80 %).
  Tables `main.md`, `scidocs_client.md`, `legal_client.md`, `calib_question.md`.
- C2. Calibration text: language before domain, and the domain effect scales with quantizer damage. `calib_question.md`,
  `review_replications.md`.
- C3. At ~200 MiB our GPTQ export is not a better file than `llama-quantize --imatrix`. `imatrix_compare.md`.
- C4. The file runs in the browser with measured latency and memory, and file size is not memory footprint.
  `browser_client.md`, `wins.md` §2.1b.
- C5 (negative, secondary; numbers only, the rotation code is not in the public package). Two-sided Hadamard rotation helps only where the quantizer is badly damaged (bge-m3 at
  2.6 bits: +0.106; vector quantization at 1.58 bpw: +0.32 to +0.38); on the strong scalar grid its effect is not
  stable across random seeds. `rot_ci.md`, `rotation_side.md`.

**What we do not claim.** No new quantization algorithm. No "best quantization at 200 MiB". No universality of the
calibration effect (with llama.cpp's importance-matrix quantizer the domain effect is within ±0.005). No numbers on the
9.58-million-text production index of the Czech application (only on a 55k evaluation subset whose vectors we
reproduce to cosine 1.0000).

## 2. Method (what is run)

1. **Grid.** GPTQ (Hessian-weighted, act-order, per-sample calibration sequences cut at 512 tokens, 300k tokens total)
   rounds each block linear directly onto the llama.cpp Q2_K (2.625 bpw) or Q3_K (3.4375 bpw) block format: super-blocks
   of 16-element sub-blocks with 4-bit scale and minimum. The token embedding table is Q2_K (48.7 MiB); there is no output
   head. The packed bytes are checked against `gguf.quants.dequantize` (zero deviation); the runtime reproduces the torch
   simulation to ±0.002 nDCG@10.
2. **Calibration text** (one of): generic Wikipedia in English (`generic_wikitext`) or Czech (`generic_cs`: 3 000
   paragraphs of cs-Wikipedia, deterministic sample), documents of the target corpus, or synthetic queries generated from
   held-out corpus documents with Qwen3-1.7B (doc2query, 2 per document). Real queries are never used for calibration.
3. **Verification against the index.** Test queries encoded by the GGUF with `llama-embedding` (last-token pooling, the
   model's own query prompt, L2-normalised), scored against the fp32/int8 index: nDCG@10, recall@100, mean cosine to the
   fp32 query vector, mean top-10 overlap with the fp32 ranking. Differences between two files are paired bootstrap over
   queries (10 000 draws, seed 0). ArguAna ignores the query's own document (MTEB `ignore_identical_ids`).
4. **Export and browser.** The same GGUF is cut into 24 MiB chunks, reassembled in OPFS, run by wllama 3.6.1
   (llama.cpp in WebAssembly; WebGPU through llama.cpp's backend where `shader-f16` exists); the page scores the index in
   JavaScript and reports per-query agreement with the native run.

## 3. Results

### 3.1 English corpora, harrier-0.6b (fp32 index of the same model, test split, native llama.cpp)

| corpus | fp32 | Q3_K 235 MiB (generic EN) | Q3_K 288 MiB (synthetic, Q5_0 table) | Q2_K 192 MiB (corpus) | Q2_K 192 MiB (synthetic) | Q2_K 192 MiB (generic EN) |
|---|---|---|---|---|---|---|
| SciFact (300 q) | 0.7559 | 0.7475 (98.9 %) | 0.7538 (99.7 %) | **0.7440 (98.4 %)** | 0.7446 (98.5 %, 246 MiB) | 0.7284 (96.4 %) |
| NFCorpus (323 q) | 0.3808 | 0.3782 (99.3 %) | 0.3801 (99.8 %) | 0.3598 (94.5 %, SciFact text) | 0.3646 (95.7 %, 246 MiB) | – |
| ArguAna (700 q) | 0.6665 | 0.6750 (101.3 %) | 0.6745 (synthetic, 235 MiB) | 0.6392 (95.9 %, leak-free) | 0.6664 (186 MiB) | 0.6213† |
| SciDocs (500 q) | 0.2269 | 0.2250 (99.2 %) | – | 0.2115 (93.2 %) | 0.2145 (94.5 %) | 0.2046 (90.1 %) |

† all real queries (fp32 0.6451, 96.3 %), not the test split. Cosine to the fp32 query vector for the Q3_K generic
file: 0.962 (SciFact), 0.959 (NFCorpus), 0.966 (ArguAna), 0.964 (SciDocs); for the Q2_K generic file 0.848 (SciFact),
0.856 (SciDocs). The Q3_K file calibrated on generic English text is therefore a "download once" client: 98.9–101 % on
all four corpora.

The 2.6-bit SciFact number (98.4 %) was the reference of this project for weeks; SciDocs, a corpus of the same genre,
falsified the pre-registered "≥ 97 %" (93.2 %), and domain calibration on SciDocs did nothing: the client calibrated on
SciFact text scores 0.2123 on SciDocs, the one calibrated on SciDocs text 0.2115.

### 3.2 Czech legal index, jina-embeddings-v5-text-small (customer index unchanged)

Index: 55 071 reasoning segments of Czech Supreme Court decisions (civil agenda, from 2020), vectors taken from the
customer's Elasticsearch export and reproduced by our pipeline to cosine 0.9999. Evaluation: 1 137 synthetic Czech queries
(Qwen3-1.7B) for 600 held-out segments, relevant = source segment. This is an easy, label-free test; the carrying
readouts are the ratio to fp32, the cosine and the overlap. fp32 nDCG@10 0.3190; the official f16 GGUF gives 0.3191.

| calibration text (300k tokens) | Q2_K 192 MiB nDCG@10 (% fp) | Q2_K cos / overlap | Q3_K 235 MiB nDCG@10 (% fp) | Q3_K cos / overlap |
|---|---|---|---|---|
| generic English (wikitext) | 0.1624 (50.9) | 0.640 / 0.225 | 0.2998 (94.0) | 0.913 / 0.652 |
| generic Czech (cs-Wikipedia) | 0.2537 (79.5) | 0.786 / 0.455 | **0.3092 (96.9)** | 0.942 / 0.728 |
| corpus documents (2 000 segments outside the index) | 0.2638 (82.7) | 0.841 / 0.531 | 0.3069 (96.2) | 0.955 / 0.752 |
| synthetic queries from those documents | **0.2752 (86.2)** | 0.905 / 0.615 | 0.3071 (96.3) | 0.969 / 0.789 |

Paired differences (95 % CI over the 1 137 queries): Czech − English generic at Q2_K **+0.0912 [+0.0762; +0.1064]**, at
Q3_K +0.0094 [+0.0018; +0.0174]; synthetic − generic Czech at Q2_K **+0.0215 [+0.0114; +0.0319]**, at Q3_K −0.0021
[−0.0087; +0.0044]; synthetic − documents at Q2_K +0.0113 [+0.0005; +0.0210]. Pre-registered predictions for the Q2_K
client (≥ 97 % fp, cosine ≥ 0.85, overlap ≥ 0.75) all failed; the falsification threshold (< 95 %) was hit at 82.7 %.
Q3_K is what the application's demo ships.

Two caveats were recorded before the runs. (1) The Czech generic paragraphs are ~4× longer in tokens than the English
ones under the same budget, so language and sequence shape were confounded; the pre-registered control (E2: Czech
paragraphs of 62 words, 1 608 × ~186 tokens, vs 779 × ~385) gives Q2_K 0.2542 vs 0.2537 (+0.0005 [−0.0084; +0.0095])
and Q3_K 0.3081 vs 0.3092 (−0.0011 [−0.0063; +0.0042]): shape carries nothing, the +0.09 is the language. (2) The test
queries come from the same generator as the synthetic calibration arm (its advantage may be overstated; the document arm
has no such issue).

**Fair comparison on the Czech branch with `llama-quantize --imatrix`** (Czech imatrix from the same 300k tokens of
generic Czech text, token table Q2_K, same runtime, paired against our files; `results/tables/release_cs.md`):

| file | MiB | nDCG@10 (% fp) | cos | overlap | Δ vs our GPTQ Q3_K generic-CS [CI] |
|---|---|---|---|---|---|
| ours: GPTQ Q3_K, generic Czech | 235.1 | 0.3092 (96.9) | 0.942 | 0.728 | – |
| Q3_K mixture, Czech imatrix | 258.0 | 0.3092 (96.9) | 0.942 | 0.708 | −0.000 [−0.007; +0.007] |
| Q3_K `--pure` (our format), Czech imatrix | 235.1 | 0.2990 (93.7) | 0.897 | 0.655 | −0.010 [−0.018; −0.002] |
| Q3_K `--pure`, synthetic-query imatrix | 235.1 | 0.3007 (94.3) | 0.898 | 0.652 | −0.009 [−0.017; −0.000] |
| Q3_K `--pure`, no imatrix (RTN) | 235.1 | 0.2857 (89.6) | 0.865 | 0.585 | −0.024 [−0.033; −0.014] |
| IQ3_XXS, Czech imatrix | 212.8 | 0.2807 (88.0) | 0.862 | 0.564 | −0.029 [−0.039; −0.018] |
| IQ2_M, Czech imatrix | 199.3 | 0.2718 (85.2) | 0.830 | 0.518 | −0.037 [−0.049; −0.026] |

Reading: at the same size our GPTQ export is +0.010 better than llama.cpp's quantizer in the same format (CI excludes
zero, but the effect equals the calibration-draw noise of ~0.01, one draw per arm); the K-quant mixture reaches the same
quality with 23 MiB more, in one command. Nothing at ~200 MiB is usable on this index (IQ2_M 85 %, our Q2_K 80–86 %).
The imatrix quantizer is again indifferent to the calibration text (synthetic vs generic ≤ +0.002), GPTQ at Q2_K is not
(+0.022). Both recipes are documented; since the jina weights are non-commercial, neither file is distributed.

**Commercially distributable base (Qwen3-Embedding-0.6B, Apache-2.0).** jina-v5-small is CC BY-NC 4.0, so a
distributable Czech-calibrated client needs another base. Qwen/Qwen3-Embedding-0.6B (same Qwen3-0.6B architecture,
Apache-2.0, 100+ languages) was indexed with its **own** fp32 index of the same 55 071 segments (it is not compatible with
the jina index; replacing the base in a deployed system means re-indexing, a separate decision). On the same 1 137
synthetic queries its fp32 nDCG@10 is 0.3186 (jina: 0.3190; the f16 GGUF of Qwen3-Embedding reproduces its fp32
ranking less exactly than jina's, top-10 overlap 0.942 vs 0.986). Its Q3_K client calibrated on generic Czech text
holds only **0.2868 = 90.0 %** of its own fp32 (cosine 0.881, top-10 overlap 0.547, Δ −0.032 [−0.043; −0.021]),
against 96.9 % for jina with the same recipe. The pre-registered prediction (≥ 96 %) failed and the kill rule (< 95 %)
fired: "generic text in the corpus language is enough at 3.4 bits" is a finding on one model (jina), not a property of
the recipe. Two hypotheses, untested: jina's merged retrieval adapter makes the query side more tolerant to weight
noise, or the plain multilingual base spreads Czech over weights that a Czech Wikipedia sample does not exercise.
Compression quality (client vs its own fp32) and base-model quality (fp32 vs the deployed model) are reported separately
and must not be traded against each other: keeping 96 % of a weaker base is not a replacement argument, and here the
base is equal while the compression is not.

The fragility is not specific to GPTQ. `llama-quantize` with the same Czech imatrix on Qwen3-Embedding-0.6B (token table
Q2_K as everywhere above): Q3_K `--pure` 80.2 %, IQ3_XXS 69.8 %, Q3_K mixture 86.2 %, Q4_K_M (305 MiB) 94.7 % with
top-10 overlap 0.649. A pre-registered follow-up isolated the token table: with an intact (q8_0) table the Q3_K mixture
rises to 92.8 % (367 MiB) and **Q4_K_M reaches 99.3 %** (0.3164, cosine 0.967, overlap 0.785) at 414 MiB. So on this
model the Q2_K token table costs 4.6–6.6 nDCG points on Czech (rare Czech tokens sit in damaged rows) and the 3-bit blocks
cost the rest; jina, a retrieval fine-tune of the same architecture, tolerates both (96.9 % at 235 MiB). The
Qwen3-Embedding files that meet our release rule on the Czech index are Q4_K_M with a q4_0 table (**340 MiB, 98.3 %**,
cosine 0.964, overlap 0.773, Δ vs fp32 −0.005 [−0.012; +0.001]) and with a q8_0 table (414 MiB, 99.3 %), both made with
`llama-quantize` alone; nothing at or below 3 bits per weight does, whatever the quantizer.

On the four English corpora with its own fp32 indices (E3/E7, `results/tables/release_en.md`) the same base behaves
the same way, only less sharply: GPTQ Q3_K with generic English text keeps 96.6 / 96.6 / 100.0 / 95.1 % of nDCG@10 on
SciFact / NFCorpus / ArguAna / SciDocs but its cosine to the fp32 query vector is 0.905–0.945 (harrier's Q3_K: 0.96),
while `llama-quantize` Q4_K_M with an English imatrix keeps 98.7 / 98.2 / 100.8 / 100.2 % at cosine 0.95–0.97 with the
Q2_K token table (305 MiB) and 100.0 / 99.4 / 100.1 / 99.3 % at cosine 0.97–0.98 with a q4_0 table (340 MiB). The general
Qwen3-Embedding client we release is therefore a 4.5-bit file with a 4-bit token table, the general harrier client a
3.4-bit file; the Czech-calibrated Qwen3 file uses the same recipe with a Czech imatrix (98.7 % on the Czech index).
Czech synthetic-query calibration lifts the Qwen3 Q3_K file by +1.6 points (91.9 %), not enough to release it. Release
rule for any Qwen3 file: ≥ 95 % of its own fp32 with the CI of the difference above −0.02, cosine ≥ 0.94, top-10 overlap
≥ 0.75; a "legal" variant only if it beats the generic Czech calibration by ≥ +0.01 with a CI excluding zero, labelled a
quantization calibrated on Czech (legal) text, never a legal-domain model.

**A second distributable base that tolerates 3 bits (bge-m3, MIT).** The Czech fragility is a property of the
Qwen3-0.6B decoder-style embedding models, not of Czech. BAAI/bge-m3 (XLM-RoBERTa large, CLS pooling, 1024-d, MIT) was
indexed the same way with its own fp32 index of the 55 071 segments (fp32 nDCG@10 0.3169; the synthetic test is
saturated, every base we tried scores 0.317–0.319, so only the ratio, cosine and overlap are informative). With a Czech
imatrix and `llama-quantize` alone (encoders need a copy with `add_eos_token` cleared for `llama-imatrix` and
`-b 512 -ub 512`), its clients hold **Q4_K_M + q4_0 table 355 MiB: 99.2 %** (cosine 0.990, top-10 overlap 0.864) and
**Q3_K + q4_0 table 321 MiB: 98.3 %** (cosine 0.970, overlap 0.780), i.e. the 3-bit file of this encoder is as good as
jina's and better than any Qwen3-Embedding file below 4.5 bits (R2/R2b, `results/tables/release2.md`). The 4-bit token
table (half of the file: 250 002 × 1024) costs ≤ 0.2 nDCG points against an 8-bit one on this model, against 1.0 on
Qwen3-Embedding. The Czech calibration does not hurt English: the same files keep 99.6 / 100.0 % on SciFact. Both are
released (`thinletter/bge-m3-query-clients`); Q5_K_M (505 MiB, 99.6 %) adds 0.4 points and is not. Across the four
families measured on this index the quantization tolerance orders as bge-m3 (CLS encoder) ≥ jina-v5-small (retrieval
fine-tune) > Qwen3-Embedding-0.6B (plain multilingual base); multilingual-e5-small could not be measured (the GGUF
converter mismatches its tokenizer). For Qwen3-Embedding the 5.7-bit file with the Czech imatrix, Q5_K_M + q4_0 table (385 MiB), is
released as the higher-quality option for Czech: 99.5 % on the Czech index (cosine 0.983, overlap 0.839), +0.8 points
over Q4_K_M for 45 MiB (R1). The same file with the English imatrix keeps 99.8 / 100.1 / 100.1 / 100.1 % on SciFact /
NFCorpus / ArguAna / SciDocs at cosine 0.988–0.992: +0.3 points and +0.010–0.015 cosine over the released Q4_K_M, below the
pre-registered threshold for a second English file (+0.5 points, or +0.01 cosine on every corpus; SciDocs +0.0097); it is
published anyway as the "closer to fp32" option on the author's decision, with the numbers, not as a recommendation.

### 3.3 Fair comparison with llama.cpp's own quantizer (harrier-0.6b, ~200 MiB)

Same f16 source, same calibration text and token budget (585 × 512 tokens, `-c 512 --chunks 585`), token table Q2_K,
one runtime for every row (upstream llama.cpp b81c99b, Windows), test split, paired bootstrap against our file.

| file | MiB | calibration | SciFact | Δ vs ours [CI] | NFCorpus | Δ vs ours [CI] | cos to fp32 | ms/query idle |
|---|---|---|---|---|---|---|---|---|
| **ours: GPTQ Q2_K act-order** | 192.4 | SciFact corpus | 0.7400 | – | 0.3583 | – | 0.826 / 0.781 | **160** |
| IQ2_M, imatrix | 199.3 | SciFact corpus | 0.7418 | +0.002 [−0.015; +0.019] | 0.3743 | **+0.016 [+0.007; +0.026]** | 0.915 / 0.910 | **103** |
| IQ2_M, imatrix | 199.3 | generic EN | 0.7371 | −0.003 [−0.022; +0.015] | 0.3750 | +0.017 [+0.007; +0.026] | 0.910 / 0.905 | 103 |
| Q2_K mixture, imatrix | 209.5 | SciFact corpus | 0.7377 | −0.002 [−0.019; +0.014] | 0.3757 | +0.017 [+0.009; +0.026] | 0.909 / 0.910 | 175–179 |
| Q2_K `--pure` (our format), imatrix | 192.4 | SciFact corpus | 0.7347 | −0.005 [−0.024; +0.015] | 0.3636 | +0.005 [−0.004; +0.015] | 0.865 / 0.866 | 154–163 |
| Q2_K `--pure`, no imatrix (RTN) | 192.4 | – | 0.6239 | **−0.116 [−0.145; −0.091]** | 0.3007 | −0.058 [−0.075; −0.042] | 0.692 / 0.701 | 154–163 |
| IQ2_XS, imatrix | 177.5 | SciFact corpus | 0.7097 | −0.030 [−0.050; −0.011] | 0.3528 | −0.006 [−0.020; +0.007] | 0.844 / 0.836 | 106 |

Reading: in-domain tie, out-of-domain loss of 0.016 with CIs excluding zero, embeddings farther from fp32 (cosine 0.83 vs
0.91), GPTQ clearly beats only round-to-nearest. Peak RSS is 888–921 MB for every file (context 512): file size decides a
part of the footprint, the KV cache and compute buffers the rest. The calibration text barely matters for
`llama-quantize` (domain vs generic within ±0.005), which is why the domain effect is a property of GPTQ/GPTVQ, not of
calibration in general.

### 3.4 Browser

192 MiB harrier file, Chrome (headless), wllama 3.6.1, 300 SciFact test queries, idle machine (`browser_client.md`):

| backend | p50 | p95 | load | memory after load | peak RSS (all Chrome processes) | nDCG@10 (Δ vs native 0.7440) |
|---|---|---|---|---|---|---|
| WebGPU, Intel Xe integrated GPU, f16 shaders | **405 ms** | 680 ms | 4.7 s | 530 MiB (weights on GPU) | 2 124 MiB | 0.7413 (−0.0027, 13/300 queries differ) |
| WASM, 8 threads, prefix cache | 1 669 ms | 3 363 ms | 4.4 s | 1 147 MiB | 1 228 MiB | 0.7425 (−0.0015, 17/300 differ) |
| WASM, 1 thread (60 queries) | 6 964 ms | 12 268 ms | 4.9 s | 917 MiB | 1 209 MiB | 0.7382 on the subset |

Czech Q3_K client (235 MiB) in the browser, WebGPU, 50 test queries: p50 474 ms, load 4.5 s, 1 019 MiB after load,
nDCG@10 0.2539 vs 0.2572 native on the same queries (43/50 identical). Native laptop reference: 151 ms per query
(bitnet.cpp fork) / 160 ms (upstream), RSS 0.93 GB at context 512 and 4.08 GB at the model's default 128k context —
the context must be capped to the query length or the client is not deployable in a tab.

### 3.4b Idle comparison of browser runtimes (pre-registered V3, measured 2026-09-09)

Same laptop, same integrated GPU (Intel Xe), same headless Chrome, same fp32 SciFact index and 300 test queries, machine
idle. wllama 3.6.1 = llama.cpp's WebGPU backend in WebAssembly; "VQ" = our standalone WebGPU runtime for the
vector-quantised container (non-public, §3.6). `results/tables/browser_client.md`, rows `idle2*`.

| runtime / file | MiB | load | memory after load | peak RSS (all Chrome processes) | p50 | p95 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **VQ 2.10 bpw** | 119.5 | 2.4 s | 181 MiB + 151 MiB GPU | 1 592 MiB | **103 ms** | **113 ms** | 0.7375 |
| **VQ 1.83 bpw** | 105.3 | 2.5 s | 167 + 137 MiB | 1 501 MiB | 102 ms | 112 ms | 0.7109 |
| wllama WebGPU Q3_K generic-EN | 235.1 | 4.2 s | 530 MiB | 2 080 MiB | 444 ms | 754 ms | 0.7481 |
| wllama WebGPU IQ2_M | 199.3 | 5.2 s | 530 MiB | 1 983 MiB | 508 ms | 1 012 ms | 0.7387 |
| wllama WebGPU Q2_K | 192.4 | 4.7 s | 530 MiB | 2 124 MiB | 405 ms | 680 ms | 0.7413 |
| wllama WASM 8 threads Q2_K | 192.4 | 4.4 s | 1 147 MiB | 1 228 MiB | 1 669 ms | 3 363 ms | 0.7425 |

Reading, limited to this device, model and corpus: at half the file the VQ runtime answers 4.3× faster at p50 and 6.7×
at p95 than llama.cpp's WebGPU path, loads 1.8× faster and needs 77 % of the peak process memory (1 592 vs 2 080 MiB, all Chrome processes including the GPU process; the "memory after load" column is CPU-side only and does not contain wllama's GPU buffers, so 332 vs 530 MiB is not like-for-like), at −0.011
nDCG@10 against the 3.4-bit file and −0.004 against the 2.6-bit file. The pre-registered "≤ 40 % memory" did not hold (77 % of peak RSS);
the kill rule for the "smaller footprint, stable latency" claim (a wllama row within 1.3× of our memory and faster) did
not fire. Part of the gap is the format (fewer bytes to read), part the runtime (no output head, no 309 MiB llama.cpp
compute buffer, one decode of the weights per query); the accounting split is in the SDK notes. Not measured: a
discrete GPU, Transformers.js/ONNX Runtime Web, other machines.

### 3.5 Where calibration and rotation matter (secondary)

The domain effect of the calibration text replicates on every GPTQ/GPTVQ configuration we measured (3 models, 4
datasets), is stable across three paired calibration draws on Q2_K (+0.0089, sd 0.0015), and grows with quantizer damage
on one model and one vector grid (SciFact, harrier, 2.12 / 1.84 / 1.58 bpw: +0.020 / +0.042 / +0.137, recovering 52–62 %
of the loss to fp32). "Six documents beat 300k tokens of Wikipedia" did **not** replicate (ten draws of ~3k tokens:
0.7308 ± 0.0064 < generic 0.7385). Two-sided Hadamard rotation: bge-m3 at Q2_K +0.106 [+0.092; +0.120]; vector
quantization at 1.58 bpw +0.32/+0.38 across draws (the input side carries 66–96 % of the effect, three seeds per rate);
on harrier Q2_K two seeds give +0.012 and −0.000 — not a demonstrated stable effect. Fusing the rotation into RMSNorm
costs −0.132; cancelling it online per matrix does not, at +13 % latency (AVX2 kernel equal to torch to 0.0).

### 3.6 Sub-2-bit vector quantization in the browser (preview; depends on a non-public backend)

*Reproducibility note.* The quantizer, the container format and the WebGPU runtime behind this section are not part of
the public repository (they are a separate component being prepared as an SDK). The numbers are reported for
completeness and can be demonstrated, not independently reproduced from the released package; every other section can.

llama.cpp has no format below IQ2_XS (177.5 MiB, 0.7097 on SciFact). Our GPTVQ-style container (4-d codebooks per
256-column block, frozen row scales, input-side rotation, trimmed vocabulary) at 2.10 bpw is 119.5 MiB and gives 0.7375
in the browser through a standalone WebGPU runtime (torch simulation 0.7380; per-query cosine to torch 1.00000); at
1.83 bpw 105 MiB and 0.7109. Paired against BitNet-270m (140 MiB, ternary QAT): SciFact and NFCorpus indistinguishable,
ArguAna +0.044 [+0.023; +0.065] after removing leaked documents from calibration. On the Czech index the same
container format at 2.10 bpw holds 82.9 % with jina-v5-small (112 MiB; its scalar Q2_K: 86.2 %) and 64.3 % with
Qwen3-Embedding-0.6B (torch simulation, 1 137 queries): the format and runtime carry over to the other two models of
the architecture, the quality below 2.2 bits per weight does not, so a Czech deployment needs a ~3-bit variant of the
format that does not exist yet. Kernel performance after the M4 kernel: p50 142 ms under load; idle numbers in §3.4b.
Not part of this release beyond the numbers.

### 3.7 The query prompt's keys and values in full precision (pre-registered 2026-09-10)

The query prompt (harrier: the E5 instruction, 19 tokens; jina: `Query: `, 2 tokens) is the same text in front of every
query. Its keys and values in every block can therefore be computed once by the full-precision model and shipped with the
client as data (harrier: 28 blocks × K and V × 19 tokens × 1024 × f16 ≈ 2.2 MB; the first token alone 112 KB); the quantized
model then processes only the user's tokens and attends to an exact prompt. We evaluated the *same* quantized model three
ways on the same queries (plain; prompt K/V from fp32; only the first token from fp32), so the readouts are paired by
construction (`results/tables/prefix_kv.md`; scalar rows reproducible with `scripts/gptq_export_gguf.py --no_gguf
--prefix_kv_fp_tokens 0 1`, helper `scripts/prefix_kv.py`).

| client | plain nDCG@10 (% fp) | prompt K/V from fp32: Δ nDCG [CI] / Δ cos | first token only: Δ nDCG / Δ cos |
|---|---|---|---|
| harrier-0.6b VQ 1.84 bpw, SciFact, three rotation seeds | 0.726–0.731 (94 %) | **+0.009 / +0.015 / +0.020**, every CI above zero; cos +0.09 to +0.11 | +0.007 / +0.018 / +0.011; cos +0.03 to +0.05 |
| harrier-0.6b VQ 1.58 bpw (83 MiB), SciFact | 0.676 (87.5 %) | **+0.037 [+0.025; +0.048]** → 0.713 (92.3 %); cos +0.14 | +0.014; cos +0.05 |
| harrier-0.6b GPTQ Q2_K / Q3_K, SciFact | 0.753 / 0.770 | +0.001 / +0.000 (n.s.); cos +0.006 / +0.001 | +0.001 / −0.001 (n.s.) |
| jina-v5-small VQ 1.84 bpw / Q2_K, Czech index | 0.229 / 0.280 | +0.002 / −0.001 (n.s.); cos +0.01 | +0.003 / −0.002 (n.s.) |

The effect grows with the damage: nothing at 3.4 bits, +0.001 at Q2_K, +0.009 to +0.020 at 1.84 bpw, +0.037 at 1.58 bpw,
where the 83 MiB file with 2 MB of prompt data ends above the 1.84 bpw file without it. On the scalar grid and on the Czech
index the first token alone carries the whole (small) effect; on the vector grid it carries most of the nDCG gain and a third
of the cosine gain. The mechanism is the attention sink: a 2-bit model computes the first token's keys and values
differently, the "sink" that later tokens park their attention on does not form correctly, and every later token's
attention is displaced (diagnosed on decoders by the `llm-weight-compression` project, 2026-09-09; here measured on the
encoder). Shipping the exact first token repairs it for 112 KB. This goes into the vector-quantized runtime (the container
carries the prompt's K/V and the attention kernel starts from it, which also removes two thirds of the tokens per query);
for the GGUF clients it would need a KV-state import in llama.cpp and is not worth +0.001 at Q2_K. The pre-registered
prediction (cosine ≥ +0.005 on two of three seeds) held by an order of magnitude; the prediction that the first token carries
at least half of the gain held for nDCG and failed for the cosine.

*Addendum (2026-09-10, evening): the gain depends on the calibration.* All rows above were measured on quantizations
calibrated on documents (`scifact_corpus_only`), which never see the prompt during calibration. On containers calibrated
on synthetic queries (the prompt included, which is our recommended recipe at these rates and what the released files
use) the same prompt K/V gives −0.003 [−0.007; −0.000] on the SciDocs 1.83 bpw file and +0.000 [−0.002; +0.003] on the
2.10 bpw file, while a document-calibrated SciFact 1.83 bpw file gains +0.011 [+0.003; +0.019] (`scripts/vqw_add_prefix.py
--sim`, same file, paired). A query-calibrated model has already learnt to reproduce the prompt itself, and its later blocks
are fitted to its own prompt representation, so the fp K/V help only where the calibration left the prompt out. As it
stands the result is a repair for calibration without the prompt, not a gain on top of the best recipe. Whether calibrating
every block with the fp prompt K/V in place combines the two is being measured (pre-registered).

*Addendum 2 (2026-09-11): prefix-aware calibration.* Calibrating every block on the query tokens only, with the fp prompt's
K/V as the past (what a client with the shipped prefix computes), and shipping that prefix does combine the two on the
1.8-bit vector grid: on SciFact, three rotation seeds, +0.013 [+0.006; +0.021], +0.009 [+0.002; +0.017] and +0.008 [+0.001;
+0.016] nDCG@10 over the best synthetic-query-calibrated quantization of the same seed (95.3 → 96.5 % of fp32; cosine
0.90 → 0.92); the first token alone from fp32 does as well as the whole prompt. It is not detectable elsewhere: 2.1 bpw
+0.002 [−0.004; +0.009] (there is little left to repair at 97.5 %), 1.6 bpw +0.001 [−0.008; +0.010], SciDocs 1.8 bpw
+0.002 / +0.004 / +0.000 on three seeds (500 title-like queries), NFCorpus −0.000 [−0.003; +0.002] / +0.002 (3 237 queries,
two seeds), ArguAna +0.001 / −0.009 (two seeds; the plain file is at 98–100 %). The cosine to fp32 rises by 0.011–0.017 on
every corpus; the ranking follows only on SciFact. Per query the change is a redistribution, not a uniform repair: queries
whose relevant document sat at rank 2–3 or outside the top 10 gain, queries at rank 1 lose, and on SciFact that sums to
+0.010 while elsewhere it sums to zero. A model calibrated this way needs the prefix at inference (without it, 75–84 % of
fp32). The published containers are therefore unchanged; the recipe stays in the runtime as an option, not a default, and
we do not have a mechanism for why SciFact alone benefits.

## 4. Practical guidance

1. Compile both grids, verify against your index, keep the smallest file that passes: nDCG@10 ≥ 95 % of fp32 with the
   CI of the difference above −0.02, cosine ≥ 0.94, top-10 overlap ≥ 0.75. Expect Q3_K to pass everywhere we looked,
   Q2_K only on some (model, corpus) pairs.
2. Calibrate with text in the corpus language; generic text is enough at Q3_K. Synthetic queries from your documents are
   worth +0.01–0.02 only at Q2_K.
3. Start with `llama-quantize --imatrix`; run the GPTQ arm as a second candidate and keep whichever verifies better.
4. In the browser prefer WebGPU (4× faster than WASM here), cap the context to the query length, deliver the file in
   chunks under the host's per-asset limit, and re-verify per-query agreement with the native run on the same queries.
5. Read differences under 0.01 nDCG as ties; the cosine to the fp32 query vector is the stable readout.

## 5. Limitations

One size class (0.6B) and one architecture family for the file-level claims (both teachers are Qwen3-0.6B fine-tunes;
bge-m3 and Qwen3-4B appear only in grid-level simulations). One Czech corpus, evaluated with synthetic queries and no
human relevance. Paired CIs over queries omit calibration-draw variance (~0.01) and between-machine variance (±0.006).
No comparison with ONNX int8/int4 export through transformers.js in the browser. No measurement on a discrete GPU in the
browser. The production-scale index (9.58 M texts) was not measured.

## 6. Reproducibility

The public repository contains the verification recipe (data layer, metrics, native evaluation, paired CIs), the GPTQ
exporter onto the K-quant grids, the `llama-quantize` recipes of the released Qwen3 files, the comparison scripts and the
cited tables: §3.1–3.4 and the calibration results of §3.5 are reproducible from it. It does not contain the structured
rotation code, the vector-quantization code (quantizer, container, runtime), the raw per-run results or the research log;
the rotation rows of §3.5 and all of §3.6 report numbers only; of §3.7 the GGUF rows reproduce (`scripts/prefix_kv.py`,
`gptq_export_gguf.py --no_gguf --prefix_kv_fp_tokens 0 1`), the vector-quantized rows report numbers only.
Released files (MIT, derived from harrier-0.6b): Q3_K generic-EN 235.1 MiB (sha256 a06e72ce…), Q2_K SciDocs-synthetic
192.4 MiB (c75c22b6…), Q2_K generic-EN 192.4 MiB (db7e7760…). Base-model and dataset licences in the model card. The
Czech customer data and the jina-derived files (CC BY-NC 4.0) are not distributed.

## 7. Related work (short)

*Prompt K/V in full precision (§3.7).* PrefixQuant (Chen et al., 2024) prepends high-frequency outlier tokens as a fixed
prefix so that activation quantization does not see them; attention sinks (Xiao et al., 2023) and massive activations
(Sun et al., 2024) describe why the first token matters. What we ship is the exact keys and values of a constant prompt
under weight-only quantization, evaluated paired against the same quantized model; the sink diagnosis on 2-bit decoders
is from the `llm-weight-compression` project (2026-09-09).

Asymmetric retrieval with a frozen document index: Query Encoder Distillation via Embedding Alignment; KALE (Campos et
al., 2023). Post-training quantization: GPTQ; llama.cpp's importance-matrix quantizer (the baseline of §3.3); calibration
data studies for GPTQ (Williams & Aletras, 2024; "Understanding and Selecting Calibration Data", 2026) measure perplexity
on causal LMs, not retrieval. Rotations: QuIP# (Tseng et al., 2024), QuaRot (Ashkboos et al., 2024), SpinQuant (Liu et
al., 2024). Codebook vector quantization: GPTVQ (van Baalen et al., 2024), AQLM (Egiazarian et al., 2024). Browser
inference: wllama (llama.cpp in WebAssembly/WebGPU), transformers.js (ONNX Runtime Web). Ternary QAT embedding models
used as external baselines: BitNet-embedding-270m/0.6b (Microsoft). Full list with what is ours and what is not:
`research/prior_art.md`.

## 8. Pending runs at draft time (pre-registered 2026-09-08)

- imatrix-CS: `llama-quantize` with a Czech imatrix on jina/legal-cs (Q3_K mixture, Q3_K pure, IQ2_M, IQ3_XXS, RTN;
  generic Czech and synthetic imatrix). Prediction: Q3_K pure within ±0.01 of our GPTQ Q3_K (CI over zero); IQ2_M ≥ our
  Q2_K generic-Czech + 0.03. Kill rule for "our export is the better Czech file": any llama-quantize file ≤ 236 MiB within
  0.01 of ours.
- E1: done — fp32 0.3186 (prediction held), Q3_K generic-Czech 90.0 % (prediction failed, kill rule fired; §3.2).
- E3 / E4 (pre-registered after E1): Qwen3-Embedding-0.6B general client on the four English corpora (Q3_K generic
  English; prediction ≥ 97 %, kill < 95 % on two corpora), Czech synthetic-query calibration (prediction ≥ +0.01 over
  generic Czech, CI excluding zero) and `llama-quantize` Q4_K_M / IQ3_XXS arms with Czech and English importance
  matrices as the size/quality compromises.
- E2: done — shape effect +0.0005 (Q2_K) / −0.0011 (Q3_K), prediction held (§3.2).
- SciDocs Q3_K generic-EN row of §3.1: done (0.2250, 99.2 %).
