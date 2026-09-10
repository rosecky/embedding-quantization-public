# Technical brief — query-side quantization against an unchanged embedding index

Jan Rosecký, September 2026 · repository `github.com/rosecky/embedding-quantization-public` · one page, forwardable.

**Problem.** Systems that already hold an embedding index (pgvector, Elasticsearch, Qdrant) can move query encoding to
the client — browser, laptop, phone — without re-indexing, if a small query encoder ranks documents like the full model.

**What was done.** The query side of two Qwen3-0.6B embedding models (microsoft/harrier-oss-v1-0.6b, MIT;
jinaai/jina-embeddings-v5-text-small, CC BY-NC) was quantized post hoc with GPTQ directly onto llama.cpp's K-quant grids
(2.6 and 3.4 bits/weight), exported as bit-exact GGUF (192 / 235 MiB vs 1 143 MiB fp16) and verified against the unchanged
fp32 index: nDCG@10, cosine to the fp32 query vector, top-10 overlap, paired bootstrap over queries, pre-registered
predictions with kill rules. Corpora: SciFact, NFCorpus, ArguAna, SciDocs (English) and a Czech legal index of 55 071
supreme-court reasoning segments (customer index, vectors reproduced to cosine 0.9999).

**Results.**

| | 3.4-bit file (235 MiB) | 2.6-bit file (192 MiB) |
|---|---|---|
| harrier / SciFact · NFCorpus · ArguAna · SciDocs, % of fp32 nDCG@10 | 98.9 · 99.3 · 101 · 99.2 (generic English text, one file) | 98.4 · 94.5 · 96 · 93–94 (corpus / synthetic text) |
| jina / Czech legal, generic **English** calibration | 94.0 % | **50.9 %** |
| jina / Czech legal, generic **Czech** calibration | **96.9 %** | 79.5 % |
| jina / Czech legal, synthetic queries from the corpus | 96.3 % | 86.2 % |
| Qwen3-Embedding-0.6B (Apache-2.0, own index, fp32 0.3186 ≈ jina) / Czech legal, generic **Czech** calibration | **90.0 %** (same recipe, different model); `llama-quantize` Q3_K 80–86 %; first file to pass: Q4_K_M + q4_0 token table, 340 MiB, 98.3 % (q8_0 table: 414 MiB, 99.3 %) | – |

1. *Language before domain.* Czech instead of English calibration text: +0.091 nDCG@10 [+0.076; +0.106] at 2.6 bits.
   Synthetic queries from the corpus add +0.022 [+0.011; +0.032] at 2.6 bits and nothing at 3.4 bits (−0.002 [−0.009; +0.004]).
2. *The size/quality point is per deployment.* 2.6 bits holds 98 % on SciFact, 93 % on SciDocs, 80–86 % on the Czech index;
   3.4 bits holds ≥ 96 % everywhere measured.
3. *Negative result kept in.* Against `llama-quantize --imatrix` with the same text and budget, our GPTQ export ties
   in-domain (SciFact +0.002 [−0.015; +0.019]), loses out-of-domain (NFCorpus −0.016 [−0.026; −0.007]) and is 1.5× slower
   (160 vs 103 ms). The pipeline therefore uses llama.cpp's quantizer as its first arm. Czech branch at 3.4 bits: the
   K-quant mixture with a Czech imatrix ties our export (0.3092 = 0.3092, +23 MiB); in the identical format ours is
   +0.010 [+0.002; +0.018], within calibration-draw noise.
4. *Browser.* Same 192 MiB file in Chrome: WebGPU (integrated GPU) p50 405 ms, 530 MiB in the tab; WebAssembly 1.7 s,
   1.15 GB; nDCG within ±0.003 of native. File size ≠ memory: 0.93 GB RSS at context 512, 4 GB at the default context.
5. *Preview (non-public runtime).* A standalone WebGPU runtime for a 2.1-bit vector-quantized container (119.5 MiB)
   reaches 0.7375 on SciFact in the browser and, on the same idle laptop iGPU, answers in 103 ms p50 / 113 ms p95 vs
   444 / 754 ms for llama.cpp's WebGPU path (Q3_K, 235 MiB), peak process memory 1 592 vs 2 080 MiB. The format carries over to
   jina and Qwen3-Embedding, its quality below 2.2 bpw on the Czech index does not (83 % / 64 %).

**What is new and what is not.** Asymmetric retrieval, GPTQ, importance matrices, Hadamard rotations and codebook VQ are
prior work (KALE, QuIP#, QuaRot, SpinQuant, GPTVQ, AQLM). Contributed here: the reproducible chain compile → verify against
the unchanged index → export → browser, the calibration map (language first, domain effect growing with quantizer damage,
absent with the imatrix quantizer), the negative results, and the WebGPU VQ runtime.

**Caveats.** One size class; both teachers are Qwen3-0.6B fine-tunes; Czech evaluated with synthetic queries and no human
labels; paired CIs omit calibration-draw variance (~0.01) and between-machine variance (±0.006); no ONNX/transformers.js
comparison; production-scale index (9.6 M texts) not measured.

**Ask.** Run the verification recipe (README, five commands) on your own index and send the table; or tell me which
conclusion you would challenge first. Contact: issues at https://github.com/rosecky/embedding-quantization-public (a contact address follows).
