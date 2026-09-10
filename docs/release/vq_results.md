# Vector-quantised query clients: results (numbers only)

The vector-quantised container (4-d codebooks per 256-column block, frozen row scales, an input-side structured
rotation) and its WebGPU runtime are **not part of the public repository**; they are being prepared as an SDK. This page
collects the measured numbers so that the claims in the report can be read against them. The SciDocs-calibrated
containers are published on Hugging Face (`thinletter/harrier-0.6b-vq-clients`, MIT, 127 / 113 MiB; 94.7 % / 91.3 % of
fp32 on SciDocs, browser-verified) and a live demo with the VQ clients runs behind access control at
`https://lab.thinletter.io`; ask for an invitation.

All numbers are ours (`OUR_MEASUREMENT`), nDCG@10 against the unchanged fp32 index of the same model, test split,
paired bootstrap over queries where a CI is given (10 000 draws). "sim" = torch simulation of the quantised weights;
"browser" = the WebGPU runtime in headless Chrome on the same queries.

## 1. Size × quality, harrier-0.6b (SciFact fp32 0.7559, 300 test queries)

| client | file | nDCG@10 | % fp32 | how measured |
|---|---|---|---|---|
| VQ 2.10 bpw (K=256), synthetic-query calibration | 119.5 MiB | 0.7375 browser / 0.7380 sim | 97.6 | browser, 2/300 queries differ from sim |
| VQ 1.83 bpw (K=128) | 105.3 MiB | 0.7109 browser = sim | 94.0 | browser, 0/300 differ |
| VQ 1.58 bpw (K=64) | ~91 MiB | 0.7074 sim | 93.6 | sim; ±0.03 across rotation seeds |
| llama.cpp IQ2_XS (smallest llama.cpp format), domain imatrix | 177.5 MiB | 0.7097 | 93.9 | native |
| our GPTQ Q2_K | 192.4 MiB | 0.7440 | 98.4 | native |
| llama.cpp IQ2_M, domain imatrix | 199.3 MiB | 0.7418 | 98.1 | native |
| BitNet-embedding-270m (ternary QAT, EXTERNAL) | 140 MiB | 0.7350 | – | native |

Paired against BitNet-270m at 2.12 bpw (leak-free calibration): SciFact +0.007 [−0.014; +0.028] and NFCorpus −0.005
[−0.020; +0.009] indistinguishable, ArguAna +0.044 [+0.023; +0.065]. Calibration on documents is worse than on synthetic
queries on six of seven points; on NFCorpus document-calibrated VQ clients are below BitNet (−0.016 [−0.031; −0.001]).

## 2. Idle comparison of browser runtimes (same laptop, Intel Xe integrated GPU, headless Chrome, 300 SciFact queries)

| runtime / file | MiB | load | memory after load (CPU side) | peak RSS, all Chrome processes | p50 | p95 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **VQ runtime, 2.10 bpw** | 119.5 | 2.4 s | 181 MiB + 151 MiB GPU buffers | 1 592 MiB | **103 ms** | **113 ms** | 0.7375 |
| **VQ runtime, 1.83 bpw** | 105.3 | 2.5 s | 167 + 137 MiB | 1 501 MiB | 102 ms | 112 ms | 0.7109 |
| wllama 3.6.1 WebGPU, Q3_K generic-EN | 235.1 | 4.2 s | 530 MiB (GPU buffers not included) | 2 080 MiB | 444 ms | 754 ms | 0.7481 |
| wllama WebGPU, IQ2_M | 199.3 | 5.2 s | 530 MiB | 1 983 MiB | 508 ms | 1 012 ms | 0.7387 |
| wllama WebGPU, Q2_K | 192.4 | 4.7 s | 530 MiB | 2 124 MiB | 405 ms | 680 ms | 0.7413 |
| wllama WASM 8 threads, Q2_K | 192.4 | 4.4 s | 1 147 MiB | 1 228 MiB | 1 669 ms | 3 363 ms | 0.7425 |

Reading, limited to this device, model and corpus: at half the file the VQ runtime answers 4.3× faster at p50 and 6.7×
at p95 than llama.cpp's WebGPU path, loads 1.8× faster and peaks at 77 % of the process memory, at −0.011 nDCG@10
against the 3.4-bit file. GPU time per query at T=33 tokens: 96.6 ms, 92 % in the codebook matmul. Not measured: a
discrete GPU, Transformers.js / ONNX Runtime Web, other machines. Correctness: per-query cosine to the torch reference
1.00000 on 20 queries, layer tests within f16 tolerance (T=86: five norm tensors at 1.0–1.5e-3 vs 1e-3).

## 3. Other models and the Czech index (torch simulation, 1 137 synthetic queries, index unchanged)

| model | fp32 nDCG@10 | VQ 2.10 bpw (112.3 MiB) | scalar reference |
|---|---|---|---|
| jina-embeddings-v5-text-small | 0.3190 | 0.2646 (**82.9 %**), cos 0.891 | Q2_K synthetic 86.2 % (192 MiB), Q3_K 96.9 % (235 MiB) |
| Qwen3-Embedding-0.6B | 0.3186 | 0.2048 (**64.3 %**), cos 0.781 | Q3_K 90 %; first passing file Q4_K_M + q4_0 table 98.7 % (340 MiB) |

The container format and runtime carry over to the other two models of the architecture unchanged; the quality below
2.2 bits per weight on Czech does not. A Czech deployment needs a ~3-bit variant of the format, which does not exist yet.

## 4. Where the rotation matters (harrier, SciFact, all real queries, three seeds per rate)

| bpw | no rotation | input side only | both sides | effect of both sides (seed spread) | share carried by the input side |
|---|---|---|---|---|---|
| 2.12 | 0.7442 | 0.7521 | 0.7540 / 0.7605 / 0.7538 | +0.012 (0.007) | 66 % |
| 1.84 | 0.7004 | 0.7257 / 0.7291 / 0.7309 | 0.7383 / 0.7328 / 0.7339 | +0.035 (0.006) | 81 % |
| 1.58 | 0.3082 | 0.6760 / 0.6069 / 0.6637 | 0.6877 / 0.6294 / 0.6761 | +0.356 (0.058) | 96 % |

Below 2 bits the rotation is the difference between a dead and a working model; at 2.12 bpw it is a small improvement.
The domain-calibration "cliff" below 2 bits (generic text falling to 71 % of fp32 at 1.58 bpw) was found on SciFact only;
on ArguAna generic text keeps 90 % and corpus calibration adds nothing measurable.

## 5. What is and is not established

Established: a real file of 105–120 MiB with browser-verified quality, an order-of-magnitude latency advantage over the
llama.cpp WebGPU path on one integrated GPU, and transfer of the format to two more models of the same architecture.
On SciDocs (second English corpus, fp32 0.2269, 500 test queries) the published containers keep 94.7 % (2.10 bpw,
127 MiB) and 91.3 % (1.83 bpw, 113 MiB), the 2.10 bpw file matching the 2.6-bit GGUF (94.5 %, 192 MiB); the browser
reproduces the simulation on 50 queries to −0.0002 / +0.0009. Not established: any Czech point at ≤ 2.2 bpw,
discrete-GPU or mobile numbers, and a comparison with ONNX Runtime Web.

## 6. The prompt's keys and values in full precision (measured 2026-09-10, paired on the same quantised model)

The query prompt (19 tokens for harrier) is constant, so its K/V in every block can be computed by the fp32 model once and
shipped with the container (≈ 2.2 MB; first token alone 112 KB); the quantised model then processes only the user's tokens.
Same quantised model, same queries, three readouts; paired bootstrap over queries, 10 000 draws (`results/tables/prefix_kv.md`).

| container (SciFact, all 1 109 queries, fp32 0.7723) | plain | prompt K/V from fp32 | first token only from fp32 |
|---|---|---|---|
| 1.84 bpw, rotation seed 0 / 7 / 13 | 0.7257 / 0.7291 / 0.7309 | 0.7408 / 0.7493 / 0.7399 (+0.015 / +0.020 / +0.009, CIs above zero); cos to fp32 0.78 → 0.88–0.89 | 0.7432 / 0.7398 / 0.7378; cos 0.82–0.83 |
| 1.58 bpw (83 MiB), seed 0 | 0.6760 (87.5 %) | **0.7126 (92.3 %), +0.037 [+0.025; +0.048]**; cos 0.68 → 0.83 | 0.6904 (+0.014); cos 0.73 |

The same readout on the scalar GGUF clients of this model is +0.001 (Q2_K) and 0.000 (Q3_K), and on the Czech index with
jina-v5-small (prompt 2 tokens) +0.002 / −0.001, not significant. Mechanism: the attention sink on the first token forms
wrongly in the 2-bit model; an exact first token repairs most of the ranking loss. Not in the published containers yet
(the runtime has to start the attention from a shipped cache); report §3.7.
