---
license: mit
base_model: microsoft/harrier-oss-v1-0.6b
language:
  - en
pipeline_tag: feature-extraction
tags:
  - embeddings
  - retrieval
  - quantization
  - vector-quantization
  - webgpu
  - query-encoder
  - asymmetric-retrieval
---

# harrier-0.6b vector-quantised query clients (`.vqw`, 105–120 MiB)

Query encoders for [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) at
**1.8–2.1 bits per weight**, below the smallest format llama.cpp offers (IQ2_XS, 177.5 MiB). They are meant for the
same use as the [GGUF clients](https://huggingface.co/thinletter/harrier-0.6b-query-clients): your document index
built with harrier-0.6b stays as it is, only the query encoder moves to the client. These files need the **Thinletter
WebGPU runtime**, which is not published yet: it runs in the invitation-only demo at https://lab.thinletter.io and is
being prepared as an SDK. The files are published so that the numbers can be checked against real artifacts (sizes,
hashes, container contents) and so that they are ready the day the runtime is.

| file | bits/weight (blocks) | nDCG@10 on SciDocs (% of fp32) | browser check |
|---|---|---|---|
| `harrier-0.6b-vq2.0-scidocs.vqw` (127.0 MiB) | 2.10 (K = 256, 4-d codebooks) | 0.2150 = **94.7 %** of fp32 0.2269 (cosine 0.935), torch simulation on 500 test queries | browser (WebGPU, integrated GPU) reproduces the simulation on 50 queries to −0.0002, 49/50 identical |
| `harrier-0.6b-vq1.75-scidocs.vqw` (112.8 MiB) | 1.83 (K = 128) | 0.2073 = **91.3 %** (cosine 0.906) | browser +0.0009, 47/50 identical |

Measured on SciFact with the earlier (non-distributable) SciFact-calibrated containers of the same recipe: 2.10 bpw
0.7375 in the browser (97.6 % of fp32 0.7559; torch simulation 0.7380), 1.83 bpw 0.7109 (94.0 %); llama.cpp's IQ2_XS at
177.5 MiB: 0.7097. On one laptop with an Intel integrated GPU, idle, 300 queries: 2.10 bpw p50 103 ms / p95 113 ms, load
2.4 s, 181 MiB + 151 MiB GPU buffers in the tab, peak process memory 1 592 MiB; llama.cpp's WebGPU path in wllama with the
235 MiB Q3_K file: 444 / 754 ms, 4.2 s, 530 MiB, 2 080 MiB. Full tables and caveats:
https://github.com/rosecky/embedding-quantization-public/blob/main/docs/release/vq_results.md.

## What is in a `.vqw` file

One binary: an 8-byte magic, a JSON header (model shape, tokenizer and prompt settings, rotation seeds, a tensor table)
and 64-byte-aligned tensor data. Per block linear: f16 codebooks (one per 256-column block of the rotated input space,
K × 4 entries), packed 6/7/8-bit indices (one per 4 consecutive columns), f16 per-row scales. Norm weights f16, the
token table in llama.cpp's Q2_K block layout trimmed to the vocabulary of the calibration corpus plus a byte fallback,
and the sign/permutation data of the input-side structured rotation. The header is readable with any JSON parser; the
quantiser and the WebGPU kernels that consume the file are not part of this release.

## Compatibility

Document vectors from `microsoft/harrier-oss-v1-0.6b`, no document prompt, last-token pooling, L2, 1024-d. Query prompt
(stored in the header): the E5-style web-search instruction, EOS appended. Not compatible with other models' indexes.

## Limits

One model, two corpora; quality below 2.2 bits per weight on a Czech index with other models of the same architecture
was 83 % (jina-v5-small) and 64 % (Qwen3-Embedding-0.6B), so this rate is for English corpora and this model until a
~3-bit variant exists. Latency and memory were measured on one integrated GPU; no discrete-GPU or mobile numbers.

sha256: `e5891a4a54351156fa63681aec4c3d864b22d66b2299bd2cec7385543ef56cdf` (2.10 bpw), `0228e0ab4d820f7983f5c124dab84a8c2367f8ad002b48f3a19bc34cac548326` (1.83 bpw).
For comparison on the same index, the scalar GGUF clients of this model: Q3_K 235 MiB 99.2 %, Q2_K (SciDocs-synthetic)
192 MiB 94.5 %, i.e. the 2.10 bpw container matches the 2.6-bit GGUF at two thirds of its size.

## The prompt's K/V in full precision: measured, not an improvement for these files (2026-09-10)

The query prompt is the same 19 tokens in front of every query, so its keys and values in every block can be computed once
by the full-precision model and shipped with the client (≈ 2.2 MB; the first token alone 112 KB). On the same quantized
model this gives +0.009 to +0.020 nDCG@10 at 1.84 bpw (three seeds, every paired CI above zero) and +0.037 at 1.58 bpw
(87.5 → 92.3 % of fp32 on SciFact); the first token alone carries most of it (the 2-bit model forms the attention sink
wrongly, an exact first token repairs it). The containers here do not carry the prompt K/V yet and the runtime does not read
it yet; the numbers are in the report (§3.7) and in `results/tables/prefix_kv.md` of the public repository. On the scalar
GGUF clients the same trick is worth +0.001 and is not planned.

**Addendum (same day, evening):** those gains were measured on quantizations calibrated on documents. The files in this
repository are calibrated on synthetic queries (the prompt included), and on them the same prompt K/V gives −0.003
[−0.007; −0.000] (1.83 bpw) and +0.000 [−0.002; +0.003] (2.10 bpw): a model calibrated with the prompt already reproduces
it. So this is not an improvement pending for these files; whether calibrating with the exact prompt in place changes
that is being measured, and the files here will only be replaced if it does (≥ +0.005 nDCG@10, paired CI above zero).

**Addendum 2 (2026-09-11):** measured. Calibrating every block with the exact prompt in place and shipping it gives
+0.008 to +0.013 nDCG@10 over the best query-calibrated quantization at 1.83 bpw on SciFact (three seeds, every paired CI
above zero), but on SciDocs, the corpus of these files, +0.002 / +0.004 / +0.000 on three seeds, and nothing at 2.10 bpw.
The release rule is not met, so **these files stay as they are**; the recipe lives in the runtime for deployments whose
queries look like SciFact's. Numbers: report §3.7 addendum 2, `docs/release/vq_results.md` §6.

## Licence

Weights derive from `microsoft/harrier-oss-v1-0.6b` (MIT) and are released under MIT; calibration text: synthetic queries
generated from SciDocs documents (SciDocs, CC BY 4.0). The container format and the runtime are separate works under
their own terms.
