---
license: mit
base_model: microsoft/harrier-oss-v1-0.6b
language:
  - en
library_name: gguf
pipeline_tag: feature-extraction
tags:
  - embeddings
  - retrieval
  - quantization
  - gguf
  - llama.cpp
  - wllama
  - query-encoder
  - asymmetric-retrieval
---

# harrier-0.6b query-side clients (GGUF, 192–235 MiB)

Small **query encoders** for [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)
(Qwen3-0.6B retrieval fine-tune, last-token pooling, 1024-d). Use one of them wherever you already have a document index
built with harrier-0.6b and want to encode queries on the client (laptop, browser tab, edge device) instead of on a
server: the index stays as it is, the query vectors stay compatible, and the file is 235 MiB instead of 1 143 MiB (fp16).
Runs unchanged in llama.cpp (`llama-embedding`) and in the browser (wllama 3.6.1, WebGPU or WebAssembly). Live demo:
https://thinletter.io/demo.

| file | use it for | quality vs the fp32 model on its own index (nDCG@10) |
|---|---|---|
| `harrier-0.6b-gptq-Q3_K-generic_wikitext-ps20000-t300k-ao-tabQ2_K.gguf` (235.1 MiB) | **the default**: English retrieval, any corpus | SciFact 98.9 % · NFCorpus 99.3 % · ArguAna 101.3 % · SciDocs 99.2 % (cosine to the fp32 query vector 0.96) |
| `harrier-0.6b-gptq-Q2_K-scidocs_synth_only-ps20000-t300k-ao-tabQ2_K.gguf` (192.4 MiB) | scientific-paper corpora when 43 MiB less matters | SciDocs 94.5 % (cosine 0.935); quantization calibrated on synthetic queries from SciDocs |
| `harrier-0.6b-gptq-Q2_K-generic_wikitext-ps20000-t300k-ao-tabQ2_K.gguf` (192.4 MiB) | the 2.6-bit point with generic calibration, for comparison | SciFact 96.4 % · SciDocs 90.1 % (cosine 0.85) |

Take the Q3_K file unless you have verified on your own index that a Q2_K file passes (≥ 95 % of fp32, cosine ≥ 0.94,
top-10 overlap ≥ 0.75): 2.6-bit files do not generalise across corpora (98 % on SciFact, 93 % on SciDocs).

## Compatibility

Compatible with document vectors from `microsoft/harrier-oss-v1-0.6b` produced with no document prompt, last-token
pooling, L2 normalisation, 1024 dimensions, stored as fp32, fp16 or int8 (the demo uses int8 with a per-row scale,
Δ nDCG −0.0003). **Not compatible** with indexes of other models, including other Qwen3-0.6B fine-tunes
(Qwen3-Embedding-0.6B, jina-embeddings-v5-text-small): same dimension does not mean same space. Query prompt = the
E5-style instruction the base model uses, EOS appended by the tokenizer:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: <your query>
```

## How to use

```
llama-embedding -m harrier-0.6b-gptq-Q3_K-generic_wikitext-ps20000-t300k-ao-tabQ2_K.gguf -c 512 --pooling last \
    --embd-normalize 2 -p "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <your query>"
```

Set the context to the query length (`-c 512`): at the model's default 128k context llama.cpp allocates a 4 GB KV
cache; at 512 the process peaks at 0.93 GB. Laptop latency (Core Ultra 7 155H, 8 threads): 151–160 ms per query.
Browser (Chrome, wllama 3.6.1, 300 queries, idle machine, Q2_K file): WebGPU on an Intel integrated GPU p50 405 ms,
530 MiB in the tab; WebAssembly with 8 threads p50 1 669 ms, 1 147 MiB; nDCG within ±0.003 of native. The demo at
https://thinletter.io/demo shows the chunked download and the per-query verification.

## Results

Test split, native `llama-embedding`, fp32 document index of the same model. nDCG@10 (% of fp32) and mean cosine of
the client's query vector to the fp32 query vector.

| corpus (queries) | fp32 | Q3_K generic-EN | Q2_K SciDocs-synthetic | Q2_K generic-EN |
|---|---|---|---|---|
| SciFact (300) | 0.7559 | 0.7475 (98.9 %) · cos 0.962 | – | 0.7284 (96.4 %) · cos 0.848 |
| NFCorpus (323) | 0.3808 | 0.3782 (99.3 %) · 0.959 | – | – |
| ArguAna (700, self-document ignored) | 0.6665 | 0.6750 (101.3 %) · 0.966 | – | 0.6213 (96.3 % of 0.6451, all queries) |
| SciDocs (500) | 0.2269 | 0.2250 (99.2 %) · 0.964 | 0.2145 (94.5 %) · 0.935 | 0.2046 (90.1 %) · 0.856 |

Read differences under 0.01 nDCG@10 as ties: a paired interval over queries does not include the variance of the
calibration draw (~0.01) or between-machine variation (±0.006). Evaluated with the generic E5 instruction, not the
per-task MTEB instructions; these are not leaderboard numbers.

## How they were made

GPTQ (act-order, per-sample calibration sequences, 300k tokens) rounding each block linear directly onto the llama.cpp
Q3_K (3.44 bpw) or Q2_K (2.625 bpw) grid; token table Q2_K; packed bytes identical to `gguf.quants`, runtime reproduces
the torch simulation to ±0.002 nDCG@10. Tag: `ps20000` per-sample cap, `t300k` calibration tokens, `ao` act-order,
`tabQ2_K` token table. At the same size, llama.cpp's own `llama-quantize --imatrix` with the same text produces files of
the same in-domain quality (and better out-of-domain), so these are released as the measured files of the report with a
verification recipe, not as a better quantizer. Exporter, harness and report:
https://github.com/rosecky/embedding-quantization-public.

sha256: `a06e72cebe4a586e88e13f6260a3db1ae6e92bd5c2571a3cfdbb23310e80a0e8` (Q3_K generic-EN),
`c75c22b659fd341df988b0d1b2975c2a872da51a6f96e0a4ced250d009557a1f` (Q2_K SciDocs-synthetic),
`db7e7760929dfbcb62dcf4eff9670bdd58b020c83fffa3e3547a5285d5ba2981` (Q2_K generic-EN).

## Licence

Weights derive from `microsoft/harrier-oss-v1-0.6b` (MIT; a fine-tune of Qwen3-0.6B, Apache-2.0) and are released
under MIT. Calibration texts: wikitext-2 (CC BY-SA), SciDocs (CC BY 4.0).
