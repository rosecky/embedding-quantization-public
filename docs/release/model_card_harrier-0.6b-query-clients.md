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

# harrier-0.6b query-side clients (GGUF, Q2_K / Q3_K)

Quantized **query encoders** for `microsoft/harrier-oss-v1-0.6b` (Qwen3-0.6B decoder, last-token pooling, 1024-d),
meant to be used **against an existing fp32/int8 document index computed by the unquantized model**. Documents are not
re-encoded; only the query side is replaced by one of these files. They run unchanged in llama.cpp (`llama-embedding`)
and in the browser (wllama 3.6.1, WebGPU or WebAssembly).

Method, evaluation protocol and the honest comparison with `llama-quantize --imatrix` (which ties these files in-domain
and beats them out-of-domain at the same size) are in the technical report:
`https://github.com/rosecky/embedding-quantization-public/blob/main/docs/release/technical_report.md`.

## Compatibility

These files replace the **query** side only. They are compatible with document vectors produced by
`microsoft/harrier-oss-v1-0.6b` with: no document prompt, last-token pooling, L2 normalisation, 1024 dimensions, fp32,
fp16 or int8 storage (the demo uses int8 with per-row scale, Δ nDCG −0.0003). They are **not** compatible with indexes
of other models, including other Qwen3-0.6B fine-tunes (Qwen3-Embedding-0.6B, jina-embeddings-v5-text-small): same
dimension does not mean same space. Query prompt: the E5-style instruction the base model uses (below); EOS appended.

## Files

| file | MiB | grid | calibration text (300k tokens) | sha256 |
|---|---|---|---|---|
| `harrier-0.6b-gptq-Q3_K-generic_wikitext-ps20000-t300k-ao-tabQ2_K.gguf` | 235.1 | Q3_K blocks (3.44 bpw), token table Q2_K | generic English (wikitext) — **download-once client** | `a06e72cebe4a586e88e13f6260a3db1ae6e92bd5c2571a3cfdbb23310e80a0e8` |
| `harrier-0.6b-gptq-Q2_K-scidocs_synth_only-ps20000-t300k-ao-tabQ2_K.gguf` | 192.4 | Q2_K blocks (2.625 bpw), token table Q2_K | synthetic queries generated from SciDocs documents | `c75c22b659fd341df988b0d1b2975c2a872da51a6f96e0a4ced250d009557a1f` |
| `harrier-0.6b-gptq-Q2_K-generic_wikitext-ps20000-t300k-ao-tabQ2_K.gguf` | 192.4 | Q2_K blocks, token table Q2_K | generic English (wikitext) | `db7e7760929dfbcb62dcf4eff9670bdd58b020c83fffa3e3547a5285d5ba2981` |

Tag semantics: `ps20000` per-sample calibration sequences (cap), `t300k` 300 000 calibration tokens, `ao` GPTQ act-order,
`tabQ2_K` token embedding table in Q2_K. Blocks are quantized by GPTQ directly onto the llama.cpp K-quant grid; the packed
bytes match `gguf.quants.dequantize` exactly and the runtime reproduces the torch simulation to ±0.002 nDCG@10.

## Quality against the unchanged fp32 index (nDCG@10, test split, native `llama-embedding`)

| corpus (queries) | fp32 model | Q3_K generic-EN | Q2_K SciDocs-synthetic | Q2_K generic-EN |
|---|---|---|---|---|
| SciFact (300) | 0.7559 | 0.7475 (98.9 %), cos 0.962 | – | 0.7284 (96.4 %), cos 0.848 |
| NFCorpus (323) | 0.3808 | 0.3782 (99.3 %), cos 0.959 | – | – |
| ArguAna (700, self-document ignored) | 0.6665 | 0.6750 (101.3 %), cos 0.966 | – | 0.6213 (96.3 % of 0.6451)† |
| SciDocs (500) | 0.2269 | 0.2250 (99.2 %), cos 0.964 | 0.2145 (94.5 %), cos 0.935 | 0.2046 (90.1 %), cos 0.856 |

† measured on all real queries rather than the test split. "cos" = mean cosine between the client's query vector and
the fp32 query vector. Differences under 0.01 nDCG@10 are ties: a paired CI over queries does not include the variance of
the calibration draw (~0.01) or between-machine variation (±0.006).

Which file to use: **Q3_K generic-EN** unless you have verified on your own index that Q2_K passes (≥ 95 % of fp32,
cosine ≥ 0.94, top-10 overlap ≥ 0.75). On SciFact Q2_K passes (98.4 % with corpus calibration); on SciDocs it does not
(93–94 %). A Czech legal index with a different Qwen3-0.6B fine-tune needed Q3_K (96.9 % vs 79.5 % at Q2_K).

## Speed and memory

Laptop (Core Ultra 7 155H, 8 threads, context 512, one process per query): Q2_K 151–160 ms per query, Q3_K similar;
`llama-quantize` IQ2_M of the same model is faster (103 ms). Peak RSS 0.93 GB at context 512 — **set the context to the
query length** (`-c 512`); at the model's default 128k context llama.cpp allocates a 4 GB KV cache.

Browser (Chrome, wllama 3.6.1, 300 queries, idle machine, Q2_K file): WebGPU on an Intel integrated GPU p50 405 ms,
530 MiB in the tab after load; WebAssembly 8 threads p50 1 669 ms, 1 147 MiB. nDCG within ±0.003 of native.

## How to use

```
# native
llama-embedding -m harrier-0.6b-gptq-Q3_K-generic_wikitext-ps20000-t300k-ao-tabQ2_K.gguf -c 512 --pooling last \
    --embd-normalize 2 -p "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <your query>"
```

Query prompt = the E5-style instruction the base model uses; the tokenizer appends EOS. Documents keep their existing
vectors from `microsoft/harrier-oss-v1-0.6b` (no document prompt, last-token pooling, L2-normalised). Browser: see
`demo/` in the repository (chunked download, OPFS, WebGPU/WASM selection) — `python demo/serve.py` runs it locally.

## Limitations

- Query side only. Quantizing both sides on the same 2-bit grid collapses (0.261 vs 0.596 nDCG@10 in our early tests).
- English calibration text. For a corpus in another language calibrate with text in that language: on a Czech index,
  English vs Czech generic text was −0.091 nDCG@10 at Q2_K and −0.009 at Q3_K.
- Evaluated on four BEIR corpora with the generic E5 instruction, not the per-task MTEB instructions; numbers are not
  MTEB leaderboard numbers.
- These files are not better than `llama-quantize --imatrix` at the same size; they are released because they are the
  files the report measures, with a verification recipe attached.

## Licences and provenance

Weights derive from `microsoft/harrier-oss-v1-0.6b` (MIT; itself a fine-tune of Qwen3-0.6B, Apache-2.0) and are released
under MIT. Calibration texts: wikitext (CC BY-SA), SciDocs (CC BY 4.0). No SciFact-calibrated file is released (SciFact is
non-commercial). Evaluation code and tables: Apache-2.0, `https://github.com/rosecky/embedding-quantization-public`.

## Citation

Rosecký, J. (2026). Keep the index, shrink the query encoder: what holds at 2–3.5 bits for the query side of a 0.6B
embedding model. Technical report, `github.com/rosecky/embedding-quantization-public`.
