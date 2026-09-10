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
  - wllama
  - query-encoder
  - asymmetric-retrieval
---

# Qwen3-Embedding-0.6B query-side clients (GGUF, 340–385 MiB)

Small **query encoders** for [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). Use one of
them wherever you already have a document index built with Qwen3-Embedding-0.6B and want to encode queries on the
client (laptop, browser tab, edge device) instead of on a server: the index stays as it is, the query vectors stay
compatible, and the file is 340–385 MiB instead of 1 142 MiB (fp16). Runs unchanged in llama.cpp (`llama-embedding`) and in
the browser (wllama 3.6.1, WebGPU or WebAssembly). Live demo: https://thinletter.io/demo.

| file | use it for | quality vs the fp32 model on its own index (nDCG@10) |
|---|---|---|
| `qwen3-0.6b-imx-Q4_K_M-generic_wikitext-tabq4_0.gguf` (339.9 MiB) | **English and general multilingual retrieval** | SciFact 100.0 % · NFCorpus 99.4 % · ArguAna 100.1 % · SciDocs 99.3 % (cosine to the fp32 query vector 0.97–0.98) |
| `qwen3-0.6b-imx-Q4_K_M-generic_cs-tabq4_0.gguf` (339.9 MiB) | **Czech corpora** (quantization calibrated on Czech text) | Czech supreme-court index, 55 071 segments: 98.7 % (cosine 0.966, top-10 overlap 0.775) |
| `qwen3-0.6b-imx-Q5_K_M-generic_cs-tabq4_0.gguf` (385.4 MiB) | Czech corpora, when 45 MiB more is acceptable | Czech index: 99.5 % (cosine 0.983, overlap 0.839) |
| `qwen3-0.6b-imx-Q5_K_M-generic_wikitext-tabq4_0.gguf` (385.4 MiB) | English / multilingual, when 45 MiB more is acceptable | SciFact 99.8 % · NFCorpus 100.1 % · ArguAna 100.1 % · SciDocs 100.1 % (cosine 0.988–0.992, overlap 0.91–0.96) |

All files are the same recipe (`llama-quantize` K-quant mixture with a 4-bit token table) and differ in the language of
the importance-matrix text and in the bit width of the blocks (Q4_K_M 4.5 bits, Q5_K_M 5.7 bits). The Q5_K_M files sit
0.01 closer in cosine to the fp32 query vector and rank 2–4 % more of the top-10 identically; in nDCG@10 the difference is
within noise on English (+0.3 points) and +0.8 points on Czech. If your corpus is Czech, take the second (or fourth);
otherwise the first (or third).

The file badge on Hugging Face shows the type of the token table (`Q4_0`), which all files here share; the bit width of
the layer blocks (Q4_K_M = 4.5 bits, Q5_K_M = 5.7 bits) is in the file name and in the GGUF header (`general.file_type`
15 / 17), and is what the sizes differ by.

## Compatibility

Compatible with document vectors from `Qwen/Qwen3-Embedding-0.6B` produced **without** an instruction on the document
side, last-token pooling, L2 normalisation, 1024 dimensions. Matryoshka truncation works the same way for the client's
output as for the original (truncate, then renormalise). Queries use the model's instruction format; the files were
evaluated with the generic web-search instruction and **no trailing space** after `Query:`, which is what the official
sentence-transformers prompt produces:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:<your query>
```

**Not compatible** with indexes of `jina-embeddings-v5-text-small`, `harrier-oss-v1-0.6b` or any other model, even
though the architecture and the dimension are the same.

## How to use

```
llama-embedding -m qwen3-0.6b-imx-Q4_K_M-generic_wikitext-tabq4_0.gguf -c 512 --pooling last --embd-normalize 2 \
    -p "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:<your query>"
```

Set the context to the query length (`-c 512`); at the model's default 32k context llama.cpp allocates a KV cache of
several GB. In the browser, wllama 3.6.1 loads the file directly (WebGPU where `shader-f16` is available, WebAssembly
otherwise); the demo at https://thinletter.io/demo shows the chunked download and the per-query verification.

## Results

Test split, native `llama-embedding`, fp32 document index of the same model. nDCG@10 (% of fp32), mean cosine of the
client's query vector to the fp32 query vector, mean top-10 overlap with the fp32 ranking.

| corpus (queries) | fp32 | Q4_K_M + q4_0, English imatrix | Q5_K_M + q4_0, English imatrix | Q4_K_M + q4_0, Czech imatrix | Q5_K_M + q4_0, Czech imatrix |
|---|---|---|---|---|---|
| SciFact (300) | 0.7004 | 0.7004 (100.0 %) · cos 0.981 · overlap 0.898 | 0.6989 (99.8 %) · 0.992 · 0.928 | – | – |
| NFCorpus (323) | 0.3537 | 0.3514 (99.4 %) · 0.973 · 0.869 | 0.3539 (100.1 %) · 0.988 · 0.912 | – | – |
| ArguAna (700, self-document ignored) | 0.7034 | 0.7044 (100.1 %) · 0.980 · 0.937 | 0.7040 (100.1 %) · 0.991 · 0.959 | – | – |
| SciDocs (500) | 0.2168 | 0.2154 (99.3 %) · 0.983 · 0.898 | 0.2171 (100.1 %) · 0.992 · 0.936 | – | – |
| Czech supreme-court segments (1 137 synthetic queries) | 0.3186 | – | – | 0.3145 (98.7 %) · 0.966 · 0.775 | 0.3171 (99.5 %) · 0.983 · 0.839 |

Why 4.5 bits and not 3 (and 5.7 bits only as an option): every ≤ 3-bit file of this model failed our release rule (≥ 95 % of fp32, cosine ≥ 0.94, top-10
overlap ≥ 0.75). GPTQ Q3_K keeps 95–100 % nDCG on English but its cosine drops to 0.90–0.94; on Czech it keeps 90 %,
`llama-quantize` Q3_K 80–86 %, IQ3_XXS 70 %; a 2-bit token table alone costs 4.6–6.6 points on Czech. For comparison,
`microsoft/harrier-oss-v1-0.6b`, the same architecture as a retrieval fine-tune, holds 99 % at 3.4 bits
([harrier clients](https://huggingface.co/thinletter/harrier-0.6b-query-clients)).

Read differences under 0.01 nDCG@10 as ties: a paired interval over queries does not include the variance of the
calibration draw (~0.01) or between-machine variation (±0.006). The Czech evaluation uses synthetic queries
(Qwen3-1.7B doc2query, relevant = source segment), no human relevance judgements.

## Recipe

```
llama-imatrix -m qwen3-embedding-0.6b-f16.gguf -f generic_cs.txt -c 512 --chunks 585 -o imatrix.gguf   # or generic_wikitext.txt
llama-quantize --imatrix imatrix.gguf --token-embedding-type q4_0 qwen3-embedding-0.6b-f16.gguf out.gguf Q4_K_M   # or Q5_K_M
```

`generic_cs.txt` = 3 000 paragraphs of Czech Wikipedia (sampling script in the repository), `generic_wikitext.txt` =
wikitext-2. The verification harness (import your index, synthetic queries, nDCG / cosine / overlap, paired bootstrap)
and the technical report are at https://github.com/rosecky/embedding-quantization-public.

sha256: `71676d1baa7fee867a680a7972ae88e9570c4fc3c99f33b0392b9b5fc508836f` (English imatrix),
`f76f1112bc43a1cd35c7c55c163e047caa3cde7b21b05b1e6891033db8dd29f4` (Czech imatrix),
`addc435fac1faba1259177e5c09b96ed578b70b45b8bbe2774f0272be5f3e9cb` (Q5_K_M, Czech imatrix),
`4f5cd5878269573272dab142c1c5303bc24a70ce545f6c4b7d708970aff02938` (Q5_K_M, English imatrix).

## Licence

Weights derive from `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0) and are released under Apache-2.0. Calibration texts:
wikitext-2 (CC BY-SA), Czech Wikipedia (CC BY-SA; sampling script published, text not redistributed).
