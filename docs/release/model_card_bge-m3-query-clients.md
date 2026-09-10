---
license: mit
base_model: BAAI/bge-m3
language:
  - cs
  - en
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

# bge-m3 query-side clients (GGUF, 321–355 MiB, calibrated on Czech text)

Small **query encoders** for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (XLM-RoBERTa large, CLS pooling, 1024-d,
100+ languages; dense retrieval only). Use one of them wherever you already have a document index built with bge-m3's
dense embeddings and want to encode queries on the client (laptop, browser tab, edge device): the index stays as it is,
the query vectors stay compatible, and the file is 321–355 MiB instead of 1 104 MiB (fp16). The quantization is
calibrated on Czech text; it holds equally on English (SciFact) because the encoder tolerates the grid well. Runs
unchanged in llama.cpp (`llama-embedding --pooling cls`) and in the browser (wllama 3.6.1).

| file | use it for | quality vs the fp32 model on its own index (nDCG@10) |
|---|---|---|
| `bge-m3-imx-Q4_K_M-generic_cs-tabq4_0.gguf` (354.6 MiB) | **the default**: Czech and multilingual retrieval | Czech supreme-court index (55 071 segments): **99.2 %**, cosine 0.990, top-10 overlap 0.864 · SciFact: 99.6 %, cosine 0.991 |
| `bge-m3-imx-Q3_K-generic_cs-tabq4_0.gguf` (320.6 MiB) | when 34 MiB less matters | Czech index: 98.3 %, cosine 0.970, overlap 0.780 · SciFact: 100.0 %, cosine 0.974 |

Half of a bge-m3 file is the 250 002 × 1024 token table; both files keep it in 4-bit (q4_0), which costs ≤ 0.2 nDCG
points against an 8-bit table on this model.

## Compatibility

Compatible with document vectors from `BAAI/bge-m3` **dense** embeddings: CLS pooling, L2 normalisation, 1024
dimensions, no prompt on either side (bge-m3 uses none). Not compatible with bge-m3's sparse or ColBERT outputs, nor
with indexes of any other model.

## How to use

```
llama-embedding -m bge-m3-imx-Q4_K_M-generic_cs-tabq4_0.gguf -c 512 --pooling cls --embd-normalize 2 -p "<your query>"
```

Set the context to the query length (`-c 512`). In the browser wllama 3.6.1 loads the file directly (pooling `cls`).

## Results

Test split, native `llama-embedding`, fp32 document index of the same model. nDCG@10 (% of fp32), mean cosine of the
client's query vector to the fp32 query vector, mean top-10 overlap with the fp32 ranking.

| corpus (queries) | fp32 | Q4_K_M + q4_0 (355 MiB) | Q3_K + q4_0 (321 MiB) | Q5_K_M + q8_0 (505 MiB, not published) |
|---|---|---|---|---|
| Czech supreme-court segments (1 137 synthetic queries) | 0.3169 | 0.3144 (99.2 %) · 0.990 · 0.864 | 0.3114 (98.3 %) · 0.970 · 0.780 | 0.3156 (99.6 %) · 0.995 · 0.907 |
| SciFact (300) | 0.6414 | 0.6388 (99.6 %) · 0.991 · 0.910 | 0.6416 (100.0 %) · 0.974 · 0.854 | 0.6443 (100.5 %) · 0.996 · 0.937 |

Read differences under 0.01 nDCG@10 as ties (calibration-draw variance ~0.01, between-machine ±0.006). The Czech
evaluation uses synthetic queries (Qwen3-1.7B doc2query, relevant = source segment), no human relevance judgements;
its fp32 level is the same for every base model we tried (0.317–0.319), so it measures the compression, not the base.

## Recipe

```
# llama-imatrix refuses encoders that append EOS: collect the matrix on a copy with the flag cleared
python gguf-py/gguf/scripts/gguf_set_metadata.py bge-m3-f16-noeos.gguf tokenizer.ggml.add_eos_token "" --force
llama-imatrix -m bge-m3-f16-noeos.gguf -f generic_cs.txt -c 512 -b 512 -ub 512 --chunks 585 -o imatrix.gguf
llama-quantize --imatrix imatrix.gguf --token-embedding-type q4_0 bge-m3-f16.gguf out.gguf Q4_K_M
```

`generic_cs.txt` = 3 000 paragraphs of Czech Wikipedia (sampling script in the repository). Verification harness and
report: https://github.com/rosecky/embedding-quantization-public.

sha256: `fe1011d5ec5cff995cb687f42afc47edcaf68cc8296dda11aeb7bc9640b8e295` (Q4_K_M + q4_0), `b977fd649e0750fa8092970ff8bb8d8f955e684d9cbbef055bbe89061af38fdc` (Q3_K + q4_0).

## Licence

Weights derive from `BAAI/bge-m3` (MIT) and are released under MIT. Calibration text: Czech Wikipedia (CC BY-SA;
sampling script published, text not redistributed).
