# Thinletter demo: scientific search with a 192 MiB query encoder in the browser

Two corpora, one page: **SciFact** (5 183 abstracts, CC BY-NC, the original reference client) and **SciDocs** (BEIR,
25 656 scientific title+abstract documents, CC BY 4.0, the open corpus for the public demo). The page reads
`data/index.json` for the list, `?ds=<id>` (or the dropdown) selects `data/<id>/meta.json`; each corpus has its own
client GGUF under `model/<id>/`, its own index (`corpus.f16.NNN` shards, or `corpus.i8.NNN` + `corpus.scale.f32` when
row-scaled int8 provably changes nDCG@10 by less than 0.001 -- SciDocs), document shards `docs.NNN.json`, test queries
and reference numbers. Export one corpus with `scripts/demo_export.py --dataset <id> --client <gguf>`; the SciDocs
numbers are in `results/tables/scidocs_client.md`. The rest of this README describes the SciFact reference client.


The query side of the embedding model `harrier-0.6b` ([microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b),
a Qwen3-0.6B decoder, last-token pooling, 1024-d) compressed by GPTQ onto llama.cpp's Q2_K grid, calibrated on 585 SciFact
corpus documents, token table Q2_K, into one GGUF file of 192 MiB. It runs in the browser with llama.cpp compiled to
WebAssembly ([wllama](https://github.com/ngxson/wllama) 3.6.1) -- **on the GPU through llama.cpp's own WebGPU backend
where the browser offers it (the default, 4x faster than WASM on an integrated Intel GPU), WASM with 8 threads otherwise**
-- and searches the **unchanged** full-precision document index of the original model (5 183 SciFact abstracts) directly
in the tab. This directory is a static site (no build step, no
framework): `index.html`, `app.js`, `style.css`, the vendored runtime, the exported data and the model in 24 MiB chunks.

> **Evaluation only.** Model weights are not licensed for redistribution yet; base model microsoft/harrier-oss-v1-0.6b is
> MIT, Qwen3 base Apache-2.0, runtime llama.cpp/wllama MIT, SciFact corpus CC BY-NC 2.0 (non-commercial).

## Numbers

| | value | source |
|---|---|---|
| Model file | 192.4 MiB (201 771 840 bytes), `harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf` | `models/gguf/` |
| nDCG@10, SciFact test (300 queries), native llama.cpp | **0.7440** | `results/raw/gptq_export/results.jsonl` |
| nDCG@10, original fp16 model (1 143 MiB) | 0.7559 | same protocol, `results/tables/main.md` |
| Latency, native `llama-embedding`, laptop (Core Ultra 7 155H), 8 threads, ctx 512, idle | **151 ms** per query (bitnet.cpp fork), 160 ms (upstream llama.cpp, Windows build), 928 MiB peak RSS | `research/wins.md` §2.1b, `results/raw/imatrix_local/latency_idle_win.tsv` |
| Latency in the browser, your device | measured on your device by the page ("On this device" panel) | |

Browser runs on the same laptop, machine idle, headless Chrome, 300 SciFact test queries, the same 192 MiB file
(`scripts/browser_run.py --tag idle`, `results/raw/browser_local/browser_idle_*.json`, `browser_webgpu_mt.json`):

| run | p50 | p95 | load | memory after load (`measureUserAgentSpecificMemory`, whole origin) | peak RSS (all Chrome processes) | nDCG@10 (delta vs native 0.7440) |
|---|---|---|---|---|---|---|
| WASM, 8 threads, prefix cache | **1 669 ms** | 3 363 ms | 4.4 s | 1 147 MiB | 1 228 MiB | 0.7425 (-0.0015, 17/300 queries differ) |
| WASM, 8 threads, no prefix cache | 3 361 ms | 5 015 ms | 3.6 s | 1 147 MiB | 1 232 MiB | 0.7425 (-0.0015) |
| WASM, 1 thread (60 queries) | 6 964 ms | 12 268 ms | 4.9 s | 917 MiB | 1 209 MiB | 0.7382 on that subset (-0.0060) |
| **WebGPU** (llama.cpp WebGPU backend, integrated Intel Xe GPU, 29/29 layers offloaded, f16 shaders) | **405 ms** | 680 ms | 4.7 s | 530 MiB (weights on the GPU) | 2 124 MiB | 0.7413 (-0.0027, 13/300 queries differ) |

WebGPU is 4x faster than multi-threaded WASM, so the page uses WebGPU by default where the browser offers an adapter with
f16 shaders and falls back to WASM otherwise (the "Backend" line of the stats panel says which one actually ran, with the
adapter name and the number of offloaded layers, and links to a reload with the other backend); the discrete RTX 4050 of
this laptop has not been measured yet (headless Chrome picked the integrated GPU).
The query instruction is the generic E5 web-search instruction
`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ` (the protocol of the
native reference), not the MTEB SciFact task instruction.

**Second corpus, SciDocs** (CC BY 4.0, 25 656 documents, 500 test queries, fp32 nDCG@10 0.2269; `results/tables/scidocs_client.md`):
its own Q2_K client of 192.4 MiB calibrated on the SciDocs corpus gives **0.2115** (93.2 % of fp32) natively; the
synthetic-query calibration 0.2145 (94.5 %); the SciFact client used out of domain 0.2123 (93.6 %); BitNet-270m 0.1946.
In the browser (50 test queries) 0.1838 vs 0.1820 natively on the same queries (delta +0.0018). Note that the 98 % of
SciFact does not carry over: on SciDocs the same recipe keeps 93-94 % of the original, and domain calibration brought
nothing there. The SciDocs index is shipped as row-scaled int8 (25.1 MiB, nDCG@10 delta -0.0003 vs fp32).

**Third corpus: Czech case law (private).** 55 071 reasoning segments of Czech Supreme Court decisions (facet
`court_argument`, civil domains since 2020), exported from the owner's Elasticsearch together with the owner's own index
vectors from `jinaai/jina-embeddings-v5-text-small` (a Qwen3-0.6B-Base fine-tune, retrieval adapter merged, prompts
`Query: ` / `Document: `). Our pipeline reproduces those vectors (cosine 0.9999), so the index is used **unchanged**.
The data (`data/legal-cs/`, `demo/data/legal-cs/`, `demo/model/legal-cs/`) belong to the owner and are **never
committed**; `?ds=legal-cs` works only on a machine that ran the export. Evaluation: 1 137 synthetic Czech queries
(Qwen3-1.7B doc2query) to 600 held-out segments, relevant = source segment (no human judgments); fp32 nDCG@10 0.3190.
The 192 MiB Q2_K client keeps only **82.7 %** (corpus calibration) / **86.2 %** (synthetic) of fp32 with cosine 0.84 / 0.90
to the fp32 query vectors, so the demo ships the **Q3_K client (235.1 MiB, synthetic calibration, Q2_K token table):
0.3071 = 96.3 %** of fp32, cosine 0.969, top-10 overlap 0.789 (`results/tables/legal_client.md`). Browser check with the
Q2_K client (WASM, 50 queries): 44/50 identical to native, delta +0.012 (outside the ±0.002 criterion; jina appends no
EOS, int8 index); the browser check of the Q3_K client is done: WebGPU (Intel iGPU), 50 test queries, nDCG@10 0.2539 in the browser vs 0.2572 native on the same queries (delta -0.0033; 43/50 per-query identical), p50 474 ms, load 4.5 s, memory after load 1 019 MiB. Index shipped as row-scaled int8 (53.8 MiB,
nDCG@10 delta -0.0002). **Evaluation only:** jina-embeddings-v5-text-small is CC BY-NC 4.0 (non-commercial), the
quantized weights derived from it are not licensed for redistribution.

## What the page does

* **Backend, chosen automatically**: before loading, the page asks the browser for a WebGPU adapter
  (`navigator.gpu.requestAdapter()`); if one exists and has the `shader-f16` feature, the model is loaded with
  `n_gpu_layers: 99999` and llama.cpp's own WebGPU backend (compiled into the same wllama wasm) runs the 28 transformer
  layers and the output head on the GPU; otherwise (no `navigator.gpu`, no adapter, no f16 shaders) it is loaded with
  `n_gpu_layers: 0` and runs on the CPU in WebAssembly with min(8, cores) threads. The URL parameter
  `backend=webgpu|wasm|auto` overrides the choice, and the page shows a one-click link to reload with the other backend
  (the corpus and the other parameters are preserved). After loading, the page reads llama.cpp's own log: the
  `ggml_webgpu: adapter_info` line and the `offloaded N/29 layers to GPU` line are the evidence that the GPU is really
  used; if WebGPU was requested but those lines are missing, the stats panel says "WebGPU requested, fell back to CPU".
* **Load model**: fetches the GGUF as nine byte chunks of at most 24 MiB from `model/` (same origin), re-assembles the file
  in the browser's private origin file system (OPFS) so that the next visit needs no download, and hands it to wllama
  (`n_ctx` 512, pooling `last`, embeddings on). Shows download time and speed, load time, the backend actually used
  (WebGPU with the adapter's vendor/architecture, layers offloaded and graph splits, or WebAssembly with the thread count),
  cross-origin isolation and memory after load (with WebGPU the weights live in GPU memory, so the tab's own figure is
  lower and the whole process uses more; the table above has both).
* **Search**: prompt + query -> embedding (L2-normalised) -> dot products against the fp16 index (decoded to fp32 once) ->
  top-10 with score, title and abstract (click to expand). Encode and search latency are shown separately, with a running p50.
* **Verify on this device**: the 300 test queries one by one, nDCG@10 per query with the same code as the search box, mean
  against the native 0.7440 and fp16 0.7559, p50/p95 latency, JSON report to the clipboard. Abortable.
* Multi-threaded WebAssembly needs cross-origin isolation (COOP/COEP headers, `_headers`); without them the page says so and
  runs single-threaded. Devices reporting less than 4 GB of memory get a warning before loading.

URL parameters: `?ds=<corpus id>` (scifact | scidocs), `backend=auto|webgpu|wasm` (default auto), `?model=<url>` (a `.gguf` URL loaded by wllama directly, or a `.chunks.json` manifest), `threads=<n>`,
`autoload=1`, `autoquery=<text>`, `autoverify=1`, `verify_n=<n>`, `smoke=1` (headless test mode).

## M5 / VQ client: the vector-quantised encoder next to the scalar one

Since M5 of the VQ WebGPU runtime (`client/vqweb/README.md`) a corpus can offer **several clients** and the load panel has a
"Client" selector next to the corpus. SciFact ships three (the only corpus with VQ containers so far; jina/legal-cs
containers do not exist yet, the selector just lists zero or more VQ entries per corpus):

| client id | label | runtime | file | bpw | nDCG@10 test, reference | browser, our run |
|---|---|---|---|---|---|---|
| `scalar` | Q2_K scalar, 192.4 MiB (llama.cpp WASM/WebGPU) | `wllama` | `...tabQ2_K.gguf` (9 chunks) | 2.625 | 0.7440 (native llama.cpp) | 0.7425 (WASM) / 0.7413 (WebGPU) |
| `vq2.0` | VQ 2.10 bpw, 119.5 MiB (WebGPU only) | `vqweb` | `harrier-0.6b-vq2.0-scifact.vqw` (5 chunks) | 2.104 | 0.7380 (torch simulation of the file, test split; 0.7465 on all 1 109 queries) | 0.7375 |
| `vq1.75` | VQ 1.83 bpw, 105 MiB (WebGPU only) | `vqweb` | `harrier-0.6b-vq1.75-scifact.vqw` (5 chunks) | 1.833 | 0.7109 (simulation, test) | 0.7109 |

**Declaration.** `data/<ds>/meta.json` carries `clients: [{id, label, runtime: "wllama" | "vqweb", model_url, model_file,
size_bytes, sha256, bpw, ndcg_test, ndcg_browser, note, ...}]` plus `default_client`; the VQ entries also carry the
per-query reference values (`ndcg_test_per_query` = the exporter's torch simulation of exactly this file, aligned by qid
from `results/raw/vqweb/perq/<stem>_scifact_train+dev+test.npz`; `ndcg_browser_per_query` and `ndcg_browser_run` from
`results/raw/browser_local/browser_<id>_vqweb.json`). `data/index.json` lists the same entries in compact form per corpus.
The top-level `model_file` / `model_url` / `reference` fields stay as they are: a `meta.json` **without** `clients` (an
older export, or a corpus with only the GGUF) behaves exactly as before, the page synthesises the scalar entry from them.
The list is written by `demo/vq_clients.py` (`scripts/demo_export.py` regenerates `meta.json` and knows only the GGUF, so
run this afterwards; it is idempotent):

```
.venv/Scripts/python.exe demo/vq_clients.py --ds scifact --chunk \
    --vqw models/vqw/harrier-0.6b-vq2.0-scifact.vqw models/vqw/harrier-0.6b-vq1.75-scifact.vqw
```

`--chunk` cuts each container into the same 24 MiB byte-chunk manifest as the GGUF (`model/scifact/<file>.vqw.chunk000 …`
+ `<file>.vqw.chunks.json`, produced by the very function `scripts/demo_export.py: chunk_model`) and copies
`tokenizer.json` / `tokenizer_config.json` from `models/vqw/` next to it, because `client/vqweb/tokenizer.js` fetches them
relative to the container URL (`header.tokenizer.file`); without `--chunk` an existing manifest is reused (sha256 checked
against the container).

**Selection.** `?client=<id>` (kept alongside `ds=`, `model=`, `backend=` …; `model=` still overrides the selected
client's URL, `.gguf` / `.chunks.json` for the scalar path, `.vqw` / `.chunks.json` for the VQ path). The dropdown
changes the client in place before loading (the URL is updated) and reloads the page after a model is loaded. VQ entries
are **disabled with the reason** when `navigator.gpu` is missing or the adapter has no `shader-f16` (the same probe as
the scalar WebGPU choice, `loader.probeWebGPU`); the standalone runtime has no CPU fallback.

**Loading the VQ client.** The same `loader.assembleFromManifest` as the GGUF: chunks → one file in OPFS (cached for the
next visit, same progress bar and "cached" detection) → `ArrayBuffer` → `container.js` (header parse, every tensor
uploaded as its own storage buffer) → `tokenizer.js` (`tokenizer.json` next to the container, trimmed-vocabulary byte
fallback) → `runtime.js` (`requestDevice()` with `shader-f16`, `powerPreference: 'low-power'` = the integrated GPU where
there is a choice; 13 pipelines) → the file's buffer is released (`container.release()`), exactly the sequence of
`client/vqweb/bench.js`. The runtime modules are imported lazily from `./vqweb/` (a mount of `client/vqweb/` in
`serve.py`; a static host needs the directory copied to `demo/vqweb/`), so the scalar client never depends on them.

**Search / verify.** For the VQ client the page hands the **raw query** to `VqwTokenizer.encode()`, which applies the
**container header's** prompt and `add_eos` (checked against `meta.prompt` at load: `prompt_match`), and L2-normalises the
runtime's 1024-d output; scores, top-10, qrels highlighting and the nDCG@10 code are shared with the scalar client. The
stats panel shows the same ids with the runtime's meaning: load = GPU upload + tokenizer + pipelines, backend = "WebGPU
(vqweb standalone runtime)" with the adapter, memory = the tab's `measureUserAgentSpecificMemory` **plus** the GPU buffers
(weights + activations for 512 tokens, invisible to the tab's measurement), encode / search last and p50. The verify
table's reference rows switch to the selected client's own numbers: "Torch simulation, same file" (`ndcg_test`), "Browser,
our measurement" (`ndcg_browser`, with the adapter and p50 of that run) and the fp16 original; the per-query comparison and
the JSON report (`reference_kind: "simulation"`, `client`, `runtime`, `vq: {...}`) use the per-query simulation values.

**Smoke (2026-09-08, headless Chrome, integrated Intel Xe-LPG GPU, machine loaded by the other session's queues —
timings are not citable).** Scalar client, the unchanged procedure (`scripts/demo_smoke.py --ds scifact --verify_n 5
--backend wasm`) before and after the change: identical per-query tuples (qids 1, 100, 1012, 1014, 1019 → nDCG@10 0, 1,
1, 1, 1, same top-1 docs, 5/5 identical to native) and an identical top-10 for the search query, p50 1.9 s before / 3.1 s
after (load noise). VQ clients (`?client=vq2.0`, `verify_n=20`, Chrome with `--enable-unsafe-webgpu --ignore-gpu-blocklist
--use-angle=d3d11 --enable-features=SharedArrayBuffer,Vulkan,WebGPU`): **vq2.0** nDCG@10 0.6783 on the first 20 test
queries = the simulation on the same queries to 1.5e-8 (20/20 identical, and identical to our 300-query browser run),
p50 156 ms / p95 177 ms, load 5.0 s (download 1.8 s, upload 0.15 s, tokenizer 0.9 s, pipelines 3.9 s), memory 71 MiB in the
tab + 151 MiB GPU buffers (111 weights + 40 activations); **vq1.75** 0.6126 = simulation to 3e-9 (20/20), p50 157 ms / p95
221 ms, load 5.0 s, 71 MiB + 137 MiB GPU. The second visit loads the container from OPFS without downloading. (The p50 is
half the 326 ms of the M3 run because the M4 kernel work landed in `client/vqweb/` in the meantime; both numbers are
under load.)

## Legal CS over the full production index (`legal.html`)

`legal.html` + `legal.js` is the page for the Czech legal corpus that does **not** search the local 55 071-segment
sample: the quantised query encoder (the same Q3_K client, 235 MiB, served as the chunk manifest of `data/legal-cs/meta.json`)
runs in the tab, the 1024-d query vector is POSTed to a **server bridge** that runs a kNN search over the **full
Elasticsearch index** (`nsoud-decision-segments-v1`, 2 686 582 segments, 13 facets), and in parallel the bridge encodes
the same query with the **original fp32 model on the server CPU** and runs the same kNN. The page shows the two result
lists side by side (hits present in both lists are highlighted with their rank in the other list) with cosine(client,
fp32), top-k / top-5 overlap, top-1 agreement and the latencies (browser encode, ES kNN, server fp32 encode + kNN), and a
session table (encode / kNN / fp32 p50, mean cosine and mean top-10 overlap over the session's queries). The header of
`index.html` links to it. The index is unchanged; the query text and vector go only to this server.

Search form: query (placeholder = `meta.query_hint`), facet checkboxes from `/es/status` with counts (default:
`court_argument` only; "vše" = all, which sends no facet filter), k = 10 / 20 / 50, optional "od data" (`since`,
`decision_date >= YYYY-MM-DD`), three example queries from the legal-cs test set.

**Bridge contract** (`demo/es_bridge.py`, mounted by `demo/serve.py --es`; all JSON, errors are non-200 with a text body
which the page shows in its status line):

| endpoint | body | answer |
|---|---|---|
| `GET /es/status` | – | `{ok, index, host, count, facets: [[name, count], ...], fp_model_loaded, fp_model}` |
| `POST /es/knn` | `{vector: [1024], k <= 50, facets: [..] (empty = all), num_candidates?, since?: "YYYY-MM-DD"}` | `{hits, took_ms, es_took_ms, k, num_candidates, filter}` |
| `POST /es/fp` | `{text, k, facets, num_candidates?, since?}` | `{vector: [1024], embed_ms, tokens, hits, took_ms, model}` |

`hit = {segment_id, decision_id, decision_date, facet, case_domain, path, text, score}`. Without `--es` the three
endpoints answer 404: the page says "Elasticsearch bridge not enabled on this server", still loads the model and encodes
the query, but shows no lists.

Start it:

```
.venv/Scripts/python.exe demo/serve.py --port 8766 --es          # add --es-preload to load the fp32 model at start
```

then open http://localhost:8766/legal.html. The Elasticsearch API key is read by the bridge at run time from the fragmea
deployment env file and **stays on the server**: the browser only ever talks to `/es/*` on this origin, nothing is
sent anywhere else. URL parameters as `index.html` (`model=`, `threads=`, `backend=`, `autoload=1`, `autoquery=<text>`)
plus `smoke=1` (headless: autoload + autoquery, then a report `{ok, cosine, overlap10, overlap5, top1_same, hits_left,
hits_right, latencies, ...}` POSTed to `/smoke` when the server has `--smoke-out`):
`legal.html?autoload=1&autoquery=promlčení nároku na náhradu škody&smoke=1&backend=wasm&threads=4`.

The model-loading code shared by both pages (chunk manifest -> OPFS assembly -> `wllama.loadModel`, WebGPU probe, GPU
evidence from the llama.cpp log, timings) lives in `loader.js`; `app.js` and `legal.js` import it.

## Run locally

```
.venv/Scripts/python.exe scripts/demo_export.py      # writes demo/data/scifact/ and demo/model/scifact/ (needs the fp32 index and the GGUF)
.venv/Scripts/python.exe scripts/demo_export.py --dataset scidocs --client models/gguf/harrier-0.6b-gptq-Q2_K-scidocs_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf
.venv/Scripts/python.exe demo/serve.py               # http://localhost:8766/  (COOP/COEP headers, ../models mounted at /models)
```

Open http://localhost:8766/. To load the un-chunked file straight from `models/gguf/` instead of the chunks:
`http://localhost:8766/?model=/models/harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf`.

Headless smoke test (`scripts/demo_smoke.py --ds <corpus> --verify_n <n> [--backend wasm|webgpu|auto]` does all of this and compares the per-query nDCG@10 with the native values; with `--backend webgpu` Chrome gets `--enable-unsafe-webgpu --ignore-gpu-blocklist --use-angle=d3d11 --enable-features=Vulkan,WebGPU` and the summary records whether the adapter line appeared): start `demo/serve.py --smoke-out <file>`, open
`http://localhost:8766/?ds=scifact&smoke=1&threads=8&autoquery=<query>&verify_n=20&backend=wasm` in `chrome --headless=new --enable-features=SharedArrayBuffer`,
and read the report the page POSTs to `/smoke` (also printed to the console as `[smoke]` lines). On 2026-09-07 (machine
under load) this loaded the chunked model in 2.7 s download + 30 s load, returned 10 results, and reproduced the recorded
browser benchmark exactly on the first 20 test queries (top-1 20/20, per-query nDCG@10 identical to 1e-6); a second visit
loaded from OPFS without downloading.

## Deploy

**Cloudflare Pages** (the site root is this directory; every file is below the 25 MiB per-asset limit, the model chunks
are 24 MiB):

```
npx wrangler pages deploy demo --project-name thinletter
```

or connect the repository and set the build output directory to `demo` (no build command). Note that `demo/data/` and
`demo/model/` are git-ignored (root rules `data/`, `demo/model/`, `*.gguf`); a repository-connected build only sees them
if they are committed or produced by a build step, so `wrangler pages deploy` from a machine that ran
`scripts/demo_export.py` is the simpler path. `_headers` sets COOP/COEP for the whole site (multi-threaded wasm),
`Cross-Origin-Resource-Policy: same-origin` and long caching for `vendor/`, `data/` and `model/`. After re-exporting
data or model, purge the Pages cache or bump the file names.

**Access control.** Put the whole site behind Cloudflare Access: Zero Trust -> Access -> Applications -> Add an
application -> Self-hosted; application domain `thinletter.io` (and `www` if used); policy "Allow" with rule
"Emails" listing the reviewers; identity provider "One-time PIN". Because the model chunks are on the same host, the
Access cookie gates their download as well; nothing is fetched from a third-party host.

**Why byte chunks and not `llama-gguf-split`.** The Q2_K token table (`token_embd.weight`, 151 936 × 1024) alone is
48.7 MiB, a GGUF shard cannot split a tensor, and Pages caps every asset at 25 MiB. `scripts/demo_export.py` therefore
cuts the file into raw byte ranges (`model/<file>.chunk000` …) and writes a manifest (`model/<file>.chunks.json` with the
sizes and the SHA-256 of the whole file); the page concatenates the ranges back into the identical file. If a host without
a per-file limit is used, `?model=<url>.gguf` loads the whole file through wllama's own downloader and cache.

## Reproduction

All commands run from the repository root with the project's virtual environment; the GPTQ export needs a CUDA GPU and
`gguf-py` from upstream llama.cpp on `PYTHONPATH`.

1. The GGUF (tag `Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K` is built by `scripts/gptq_export_gguf.py` from
   these flags: `--type Q2_K`, `--calib scifact_corpus_only`, `--per_sample` + `--n_seq 20000` -> `ps20000`,
   `--calib_tokens 300000` -> `t300k`, `--act_order` -> `ao`, `--table_type Q2_K` -> `tabQ2_K`):

   ```
   PYTHONPATH=third_party/llama.cpp/gguf-py python scripts/gptq_export_gguf.py --type Q2_K --calib scifact_corpus_only \
       --per_sample --n_seq 20000 --calib_tokens 300000 --act_order --table_type Q2_K --datasets scifact nfcorpus
   ```

   Output: `models/gguf/harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf` plus a torch-side row
   in `results/raw/gptq_export/results.jsonl`.

2. Native evaluation of the file (llama.cpp `llama-embedding`, pooling last, 8 threads; the 0.7440):

   ```
   python scripts/quant_eval_queries.py --teacher harrier-0.6b --datasets scifact \
       --ggufs models/gguf/harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf --threads 8
   ```

   Per-query nDCG@10 goes to `results/raw/gptq_export/perq/scifact_<file>.npz` (the page's verification compares
   against these values, exported into `data/meta.json`).

3. Browser benchmark (headless Chrome, `client/browser/`, writes `results/raw/browser_local/browser_<tag>_<build>.json`
   and `results/tables/browser_client.md`; run `--tag idle` on an idle machine for latency):

   ```
   .venv/Scripts/python.exe scripts/browser_run.py --tag idle --builds mt st
   ```

4. Data export for this demo (fp16 index, documents, test queries, meta, model chunks):

   ```
   .venv/Scripts/python.exe scripts/demo_export.py
   ```

## Licences

* Base model `microsoft/harrier-oss-v1-0.6b`: **MIT** according to its Hugging Face model card (`license: mit` in the card
  metadata; the API's top-level `license` field is empty). It is a fine-tune of Qwen3-0.6B (Apache-2.0).
* The quantized weights in `model/` are derived from it; their redistribution licence has **not** been decided, hence
  evaluation only and access control.
* Runtime: llama.cpp and wllama, MIT (`vendor/wllama/LICENCE`).
* Data: SciFact (Wadden et al., 2020), **CC BY-NC 2.0**, non-commercial; the abstracts in `data/scifact/docs.*.json` and the
  index built from them are subject to it. SciDocs (Cohan et al., 2020, via BEIR), **CC BY 4.0**: `data/scidocs/`.
* Second teacher `jinaai/jina-embeddings-v5-text-small`: **CC BY-NC 4.0** (non-commercial); base Qwen3-0.6B-Base
  Apache-2.0. The legal-cs client in `model/legal-cs/` is derived from it: evaluation only. The legal-cs corpus and
  index are the owner's data (Czech supreme-court decisions are public documents; the export and the index vectors
  are provided by the owner) and are not part of the repository.
* The demo code in this directory follows the repository's licence.
