# SciDocs: druhý referenční klient (Q2_K GPTQ, 192 MiB) a druhý korpus dema

Protokol jako u SciFact klienta: model harrier-0.6b, GPTQ act-order na mřížce Q2_K (per-sample, rozpočet 300k tokenů), tabulka tokenů Q2_K; **kvantizovaná strana dotazu, fp32 index dokumentů téhož modelu** (`data/emb/scidocs/harrier-0.6b`, generická E5 instrukce), SciDocs test split (500 dotazů z 1 000; BEIR SciDocs je test-only, registr dělí 40/10/50 se seedem 0 jako u ArguAny), nDCG@10, pooling last, L2. Runtime kvality: upstream llama.cpp `llama-embedding.exe` (Windows build-clang), 8 vláken, ctx 1024 (`scripts/quant_eval_queries.py --bin third_party/llama.cpp/build-clang/bin/llama-embedding.exe`); rozdíl proti bitnet.cpp/WSL je ≤ 0,004 v pořadí remíz (viz 2026-09-07).

fp32 reference (harrier-0.6b, bf16 na GPU, test): nDCG@10 = **0.2269**; všech 1 000 dotazů: 0.2329; BitNet-270m (fp výstup, poslední token, E5): 0.1946.

| soubor | MiB | kalibrace (skutečný rozpočet) | nDCG@10 test (% fp) | Δ vs fp32 [95 % CI] | nDCG@10 všech 1 000 (% fp) | cos(q, fp) | top-10 překryv s fp | torch-side nDCG@10 SciDocs / SciFact | GPTQ čas |
|---|---|---|---|---|---|---|---|---|---|
| klient Q2_K, kalibrace SciDocs korpus (`scidocs_corpus_only`) | 192.4 | 585 oken × 512 = 299 520 tokenů | 0.2115 (93.2) | -0.0154 [-0.0223, -0.0084] | 0.2148 (92.2) | 0.8456 | 0.709 | 0.2150 / 0.7296 | 434 s |
| klient Q2_K, kalibrace syntetické dotazy (`scidocs_synth_only`) | 192.4 | 2997 dotazů = 99 523 tokenů | 0.2145 (94.5) | -0.0124 [-0.0184, -0.0063] | 0.2187 (93.9) | 0.9349 | 0.779 | 0.2141 / 0.7393 | 307 s |
| kontrola: SciFact klient (`scifact_corpus_only`, mimo doménu) | 192.4 | – | 0.2123 (93.6) | -0.0146 [-0.0228, -0.0064] | 0.2132 (91.5) | 0.8281 | 0.677 | – / – | – |

Rozdíl kalibrací (syntetické − korpusové, párový bootstrap přes 500 dotazů, 2 000 losů): **+0.0030** [-0.0036, +0.0096].

## Prohlížeč (demo, wllama 3.6.1, headless Chrome, `scripts/demo_smoke.py`)

- SciDocs: 50 testovacích dotazů, nDCG@10 v prohlížeči **0.1838** vs nativně na týchž dotazech 0.1820 (Δ +0.0018; 43/50 per-query hodnot shodných do 1e-6, 7 se liší o > 0,001, max |Δ| 0.1021); vyhledávání vrátilo 10 výsledků; 8 vláken (mt=True), stažení 1.8 s, načtení 2.3 s, p50 1174 ms (stroj po noční frontě, latence orientační), paměť po načtení 1267 MiB.

## Soubory dema (Cloudflare Pages, limit 25 MiB na soubor)

- Index: 25656 × 1024, uložen jako **int8_rowscale** v 2 souborech (25.1 MiB); kontrola na testovacích dotazech (fp32 dotazy z cache): nDCG@10 fp32 0.226927 / f16 0.227024 / int8 0.226634 (int8 Δ -0.000294, tolerance 0.001).
- Dokumenty: 6 JSON shardů; klient `harrier-0.6b-gptq-Q2_K-scidocs_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf` 192.4 MiB v 9 bajtových chuncích (sha256 efc11a2b5465…).
- Velikosti: `corpus.i8.000` 24.00 MiB, `corpus.i8.001` 1.05 MiB, `corpus.scale.f32` 0.10 MiB, `docs.000.json` 6.06 MiB, `docs.001.json` 6.06 MiB, `docs.002.json` 6.09 MiB, `docs.003.json` 6.15 MiB, `docs.004.json` 6.11 MiB, `docs.005.json` 0.77 MiB, `meta.json` 0.01 MiB, `test_queries.json` 0.17 MiB; chunky modelu 10 × ≤ 24 MiB.
- Licence korpusu: SciDocs: CC BY 4.0 (SciFact je CC BY-NC, proto druhý korpus).

## Pre-registrované predikce (research/lab_log.md, 2026-09-07 večer)

- P1a korpusová kalibrace ≥ 97 % fp: 93.2 % → **neplatí**
- P1b syntetické dotazy ≥ 97.5 % fp: 94.5 % → **neplatí**
- P2 |rozdíl kalibrací| < 0.01: 0.0030 (synth − korpus +0.0030 [-0.0036, +0.0096]) → **platí**
- P3 prohlížeč vs nativní do ±0.002: Δ +0.0018 na 50 dotazech (43/50 per-query shodných do 1e-6) → **platí**
- P4 klient 192 MiB, index int8 s dopadem < 0.001: soubor 192.4 MiB; int8 Δ nDCG -0.0003 → uloženo jako int8_rowscale → **platí**
- Falzifikační práh (< 95 % fp): NASTAL (93.2 %).

Zdroj: `results/raw/scidocs_local/` (results.jsonl, gptq_export/exports.jsonl, student.jsonl, smoke_*.json, perq/*.npz; vše OUR_MEASUREMENT), fronta `scripts/scidocs_queue.sh` (log `results/raw/scidocs_local/queue.log`), tabulka `scripts/scidocs_table.py`.
