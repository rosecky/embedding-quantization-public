# legal-cs: zákaznický index (jina-embeddings-v5-text-small) a klient Q2_K strany dotazu

Korpus: 55071 segmentů odůvodnění Nejvyššího soudu (court_argument, civilní agenda, od 2020-01-01), **index = vektory zákazníka** (jina-v5-small, 1024 d, `data/emb/legal-cs/jina-v5-small/corpus.npy`, nepřekódováno). Dotazy: syntetické (doc2query, Qwen3-1.7B, česky) k 600 náhodným segmentům z korpusu (1137 dotazů, relevantní = zdrojový segment, stupeň 1; **ne lidská hodnocení**). Kalibrace: 2 000 segmentů z jiných agend/let mimo demo korpus (`legal-cs_corpus_only`, prefix `Document: `) a syntetické dotazy k 1 500 z nich (`legal-cs_synth_only`, prefix `Query: `). GPTQ act-order na mřížce Q2_K (per-sample, rozpočet 300k tokenů), tabulka Q2_K. Runtime: upstream `llama-embedding.exe` (Windows build-clang), pooling last, ctx 512, 8 vláken (`scripts/quant_eval_queries.py`).

fp32 reference (dotazy jina-v5-small bf16 na GPU proti zákaznickému indexu, test): nDCG@10 = **0.3190** (1137 dotazů).

| soubor | MiB | kalibrace (skutečný rozpočet) | nDCG@10 test (% fp) | Δ vs fp32 [95 % CI] | R@100 | cos(q, fp) | top-10 překryv s fp | torch-side nDCG@10 | bpw bloků | GPTQ čas |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 soubor (oficiální GGUF, reference runtime) | 1142.7 | – | 0.3191 (100.0) | +0.0001 [-0.0009, +0.0011] | 0.6060 | 0.9999 | 0.986 | – | – | – |
| klient Q2_K, kalibrace korpus (`legal-cs_corpus_only`, 2 000 segmentů mimo demo korpus) | 192.4 | 523 oken × 512 = 267 776 tokenů | 0.2638 (82.7) | -0.0552 [-0.0682, -0.0426] | 0.5488 | 0.8413 | 0.531 | 0.2674 | 2.625 | 408 s |
| klient Q2_K, kalibrace syntetické dotazy (`legal-cs_synth_only`) | 192.4 | 3000 sekvencí = 89 736 tokenů | 0.2752 (86.2) | -0.0439 [-0.0546, -0.0334] | 0.5550 | 0.9046 | 0.615 | 0.2795 | 2.625 | 308 s |
| klient Q3_K (3,44 bpw), kalibrace syntetické dotazy (`legal-cs_synth_only`), tabulka Q2_K | 235.1 | – | 0.3071 (96.3) | -0.0120 [-0.0186, -0.0054] | 0.5875 | 0.9694 | 0.789 | 0.3115 | 3.438 | 277 s |

Rozdíl kalibrací (syntetické − korpusové, párový bootstrap přes 1137 dotazů, 2 000 losů): **+0.0113** [+0.0005, +0.0210].

## Prohlížeč (demo, wllama, headless Chrome, `scripts/demo_smoke.py --dataset legal-cs`)

- 50 testovacích dotazů, nDCG@10 v prohlížeči **0.2539** vs nativně na týchž dotazech 0.2572 (Δ -0.0033; 43/50 per-query shodných do 1e-6, 7 se liší o > 0,001, max |Δ| 0.3155); vyhledávání vrátilo 10 výsledků; 8 vláken (mt=True), stažení 2.0 s, načtení 4.5 s, p50 474 ms, paměť po načtení 1019 MiB.

## Soubory dema (limit 25 MiB na soubor)

- Index: 55071 × 1024, uložen jako **int8_rowscale** v 3 souborech (53.8 MiB); kontrola na testovacích dotazech: nDCG@10 fp32 0.319019 / f16 0.319147 / int8 0.318852 (int8 Δ -0.000167, tolerance 0.001).
- Dokumenty: 12 JSON shardů; klient `jina-v5-small-gptq-Q3_K-legal-cs_synth_only-ps20000-t300k-ao-tabQ2_K.gguf` 235.1 MiB v 10 bajtových chuncích (sha256 d24428e08bd3…); prompt `Query: `; model jinaai/jina-embeddings-v5-text-small (retrieval), query side, GPTQ on the llama.cpp Q2_K grid; licence: Evaluation only. Model weights are not licensed for redistribution; jinaai/jina-embeddings-v5-text-small is CC BY-NC 4.0 (non-commercial), base Qwen3-0.6B-Base Apache-2.0, runtime llama.cpp/wllama MIT, corpus Czech supreme-court decisions (public documents); index provided by the owner.
- Velikosti: `corpus.i8.000` 24.00 MiB, `corpus.i8.001` 24.00 MiB, `corpus.i8.002` 5.78 MiB, `corpus.scale.f32` 0.21 MiB, `docs.000.json` 1.61 MiB, `docs.001.json` 1.61 MiB, `docs.002.json` 1.61 MiB, `docs.003.json` 1.60 MiB, `docs.004.json` 1.61 MiB, `docs.005.json` 1.61 MiB, `docs.006.json` 1.60 MiB, `docs.007.json` 1.61 MiB, `docs.008.json` 1.61 MiB, `docs.009.json` 1.60 MiB, `docs.010.json` 1.61 MiB, `docs.011.json` 0.02 MiB, `meta.json` 0.02 MiB, `test_queries.json` 0.20 MiB; chunky modelu 21 × ≤ 24 MiB.
- Korpus (notice na stránce): Czech supreme-court decisions (public documents); index provided by the owner

## Pre-registrované predikce (research/lab_log.md, 2026-09-07 pozdě večer)

- (i) klient s korpusovou kalibrací ≥ 97 % fp32 nDCG@10: 82.7 % → **neplatí**
- (ii) cos(q, fp) ≥ 0.85 a top-10 překryv ≥ 0.75: 0.8413 / 0.531 → **neplatí**
- (iii) |korpus − syntetické| < 0.01: 0.0113 (synth − korpus +0.0113 [+0.0005, +0.0210]) → **neplatí**
- (iv) velikost klienta ≈ 192 MiB (189–195): 192.4 MiB → **platí**
- kontrola prohlížeče (±0.002): Δ -0.0033 na 50 dotazech (43/50 per-query shodných do 1e-6) → **MIMO toleranci**
- Falzifikační práh (< 95 % fp): NASTAL (82.7 %).

Zdroj: `results/raw/legal_local/` (results.jsonl, gptq_export/exports.jsonl, smoke_legal-cs.json, perq/*.npz; vše OUR_MEASUREMENT), fronta `scripts/legal_queue.sh` (log `results/raw/legal_local/queue.log`), tabulka `scripts/legal_table.py`. Zákaznická data nejsou v repu (data/legal-cs/, demo/data/legal-cs/ necommitovat).
