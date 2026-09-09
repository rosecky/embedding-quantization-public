# Rozhodovací tabulky prvního zveřejnění: česká větev (legal-cs, 1 137 syntetických dotazů, index nezměněný)

Generuje `scripts/release_tables.py`; pre-registrace `research/lab_log.md` 2026-09-08 (imatrix-CS, E1, E2). Párový bootstrap přes dotazy, 10 000 losů, seed 0; `*` = 95% CI mimo nulu. Rozptyl opakované kvantizace (~±0,01) v CI není.

## imatrix-CS: `llama-quantize --imatrix` (česká imatrix) proti našemu GPTQ exportu, jina-v5-small (fp32 nDCG@10 0.3190)

| soubor | MiB | kalibrace | nDCG@10 (% fp) | cos k fp | překryv top-10 | Δ vs náš Q3_K generic_cs | Δ vs náš Q2_K generic_cs | Δ vs náš Q2_K synth |
|---|---|---|---|---|---|---|---|---|
| **náš GPTQ Q3_K** | 235.1 | generic_cs | 0.3092 (96.9) | 0.942 | 0.728 | – | +0.0555 [+0.0440; +0.0675] * | +0.0341 [+0.0246; +0.0436] * |
| **náš GPTQ Q2_K** | 192.4 | generic_cs | 0.2537 (79.5) | 0.786 | 0.455 | -0.0555 [-0.0675; -0.0440] * | – | -0.0215 [-0.0319; -0.0114] * |
| náš GPTQ Q2_K | 192.4 | synth | 0.2752 (86.2) | 0.905 | 0.615 | -0.0341 [-0.0436; -0.0246] * | +0.0215 [+0.0114; +0.0319] * | – |
| Q3_K směs (A) | 258.0 | generic_cs imatrix | 0.3092 (96.9) | 0.942 | 0.708 | -0.0000 [-0.0071; +0.0070] | +0.0555 [+0.0444; +0.0673] * | +0.0340 [+0.0244; +0.0438] * |
| Q3_K `--pure` (B, náš formát) | 235.1 | generic_cs imatrix | 0.2990 (93.7) | 0.897 | 0.655 | -0.0102 [-0.0184; -0.0023] * | +0.0453 [+0.0342; +0.0572] * | +0.0238 [+0.0139; +0.0340] * |
| IQ2_M (C) | 199.3 | generic_cs imatrix | 0.2718 (85.2) | 0.830 | 0.518 | -0.0374 [-0.0485; -0.0264] * | +0.0182 [+0.0075; +0.0290] * | -0.0033 [-0.0141; +0.0075] |
| IQ3_XXS (D) | 212.8 | generic_cs imatrix | 0.2807 (88.0) | 0.862 | 0.564 | -0.0285 [-0.0387; -0.0181] * | +0.0271 [+0.0167; +0.0379] * | +0.0056 [-0.0045; +0.0156] |
| Q3_K `--pure` bez imatrix (E, RTN) | 235.1 | – | 0.2857 (89.6) | 0.865 | 0.585 | -0.0235 [-0.0329; -0.0141] * | +0.0321 [+0.0210; +0.0436] * | +0.0106 [+0.0001; +0.0209] * |
| IQ2_M (F) | 199.3 | synth imatrix | 0.2738 (85.8) | 0.832 | 0.516 | -0.0354 [-0.0466; -0.0238] * | +0.0202 [+0.0089; +0.0318] * | -0.0013 [-0.0125; +0.0100] |
| Q3_K `--pure` (G) | 235.1 | synth imatrix | 0.3007 (94.3) | 0.898 | 0.652 | -0.0085 [-0.0169; -0.0004] * | +0.0471 [+0.0362; +0.0584] * | +0.0256 [+0.0156; +0.0357] * |

## E1: Qwen3-Embedding-0.6B (Apache-2.0) jako učitel na legal-cs s vlastním fp32 indexem

fp32 nDCG@10 Qwen3-Embedding-0.6B: **0.3186** (jina-v5-small na týchž dotazech, jiný index: 0.3190; poměr 99.9 %)

| soubor | MiB | nDCG@10 (% vlastního fp) | cos k fp | překryv top-10 | Δ vs vlastní fp32 |
|---|---|---|---|---|---|
| f16 soubor (reference runtime) | 1142.1 | 0.3196 (100.3) | 1.000 | 0.978 | +0.0010 [-0.0014; +0.0035] |
| náš GPTQ Q3_K generic_cs, tabulka Q2_K | 235.0 | 0.2876 (90.3) | 0.880 | 0.546 | -0.0310 [-0.0417; -0.0202] * |
| `gptq-Q3_K-generic_cs-tabQ8_0` (E4, GPTQ) | 343.8 | 0.2997 (94.1) | 0.914 | 0.630 | -0.0189 [-0.0282; -0.0094] * |
| `gptq-Q3_K-legal-cs_synth_only-tabQ2_K` (E4, GPTQ) | 235.0 | 0.2927 (91.9) | 0.918 | 0.625 | -0.0259 [-0.0361; -0.0160] * |
| `gptq-Q3_K-legal-cs_synth_only-tabQ8_0` (E4, GPTQ) | 343.8 | 0.3022 (94.8) | 0.948 | 0.715 | -0.0164 [-0.0244; -0.0085] * |
| `imx-IQ3_XXS-generic_cs-tabQ2_K` (E4/E5, llama-quantize) | 212.7 | 0.2218 (69.6) | 0.728 | 0.316 | -0.0968 [-0.1128; -0.0814] * |
| `imx-Q3_K-generic_cs-tabq2_k` (E4/E5, llama-quantize) | 257.9 | 0.2726 (85.6) | 0.867 | 0.512 | -0.0460 [-0.0584; -0.0335] * |
| `imx-Q3_K-generic_cs-tabq8_0` (E4/E5, llama-quantize) | 366.7 | 0.2963 (93.0) | 0.913 | 0.633 | -0.0223 [-0.0325; -0.0121] * |
| `imx-Q3_Kpure-generic_cs-tabQ2_K` (E4/E5, llama-quantize) | 235.0 | 0.2550 (80.0) | 0.775 | 0.391 | -0.0637 [-0.0785; -0.0490] * |
| `imx-Q4_K_M-generic_cs-tabQ2_K` (E4/E5, llama-quantize) | 305.2 | 0.3009 (94.4) | 0.928 | 0.654 | -0.0178 [-0.0265; -0.0090] * |
| `imx-Q4_K_M-generic_cs-tabq4_0` (E4/E5, llama-quantize) | 339.9 | 0.3145 (98.7) | 0.966 | 0.775 | -0.0041 [-0.0108; +0.0023] |
| `imx-Q4_K_M-generic_cs-tabq8_0` (E4/E5, llama-quantize) | 414.0 | 0.3172 (99.5) | 0.969 | 0.788 | -0.0015 [-0.0076; +0.0048] |

## E2: kontrola tvaru sekvencí — jina, `generic_cs_short` (62 slov/odst., 1 608 × ~186 tok.) proti `generic_cs` (779 × ~385 tok.)

| mřížka | short nDCG@10 (% fp) | dlouhý nDCG@10 (% fp) | generic EN (% fp) | short − dlouhý | short − EN | cos short / dlouhý |
|---|---|---|---|---|---|---|
| Q2_K | 0.2542 (79.7) | 0.2537 (79.5) | 0.1624 (50.9) | +0.0005 [-0.0084; +0.0095] | +0.0917 [+0.0766; +0.1068] * | 0.796 / 0.786 |
| Q3_K | 0.3081 (96.6) | 0.3092 (96.9) | 0.2998 (94.0) | -0.0011 [-0.0063; +0.0042] | +0.0083 [+0.0009; +0.0161] * | 0.945 / 0.942 |

Zdroje: `results/raw/imatrix_cs_local/`, `results/raw/release_local/`, `results/raw/calibq_local/` (vše OUR_MEASUREMENT; jina f16 = oficiální GGUF).
