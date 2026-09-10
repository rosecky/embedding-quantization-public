# Vydání 2: Qwen3-Embedding-0.6B Q5_K_M a bge-m3 (česká imatrix) — nativní eval, vlastní fp32 index, test split

Generuje `scripts/release_tables.py --r2`; pre-registrace lab log 2026-09-10 (R1, R2, R2b). Kritérium vydání: ≥ 95 % vlastního fp32, cos ≥ 0,94, překryv ≥ 0,75; „vyšší kvalita“ jen při ≥ +0,5 bodu proti vydanému Q4_K_M.

## bge-m3

| soubor | MiB | legal-cs nDCG@10 (% fp) / cos / překryv | scifact nDCG@10 (% fp) / cos / překryv |
|---|---|---|---|
| `bge-m3-f16.gguf` | 1104.0 | 0.3171 (100.1) / 1.000 / 0.988 | 0.6413 (100.0) / 1.000 / 0.999 |
| `bge-m3-imx-Q3_K-generic_cs-tabq4_0.gguf` | 320.7 | 0.3114 (98.3) / 0.970 / 0.780 | 0.6416 (100.0) / 0.974 / 0.854 |
| `bge-m3-imx-Q3_K-generic_cs-tabq8_0.gguf` | 442.8 | 0.3119 (98.4) / 0.970 / 0.781 | 0.6402 (99.8) / 0.975 / 0.854 |
| `bge-m3-imx-Q4_K_M-generic_cs-tabq4_0.gguf` | 354.6 | 0.3144 (99.2) / 0.990 / 0.864 | 0.6388 (99.6) / 0.991 / 0.910 |
| `bge-m3-imx-Q4_K_M-generic_cs-tabq8_0.gguf` | 476.6 | 0.3136 (99.0) / 0.990 / 0.869 | 0.6407 (99.9) / 0.992 / 0.904 |
| `bge-m3-imx-Q5_K_M-generic_cs-tabq8_0.gguf` | 505.1 | 0.3156 (99.6) / 0.995 / 0.907 | 0.6443 (100.5) / 0.995 / 0.937 |

fp32 bge-m3: legal-cs 0.3169, scifact 0.6414.

## me5-small

| soubor | MiB | legal-cs nDCG@10 (% fp) / cos / překryv | scifact nDCG@10 (% fp) / cos / překryv |
|---|---|---|---|
| `me5-small-f16.gguf` | 231.1 | 0.3186 (100.6) / 0.999 / 0.929 | 0.6754 (99.7) / 0.999 / 0.949 |

fp32 me5-small: legal-cs 0.3168, scifact 0.6777.

## qwen3-0.6b

| soubor | MiB | legal-cs nDCG@10 (% fp) / cos / překryv | scifact nDCG@10 (% fp) / cos / překryv | nfcorpus nDCG@10 (% fp) / cos / překryv | arguana nDCG@10 (% fp) / cos / překryv | scidocs nDCG@10 (% fp) / cos / překryv |
|---|---|---|---|---|---|---|
| `qwen3-0.6b-imx-Q5_K_M-generic_cs-tabq4_0.gguf` | 385.4 | 0.3171 (99.5) / 0.983 / 0.839 | – | – | – | – |
| `qwen3-0.6b-imx-Q5_K_M-generic_cs-tabq8_0.gguf` | 459.5 | 0.3163 (99.3) / 0.987 / 0.864 | – | – | – | – |
| `qwen3-0.6b-imx-Q5_K_M-generic_wikitext-tabq4_0.gguf` | 385.4 | – | 0.6989 (99.8) / 0.992 / 0.928 | 0.3539 (100.1) / 0.988 / 0.912 | 0.7040 (100.1) / 0.991 / 0.959 | 0.2171 (100.1) / 0.992 / 0.936 |
| `qwen3-0.6b-imx-Q5_K_M-generic_wikitext-tabq8_0.gguf` | 459.5 | – | 0.6992 (99.8) / 0.993 / 0.932 | 0.3534 (99.9) / 0.990 / 0.928 | 0.7047 (100.2) / 0.992 / 0.959 | 0.2156 (99.4) / 0.993 / 0.936 |

fp32 qwen3-0.6b: legal-cs 0.3186, scifact 0.7004, nfcorpus 0.3537, arguana 0.7034, scidocs 0.2168.

Zdroje: `results/raw/release2_local/` (OUR_MEASUREMENT, llama-quantize --imatrix; legal-cs = 1 137 syntetických dotazů, saturovaný test: fp32 všech bází 0,317–0,319, vypovídá jen poměr / cos / překryv).
