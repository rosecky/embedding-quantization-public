# E3: Qwen3-Embedding-0.6B (Apache-2.0) obecný dotazový klient na anglických korpusech (vlastní fp32 index, test split)

Generuje `scripts/release_tables.py` (`main_en`); pre-registrace lab log 2026-09-08 večer (E3). Podmínka vydání Q1: ≥ 95 % vlastního fp32 s CI rozdílu nad −0,02, cos ≥ 0,94, překryv ≥ 0,75 na ≥ 3 ze 4 korpusů.

| soubor | MiB | scifact nDCG@10 (% fp) / cos / překryv | nfcorpus nDCG@10 (% fp) / cos / překryv | arguana nDCG@10 (% fp) / cos / překryv | scidocs nDCG@10 (% fp) / cos / překryv |
|---|---|---|---|---|---|
| `qwen3-0.6b-gptq-Q3_K-generic_wikitext-ps20000-t300k-ao-tabQ2_K.gguf` | 235.0 | 0.6764 (96.6) / 0.933 / 0.793 | 0.3418 (96.6) / 0.905 / 0.741 | 0.7032 (100.0) / 0.945 / 0.894 | 0.2063 (95.1) / 0.936 / 0.779 |
| `qwen3-0.6b-imx-IQ3_XXS-generic_wikitext-tabQ2_K.gguf` | 212.7 | 0.6652 (95.0) / 0.857 / 0.717 | 0.3170 (89.6) / 0.810 / 0.638 | 0.6603 (93.9) / 0.832 / 0.835 | 0.1984 (91.5) / 0.868 / 0.670 |
| `qwen3-0.6b-imx-Q3_Kpure-generic_wikitext-tabQ2_K.gguf` | 235.0 | 0.6733 (96.1) / 0.874 / 0.735 | 0.3337 (94.4) / 0.841 / 0.675 | 0.6895 (98.0) / 0.869 / 0.847 | 0.2001 (92.3) / 0.889 / 0.694 |
| `qwen3-0.6b-imx-Q4_K_M-generic_wikitext-tabQ2_K.gguf` | 305.2 | 0.6916 (98.7) / 0.968 / 0.856 | 0.3473 (98.2) / 0.952 / 0.812 | 0.7091 (100.8) / 0.973 / 0.929 | 0.2173 (100.2) / 0.972 / 0.860 |
| `qwen3-0.6b-imx-Q4_K_M-generic_wikitext-tabq4_0.gguf` | 339.9 | 0.7004 (100.0) / 0.981 / 0.898 | 0.3514 (99.4) / 0.973 / 0.869 | 0.7044 (100.1) / 0.980 / 0.937 | 0.2154 (99.3) / 0.983 / 0.898 |
| `qwen3-embedding-0.6b-f16.gguf` | 1142.1 | 0.7001 (100.0) / 1.000 / 0.989 | 0.3542 (100.1) / 1.000 / 0.987 | 0.7046 (100.2) / 1.000 / 0.993 | 0.2162 (99.7) / 1.000 / 0.990 |

fp32 Qwen3-Embedding-0.6B: scifact 0.7004, nfcorpus 0.3537, arguana 0.7034, scidocs 0.2168 (harrier-0.6b: scifact 0.7559, nfcorpus 0.3808, arguana 0.6665, scidocs 0.2269 — jiný model, jiný index, jen orientačně).

