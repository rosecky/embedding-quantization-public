# Přidává kalibrace na korpusu něco proti generickému textu? (rozhodnutí pro produkt)

Strana dotazu kvantizovaná (GPTQ act-order, per-sample, rozpočet 300k tokenů, tabulka Q2_K), fp32 index učitele, test split, nativní `llama-embedding.exe` (Windows), 8 vláken. Δ = párový bootstrap přes dotazy proti generickému EN textu (`eq.graph_metrics.paired_bootstrap`, 10 000 losů, seed 0), 95% CI.
Pravidlo (pre-registrace 2026-09-08): syntetické − generický (jazykově odpovídající: `generic_cs` na legal-cs, jinak `generic_wikitext`) ≥ +0.01 nDCG@10 a CI bez nuly → „kalibrace se nabízí“; jinak „generická síť stačí“. Chybějící rameno = „–“.

## jina-v5-small / legal-cs — fp32 nDCG@10 0.3190, 1137 testovacích dotazů

| kalibrační text | Q2_K nDCG@10 (% fp) | Q2_K Δ vs generický EN [95 % CI] | Q2_K cos(q, fp) | Q2_K překryv top-10 | Q3_K nDCG@10 (% fp) | Q3_K Δ vs generický EN [95 % CI] | Q3_K cos(q, fp) | Q3_K překryv top-10 |
|---|---|---|---|---|---|---|---|---|
| generický EN (`generic_wikitext`) | 0.1624 (50.9) | 0 (reference) | 0.6400 | 0.225 | 0.2998 (94.0) | 0 (reference) | 0.9129 | 0.652 |
| generický CS (`generic_cs`, cs-wiki) | 0.2537 (79.5) | +0.0912 [+0.0762, +0.1064] | 0.7859 | 0.455 | 0.3092 (96.9) | +0.0094 [+0.0018, +0.0174] | 0.9416 | 0.728 |
| dokumenty korpusu (`legal-cs_corpus_only`) | 0.2638 (82.7) | +0.1014 [+0.0841, +0.1182] | 0.8413 | 0.531 | 0.3069 (96.2) | +0.0071 [-0.0008, +0.0151] | 0.9552 | 0.752 |
| syntetické dotazy (`legal-cs_synth_only`) | 0.2752 (86.2) | +0.1127 [+0.0964, +0.1293] | 0.9046 | 0.615 | 0.3071 (96.3) | +0.0073 [-0.0007, +0.0153] | 0.9694 | 0.789 |

- **jina-v5-small / legal-cs / Q2_K: kalibrace se nabízí** — syntetické − generic_cs: +0.0215 [+0.0114, +0.0319]; dokumenty − generic_cs: +0.0101 [-0.0005, +0.0209]; generický CS − generický EN: +0.0912 [+0.0762, +0.1064].
- **jina-v5-small / legal-cs / Q3_K: generická síť stačí** — syntetické − generic_cs: -0.0021 [-0.0087, +0.0044]; dokumenty − generic_cs: -0.0023 [-0.0084, +0.0036]; generický CS − generický EN: +0.0094 [+0.0018, +0.0174].

## harrier-0.6b / scidocs — fp32 nDCG@10 0.2269, 500 testovacích dotazů

| kalibrační text | Q2_K nDCG@10 (% fp) | Q2_K Δ vs generický EN [95 % CI] | Q2_K cos(q, fp) | Q2_K překryv top-10 | Q3_K nDCG@10 (% fp) | Q3_K Δ vs generický EN [95 % CI] | Q3_K cos(q, fp) | Q3_K překryv top-10 |
|---|---|---|---|---|---|---|---|---|
| generický EN (`generic_wikitext`) | 0.2046 (90.1) | 0 (reference) | 0.8561 | 0.645 | – | – | – | – |
| dokumenty korpusu (`scidocs_corpus_only`) | 0.2115 (93.2) | +0.0069 [-0.0008, +0.0147] | 0.8456 | 0.709 | – | – | – | – |
| syntetické dotazy (`scidocs_synth_only`) | 0.2145 (94.5) | +0.0100 [+0.0024, +0.0177] | 0.9349 | 0.779 | – | – | – | – |

- **harrier-0.6b / scidocs / Q2_K: generická síť stačí** — syntetické − generic_en: +0.0100 [+0.0024, +0.0177]; dokumenty − generic_en: +0.0069 [-0.0008, +0.0147].
- **harrier-0.6b / scidocs / Q3_K:** nelze rozhodnout — chybí syntetické, generic_en.

## harrier-0.6b / arguana — fp32 nDCG@10 0.6665, 700 testovacích dotazů

| kalibrační text | Q2_K nDCG@10 (% fp) | Q2_K Δ vs generický EN [95 % CI] | Q2_K cos(q, fp) | Q2_K překryv top-10 | Q3_K nDCG@10 (% fp) | Q3_K Δ vs generický EN [95 % CI] | Q3_K cos(q, fp) | Q3_K překryv top-10 |
|---|---|---|---|---|---|---|---|---|
| generický EN (`generic_wikitext`) | – | – | – | – | 0.6750 (101.3) | 0 (reference) | 0.9659 | 0.915 |
| dokumenty korpusu (`arguana_corpus_only`) | – | – | – | – | – | – | – | – |
| syntetické dotazy (`arguana_synth_only`) | – | – | – | – | 0.6745 (101.2) | -0.0006 [-0.0082, +0.0070] | 0.9773 | 0.935 |

- **harrier-0.6b / arguana / Q2_K:** nelze rozhodnout — chybí syntetické, generic_en.
- **harrier-0.6b / arguana / Q3_K: generická síť stačí** — syntetické − generic_en: -0.0006 [-0.0082, +0.0070].

## harrier-0.6b / nfcorpus — fp32 nDCG@10 0.3808, 323 testovacích dotazů

| kalibrační text | Q2_K nDCG@10 (% fp) | Q2_K Δ vs generický EN [95 % CI] | Q2_K cos(q, fp) | Q2_K překryv top-10 | Q3_K nDCG@10 (% fp) | Q3_K Δ vs generický EN [95 % CI] | Q3_K cos(q, fp) | Q3_K překryv top-10 |
|---|---|---|---|---|---|---|---|---|
| generický EN (`generic_wikitext`) | – | – | – | – | 0.3782 (99.3) | 0 (reference) | 0.9592 | 0.846 |
| dokumenty korpusu (`nfcorpus_corpus_only`) | – | – | – | – | – | – | – | – |
| syntetické dotazy (`nfcorpus_synth_only`) | – | – | – | – | 0.3780 (99.3) | -0.0002 [-0.0060, +0.0053] | 0.9773 | 0.862 |

- **harrier-0.6b / nfcorpus / Q2_K:** nelze rozhodnout — chybí syntetické, generic_en.
- **harrier-0.6b / nfcorpus / Q3_K: generická síť stačí** — syntetické − generic_en: -0.0002 [-0.0060, +0.0053].

Zdroj: `results/raw/calibq_local/` (fronta `scripts/calib_question_queue.sh`), `results/raw/legal_local/`, `results/raw/scidocs_local/` (results.jsonl, perq/*.npz; vše OUR_MEASUREMENT); tabulka `scripts/calib_question_table.py`.
