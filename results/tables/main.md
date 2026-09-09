# Hlavní tabulka: každý klient s plnou proveniencí

Generuje `scripts/main_table.py`. `měřeno jako`: **file** = skutečný GGUF přes llama-embedding, 
**grid** = dekvantizované váhy v torchi (runtime to reprodukuje do ±0,002), **sim** = pro tenhle formát 
runtime neexistuje. Všechna čísla jsou nDCG@10 na testovacím splitu proti fp32 indexu téhož modelu.

| klient | MiB | dataset | nDCG@10 | % fp | vs BitNet | split | kalibrace | měřeno jako | pozn. |
|---|---|---|---|---|---|---|---|---|---|
| VQ dim4 1.58 bpw + rotace + oříznutá tabulka | 94 | arguana | 0.6370 | 95.6 | +0.0310 | test | arguana_synth_only (3000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 1.84 bpw + rotace + oříznutá tabulka | 108 | arguana | 0.6649 | 99.8 | +0.0589 | test | arguana_synth_only (3000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 1.84 bpw + rotace + oříznutá tabulka | 108 | arguana | 0.6610 | 99.2 | +0.0550 | test | arguana_corpus_only (2000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 2.12 bpw + rotace + oříznutá tabulka | 122 | arguana | 0.6736 | 101.1 | +0.0676 | test | arguana_synth_only (3000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 2.12 bpw + rotace + oříznutá tabulka | 122 | arguana | 0.6879 | 103.2 | +0.0819 | test | arguana_corpus_only (2000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| BitNet-270m (ternární QAT, obě strany) | 140 | arguana | 0.6060 | 90.9 | +0.0000 | test | — | **file** | externí reference |
| harrier-0.6b fp16 (serverový model) | 1143 | arguana | 0.6665 | 100.0 | +0.0605 | test | — | **file** | reference |
| VQ dim4 1.84 bpw + rotace + oříznutá tabulka | 104 | nfcorpus | 0.3489 | 91.6 | -0.0121 | test | nfcorpus_synth_only (3000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 1.84 bpw + rotace + oříznutá tabulka | 104 | nfcorpus | 0.3269 | 85.9 | -0.0341 | test | nfcorpus_corpus_only (2000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 2.12 bpw + rotace + oříznutá tabulka | 119 | nfcorpus | 0.3444 | 90.4 | -0.0166 | test | nfcorpus_corpus_only (2000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 2.12 bpw + rotace + oříznutá tabulka | 119 | nfcorpus | 0.3553 | 93.3 | -0.0057 | test | nfcorpus_synth_only (3000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| BitNet-270m (ternární QAT, obě strany) | 140 | nfcorpus | 0.3610 | 94.8 | +0.0000 | test | — | **file** | externí reference |
| GPTQ Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ | 192 | nfcorpus | 0.3598 | 94.5 | -0.0012 | test | viz název | **file** | skutečný soubor |
| GPTQ Q2_K-scifact_synth_only-ps3000 | 246 | nfcorpus | 0.3645 | 95.7 | +0.0035 | test | viz název | **file** | skutečný soubor |
| GPTQ Q2_K-scifact_synth_only-ps3000-ao | 246 | nfcorpus | 0.3646 | 95.7 | +0.0036 | test | viz název | **file** | skutečný soubor |
| GPTQ Q3_K-scifact_synth_only-ps3000 | 288 | nfcorpus | 0.3801 | 99.8 | +0.0191 | test | viz název | **file** | skutečný soubor |
| harrier-0.6b fp16 (serverový model) | 1143 | nfcorpus | 0.3808 | 100.0 | +0.0198 | test | — | **file** | reference |
| VQ dim4 1.58 bpw + rotace + oříznutá tabulka | 91 | scifact | 0.7074 | 93.6 | -0.0256 | test | scifact_synth_big (3000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 1.84 bpw + rotace + oříznutá tabulka | 105 | scifact | 0.7164 | 94.8 | -0.0166 | test | scifact_corpus_only (2000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 1.84 bpw + rotace + oříznutá tabulka | 105 | scifact | 0.7164 | 94.8 | -0.0166 | test | scifact_corpus_only (2000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 1.84 bpw + rotace + oříznutá tabulka | 105 | scifact | 0.7294 | 96.5 | -0.0036 | test | scifact_synth_big (3000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 2.12 bpw + rotace + oříznutá tabulka | 120 | scifact | 0.7315 | 96.8 | -0.0015 | test | scifact_corpus_only (2000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 2.12 bpw + rotace + oříznutá tabulka | 120 | scifact | 0.7315 | 96.8 | -0.0015 | test | scifact_corpus_only (2000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| VQ dim4 2.12 bpw + rotace + oříznutá tabulka | 120 | scifact | 0.7422 | 98.2 | +0.0092 | test | scifact_synth_big (3000 sekv.) | **sim** | bloky i tabulka v jednom běhu; bez runtimu |
| BitNet-270m (ternární QAT, obě strany) | 140 | scifact | 0.7330 | 97.0 | +0.0000 | test | — | **file** | externí reference |
| GPTQ Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ | 192 | scifact | 0.7440 | 98.4 | +0.0110 | test | viz název | **file** | skutečný soubor |
| GPTQ Q2_K-scifact_synth_only-ps3000 | 246 | scifact | 0.7351 | 97.3 | +0.0021 | test | viz název | **file** | skutečný soubor |
| GPTQ Q2_K-scifact_synth_only-ps3000-ao | 246 | scifact | 0.7446 | 98.5 | +0.0116 | test | viz název | **file** | skutečný soubor |
| GPTQ Q3_K-scifact_synth_only-ps3000 | 288 | scifact | 0.7538 | 99.7 | +0.0208 | test | viz název | **file** | skutečný soubor |
| harrier-0.6b fp16 (serverový model) | 1143 | scifact | 0.7559 | 100.0 | +0.0229 | test | — | **file** | reference |

## Co tabulka záměrně neobsahuje

- **Latenci.** `llama-bench` měří propustnost, ne latenci klienta; jediné poctivé číslo je 151 ms na dotaz
  pro soubor 192 MiB (300 reálných dotazů, kontext 512, 8 vláken) — viz `wins.md` §2.1b.
- **Paměť.** Velikost souboru není paměťová stopa: 192 MiB vah = 0,93 GB RSS při kontextu 512
  a 4,08 GB při výchozím okně modelu.
- **Řádky měřené na jiné množině dotazů.** Kde není testovací split, řádek tu není.
