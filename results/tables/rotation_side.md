# Strana rotace × sazba × seed na vektorové mřížce (SciFact, všechny reálné dotazy, fp 0,7723)

Generuje `scripts/rotation_side_table.py`. Dim 4, jedna kniha na matici, zmrazené amplitudy, per-sample, kalibrace `scifact_corpus_only` (2 000 dok.) nebo `generic_wikitext` (3 000 odstavců, strop, ne srovnaný rozpočet). Seedy 0 / 7 / 13. Pre-registrace a predikce: `lab_log.md` 2026-09-07.

| bpw | bez rotace, korpus | jen vstupní (seedy 0/7/13) | obě strany (seedy 0/7/13) | bez rotace, generický | obě strany, generický (seed 0) |
|---|---|---|---|---|---|
| 2.12 | 0.7442 | 0.7521 | 0.7540 / 0.7605 / 0.7538 | 0.7037 | 0.7339 |
| 1.84 | 0.7004 | 0.7257 / 0.7291 / 0.7309 | 0.7383 / 0.7328 / 0.7339 | 0.5667 | 0.6918 |
| 1.58 | 0.3082 | 0.6760 / 0.6069 / 0.6637 | 0.6877 / 0.6294 / 0.6761 | 0.0840 | 0.5503 |

- **2.12 bpw:** efekt obou stran +0.0119 (průměr tří seedů; rozpětí seedu 0.0067), jen vstupní +0.0079 (rozpětí 0.0000), **vstupní strana nese 66 %** efektu; výstupní strana přidává +0.0040.
- **1.84 bpw:** efekt obou stran +0.0346 (průměr tří seedů; rozpětí seedu 0.0056), jen vstupní +0.0282 (rozpětí 0.0052), **vstupní strana nese 81 %** efektu; výstupní strana přidává +0.0065.
- **1.58 bpw:** efekt obou stran +0.3562 (průměr tří seedů; rozpětí seedu 0.0583), jen vstupní +0.3407 (rozpětí 0.0691), **vstupní strana nese 96 %** efektu; výstupní strana přidává +0.0155.

**Čtení.** Na 1,84 bpw je rotace vylepšení, ne podmínka (bez rotace 0,700, tedy 90,7 % fp); na 1,58 bpw je podmínkou funkčnosti pro obě kalibrace (generický text bez rotace je mrtvý). Vstupní strana nese většinu efektu na obou sazbách; výstupní strana je na dně uvnitř rozptylu seedu. Rozptyl seedu je na 1,58 bpw velký (řádově 0,06) u obou variant, na 1,84 bpw malý (řádově 0,005). Pro nasazení to znamená: jednostranná (vstupní) online rotace stačí, cena je zhruba třetina oboustranné (transformace se sdílí mezi q/k/v a gate/up).
