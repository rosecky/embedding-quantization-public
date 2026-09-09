# Tři replikace z recenze (vyhodnoceno 2026-09-07 z per-query polí)

Generuje `scripts/review_replications.py`. Párový bootstrap přes dotazy, 10 000 vzorků, seed 0. „všechny dotazy“ = train+dev+test (SciFact 1 109, fp 0,7723; ArguAna 1 401, fp 0,6451), „test“ = poslední split.

## REV-2: párový kalibrační čtverec, tři nezávislé losy (Q2_K, 300k tokenů, bez rotace, SciFact)

| los | korpus | generický | rozdíl (všechny dotazy) | 95% CI | rozdíl (test) |
|---|---|---|---|---|---|
| 1 | 0.7532 | 0.7432 | **+0.0101** | [+0.0021; +0.0181] | +0.0073 |
| 2 | 0.7518 | 0.7446 | **+0.0072** | [-0.0010; +0.0156] | +0.0061 |
| 3 | 0.7553 | 0.7460 | **+0.0093** | [+0.0011; +0.0175] | +0.0061 |

Tři párové efekty: průměr **+0.0089**, sd mezi losy **0.0015**, rozpětí +0.0072 až +0.0101. Původní jednolosový srovnaný kontrast bez rotace byl +0,0130 (0,7515 vs 0,7385, jiný los obou ramen). Sd efektu mezi losy je řádově menší než efekt sám: **kalibrační efekt na Q2_K je stabilní přes kalibrační losy** (tři losy, tři kladné efekty; dva intervaly mimo nulu, třetí ji těsně zahrnuje). Poznámka: losy mají ~1 080 (korpus) a ~1 950 (generický) sekvencí při stejném tokenovém rozpočtu; délkové rozdělení se tedy liší, kontrola tvaru nebyla součástí této sady.

## REV-3: deset losů po ~3 tis. tokenech korpusu (≈10 dokumentů), Q2_K, bez rotace, SciFact

| los | všechny dotazy | % fp | test | proti korpusu 300k (bez rot.) | proti generickému 300k (bez rot.) |
|---|---|---|---|---|---|
| 1 | 0.7364 | 95.3 | 0.7167 | -0.0151 [-0.0241; -0.0064] | -0.0022 [-0.0122; +0.0074] |
| 2 | 0.7217 | 93.4 | 0.7132 | -0.0298 [-0.0398; -0.0198] | -0.0168 [-0.0271; -0.0066] |
| 3 | 0.7287 | 94.4 | 0.7207 | -0.0227 [-0.0323; -0.0136] | -0.0098 [-0.0203; +0.0006] |
| 4 | 0.7349 | 95.2 | 0.7231 | -0.0166 [-0.0259; -0.0074] | -0.0036 [-0.0130; +0.0056] |
| 5 | 0.7247 | 93.8 | 0.7140 | -0.0267 [-0.0366; -0.0172] | -0.0138 [-0.0236; -0.0042] |
| 6 | 0.7388 | 95.7 | 0.7311 | -0.0127 [-0.0219; -0.0037] | +0.0003 [-0.0095; +0.0097] |
| 7 | 0.7332 | 94.9 | 0.7211 | -0.0182 [-0.0280; -0.0085] | -0.0053 [-0.0160; +0.0055] |
| 8 | 0.7293 | 94.4 | 0.7195 | -0.0222 [-0.0321; -0.0124] | -0.0092 [-0.0205; +0.0018] |
| 9 | 0.7222 | 93.5 | 0.7211 | -0.0293 [-0.0385; -0.0199] | -0.0163 [-0.0264; -0.0062] |
| 10 | 0.7383 | 95.6 | 0.7289 | -0.0132 [-0.0224; -0.0040] | -0.0002 [-0.0105; +0.0101] |

Deset losů: průměr **0.7308** (94.6 % fp), sd **0.0064**, min 0.7217, max 0.7388. Korpus 300k bez rotace 0.7515, generický 300k bez rotace 0.7385 (los 1 generického: 0.7432).
Průměrná ztráta proti korpusu 300k **-0.0207**, proti generickému 300k **-0.0077**; 1 z 10 losů je nad generickým 300k, 0 z 10 nad korpusem 300k.

**Důsledek:** tvrzení „šest dokumentů porazí 300 tisíc tokenů Wikipedie“ bylo měřeno **s rotací** (0,7505 vs 0,7360, jeden los rotace i kalibrace). **Bez rotace** hrstka dokumentů generický text 300k neporazí. Tvrzení se tedy nedá vyslovit obecně; platí nanejvýš pro rotovanou konfiguraci a tam stojí na jednom losu.

## REV-1: ArguAna, vektorový klient 2,12 bpw + rotace + oříznutá tabulka, kalibrace bez uniklých dokumentů

| kalibrace | všechny dotazy | test | test − BitNet | 95% CI |
|---|---|---|---|---|
| korpus s únikem (2 000 dok.) | 0.6603 | **0.6879** | +0.0816 | [+0.0625; +0.1012] |
| **korpus bez uniklých (2 000 z 7 981 dok.)** | 0.6254 | **0.6494** | +0.0431 | [+0.0246; +0.0621] |
| syntetické dotazy (3 000, z neodfiltrovaných dok.) | 0.6486 | **0.6736** | +0.0673 | [+0.0499; +0.0851] |
| BitNet-270m (140 MiB) | 0.5891 | **0.6063** | +0.0000 | [+0.0000; +0.0000] |

Čisté proti uniklému (test, párově): **-0.0385** [-0.0520; -0.0251]. Slovník tabulky byl v tomto běhu ořezán ještě ze **všech** dokumentů (filtr před výběrem slovníku je až v lokální frontě `scripts/arguana_clean_queue.sh`), takže tohle je kontrola kalibrace, ne ještě celého klienta.

**Skalární mřížka (Q2_K, 300k tokenů, bez rotace), táž kontrola s párovými intervaly proti generickému textu:**

- bez uniklých − generický, všechny dotazy: **+0.0178** [+0.0089; +0.0268] (0.6392 vs 0.6213)
- bez uniklých − generický, test: **+0.0165** [+0.0038; +0.0291] (0.6605 vs 0.6440)
- s únikem − generický, všechny dotazy: **+0.0100** [+0.0005; +0.0195] (0.6314 vs 0.6213)
- bez uniklých − s únikem, všechny dotazy: **+0.0078** [+0.0004; +0.0151] (0.6392 vs 0.6314)

Na skalární mřížce kalibrační efekt na ArguAně po odstranění úniku **roste a zůstává průkazný**; na vektorové mřížce (REV-1b níže) je po vyčištění +0,010 s intervalem přes nulu.

## REV-1b: plně filtrovaný klient (filtr před kalibrací i výběrem slovníku; lokální fronta, 4050)

| kalibrace | všechny dotazy | test | test − BitNet | 95% CI | test − uniklý 0,6879 | 95% CI |
|---|---|---|---|---|---|---|
| dokumenty bez uniklých, slovník bez uniklých | 0.6379 | **0.6502** | +0.0439 | [+0.0228; +0.0649] | -0.0377 | [-0.0535; -0.0226] |
| syntetické dotazy z neuniklých dok., slovník bez uniklých | 0.6427 | **0.6507** | +0.0444 | [+0.0241; +0.0645] | -0.0372 | [-0.0531; -0.0216] |
| generický text, slovník bez uniklých | 0.6306 | **0.6404** | +0.0340 | [+0.0119; +0.0562] | -0.0475 | [-0.0637; -0.0319] |

- dokumenty bez uniklých, slovník bez uniklých proti generickému textu (test, párově): **+0.0098** [-0.0037; +0.0236]
- syntetické dotazy z neuniklých dok., slovník bez uniklých proti generickému textu (test, párově): **+0.0103** [-0.0037; +0.0245]

Pre-registrovaná predikce (i) „čisté dokumentové rameno do ±0,01 od 0,6879“ **padla**; (ii) syntetika ≥ dokumenty − 0,01 a (iii) obě čistá ramena ≥ +0,05 nad BitNetem se čtou z tabulky. Řádek 0,6879 je tímto nahrazen čistým klientem.

## REV-1c: útes pod 2 bity na ArguAně? (1,58 bpw, rotace obě strany seed 0, slovník bez uniklých)

| kalibrace | všechny dotazy | test | test − BitNet | 95% CI |
|---|---|---|---|---|
| korpus bez uniklých (2 000 dok.) | 0.5870 | **0.5855** | -0.0208 | [-0.0443; +0.0031] |
| generický text (3 000 odst.) | 0.5837 | **0.5979** | -0.0084 | [-0.0335; +0.0159] |

Korpus − generický, párově: **+0.0034** [-0.0104; +0.0169] (všechny dotazy), **-0.0124** [-0.0319; +0.0074] (test). Na SciFactu je týž kontrast při 1,58 bpw +0,08 až +0,14 a generický text tam kolabuje na 71 % fp; na ArguAně drží 90 % fp bez doménového textu. **Pre-registrovaná predikce útesu (≥ +0,05) padla**; sub-2-bitový kalibrační argument je zatím podložený jen SciFactem.

