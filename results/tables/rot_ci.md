
== SROVNANÉ (liší se jedna věc, stejný kalibrační rozpočet)
kontrast                                                           a       b   rozdíl                95% CI
harrier Q3_K: rotace obě strany vs bez rotace (korpus)        0.7742  0.7705  +0.0038  [-0.0012; +0.0087]
harrier Q2_K: rotace obě strany vs bez rotace (korpus)        0.7634  0.7515  +0.0119  [+0.0036; +0.0201]  *
harrier Q2_K: obě strany vs jen vstupní (korpus)              0.7634  0.7567  +0.0067  [-0.0021; +0.0155]
harrier Q2_K: obě strany vs jen výstupní (korpus)             0.7634  0.7505  +0.0129  [+0.0040; +0.0214]  *
harrier Q2_K: obě strany vs fúzovaná r1r2 (korpus)            0.7634  0.7490  +0.0144  [+0.0049; +0.0237]  *
harrier Q2_K: cena skládání RMSNorm (fold vs r1r2)            0.6168  0.7490  -0.1322  [-0.1492; -0.1154]  *
harrier Q2_K + rotace: korpus vs generický                    0.7634  0.7360  +0.0274  [+0.0182; +0.0367]  *
harrier Q3_K + rotace: korpus vs generický                    0.7742  0.7681  +0.0062  [+0.0014; +0.0110]  *
harrier Q2_K: rotace obě strany vs bez rotace (GENERICKÁ kalibrace)  0.7360  0.7385  -0.0025  [-0.0110; +0.0058]
harrier Q2_K BEZ rotace: korpus vs generický                  0.7515  0.7385  +0.0130  [+0.0049; +0.0214]  *
PRENOS: kalibrace na CIZIM korpusu (NFCorpus) vs na vlastnim (SciFact), obe 300k + rotace  0.7630  0.7634  -0.0004  [-0.0077; +0.0068]
PRENOS: kalibrace na NFCorpusu vs na Wikipedii, obe 300k + rotace  0.7630  0.7360  +0.0270  [+0.0186; +0.0357]  *
PRENOS: kalibrace na ArguAne (vzdaleny zanr) vs na vlastnim korpusu, 300k + rotace  0.7402  0.7634  -0.0231  [-0.0328; -0.0137]  *
PRENOS: kalibrace na ArguAne vs na Wikipedii, 300k + rotace   0.7402  0.7360  +0.0043  [-0.0040; +0.0124]
rozpocet 3k: rotace vs bez rotace (korpus)                    0.7505  0.7166  +0.0339  [+0.0241; +0.0441]  *
rozpocet 10k: rotace vs bez rotace (korpus)                   0.7542  0.7400  +0.0142  [+0.0050; +0.0237]  *
rozpocet 30k: rotace vs bez rotace (korpus)                   0.7557  0.7496  +0.0061  [-0.0028; +0.0152]
Qwen3-4B Q2_K: rotace obě strany vs bez rotace (korpus)       0.7559  0.7585  -0.0026  [-0.0106; +0.0052]
Qwen3-4B Q2_K bez rotace: korpus vs generický                 0.7585  0.7440  +0.0146  [+0.0057; +0.0230]  *
ArguAna Q2_K: korpus vs genericky (300k, bez rotace)          0.6314  0.6213  +0.0100  [+0.0005; +0.0195]  *
ArguAna Q2_K: synteticke DOTAZY vs DOKUMENTY korpusu (300k, srovnany rozpocet)  0.6403  0.6314  +0.0089  [-0.0008; +0.0188]
ArguAna Q2_K: synteticke dotazy vs genericky text (300k)      0.6403  0.6213  +0.0190  [+0.0098; +0.0280]  *
bge-m3 Q2_K: rotace obě strany vs bez rotace (korpus)         0.6023  0.4966  +0.1057  [+0.0915; +0.1203]  *
bge-m3 Q2_K: doména korpusu vs generický, oba v dotazovém tvaru  0.6120  0.5890  +0.0229  [+0.0143; +0.0320]  *
VQ 2,12 bpw: rotace obě strany vs bez rotace (korpus)         0.7540  0.7442  +0.0098  [+0.0009; +0.0188]  *
VQ 2,12 bpw: rotace obě strany vs bez rotace (generický)      0.7339  0.7037  +0.0302  [+0.0204; +0.0400]  *
VQ 2,12 bpw: zmrazené řádkové amplitudy vs bez nich (korpus)  0.7442  0.7345  +0.0097  [+0.0011; +0.0186]  *
VQ 1,19 bpw (dim 8): rotace obě strany vs bez rotace (korpus)  0.0443  0.0035  +0.0407  [+0.0316; +0.0505]  *
NFCorpus VQ 2,12 bpw: zmrazené škály + rotace vs ani jedno (korpus)  0.3472  0.3411  +0.0061  [+0.0023; +0.0099]  *
NFCorpus VQ 2,12 bpw: zmrazené škály + rotace vs ani jedno (generický)  0.3463  0.3304  +0.0159  [+0.0125; +0.0193]  *

== NESROVNANÉ (rozpočet nevynucen nebo se liší) — orientační, necitovat jako ablaci
kontrast                                                           a       b   rozdíl                95% CI
harrier NFCorpus Q2_K: korpus vs generický (jen strop ps3000, rozpočet nevynucen)  0.3553  0.3488  +0.0065  [+0.0035; +0.0095]  *
harrier SciFact Q2_K: korpus vs generický (jen strop ps3000, rozpočet nevynucen)  0.7540  0.7418  +0.0121  [+0.0035; +0.0206]  *
6 dokumentu korpusu (3k tok.) vs 300k tokenu Wikipedie, oboji s rotaci  0.7505  0.7360  +0.0145  [+0.0045; +0.0247]  *
objem kalibrace: 3k vs 300k tokenu korpusu, oboji s rotaci    0.7505  0.7634  -0.0128  [-0.0212; -0.0045]  *
VQ 2,12 bpw + rotace: korpus vs generický                     0.7540  0.7339  +0.0201  [+0.0111; +0.0292]  *

== INTERAKCE (2x2, rozdíl rozdílů)
harrier Q2_K, 300k: interakce rotace x kalibrace              +0.0144  [+0.0035; +0.0252]  *

* = 95% interval nepřekrývá nulu; n = 10 000 párových bootstrapů přes dotazy testovaného datasetu.
