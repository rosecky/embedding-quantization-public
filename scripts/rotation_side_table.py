"""Rotation side x rate x seed on the vector grid (pre-registered 2026-09-07, run locally on the RTX 4050).

Reads results/raw/gptvq_local/results.jsonl (+ the two seed-0 two-sided points that already existed in
results/raw/gptvq/results.jsonl) and writes results/tables/rotation_side.md: per rate the no-rotation control,
the three input-only seeds, the three two-sided seeds, the generic-text control, and the share of the two-sided
effect carried by the input side.  All numbers are nDCG@10 on all real SciFact queries (fp 0.7723).

Usage: python scripts/rotation_side_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
FP = 0.7723


def rows():
    out = []
    for f in ("results/raw/gptvq_local/results.jsonl", "results/raw/gptvq/results.jsonl"):
        p = ROOT / f
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("dataset") != "scifact" or r.get("dim") != 4 or r.get("hi_bits") or r.get("vocab", "none") != "none":
                continue
            if not r.get("frozen_scales") or r.get("rows_per_cb") != 0 or r.get("rotate", "none") != "none":
                continue
            out.append(r)
    return out


def pick(rs, bpw, calib, side, seed=None):
    hits = [r for r in rs if abs(r["eff_bpw"] - bpw) < 1e-6 and r["calib"] == calib
            and ((not r.get("rotate_matrix")) if side is None else (r.get("rotate_matrix") and r.get("rotate_side", "both") == side))
            and (seed is None or r.get("rot_seed", 0) == seed) and r.get("rot_rounds", 2) == 2]
    return [h["gt_ndcg10"] for h in hits]


def fmt(v):
    return "–" if not v else " / ".join(f"{x:.4f}" for x in v)


def main():
    rs = rows()
    L = ["# Strana rotace × sazba × seed na vektorové mřížce (SciFact, všechny reálné dotazy, fp 0,7723)", "",
         "Generuje `scripts/rotation_side_table.py`. Dim 4, jedna kniha na matici, zmrazené amplitudy, per-sample, "
         "kalibrace `scifact_corpus_only` (2 000 dok.) nebo `generic_wikitext` (3 000 odstavců, strop, ne srovnaný "
         "rozpočet). Seedy 0 / 7 / 13. Pre-registrace a predikce: `lab_log.md` 2026-09-07.", "",
         "| bpw | bez rotace, korpus | jen vstupní (seedy 0/7/13) | obě strany (seedy 0/7/13) | bez rotace, generický | obě strany, generický (seed 0) |",
         "|---|---|---|---|---|---|"]
    summary = []
    for bpw in (2.125, 1.84375, 1.578125):
        nr = pick(rs, bpw, "scifact_corpus_only", None)
        inn = [pick(rs, bpw, "scifact_corpus_only", "in", s) for s in (0, 7, 13)]
        both = [pick(rs, bpw, "scifact_corpus_only", "both", s) for s in (0, 7, 13)]
        gnr = pick(rs, bpw, "generic_wikitext", None)
        gboth = pick(rs, bpw, "generic_wikitext", "both", 0)
        inn_v = [x[0] for x in inn if x]
        both_v = [x[0] for x in both if x]
        L.append(f"| {bpw:.2f} | {fmt(nr[:1])} | {fmt(inn_v)} | {fmt(both_v)} | {fmt(gnr[:1])} | {fmt(gboth[:1])} |")
        if nr and inn_v and both_v:
            m_in, m_both = np.mean(inn_v), np.mean(both_v)
            share = (m_in - nr[0]) / (m_both - nr[0]) if m_both != nr[0] else float("nan")
            summary.append(f"- **{bpw:.2f} bpw:** efekt obou stran {m_both - nr[0]:+.4f} (průměr tří seedů; rozpětí seedu "
                           f"{max(both_v) - min(both_v):.4f}), jen vstupní {m_in - nr[0]:+.4f} (rozpětí {max(inn_v) - min(inn_v):.4f}), "
                           f"**vstupní strana nese {share * 100:.0f} %** efektu; výstupní strana přidává {m_both - m_in:+.4f}.")
    L += [""] + summary + ["",
          "**Čtení.** Na 1,84 bpw je rotace vylepšení, ne podmínka (bez rotace 0,700, tedy 90,7 % fp); na 1,58 bpw je "
          "podmínkou funkčnosti pro obě kalibrace (generický text bez rotace je mrtvý). Vstupní strana nese většinu efektu "
          "na obou sazbách; výstupní strana je na dně uvnitř rozptylu seedu. Rozptyl seedu je na 1,58 bpw velký "
          "(řádově 0,06) u obou variant, na 1,84 bpw malý (řádově 0,005). Pro nasazení to znamená: jednostranná "
          "(vstupní) online rotace stačí, cena je zhruba třetina oboustranné (transformace se sdílí mezi q/k/v a gate/up).",
          ""]
    out = ROOT / "results/tables/rotation_side.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
