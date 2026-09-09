"""Calibration question (product decision): does corpus / synthetic-query calibration add anything over GENERIC text on the
deployment grids?  One table per (model, corpus), rows = calibration text, columns = grid (Q2_K, Q3_K).

Reads the native rows of scripts/quant_eval_queries.py (test split, query side quantised, fp32 index) from
  results/raw/calibq_local/results.jsonl   (scripts/calib_question_queue.sh; wins over older rows of the same file)
  results/raw/legal_local/results.jsonl, results/raw/scidocs_local/results.jsonl   (existing arms, reused)
and the per-query nDCG@10 files next to them (perq/<dataset>_<stem>.npz) for the paired bootstrap
(eq.graph_metrics.paired_bootstrap, 10 000 resamples, seed 0) of every arm against the generic arm.
Only GGUFs with the matched settings (-ps20000-t300k-ao-tabQ2_K) count; a missing arm prints "–", nothing is invented.

Decision rule (pre-registered, research/lab_log.md 2026-09-08): per (model, corpus, grid), synthetic − generic >= +0.01
nDCG@10 with the 95 % CI excluding zero -> "kalibrace se nabízí"; otherwise "generická síť stačí".  The reference
generic is the language-matched one: generic_cs on legal-cs (falls back to generic_wikitext if missing, marked),
generic_wikitext elsewhere.

Usage: .venv/Scripts/python.exe scripts/calib_question_table.py [--table results/tables/calib_question.md]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eq.graph_metrics import paired_bootstrap  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SUF = "-ps20000-t300k-ao-tabQ2_K"
N_BOOT, SEED = 10_000, 0
RULE_MIN = 0.01
GRIDS = ["Q2_K", "Q3_K"]
BLOCKS = [("jina-v5-small", "legal-cs"), ("harrier-0.6b", "scidocs"), ("harrier-0.6b", "arguana"), ("harrier-0.6b", "nfcorpus")]
SOURCES = [ROOT / "results/raw/legal_local", ROOT / "results/raw/scidocs_local", ROOT / "results/raw/calibq_local"]  # later wins
ARMS = ["generic_en", "generic_cs", "corpus", "synth"]
LABEL = {"generic_en": "generický EN (`generic_wikitext`)", "generic_cs": "generický CS (`generic_cs`, cs-wiki)",
         "corpus": "dokumenty korpusu (`<ds>_corpus_only`)", "synth": "syntetické dotazy (`<ds>_synth_only`)"}
NAME_RE = re.compile(r"^(?P<model>.+?)-gptq-(?P<type>Q[23]_K)-(?P<calib>.+?)" + re.escape(SUF) + r"\.gguf$")


def arm_of(calib: str, ds: str):
    if calib == "generic_wikitext":
        return "generic_en"
    if calib == "generic_cs":
        return "generic_cs"
    if calib == f"{ds}_corpus_only":
        return "corpus"
    if calib in (f"{ds}_synth_only", f"{ds}_synth_noleak"):
        return "synth"
    return None


def load_rows():
    rows = {}
    for src in SOURCES:
        p = src / "results.jsonl"
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("both_sides") or r.get("splits", "test") != "test":
                continue
            m = NAME_RE.match(r.get("gguf", ""))
            if not m:
                continue
            arm = arm_of(m["calib"], r["dataset"])
            if arm is None:
                continue
            pq = src / "perq" / f"{r['dataset']}_{Path(r['gguf']).stem}.npz"
            r = dict(r, model=m["model"], grid=m["type"], calib=m["calib"], arm=arm, perq=(np.load(pq)["ndcg"] if pq.exists() else None), src=src.name)
            rows[(m["model"], r["dataset"], m["type"], arm)] = r
    return rows


def f4(x):
    return "–" if x is None else f"{x:.4f}"


def diff(a, b):
    if a is None or b is None or a.get("perq") is None or b.get("perq") is None or len(a["perq"]) != len(b["perq"]):
        return None
    return paired_bootstrap(a["perq"], b["perq"], n=N_BOOT, seed=SEED)


def fmt_d(d):
    return "–" if d is None else f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=ROOT / "results/tables/calib_question.md")
    args = ap.parse_args()
    rows = load_rows()
    L = ["# Přidává kalibrace na korpusu něco proti generickému textu? (rozhodnutí pro produkt)", "",
         "Strana dotazu kvantizovaná (GPTQ act-order, per-sample, rozpočet 300k tokenů, tabulka Q2_K), fp32 index učitele, test split, "
         "nativní `llama-embedding.exe` (Windows), 8 vláken. Δ = párový bootstrap přes dotazy proti generickému EN textu "
         f"(`eq.graph_metrics.paired_bootstrap`, {N_BOOT // 1000} 000 losů, seed {SEED}), 95% CI.",
         f"Pravidlo (pre-registrace 2026-09-08): syntetické − generický (jazykově odpovídající: `generic_cs` na legal-cs, jinak `generic_wikitext`) "
         f"≥ +{RULE_MIN:.2f} nDCG@10 a CI bez nuly → „kalibrace se nabízí“; jinak „generická síť stačí“. Chybějící rameno = „–“.", ""]
    decisions = []
    for model, ds in BLOCKS:
        blk = {(g, a): rows.get((model, ds, g, a)) for g in GRIDS for a in ARMS}
        anyrow = next((r for r in blk.values() if r), None)
        fp = anyrow["fp_ndcg10"] if anyrow else None
        n = anyrow["n_test"] if anyrow else None
        L += [f"## {model} / {ds}" + (f" — fp32 nDCG@10 {fp:.4f}, {n} testovacích dotazů" if fp is not None else " — zatím žádné měření"), ""]
        hdr = ["kalibrační text"] + [f"{g} {c}" for g in GRIDS for c in ("nDCG@10 (% fp)", "Δ vs generický EN [95 % CI]", "cos(q, fp)", "překryv top-10")]
        L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for a in ARMS:
            if a == "generic_cs" and ds != "legal-cs":
                continue
            cells = [LABEL[a].replace("<ds>", ds)]
            for g in GRIDS:
                r = blk[(g, a)]; ref = blk[(g, "generic_en")]
                if r is None:
                    cells += ["–", "–", "–", "–"]; continue
                cells += [f"{r['gt_ndcg10']:.4f} ({r['gt_ndcg10'] / r['fp_ndcg10'] * 100:.1f})",
                          "0 (reference)" if a == "generic_en" else fmt_d(diff(r, ref)), f4(r["q_cos_fp"]), f"{r['fp_top10_overlap']:.3f}"]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
        for g in GRIDS:
            syn = blk[(g, "synth")]
            ref_arm = "generic_cs" if ds == "legal-cs" else "generic_en"
            ref = blk[(g, ref_arm)]; note = ""
            if ref is None and ds == "legal-cs" and blk[(g, "generic_en")] is not None:
                ref = blk[(g, "generic_en")]; ref_arm = "generic_en"; note = " (generic_cs chybí, náhradně generický EN)"
            missing = [x for x, r in (("syntetické", syn), (ref_arm, ref)) if r is None]
            if missing:
                line = f"**{model} / {ds} / {g}:** nelze rozhodnout — chybí {', '.join(missing)}."
            else:
                d = diff(syn, ref)
                if d is None:
                    line = f"**{model} / {ds} / {g}:** nelze rozhodnout — chybí per-query soubory (rozdíl průměrů {syn['gt_ndcg10'] - ref['gt_ndcg10']:+.4f})."
                else:
                    ok = d[0] >= RULE_MIN and d[1] > 0
                    verdict = "kalibrace se nabízí" if ok else "generická síť stačí"
                    line = f"**{model} / {ds} / {g}: {verdict}** — syntetické − {ref_arm}{note}: {fmt_d(d)}"
                    cor = blk[(g, "corpus")]
                    dc = diff(cor, ref) if cor is not None else None
                    if dc is not None:
                        line += f"; dokumenty − {ref_arm}: {fmt_d(dc)}"
                    if ds == "legal-cs" and blk[(g, "generic_cs")] is not None and blk[(g, "generic_en")] is not None:
                        line += f"; generický CS − generický EN: {fmt_d(diff(blk[(g, 'generic_cs')], blk[(g, 'generic_en')]))}"
                    line += "."
            decisions.append(line)
            L.append("- " + line)
        L.append("")
    L += ["Zdroj: `results/raw/calibq_local/` (fronta `scripts/calib_question_queue.sh`), `results/raw/legal_local/`, `results/raw/scidocs_local/` "
          "(results.jsonl, perq/*.npz; vše OUR_MEASUREMENT); tabulka `scripts/calib_question_table.py`."]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"[table] {args.table}")


if __name__ == "__main__":
    main()
