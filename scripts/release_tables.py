"""Release-decision tables (pre-registered 2026-09-08): writes results/tables/release_cs.md from
  results/raw/imatrix_cs_local/results.jsonl + perq/     llama-quantize --imatrix baseline on jina-v5-small / legal-cs
  results/raw/calibq_local/results.jsonl + perq/         our GPTQ rows (generic_cs, generic_wikitext, corpus, synth) = references
  results/raw/release_local/results.jsonl + perq/        E1 (qwen3-0.6b on legal-cs) and E2 (jina generic_cs_short)
Paired bootstrap over the 1 137 legal-cs test queries (10 000 draws, seed 0). Missing inputs leave "–", nothing is invented.

Usage: .venv/Scripts/python.exe scripts/release_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from eq.graph_metrics import paired_bootstrap  # noqa: E402

DS = "legal-cs"
SRC = {"imx": ROOT / "results/raw/imatrix_cs_local", "calibq": ROOT / "results/raw/calibq_local", "rel": ROOT / "results/raw/release_local"}
OUT = ROOT / "results/tables/release_cs.md"


def rows(d: Path) -> dict[str, dict]:
    out = {}
    p = d / "results.jsonl"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("dataset") == DS:
                out[r["gguf"]] = r  # last row wins
    return out


def perq(d: Path, gguf: str):
    p = d / "perq" / f"{DS}_{Path(gguf).stem}.npz"
    return np.load(p)["ndcg"] if p.exists() else None


def ci(a, b):
    if a is None or b is None:
        return "–"
    d, lo, hi = paired_bootstrap(a, b, n=10000, seed=0)
    flag = "" if lo <= 0 <= hi else " *"
    return f"{d:+.4f} [{lo:+.4f}; {hi:+.4f}]{flag}"


def cell(r, key, fmt="{:.4f}"):
    return fmt.format(r[key]) if r and key in r else "–"


def pct(r, fp):
    return f"{r['gt_ndcg10']:.4f} ({r['gt_ndcg10'] / fp * 100:.1f})" if r else "–"


def main():
    imx, cq, rel = rows(SRC["imx"]), rows(SRC["calibq"]), rows(SRC["rel"])
    L = ["# Rozhodovací tabulky prvního zveřejnění: česká větev (legal-cs, 1 137 syntetických dotazů, index nezměněný)", "",
         "Generuje `scripts/release_tables.py`; pre-registrace `research/lab_log.md` 2026-09-08 (imatrix-CS, E1, E2). Párový bootstrap přes dotazy, 10 000 losů, seed 0; `*` = 95% CI mimo nulu. Rozptyl opakované kvantizace (~±0,01) v CI není.", ""]

    # --- imatrix-CS ---
    ours_q3 = f"jina-v5-small-gptq-Q3_K-generic_cs-ps20000-t300k-ao-tabQ2_K.gguf"
    ours_q2 = f"jina-v5-small-gptq-Q2_K-generic_cs-ps20000-t300k-ao-tabQ2_K.gguf"
    ours_q2s = f"jina-v5-small-gptq-Q2_K-legal-cs_synth_only-ps20000-t300k-ao-tabQ2_K.gguf"
    fp = next((r["fp_ndcg10"] for r in list(cq.values()) + list(imx.values()) if "fp_ndcg10" in r), float("nan"))
    L += [f"## imatrix-CS: `llama-quantize --imatrix` (česká imatrix) proti našemu GPTQ exportu, jina-v5-small (fp32 nDCG@10 {fp:.4f})", "",
          "| soubor | MiB | kalibrace | nDCG@10 (% fp) | cos k fp | překryv top-10 | Δ vs náš Q3_K generic_cs | Δ vs náš Q2_K generic_cs | Δ vs náš Q2_K synth |",
          "|---|---|---|---|---|---|---|---|---|"]
    refs = {k: (cq.get(k), perq(SRC["calibq"], k)) for k in (ours_q3, ours_q2, ours_q2s)}
    order = [(ours_q3, "**náš GPTQ Q3_K**", "generic_cs", cq), (ours_q2, "**náš GPTQ Q2_K**", "generic_cs", cq), (ours_q2s, "náš GPTQ Q2_K", "synth", cq)]
    tags = [("Q3_K-generic_cs", "Q3_K směs (A)", "generic_cs imatrix"), ("Q3_Kpure-generic_cs", "Q3_K `--pure` (B, náš formát)", "generic_cs imatrix"),
            ("IQ2_M-generic_cs", "IQ2_M (C)", "generic_cs imatrix"), ("IQ3_XXS-generic_cs", "IQ3_XXS (D)", "generic_cs imatrix"),
            ("Q3_Kpure-noimatrix", "Q3_K `--pure` bez imatrix (E, RTN)", "–"), ("IQ2_M-legal-cs_synth_only", "IQ2_M (F)", "synth imatrix"),
            ("Q3_Kpure-legal-cs_synth_only", "Q3_K `--pure` (G)", "synth imatrix")]
    for tag, label, calib in tags:
        order.append((f"jina-v5-small-imx-{tag}-tabQ2_K.gguf", label, calib, imx))
    for g, label, calib, src in order:
        r = src.get(g); a = perq(SRC["imx"] if src is imx else SRC["calibq"], g)
        L.append(f"| {label} | {cell(r, 'gguf_mib', '{:.1f}')} | {calib} | {pct(r, fp)} | {cell(r, 'q_cos_fp', '{:.3f}')} | {cell(r, 'fp_top10_overlap', '{:.3f}')} | "
                 + " | ".join(ci(a, refs[k][1]) if g != k else "–" for k in (ours_q3, ours_q2, ours_q2s)) + " |")
    L.append("")

    # --- E1 ---
    q_f16 = "qwen3-embedding-0.6b-f16.gguf"; q_q3 = "qwen3-0.6b-gptq-Q3_K-generic_cs-ps20000-t300k-ao-tabQ2_K.gguf"
    e1 = {k: rel.get(k) for k in (q_f16, q_q3)}
    fpq = next((r["fp_ndcg10"] for r in e1.values() if r), float("nan"))
    L += ["## E1: Qwen3-Embedding-0.6B (Apache-2.0) jako učitel na legal-cs s vlastním fp32 indexem", "",
          f"fp32 nDCG@10 Qwen3-Embedding-0.6B: **{fpq:.4f}** (jina-v5-small na týchž dotazech, jiný index: {fp:.4f}; poměr {fpq / fp * 100:.1f} %)" if fpq == fpq else "fp32 Qwen3-Embedding-0.6B: – (neběželo)", "",
          "| soubor | MiB | nDCG@10 (% vlastního fp) | cos k fp | překryv top-10 | Δ vs vlastní fp32 |", "|---|---|---|---|---|---|"]
    fp_perq = None
    p = SRC["rel"] / "perq" / f"{DS}_{Path(q_f16).stem}.npz"
    if p.exists():
        fp_perq = np.load(p)["ndcg_fp"]
    others = sorted(g for g in rel if g.startswith("qwen3-0.6b-") and g not in (q_f16, q_q3))
    labels = {g: g.replace("qwen3-0.6b-", "").replace("-ps20000-t300k-ao", "").replace(".gguf", "") for g in others}
    for g, label in [(q_f16, "f16 soubor (reference runtime)"), (q_q3, "náš GPTQ Q3_K generic_cs, tabulka Q2_K")] + [(g, f"`{labels[g]}` (E4/E5, llama-quantize)" if "-imx-" in g else f"`{labels[g]}` (E4, GPTQ)") for g in others]:
        r = rel.get(g); a = perq(SRC["rel"], g)
        L.append(f"| {label} | {cell(r, 'gguf_mib', '{:.1f}')} | {pct(r, fpq) if r else '–'} | {cell(r, 'q_cos_fp', '{:.3f}')} | {cell(r, 'fp_top10_overlap', '{:.3f}')} | {ci(a, fp_perq)} |")
    L.append("")

    # --- E2 ---
    L += ["## E2: kontrola tvaru sekvencí — jina, `generic_cs_short` (62 slov/odst., 1 608 × ~186 tok.) proti `generic_cs` (779 × ~385 tok.)", "",
          "| mřížka | short nDCG@10 (% fp) | dlouhý nDCG@10 (% fp) | generic EN (% fp) | short − dlouhý | short − EN | cos short / dlouhý |", "|---|---|---|---|---|---|---|"]
    for grid in ("Q2_K", "Q3_K"):
        s = f"jina-v5-small-gptq-{grid}-generic_cs_short-ps20000-t300k-ao-tabQ2_K.gguf"
        l = f"jina-v5-small-gptq-{grid}-generic_cs-ps20000-t300k-ao-tabQ2_K.gguf"
        e = f"jina-v5-small-gptq-{grid}-generic_wikitext-ps20000-t300k-ao-tabQ2_K.gguf"
        rs, rl, re_ = rel.get(s), cq.get(l), cq.get(e)
        L.append(f"| {grid} | {pct(rs, fp)} | {pct(rl, fp)} | {pct(re_, fp)} | {ci(perq(SRC['rel'], s), perq(SRC['calibq'], l))} | {ci(perq(SRC['rel'], s), perq(SRC['calibq'], e))} | {cell(rs, 'q_cos_fp', '{:.3f}')} / {cell(rl, 'q_cos_fp', '{:.3f}')} |")
    L.append("")
    L.append("Zdroje: `results/raw/imatrix_cs_local/`, `results/raw/release_local/`, `results/raw/calibq_local/` (vše OUR_MEASUREMENT; jina f16 = oficiální GGUF).")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()


def main_en():
    """E3: Qwen3-Embedding-0.6B general client on the English corpora -> results/tables/release_en.md (rows from results/raw/release_local)."""
    rel = {}
    p = SRC["rel"] / "results.jsonl"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("teacher") == "qwen3-0.6b" and r.get("dataset") in ("scifact", "nfcorpus", "arguana", "scidocs"):
                rel[(r["dataset"], r["gguf"])] = r
    files = sorted({g for (_, g) in rel})
    L = ["# E3: Qwen3-Embedding-0.6B (Apache-2.0) obecný dotazový klient na anglických korpusech (vlastní fp32 index, test split)", "",
         "Generuje `scripts/release_tables.py` (`main_en`); pre-registrace lab log 2026-09-08 večer (E3). Podmínka vydání Q1: ≥ 95 % vlastního fp32 s CI rozdílu nad −0,02, cos ≥ 0,94, překryv ≥ 0,75 na ≥ 3 ze 4 korpusů.", "",
         "| soubor | MiB | " + " | ".join(f"{d} nDCG@10 (% fp) / cos / překryv" for d in ("scifact", "nfcorpus", "arguana", "scidocs")) + " |",
         "|---|---|" + "---|" * 4]
    for g in files:
        cells = []
        mib = "–"
        for d in ("scifact", "nfcorpus", "arguana", "scidocs"):
            r = rel.get((d, g))
            if r:
                mib = f"{r['gguf_mib']:.1f}"
                fpp = r["fp_ndcg10"]
                cells.append(f"{r['gt_ndcg10']:.4f} ({r['gt_ndcg10'] / fpp * 100:.1f}) / {r['q_cos_fp']:.3f} / {r['fp_top10_overlap']:.3f}")
            else:
                cells.append("–")
        L.append(f"| `{g}` | {mib} | " + " | ".join(cells) + " |")
    fps = {d: next((r["fp_ndcg10"] for (dd, _), r in rel.items() if dd == d), float("nan")) for d in ("scifact", "nfcorpus", "arguana", "scidocs")}
    L += ["", "fp32 Qwen3-Embedding-0.6B: " + ", ".join(f"{d} {v:.4f}" for d, v in fps.items()) + " (harrier-0.6b: scifact 0.7559, nfcorpus 0.3808, arguana 0.6665, scidocs 0.2269 — jiný model, jiný index, jen orientačně).", ""]
    out = ROOT / "results/tables/release_en.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__" and "--en" in sys.argv:
    main_en()
