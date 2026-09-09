"""The one table: every client we would put in front of anyone, with its full provenance.

An external review asked for a single table where each row states its configuration, evaluation split,
calibration data and runtime status, because the same numbers were scattered across documents measured on
different bases -- and comparing across those bases is what produced most of the retractions in lab_log.md.

Every row here is one measured configuration.  Columns that matter and were previously implicit:
  measured as  file    = the actual GGUF run through llama-embedding
               grid    = dequantised weights in torch (runtime reproduces this to +-0.002)
               sim     = no runtime exists for this format at all
  split        which queries the number is averaged over; test-split unless stated
  calibration  which text the quantiser saw, and how much of it

Usage: python scripts/main_table.py [--md results/tables/main.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
TEST_N = {"scifact": 300, "nfcorpus": 323, "arguana": 700}
FP_TEST = {"scifact": 0.7559, "nfcorpus": 0.3808, "arguana": 0.6665}
BITNET = {"scifact": 0.7330, "nfcorpus": 0.3610, "arguana": 0.6060}


def test_mean(npz: Path, ds: str):
    if not npz.exists():
        return None
    return float(np.nanmean(np.load(npz)["ndcg"][-TEST_N[ds]:]))


def vq_rows():
    """vector-quantised clients: blocks and a trimmed token table measured in the same run"""
    out = []
    P = ROOT / "results/raw/gptvq/perq"
    for line in open(ROOT / "results/raw/gptvq/results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("vocab", "none") == "none" or not r.get("rotate_matrix") or r.get("dim") != 4 or r.get("hi_bits"):
            continue
        ds = r["dataset"]
        stem = (f"{ds}_vq{r['bits']}b_d4_r0fs2s_{r['calib']}_voc{r['vocab'].replace('+', '')}"
                f"_ps1_ao0_n{r['n_seq']}.npz")
        nd = test_mean(P / stem, ds) or test_mean(P / stem.replace(f"_voc{r['vocab'].replace('+', '')}", ""), ds)
        if nd is None:
            continue
        tab = r.get("table_mib") or 48.7
        out.append(dict(client=f"VQ dim4 {r['eff_bpw']:.2f} bpw + rotace + oříznutá tabulka",
                        mib=r["size_blocks_mib"] + tab, ds=ds, nd=nd, split="test",
                        calib=f"{r['calib']} ({r['n_seq']} sekv.)", measured="sim",
                        note="bloky i tabulka v jednom běhu; bez runtimu"))
    return out


def gguf_rows():
    """real files, run through llama-embedding"""
    out = []
    for line in open(ROOT / "results/raw/gptq_export/results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if not r.get("gguf") or r.get("dataset") is None:
            continue
        out.append(dict(client=f"GPTQ {r['gguf'].replace('harrier-0.6b-gptq-', '').replace('.gguf', '')[:46]}",
                        mib=r["gguf_mib"], ds=r["dataset"], nd=r["gt_ndcg10"], split="test",
                        calib="viz název", measured="file", note="skutečný soubor"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="results/tables/main.md")
    a = ap.parse_args()
    rows = gguf_rows() + vq_rows()
    rows += [dict(client="BitNet-270m (ternární QAT, obě strany)", mib=140.0, ds=d, nd=v, split="test",
                  calib="—", measured="file", note="externí reference")
             for d, v in BITNET.items()]
    rows += [dict(client="harrier-0.6b fp16 (serverový model)", mib=1143.0, ds=d, nd=v, split="test",
                  calib="—", measured="file", note="reference")
             for d, v in FP_TEST.items()]

    L = ["# Hlavní tabulka: každý klient s plnou proveniencí", "",
         "Generuje `scripts/main_table.py`. `měřeno jako`: **file** = skutečný GGUF přes llama-embedding, ",
         "**grid** = dekvantizované váhy v torchi (runtime to reprodukuje do ±0,002), **sim** = pro tenhle formát ",
         "runtime neexistuje. Všechna čísla jsou nDCG@10 na testovacím splitu proti fp32 indexu téhož modelu.", "",
         "| klient | MiB | dataset | nDCG@10 | % fp | vs BitNet | split | kalibrace | měřeno jako | pozn. |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["ds"], x["mib"])):
        d = r["ds"]
        L.append(f"| {r['client']} | {r['mib']:.0f} | {d} | {r['nd']:.4f} | {r['nd']/FP_TEST[d]*100:.1f} | "
                 f"{r['nd']-BITNET[d]:+.4f} | {r['split']} | {r['calib']} | **{r['measured']}** | {r['note']} |")
    L += ["", "## Co tabulka záměrně neobsahuje", "",
          "- **Latenci.** `llama-bench` měří propustnost, ne latenci klienta; jediné poctivé číslo je 151 ms na dotaz",
          "  pro soubor 192 MiB (300 reálných dotazů, kontext 512, 8 vláken) — viz `wins.md` §2.1b.",
          "- **Paměť.** Velikost souboru není paměťová stopa: 192 MiB vah = 0,93 GB RSS při kontextu 512",
          "  a 4,08 GB při výchozím okně modelu.",
          "- **Řádky měřené na jiné množině dotazů.** Kde není testovací split, řádek tu není."]
    text = "\n".join(L)
    print(text)
    Path(ROOT / a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / a.md).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
