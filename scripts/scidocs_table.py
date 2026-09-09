"""SciDocs second reference client: collect every measurement of the scidocs queue into one table and a lab-log entry.

Reads (all under results/raw/scidocs_local/ unless stated; missing inputs leave "–" cells, nothing is invented):
  results.jsonl                 native rows (scripts/quant_eval_queries.py, upstream llama-embedding.exe on Windows):
                                dataset, splits, gguf, gguf_mib, gt_ndcg10, fp_ndcg10, q_cos_fp, fp_top10_overlap, n_test
  perq/scidocs_<stem>[_splits].npz   per-query nDCG@10 (paired bootstrap between the two calibrations and vs fp32)
  gptq_export/exports.jsonl     torch-side rows of scripts/gptq_export_gguf.py (calibration, n_seq, file size, quant time)
  gptq_*.log                    "[calib] ..." line = the real token budget of each calibration set
  student.jsonl                 scripts/eval_student.py rows (BitNet-270m fp, teacher fp32 reference)
  smoke_scidocs.json / smoke_scifact.json   scripts/demo_smoke.py reports (browser vs native per query)
  demo/data/index.json, demo/data/scidocs/meta.json   demo file sizes and the int8 index check
Writes results/tables/scidocs_client.md, results/raw/scidocs_local/summary.json and
results/raw/scidocs_local/lab_log_entry.md (Czech, the numbers filled in); --append_log appends the entry to
research/lab_log.md.

Usage: .venv/Scripts/python.exe scripts/scidocs_table.py [--append_log]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "results/raw/scidocs_local"
TABLE = ROOT / "results/tables/scidocs_client.md"
C1 = "harrier-0.6b-gptq-Q2_K-scidocs_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf"
C2 = "harrier-0.6b-gptq-Q2_K-scidocs_synth_only-ps20000-t300k-ao-tabQ2_K.gguf"
C0 = "harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf"
LABEL = {C1: "klient Q2_K, kalibrace SciDocs korpus (`scidocs_corpus_only`)", C2: "klient Q2_K, kalibrace syntetické dotazy (`scidocs_synth_only`)",
         C0: "kontrola: SciFact klient (`scifact_corpus_only`, mimo doménu)"}
PRED = dict(corpus_pct=97.0, synth_pct=97.5, calib_diff=0.01, browser_tol=0.002, int8_tol=0.001)


def jsonl(p: Path) -> list[dict]:
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
    return rows


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f4(x, sign=False):
    if x is None:
        return "–"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def calib_tokens(name: str) -> str | None:
    """the '[calib] file: N sequences = T tokens' line of the export log (per-sample) or the window count (stream)."""
    log = OUT / f"gptq_{name}.log"
    if not log.exists():
        return None
    t = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[calib\] \S+: (\d+) sequences = (\d+) tokens", t)
    if m:
        return f"{int(m.group(1))} dotazů = {int(m.group(2)):,} tokenů".replace(",", " ")
    m = re.search(r"n_seq=(\d+) per_sample", t)
    if m:
        n = int(m.group(1))
        return f"{n} oken × 512 = {n * 512:,} tokenů".replace(",", " ")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append_log", action="store_true")
    args = ap.parse_args()
    native = {}
    for r in jsonl(OUT / "results.jsonl"):
        if r.get("dataset") == "scidocs" and not r.get("both_sides"):
            native[(r["gguf"], r.get("splits", "test"))] = r
    exports = {r["gguf"]: r for r in jsonl(OUT / "gptq_export/exports.jsonl") if r.get("gguf")}
    student = jsonl(OUT / "student.jsonl")
    bitnet = next((r for r in student if r.get("student") == "bitnet-270m" and r.get("bits") == 16 and r.get("dataset") == "scidocs"), None)
    teacher = next((r for r in student if r.get("student") == "teacher:harrier-0.6b" and r.get("dataset") == "scidocs"), None)
    smoke = {}
    for d in ("scidocs", "scifact"):
        p = OUT / f"smoke_{d}.json"
        if p.exists():
            smoke[d] = json.loads(p.read_text(encoding="utf-8")).get("summary") or {}
    demo_meta = ROOT / "demo/data/scidocs/meta.json"
    dm = json.loads(demo_meta.read_text(encoding="utf-8")) if demo_meta.exists() else {}
    demo_dir = ROOT / "demo/data/scidocs"; model_dir = ROOT / "demo/model/scidocs"
    demo_files = sorted(demo_dir.glob("*")) if demo_dir.exists() else []
    model_files = sorted(model_dir.glob("*.chunk*")) if model_dir.exists() else []

    fp = next((r["fp_ndcg10"] for r in native.values() if r.get("splits", "test") == "test"), None)
    fp_all = next((r["fp_ndcg10"] for r in native.values() if r.get("splits") == "train+dev+test"), None)
    n_test = next((r["n_test"] for r in native.values() if r.get("splits", "test") == "test"), None)
    rows = []
    for g in (C1, C2, C0):
        r = native.get((g, "test")); ra = native.get((g, "train+dev+test")); e = exports.get(g)
        row = dict(gguf=g, label=LABEL[g], mib=(r or e or {}).get("gguf_mib", (e or {}).get("file_mib")),
                   ndcg=r and r["gt_ndcg10"], pct=r and r["gt_ndcg10"] / r["fp_ndcg10"] * 100, cos=r and r["q_cos_fp"], ov=r and r.get("fp_top10_overlap"),
                   ndcg_all=ra and ra["gt_ndcg10"], pct_all=ra and ra["gt_ndcg10"] / ra["fp_ndcg10"] * 100,
                   torch=e and (e.get("torch_ndcg") or {}).get("scidocs", {}).get("ndcg10"), torch_sf=e and (e.get("torch_ndcg") or {}).get("scifact", {}).get("ndcg10"),
                   calib=calib_tokens("corpus" if g == C1 else "synth" if g == C2 else "") or (e and f"n_seq {e.get('n_seq')}") or "–",
                   quant_s=e and e.get("quant_s"), bpw=e and e.get("eff_bpw_blocks"))
        pq = OUT / "perq" / f"scidocs_{Path(g).stem}.npz"
        if pq.exists():
            z = np.load(pq); row["perq"] = z["ndcg"]; row["perq_fp"] = z["ndcg_fp"]
            m, lo, hi = paired_bootstrap(z["ndcg_fp"], z["ndcg"]); row["d_fp"] = (m, lo, hi)
        rows.append(row)
    by = {r["gguf"]: r for r in rows}
    d12 = None
    if "perq" in by[C1] and "perq" in by[C2]:
        d12 = paired_bootstrap(by[C1]["perq"], by[C2]["perq"])   # synth - corpus
    sm = smoke.get("scidocs", {}); sf = smoke.get("scifact", {})
    chk = (dm.get("index") or {}).get("check") or {}

    # ---- predictions ----
    verdicts = []
    if by[C1]["pct"] is not None:
        verdicts.append(f"P1a korpusová kalibrace ≥ {PRED['corpus_pct']:.0f} % fp: {by[C1]['pct']:.1f} % → **{'platí' if by[C1]['pct'] >= PRED['corpus_pct'] else 'neplatí'}**")
    if by[C2]["pct"] is not None:
        verdicts.append(f"P1b syntetické dotazy ≥ {PRED['synth_pct']:.1f} % fp: {by[C2]['pct']:.1f} % → **{'platí' if by[C2]['pct'] >= PRED['synth_pct'] else 'neplatí'}**")
    if d12:
        verdicts.append(f"P2 |rozdíl kalibrací| < {PRED['calib_diff']}: {abs(d12[0]):.4f} (synth − korpus {d12[0]:+.4f} [{d12[1]:+.4f}, {d12[2]:+.4f}]) → **{'platí' if abs(d12[0]) < PRED['calib_diff'] else 'neplatí'}**")
    if sm.get("delta") is not None:
        verdicts.append(f"P3 prohlížeč vs nativní do ±{PRED['browser_tol']}: Δ {sm['delta']:+.4f} na {sm['n']} dotazech ({sm.get('n_identical_1e6')}/{sm['n']} per-query shodných do 1e-6) → **{'platí' if sm.get('within_0_002') else 'neplatí'}**")
    if chk:
        verdicts.append(f"P4 klient 192 MiB, index int8 s dopadem < {PRED['int8_tol']}: soubor {(by[C1]['mib'] or 0):.1f} MiB; int8 Δ nDCG {chk['delta_int8_vs_fp32']:+.4f} → uloženo jako {chk['decision']} → **{'platí' if (chk['decision'] == 'int8_rowscale' and abs(chk['delta_int8_vs_fp32']) < PRED['int8_tol']) else 'neplatí'}**")
    falsif = None
    if by[C1]["pct"] is not None:
        falsif = f"Falzifikační práh (< 95 % fp): {'NEnastal' if by[C1]['pct'] >= 95 else 'NASTAL'} ({by[C1]['pct']:.1f} %)."

    # ---- table ----
    L = ["# SciDocs: druhý referenční klient (Q2_K GPTQ, 192 MiB) a druhý korpus dema", ""]
    L += [f"Protokol jako u SciFact klienta: model harrier-0.6b, GPTQ act-order na mřížce Q2_K (per-sample, rozpočet 300k tokenů), tabulka tokenů Q2_K; "
          f"**kvantizovaná strana dotazu, fp32 index dokumentů téhož modelu** (`data/emb/scidocs/harrier-0.6b`, generická E5 instrukce), "
          f"SciDocs test split ({n_test if n_test else '?'} dotazů z 1 000; BEIR SciDocs je test-only, registr dělí 40/10/50 se seedem 0 jako u ArguAny), "
          f"nDCG@10, pooling last, L2. Runtime kvality: upstream llama.cpp `llama-embedding.exe` (Windows build-clang), 8 vláken, ctx 1024 "
          f"(`scripts/quant_eval_queries.py --bin third_party/llama.cpp/build-clang/bin/llama-embedding.exe`); rozdíl proti bitnet.cpp/WSL je ≤ 0,004 v pořadí remíz (viz 2026-09-07).", ""]
    L += [f"fp32 reference (harrier-0.6b, bf16 na GPU, test): nDCG@10 = **{f4(fp)}**" + (f"; všech 1 000 dotazů: {f4(fp_all)}" if fp_all else "")
          + (f"; BitNet-270m (fp výstup, poslední token, E5): {bitnet['gt_ndcg@10']:.4f}" if bitnet else "; BitNet-270m: nezměřeno") + ".", ""]
    hdr = ["soubor", "MiB", "kalibrace (skutečný rozpočet)", "nDCG@10 test (% fp)", "Δ vs fp32 [95 % CI]", "nDCG@10 všech 1 000 (% fp)", "cos(q, fp)", "top-10 překryv s fp", "torch-side nDCG@10 SciDocs / SciFact", "GPTQ čas"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        if r["ndcg"] is None and r["torch"] is None:
            continue
        d = r.get("d_fp")
        L.append("| " + " | ".join([r["label"], f"{r['mib']:.1f}" if r["mib"] else "–", r["calib"],
                                    f"{f4(r['ndcg'])} ({r['pct']:.1f})" if r["ndcg"] is not None else "–",
                                    f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" if d else "–",
                                    f"{f4(r['ndcg_all'])} ({r['pct_all']:.1f})" if r["ndcg_all"] is not None else "–",
                                    f4(r["cos"]), f"{r['ov']:.3f}" if r["ov"] is not None else "–",
                                    f"{f4(r['torch'])} / {f4(r['torch_sf'])}", f"{r['quant_s']:.0f} s" if r["quant_s"] else "–"]) + " |")
    L.append("")
    if d12:
        L.append(f"Rozdíl kalibrací (syntetické − korpusové, párový bootstrap přes {len(by[C1]['perq'])} dotazů, 2 000 losů): **{d12[0]:+.4f}** [{d12[1]:+.4f}, {d12[2]:+.4f}].")
        L.append("")
    L += ["## Prohlížeč (demo, wllama 3.6.1, headless Chrome, `scripts/demo_smoke.py`)", ""]
    if sm:
        L.append(f"- SciDocs: {sm.get('n')} testovacích dotazů, nDCG@10 v prohlížeči **{f4(sm.get('ndcg10_browser'))}** vs nativně na týchž dotazech {f4(sm.get('ndcg10_native_same_n'))} "
                 f"(Δ {f4(sm.get('delta'), True)}; {sm.get('n_identical_1e6')}/{sm.get('n')} per-query hodnot shodných do 1e-6, {sm.get('n_diff_gt_1e3')} se liší o > 0,001, max |Δ| {sm.get('max_abs_diff', 0):.4f}); "
                 f"vyhledávání vrátilo {sm.get('n_results')} výsledků; {sm.get('threads')} vláken (mt={sm.get('multithread')}), stažení {sm.get('download_s', 0):.1f} s, načtení {sm.get('load_s', 0):.1f} s, "
                 f"p50 {sm.get('p50_ms', 0):.0f} ms (stroj po noční frontě, latence orientační)"
                 + (f", paměť po načtení {sm['memory_after_load']['bytes'] / 2**20:.0f} MiB" if (sm.get('memory_after_load') or {}).get('bytes') else "") + ".")
    else:
        L.append("- SciDocs: smoke test neproběhl (viz queue.log).")
    if sf:
        L.append(f"- Regrese SciFact po přestavbě stránky na více korpusů: {sf.get('n')} dotazů, nDCG@10 {f4(sf.get('ndcg10_browser'))} vs nativně {f4(sf.get('ndcg10_native_same_n'))} "
                 f"(Δ {f4(sf.get('delta'), True)}, {sf.get('n_identical_1e6')}/{sf.get('n')} shodných do 1e-6).")
    L += ["", "## Soubory dema (Cloudflare Pages, limit 25 MiB na soubor)", ""]
    if dm:
        idx = dm["index"]
        L.append(f"- Index: {idx['n_docs']} × {idx['dim']}, uložen jako **{idx['dtype_on_disk']}** v {len(idx['files'])} souborech "
                 f"({sum(f['bytes'] for f in idx['files']) / 2**20:.1f} MiB); kontrola na testovacích dotazech (fp32 dotazy z cache): nDCG@10 fp32 {chk.get('ndcg10_fp32')} / f16 {chk.get('ndcg10_f16')} / int8 {chk.get('ndcg10_int8')} "
                 f"(int8 Δ {chk.get('delta_int8_vs_fp32'):+.6f}, tolerance {chk.get('tolerance')}).")
        L.append(f"- Dokumenty: {len(dm.get('docs_files', []))} JSON shardů; klient `{dm['file']}` {dm.get('size_mib')} MiB v {dm.get('model_chunks')} bajtových chuncích (sha256 {str(dm.get('model_sha256'))[:12]}…).")
        L.append("- Velikosti: " + ", ".join(f"`{f.name}` {f.stat().st_size / 2**20:.2f} MiB" for f in demo_files) + (f"; chunky modelu {len(model_files)} × ≤ 24 MiB" if model_files else "") + ".")
        L.append(f"- Licence korpusu: {dm.get('dataset_license')} (SciFact je CC BY-NC, proto druhý korpus).")
    else:
        L.append("- demo/data/scidocs/ nebyl vytvořen (viz queue.log).")
    L += ["", "## Pre-registrované predikce (research/lab_log.md, 2026-09-07 večer)", ""] + [f"- {v}" for v in verdicts] + ([f"- {falsif}"] if falsif else [])
    L += ["", "Zdroj: `results/raw/scidocs_local/` (results.jsonl, gptq_export/exports.jsonl, student.jsonl, smoke_*.json, perq/*.npz; vše OUR_MEASUREMENT), "
          "fronta `scripts/scidocs_queue.sh` (log `results/raw/scidocs_local/queue.log`), tabulka `scripts/scidocs_table.py`."]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[table] {TABLE}")

    # ---- lab-log entry (Czech) ----
    E = ["", "### 2026-09-08 — SciDocs: druhý korpus pro demo", ""]
    E += [f"Postaveno a změřeno lokálně na RTX 4050 frontou `scripts/scidocs_queue.sh` (spuštěna po `WEBGPU_DONE`; log `results/raw/scidocs_local/queue.log`). "
          f"Dataset: BEIR SciDocs přes `BeIR/scidocs` (HF), 25 656 dokumentů po deduplikaci (1 přesný duplikát), 1 000 dotazů test-only → registr 400/100/500 (seed 0); "
          f"licence CC BY 4.0 (původní vydání; HF karta cc-by-sa-4.0 pro přebalení) — zapsáno v `src/eq/data.py`. "
          f"Kalibrace: `scidocs_corpus_only` (2 000 dokumentů; GPTQ vezme 585 oken × 512 = 299 520 tokenů) a `scidocs_synth_only` "
          f"(3 000 syntetických dotazů z Qwen3-1.7B, 1 500 dokumentů × 2, E5 prefix{'; skutečný rozpočet ' + by[C2]['calib'] if by[C2]['calib'] != '–' else ''}). "
          f"Index fp32 harrier-0.6b (`scripts/build_teacher_cache.py`, bf16 na GPU, dávka 16, max_len 512).", ""]
    E += [f"**Výsledky (SciDocs test, {n_test} dotazů, strana dotazu kvantizovaná, fp32 index, upstream llama-embedding.exe/Windows).** fp32: {f4(fp)}"
          + (f"; BitNet-270m {bitnet['gt_ndcg@10']:.4f}" if bitnet else "") + "."]
    for r in rows:
        if r["ndcg"] is None:
            continue
        d = r.get("d_fp")
        E.append(f"- {r['label']}: {r['mib']:.1f} MiB, nDCG@10 **{f4(r['ndcg'])}** ({r['pct']:.1f} % fp), Δ vs fp {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" if d else
                 f"- {r['label']}: {r['mib']:.1f} MiB, nDCG@10 **{f4(r['ndcg'])}** ({r['pct']:.1f} % fp)")
        E[-1] += f", cos k fp {f4(r['cos'])}" + (f", všech 1 000 dotazů {f4(r['ndcg_all'])} ({r['pct_all']:.1f} %)" if r["ndcg_all"] is not None else "") + f", torch-side {f4(r['torch'])} (SciFact {f4(r['torch_sf'])})."
    if d12:
        E.append(f"- Rozdíl kalibrací syntetické − korpusové: {d12[0]:+.4f} [{d12[1]:+.4f}, {d12[2]:+.4f}] (párový bootstrap).")
    if sm:
        E.append(f"- Prohlížeč (demo, headless Chrome, {sm.get('n')} dotazů): nDCG@10 {f4(sm.get('ndcg10_browser'))} vs nativně {f4(sm.get('ndcg10_native_same_n'))}, Δ {f4(sm.get('delta'), True)}, "
                 f"{sm.get('n_identical_1e6')}/{sm.get('n')} per-query shodných do 1e-6, {sm.get('n_diff_gt_1e3')} se liší o > 0,001; dotaz vrátil {sm.get('n_results')} výsledků.")
    if sf:
        E.append(f"- Regrese SciFact dema po změně stránky: {sf.get('n')} dotazů, Δ {f4(sf.get('delta'), True)}, {sf.get('n_identical_1e6')}/{sf.get('n')} shodných.")
    if dm:
        E.append(f"- Demo: index uložen jako {dm['index']['dtype_on_disk']} ({len(dm['index']['files'])} soubory, int8 Δ nDCG {chk.get('delta_int8_vs_fp32'):+.6f} na testu), "
                 f"{len(dm.get('docs_files', []))} shardů dokumentů, klient v {dm.get('model_chunks')} chuncích; stránka má přepínač korpusu (`?ds=scidocs`, `demo/data/index.json`).")
    E += ["", "**Predikce:** " + (" ".join(verdicts) if verdicts else "nevyhodnoceno (chybí měření).") + (f" {falsif}" if falsif else ""), ""]
    E += ["**Příkazy.**", "```",
          ".venv/Scripts/python.exe scripts/prepare_data.py --datasets scidocs",
          ".venv/Scripts/python.exe scripts/build_teacher_cache.py --datasets scidocs --teacher harrier-0.6b --batch_size 16 --max_vram_gb 3",
          ".venv/Scripts/python.exe scripts/eval_student.py --dataset scidocs --student bitnet-270m --pooling last --qprompt e5 --bits 16 --out results/raw/scidocs_local/student.jsonl",
          ".venv/Scripts/python.exe scripts/doc2query.py --datasets scidocs --max_docs 1500 --n_per_doc 2 --batch 12",
          ".venv/Scripts/python.exe scripts/quant_calib_texts.py --datasets scidocs --max_docs 2000 --max_synth 3000",
          "PYTHONPATH=third_party/llama.cpp/gguf-py .venv/Scripts/python.exe scripts/gptq_export_gguf.py --type Q2_K --calib scidocs_corpus_only --per_sample --n_seq 20000 \\",
          "    --calib_tokens 300000 --act_order --table_type Q2_K --datasets scidocs scifact --out results/raw/scidocs_local/gptq_export   (a totéž s --calib scidocs_synth_only)",
          ".venv/Scripts/python.exe scripts/quant_eval_queries.py --teacher harrier-0.6b --datasets scidocs --bin third_party/llama.cpp/build-clang/bin/llama-embedding.exe \\",
          "    --threads 8 --workers 1 --out results/raw/scidocs_local --ggufs models/gguf/harrier-0.6b-gptq-Q2_K-scidocs_{corpus,synth}_only-ps20000-t300k-ao-tabQ2_K.gguf models/gguf/" + C0 + "   [+ --splits train dev test]",
          ".venv/Scripts/python.exe scripts/demo_export.py --dataset scidocs --client models/gguf/" + C1,
          ".venv/Scripts/python.exe scripts/demo_smoke.py --ds scidocs --verify_n 50 --autoquery \"graph neural networks for citation recommendation\"",
          ".venv/Scripts/python.exe scripts/scidocs_table.py --append_log",
          "```", "Výstupy: `results/tables/scidocs_client.md`, `results/raw/scidocs_local/` (results.jsonl, gptq_export/, perq/, student.jsonl, smoke_*.json, *.log), "
          "`data/calib/scidocs_{corpus_only,synth_only,corpus_synth}.txt`, `data/synth/scidocs/queries.jsonl`, `data/emb/scidocs/`, `models/gguf/harrier-0.6b-gptq-Q2_K-scidocs_*.gguf`, "
          "`demo/data/{index.json,scidocs/,scifact/}`, `demo/model/{scidocs,scifact}/`.", ""]
    entry = "\n".join(E)
    (OUT / "lab_log_entry.md").write_text(entry, encoding="utf-8")
    summary = dict(fp_ndcg10=fp, fp_ndcg10_all=fp_all, n_test=n_test, bitnet=bitnet and bitnet.get("gt_ndcg@10"),
                   clients={r["gguf"]: {k: (v if not isinstance(v, np.ndarray) else None) for k, v in r.items() if k not in ("perq", "perq_fp")} for r in rows},
                   calib_diff_synth_minus_corpus=d12, smoke=smoke, index_check=chk, verdicts=verdicts, falsification=falsif, attribution="OUR_MEASUREMENT")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n".join(verdicts) if verdicts else "(no predictions evaluable yet)")
    if args.append_log:
        lab = ROOT / "research/lab_log.md"
        txt = lab.read_text(encoding="utf-8")
        if "### 2026-09-08 — SciDocs: druhý korpus pro demo" in txt:
            print("[lab_log] entry already present; not appended (see results/raw/scidocs_local/lab_log_entry.md)")
        else:
            with open(lab, "a", encoding="utf-8") as f:
                f.write(("\n" if not txt.endswith("\n") else "") + entry)
            print(f"[lab_log] appended to {lab}")


if __name__ == "__main__":
    main()
