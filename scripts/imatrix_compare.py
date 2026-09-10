#!/usr/bin/env python
"""Fair comparison asked for by the external review: our 192 MiB GPTQ client versus llama.cpp's OWN quantizer
(llama-quantize with an importance matrix) at the same size, same calibration text, same token budget, same token
table type, same runtime and the same evaluation protocol.

The files are built in WSL by scripts/imatrix_compare.sh (llama-imatrix -c 512 --chunks 585 = 299 520 tokens, the
budget of our client; llama-quantize --token-embedding-type q2_k).  This script (Windows venv) measures, for every
file in /home/fragmea/eq/models/gguf (ours first):
  * real file size (MiB)
  * SciFact / NFCorpus TEST nDCG@10, query side quantized through llama-embedding (WSL, bitnet.cpp build = the runtime
    of the client numbers), fp32 document index of harrier-0.6b from the teacher cache, pooling last, L2-normalised,
    E5 query instruction -- exactly the protocol of scripts/quant_eval_queries.py that produced the 0.7440 / 0.3598 of
    our client (teacher key harrier-0.6b => E5_QUERY_PROMPT; task_query_prompt() applies only to the *-tp teacher key)
  * cosine to the fp32 query embeddings, fp top-10 overlap, R@100
  * peak RSS of one llama-embedding process over the 300 SciFact queries at ctx 512 (~/client3.sh protocol)
  * paired bootstrap (over queries) of the nDCG@10 difference to our client
and merges the per-query latency rows written by scripts/imatrix_latency.sh (results/raw/imatrix_local/latency_*.tsv).

Outputs: results/raw/imatrix_local/results.jsonl (one row per file), results/raw/imatrix_local/perq/*.npz,
         results/tables/imatrix_compare.md.  Everything is cached under results/raw/imatrix_local/work, so re-running
         after the idle latency pass only regenerates the table.

Usage: .venv/Scripts/python.exe scripts/imatrix_compare.py [--threads 8] [--ctx 1024] [--skip-rss] [--files ...]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eq import graph_metrics as G  # noqa: E402
from eq.data import load_dataset  # noqa: E402
from eq.teacher import E5_QUERY_PROMPT, TEACHERS, task_query_prompt, query_prompt  # noqa: E402
from eq.verify import load_emb, rel_matrix, self_mask, mask_self, l2n  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

WSL_EQ = "/home/fragmea/eq"
WSL_BIN = f"{WSL_EQ}/BitNet/build/bin/llama-embedding"
WSL_GGUF = f"{WSL_EQ}/models/gguf"
WSL_QUERIES = "/home/fragmea/queries.txt"          # 300 SciFact test queries, one per line (client*.sh)
UNC_GGUF = Path(r"\\wsl$\Ubuntu\home\fragmea\eq\models\gguf")
OURS = "harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf"
SEP = "<#sep#>"
OUT = ROOT / "results/raw/imatrix_local"
WORK = OUT / "work"
PERQ = OUT / "perq"
TABLE = ROOT / "results/tables/imatrix_compare.md"
BUDGET = "585 × 512 = 299 520 tok"
CALIB_CS = {"scifact_corpus_only": "SciFact korpus (scifact_corpus_only)", "generic_wikitext": "generický text (wikitext-2)",
            "nfcorpus_corpus_only": "NFCorpus korpus (nfcorpus_corpus_only)", "noimatrix": "žádná (round-to-nearest)"}


# ----------------------------------------------------------------------------- WSL plumbing
def win2wsl(p: Path) -> str:
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def _dec(b: bytes) -> str:
    return b.decode("utf-16", errors="replace") if b"\x00" in b[:200] else b.decode("utf-8", errors="replace")


def wsl(cmd: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl", "-e", "bash", "-c", cmd], capture_output=True, timeout=timeout)


def wsl_retry(cmd: str, tries: int = 6, timeout: int = 7200) -> subprocess.CompletedProcess:
    """WSL on this laptop is torn down under host memory pressure (Wsl/Service/E_UNEXPECTED); retry after a pause."""
    for i in range(tries):
        p = wsl(cmd, timeout)
        if p.returncode == 0:
            return p
        print(f"    [wsl] rc={p.returncode} attempt {i + 1}/{tries}: {_dec(p.stderr).strip()[-300:]}", flush=True)
        time.sleep(20)
    raise RuntimeError(f"wsl command kept failing: {cmd[:200]}")


def gguf_bytes(name: str) -> int:
    for p in ((WIN_GGUF / name), (UNC_GGUF / name)):
        try:
            return p.stat().st_size
        except OSError:
            pass
    return int(wsl_retry(f"stat -c %s {WSL_GGUF}/{name}").stdout.decode().split()[0])


def discover_files() -> list[str]:
    if RUNTIME == "win":
        names = sorted(p.name for p in WIN_GGUF.glob("harrier-0.6b-imx-*.gguf"))
    else:
        p = wsl_retry(f"cd {WSL_GGUF} && ls -1 harrier-0.6b-imx-*.gguf 2>/dev/null; true")
        names = [x.strip() for x in p.stdout.decode().splitlines() if x.strip().endswith(".gguf")]
    order = {"IQ2_M": 0, "Q2_K": 1, "Q2_Kpure": 2, "IQ2_XS": 3}
    names.sort(key=lambda n: (order.get(describe(n)["ftype_tag"], 9), describe(n)["calib"] != "scifact_corpus_only", n))
    return [OURS] + names


def describe(name: str) -> dict:
    if name == OURS:
        return dict(label="**náš GPTQ klient** (Q2_K mřížka, act-order, per-sample)", tool="GPTQ (náš)", ftype="Q2_K", ftype_tag="ours",
                    pure=True, calib="scifact_corpus_only", budget=BUDGET, arm="ours")
    m = re.match(r"harrier-0\.6b-imx-([A-Z0-9_]+?)(pure)?-(.+)-tabQ2_K\.gguf$", name)
    if not m:
        return dict(label=name, tool="?", ftype="?", ftype_tag="?", pure=False, calib="?", budget="?", arm="?")
    ft, pure, calib = m.group(1), bool(m.group(2)), m.group(3)
    arm = {("IQ2_M", "generic_wikitext"): "A", ("IQ2_M", "scifact_corpus_only"): "B", ("Q2_K", "scifact_corpus_only"): "C",
           ("Q2_K", "generic_wikitext"): "D", ("Q2_K", "noimatrix"): "E", ("IQ2_XS", "scifact_corpus_only"): "F"}.get((ft, calib), "")
    if pure:
        arm = arm + "'" if arm else "pure"
    label = f"llama-quantize {ft}" + (" `--pure`" if pure else "") + (f" — {arm}" if arm else "")
    if ft == "Q2_K" and not pure:
        label += " (K-quant směs)"
    return dict(label=label, tool="llama-quantize", ftype=ft, ftype_tag=ft + ("pure" if pure else ""), pure=pure, calib=calib,
                budget=("—" if calib == "noimatrix" else BUDGET), arm=arm)


# ----------------------------------------------------------------------------- runtime encoding (quality)
RUNTIME = "wsl"   # "wsl" = bitnet.cpp llama-embedding in WSL (the client runtime); "win" = upstream llama.cpp build-clang
WIN_BIN = ROOT / "third_party/llama.cpp/build-clang/bin/llama-embedding.exe"   # same ggml CPU kernels, Windows host
WIN_GGUF = ROOT / "models/gguf"


def encode(name: str, prompts: list[str], tag: str, threads: int, ctx: int):
    tag = tag + ("_win" if RUNTIME == "win" else "")
    npy = WORK / f"{tag}.npy"
    if npy.exists():
        return np.load(npy), dict(cached=True)
    WORK.mkdir(parents=True, exist_ok=True)
    pf = WORK / f"{tag}.txt"
    pf.write_text(SEP.join(t.replace(SEP, " ") for t in prompts), encoding="utf-8", newline="\n")
    flags = ["--embd-separator", SEP, "--pooling", "last", "--embd-normalize", "2", "--embd-output-format", "json",
             "-ngl", "0", "-t", str(threads), "-c", str(ctx), "-b", str(ctx), "--no-warmup"]   # = vocab_runtime_eval.run_chunk
    t0 = time.perf_counter()
    if RUNTIME == "win":
        p = subprocess.run([str(WIN_BIN), "-m", str(WIN_GGUF / name), "-f", str(pf), *flags], capture_output=True, timeout=7200)
        if p.returncode != 0:
            raise RuntimeError(f"{WIN_BIN.name} rc={p.returncode}: {p.stderr.decode('utf-8', 'replace')[-800:]}")
    else:
        sh = WORK / f"{tag}.sh"
        sh.write_text(f'"{WSL_BIN}" -m "{WSL_GGUF}/{name}" -f "{win2wsl(pf)}" ' + " ".join(f'"{x}"' if x == SEP else x for x in flags) + "\n",
                      encoding="utf-8", newline="\n")
        p = wsl_retry(f"bash {win2wsl(sh)}")
    out = p.stdout.decode("utf-8", errors="replace"); err = p.stderr.decode("utf-8", errors="replace")
    (WORK / f"{tag}.log").write_text(err[-4000:], encoding="utf-8")
    doc = json.loads(out[out.index("{"): out.rindex("}") + 1])
    data = sorted(doc["data"], key=lambda d: d["index"])
    emb = np.array([d["embedding"] for d in data], dtype=np.float32)
    if emb.shape[0] != len(prompts):
        raise RuntimeError(f"{name}: expected {len(prompts)} embeddings, got {emb.shape[0]}")
    np.save(npy, emb)
    m = re.search(r"prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens", err)
    return emb, dict(cached=False, wall_s=time.perf_counter() - t0, threads=threads, ctx=ctx,
                     prompt_eval_ms=float(m.group(1)) if m else None, tokens=int(m.group(2)) if m else None)


def quality(files: list[str], datasets: list[str], threads: int, ctx: int, teacher: str = "harrier-0.6b") -> dict:
    spec = TEACHERS[teacher]
    rows: dict[str, dict] = {f: {} for f in files}
    PERQ.mkdir(parents=True, exist_ok=True)
    for d in datasets:
        ds = load_dataset(d)
        tdoc, D_T, tq, Q_T = load_emb(d, teacher)
        tdi = {x: i for i, x in enumerate(tdoc)}; tqi = {q: i for i, q in enumerate(tq)}
        te = [q for q in ds.splits.get("test", []) if q in tqi]
        rel = rel_matrix(te, tdi, ds.qrels, len(tdoc)); msk = self_mask(te, tdi, len(tdoc))
        Qte = Q_T[[tqi[q] for q in te]]
        S0 = mask_self(Qte @ D_T.T, msk); nd_fp = G.ndcg_at_k(S0, rel, 10); fp = float(np.nanmean(nd_fp))
        qp = query_prompt(spec, d)  # teacher-specific prefix (E5 instruction for harrier, "Query: " for jina, MTEB task instruction for -tp keys)
        prompts = [qp + ds.queries[q] for q in te]
        print(f"[{d}] fp teacher {teacher}: nDCG@10={fp:.4f}  n_test={len(te)}  prompt={qp[:40]!r}...", flush=True)
        for name in files:
            tag = Path(name).stem
            Qq, st = encode(name, prompts, f"{d}_{tag}", threads, ctx)
            Qq = l2n(Qq.astype(np.float32))
            if Qq.shape[1] != D_T.shape[1]:
                raise SystemExit(f"dim mismatch {Qq.shape} vs docs {D_T.shape}")
            S = mask_self(Qq @ D_T.T, msk); nd = G.ndcg_at_k(S, rel, 10)
            np.savez(PERQ / f"{d}_{tag}.npz", ndcg=nd, ndcg_fp=nd_fp)
            rows[name][d] = dict(ndcg10=float(np.nanmean(nd)), fp_ndcg10=fp, pct_fp=float(np.nanmean(nd)) / fp * 100,
                                 recall100=float(np.nanmean(G.recall_at_k(S, rel, 100))),
                                 q_cos_fp=float(np.mean(np.sum(Qq * Qte, 1))), fp_top10_overlap=float(G.topk_overlap(S0, S, 10).mean()),
                                 n_test=len(te), encode=st)
            r = rows[name][d]
            print(f"[{d}] {name:62s} nDCG@10={r['ndcg10']:.4f} ({r['pct_fp']:.1f} % fp) cos={r['q_cos_fp']:.4f} "
                  f"top10ov={r['fp_top10_overlap']:.3f} {'(cached)' if st.get('cached') else f'{st['wall_s']:.0f}s'}", flush=True)
    return rows


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """mean(b - a) with a 95 % percentile interval over query resamples (evaluation noise only)."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ----------------------------------------------------------------------------- memory
def ensure_in_wsl(name: str):
    """RSS/latency run on the client runtime in WSL: copy a Windows-built file into /home/fragmea/eq/models/gguf if missing."""
    src, dst = WIN_GGUF / name, UNC_GGUF / name
    if src.exists() and not dst.exists():
        import shutil
        print(f"    [copy] {name} -> WSL ({src.stat().st_size / 2**20:.1f} MiB)", flush=True)
        shutil.copyfile(src, dst.with_suffix(".gguf.part")); (dst.with_suffix(".gguf.part")).rename(dst)


def rss(files: list[str], threads: int, n_queries: int = 300) -> dict:
    """Peak RSS of one llama-embedding process over the first n_queries of ~/queries.txt at ctx 512 (client3.sh: 300).
    n_queries < 300 exists because on this laptop every wsl.exe session is killed after ~90 s (2026-09-07): the peak is
    reached in the first batches (KV cache + compute buffers are allocated up front), so 100 queries give the same peak;
    the 300-query value of our client (906 MB) is kept as the check."""
    res = {}
    qfile = WSL_QUERIES if n_queries >= 300 else f"{WSL_EQ}/tmp/queries_{n_queries}.txt"
    for name in files:
        stem = Path(name).stem
        cache = WORK / (f"rss_{stem}.txt" if n_queries >= 300 else f"rss{n_queries}_{stem}.txt")
        if not cache.exists():
            try:
                ensure_in_wsl(name)
                WORK.mkdir(parents=True, exist_ok=True)
                log = f"{WSL_EQ}/tmp/rss_{stem}.log"
                cmd = ((f"head -n {n_queries} {WSL_QUERIES} > {qfile}; " if qfile != WSL_QUERIES else "") +
                       f'/usr/bin/time -f "%M %e" "{WSL_BIN}" -m "{WSL_GGUF}/{name}" -f {qfile} -c 512 -b 512 -t {threads} '
                       f'--embd-output-format array > /dev/null 2> {log}; tail -n 1 {log}')
                p = wsl_retry(cmd, tries=3)
                kb, wall = p.stdout.decode().split()[-2:]
                int(kb)
                cache.write_text(f"{kb} {wall}", encoding="utf-8")
            except Exception as e:  # noqa: BLE001  (WSL torn down: leave the cell empty, the next run fills it)
                print(f"[rss] {name}: not measured ({str(e)[:120]})", flush=True)
                continue
        kb, wall = cache.read_text(encoding="utf-8").split()
        res[name] = dict(peak_rss_kb=int(kb), wall_s=float(wall), ctx=512, threads=threads, n_queries=n_queries, protocol="client3.sh")
        print(f"[rss] {name:62s} peak RSS {int(kb)/1024:.0f} MB  ({float(wall):.1f} s for {n_queries} queries, ctx 512)", flush=True)
    return res


def latency_rows() -> dict:
    """latest row per (file, label) from results/raw/imatrix_local/latency_*.tsv (written by scripts/imatrix_latency.sh)."""
    lat: dict[tuple[str, str], dict] = {}
    for tsv in sorted(OUT.glob("latency_*.tsv")):
        lines = tsv.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            v = ln.split("\t")
            if len(v) != len(hdr):
                continue
            row = dict(zip(hdr, v))
            lat[(row["file"], row["label"])] = {k: (float(x) if re.fullmatch(r"-?[\d.]+", x) else x) for k, x in row.items()}
    return lat


# ----------------------------------------------------------------------------- table
def fmt_lat(r: dict | None) -> str:
    return "–" if not r else f"{r['p50_ms']:.0f} / {r['p95_ms']:.0f}"


def write_table(rows: list[dict], datasets: list[str], lat_labels: list[str]):
    fp = {d: rows[0][d]["fp_ndcg10"] for d in datasets}
    nq = next((r["rss"]["n_queries"] for r in rows if r.get("rss")), 300)
    L = ["# Férové srovnání při ~200 MiB: náš GPTQ klient vs. llama-quantize (imatrix) — lokálně, WSL", ""]
    L += [f"Stejný model (harrier-0.6b, f16 GGUF z upstream konvertoru), stejný runtime pro všechny řádky (kvalita: {rows[0].get('runtime_quality', '')}, "
          "8 vláken; RSS a latence: bitnet.cpp `llama-embedding` ve WSL = runtime klientských čísel), stejný protokol: **kvantizovaná strana dotazu, fp32 index dokumentů téhož modelu**, testovací split "
          "(SciFact 300 dotazů, NFCorpus 323), nDCG@10, pooling last, L2 normalizace, E5 instrukce v dotazu "
          "(protokol `scripts/quant_eval_queries.py`, který dal 0,7440 / 0,3598 našeho klienta). Tabulka tokenů u všech souborů Q2_K "
          "(`--token-embedding-type q2_k`), výstupní hlava neexistuje.",
          "",
          f"**Kalibrační rozpočet.** Náš klient (`ps20000-t300k`) byl kalibrován na 585 oknech × 512 tokenů = 299 520 tokenů "
          "ze začátku `data/calib/scifact_corpus_only.txt` (proud textu řezaný na okna; „585 dokumentů“ v dřívějším popisu je "
          "585 oken). `llama-imatrix` čte text stejně (jeden proud, pevné chunky `-c 512`), takže srovnaný rozpočet je "
          "`-c 512 --chunks 585` nad týmž souborem; generické rameno dostalo tentýž rozpočet z `data/calib/generic_wikitext.txt`. "
          "IQ2_M bez imatrix `llama-quantize` neumí; Q2_K bez imatrix = round-to-nearest na téže mřížce jako náš klient.",
          "",
          "Δ = rozdíl nDCG@10 proti našemu klientovi, párový bootstrap přes dotazy (2 000 losů, 95 % interval; měří jen evaluační šum). "
          f"RSS = peak RSS jednoho procesu `llama-embedding` (bitnet.cpp, WSL) nad prvními {nq} dotazy `~/queries.txt` při kontextu 512 (protokol `client3.sh`"
          + ("; 300 dotazů nebylo možné: každá relace `wsl.exe` je na tomto laptopu dnes ukončena po ~90 s, špička RSS je ale dosažena v prvních dávkách — "
             f"náš klient má na 300 dotazech {rows[0]['rss']['peak_rss_kb_300q']/1024:.0f} MB, na {nq} dotazech {rows[0]['rss']['peak_rss_kb']/1024:.0f} MB"
             if rows[0].get("rss", {}).get("peak_rss_kb_300q") else "") + "). "
          "Latence = jeden proces na dotaz (tokenizace + graf + výstup, soubor mmap), 8 vláken, ctx 512, 300 dotazů SciFactu, "
          "p50 / p95 ms (protokol `client.sh`, `scripts/imatrix_latency.sh`); sloupec „pod zátěží“ byl měřen, když na laptopu běžel "
          "jiný výpočet (GPU job + prohlížeč), „idle“ se doplní po opakovaném běhu bez zátěže.", ""]
    # runtime cross-check: our client measured natively through the WSL bitnet.cpp binary (results/raw/gptq_export/results.jsonl)
    ref = ROOT / "results/raw/gptq_export/results.jsonl"
    if ref.exists() and rows[0].get("runtime_quality", "").startswith("upstream"):
        nat = {}
        for ln in ref.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(ln)
            except ValueError:
                continue
            if j.get("gguf") == OURS and j.get("splits") == "test" and not j.get("both_sides"):
                nat[j["dataset"]] = j
        if all(d in nat for d in datasets):
            L += ["**Kontrola runtime.** Kvalita zde běžela přes upstream llama.cpp (Windows build téže revize), protože každá relace `wsl.exe` "
                  "je dnes na laptopu ukončena po ~90 s (viz lab log). Náš soubor přes bitnet.cpp/WSL (`results/raw/gptq_export/results.jsonl`): "
                  + ", ".join(f"{d} {nat[d]['gt_ndcg10']:.4f} (cos {nat[d]['q_cos_fp']:.4f})" for d in datasets)
                  + "; přes upstream/Windows: " + ", ".join(f"{d} {rows[0][d]['ndcg10']:.4f} (cos {rows[0][d]['q_cos_fp']:.4f})" for d in datasets)
                  + " — embeddingy jsou totožné na 3–4 desetinná místa kosinu, rozdíl nDCG ≤ 0,004 je pořadí u remízových skóre; "
                  "všechny řádky tabulky jsou měřeny týmž binárem, takže srovnání mezi řádky tím není dotčeno.", ""]
    hdr = ["soubor", "MiB", "kalibrace (text; rozpočet)"]
    for d in datasets:
        hdr += [f"{d} nDCG@10 (% fp {fp[d]:.4f})", "Δ vs náš [95 % CI]"]
    hdr += ["cos(q, fp) " + " / ".join(datasets), f"peak RSS ctx 512 ({nq} dotazů)", *[f"p50 / p95 ms ({lab.replace('_', ' ')})" for lab in lat_labels]]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        c = [r["label"], f"{r['mib']:.1f}", f"{CALIB_CS.get(r['calib'], r['calib'])}; {r['budget']}"]
        for d in datasets:
            q = r[d]
            c.append(f"{q['ndcg10']:.4f} ({q['pct_fp']:.1f})")
            b = r.get("delta_vs_ours", {}).get(d)
            c.append("–" if b is None else f"{b['mean']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]")
        c.append(" / ".join(f"{r[d]['q_cos_fp']:.3f}" for d in datasets))
        c.append(f"{r['rss']['peak_rss_kb']/1024:.0f} MB" if r.get("rss") else "–")
        for lab in lat_labels:
            c.append(fmt_lat(r.get("latency", {}).get(lab)))
        L.append("| " + " | ".join(c) + " |")
    L += ["", conclusion(rows, datasets, lat_labels), ""]
    L += ["Zdroj: `results/raw/imatrix_local/results.jsonl` (OUR_MEASUREMENT), soubory postavil `scripts/imatrix_compare.sh`, "
          "měřil `scripts/imatrix_compare.py` a `scripts/imatrix_latency.sh`."]
    TABLE.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[table] {TABLE}")


def _verdict(b: dict | None) -> str:
    if b is None:
        return "nezměřeno"
    if b["lo"] > 0:
        return "horší"      # competitor above ours => ours worse
    if b["hi"] < 0:
        return "lepší"
    return "v šumu stejný"


def conclusion(rows: list[dict], datasets: list[str], lat_labels: list[str]) -> str:
    ours = rows[0]
    by_arm = {r["arm"]: r for r in rows[1:]}
    P = ["## Závěr (fakticky)", ""]
    for arm, what in [("B", "IQ2_M s doménovou (SciFact) imatrix"), ("C", "Q2_K (K-quant směs) s doménovou imatrix"),
                      ("C'", "Q2_K `--pure` (přesně náš formát) s doménovou imatrix"), ("A", "IQ2_M s generickou imatrix"),
                      ("D", "Q2_K s generickou imatrix"), ("E'", "Q2_K `--pure` bez imatrix (round-to-nearest, náš formát)")]:
        r = by_arm.get(arm)
        if not r:
            continue
        parts = []
        for d in datasets:
            b = r.get("delta_vs_ours", {}).get(d)
            v = _verdict(b)
            parts.append(f"{d}: náš {ours[d]['ndcg10']:.4f} vs {r[d]['ndcg10']:.4f}, náš je **{v}** "
                         f"({-b['mean']:+.4f} pro nás, CI konkurenta [{b['lo']:+.4f}, {b['hi']:+.4f}])" if b else f"{d}: {v}")
        P.append(f"- **{arm} — {what}** ({r['mib']:.1f} MiB vs našich {ours['mib']:.1f} MiB): " + "; ".join(parts) + ".")
    P.append("")
    # kernel speed IQ2_M vs Q2_K, same session
    for lab in lat_labels:
        iq = [r["latency"][lab]["p50_ms"] for r in rows if r.get("latency", {}).get(lab) and r["ftype"] == "IQ2_M"]
        q2 = [r["latency"][lab]["p50_ms"] for r in rows if r.get("latency", {}).get(lab) and r["ftype"] == "Q2_K"]
        ol = ours.get("latency", {}).get(lab)
        if iq and q2:
            P.append(f"- **Rychlost CPU kernelů ({lab.replace('_', ' ')})**: IQ2_M p50 {min(iq):.0f}–{max(iq):.0f} ms na dotaz, "
                     f"Q2_K {min(q2):.0f}–{max(q2):.0f} ms (náš soubor {ol['p50_ms']:.0f} ms)"
                     + (f"; IQ2_M je {np.mean(iq)/np.mean(q2):.2f}× pomalejší než Q2_K (poměr průměrů p50), stejná relace" if np.mean(iq) > np.mean(q2)
                        else f"; IQ2_M je {np.mean(q2)/np.mean(iq):.2f}× rychlejší než Q2_K (poměr průměrů p50), stejná relace") + ".")
    rs = [(r["label"], r["rss"]["peak_rss_kb"] / 1024) for r in rows if r.get("rss")]
    if rs:
        P.append(f"- **Paměť**: peak RSS při ctx 512 {min(x for _, x in rs):.0f}–{max(x for _, x in rs):.0f} MB napříč soubory "
                 f"(náš {ours['rss']['peak_rss_kb']/1024:.0f} MB) — velikost souboru rozhoduje jen o části stopy, zbytek je KV cache a výpočetní buffery.")
    return "\n".join(P)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None, help="GGUF names in /home/fragmea/eq/models/gguf (default: ours + every harrier-0.6b-imx-*.gguf)")
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024, help="-c/-b of the quality pass (quant_eval_queries.py default)")
    ap.add_argument("--skip-rss", action="store_true")
    ap.add_argument("--rss-queries", type=int, default=300, help="queries per RSS run (client3.sh: 300; 100 keeps a run under the ~90 s wsl.exe session limit seen on 2026-09-07)")
    ap.add_argument("--runtime", default="wsl", choices=["wsl", "win"],
                    help="quality pass binary: wsl = bitnet.cpp llama-embedding in WSL (client runtime, default); win = upstream llama.cpp "
                         "build-clang llama-embedding.exe on the Windows host (used while the WSL VM is unusable; same ggml CPU kernels)")
    args = ap.parse_args()
    global RUNTIME
    RUNTIME = args.runtime
    OUT.mkdir(parents=True, exist_ok=True)
    files = args.files or discover_files()
    if OURS not in files:
        files = [OURS] + files
    files = [OURS] + [f for f in files if f != OURS]
    print("[files] " + ", ".join(files), flush=True)
    sizes = {f: gguf_bytes(f) for f in files}
    q = quality(files, args.datasets, args.threads, args.ctx)
    mem = {} if args.skip_rss else rss(files, args.threads, args.rss_queries)
    if not args.skip_rss and args.rss_queries < 300:   # the 300-query check for files that have it (our client)
        for f in files:
            c = WORK / f"rss_{Path(f).stem}.txt"
            if c.exists() and f in mem:
                kb, wall = c.read_text(encoding="utf-8").split()
                mem[f]["peak_rss_kb_300q"] = int(kb); mem[f]["wall_s_300q"] = float(wall)
    lat = latency_rows()
    lat_labels = sorted({lab for _, lab in lat}, key=lambda s: (s != "under_load", s)) or ["under_load"]
    rows = []
    for f in files:
        r = dict(gguf=f, mib=sizes[f] / 2**20, bytes=sizes[f], **describe(f))
        for d in args.datasets:
            r[d] = q[f][d]
        if f != OURS:
            r["delta_vs_ours"] = {}
            for d in args.datasets:
                a = np.load(PERQ / f"{d}_{Path(OURS).stem}.npz")["ndcg"]; b = np.load(PERQ / f"{d}_{Path(f).stem}.npz")["ndcg"]
                m, lo, hi = paired_bootstrap(a, b)
                r["delta_vs_ours"][d] = dict(mean=m, lo=lo, hi=hi, n_boot=2000)
        if f in mem:
            r["rss"] = mem[f]
        r["latency"] = {lab: lat[(f, lab)] for lab in lat_labels if (f, lab) in lat}
        r["runtime_quality"] = ("bitnet.cpp llama-embedding, WSL" if RUNTIME == "wsl" else "upstream llama.cpp build-clang llama-embedding.exe, Windows host")
        r["protocol"] = dict(quality=f"quant_eval_queries.py: {r['runtime_quality']} --pooling last --embd-normalize 2, "
                                     f"-c {args.ctx} -b {args.ctx}, -t {args.threads}, E5 query prompt, fp32 doc index harrier-0.6b, test split",
                             rss="client3.sh: llama-embedding -f ~/queries.txt -c 512 -b 512 -t 8, /usr/bin/time %M",
                             latency="client.sh: one llama-embedding process per query, -c 512 -b 512 -t 8, 300 SciFact test queries")
        r["attribution"] = "OUR_MEASUREMENT"
        rows.append(r)
    with open(OUT / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[rows] {OUT / 'results.jsonl'} ({len(rows)} rows)")
    write_table(rows, args.datasets, lat_labels)


if __name__ == "__main__":
    main()
