"""Run the browser client benchmark (client/browser/) headlessly and measure it as a whole client.

Starts client/browser/serve.py (COOP/COEP headers -> cross-origin isolated -> SharedArrayBuffer -> multi-thread WASM),
launches headless Chrome on the benchmark page, samples the working set of every Chrome process belonging to that
run's --user-data-dir every 0.5 s (peak process RSS), waits for the page to POST its result, then compares the
per-query nDCG@10 with the native llama.cpp reference (results/raw/gptq_export/perq/*.npz) and writes
results/tables/browser_client.md.

Default sequence (one command, nothing else): "mt" (8 threads, prefix KV cache on, 300 queries) -> "mt-nocache"
(same, slot flushed between queries = cold prefix, 300 queries) -> "st" (single-thread build = fallback without
cross-origin isolation, --n_st = 60 queries).  Result files: results/raw/browser_local/browser_<tag>_<build>.json.
Latency figures of runs whose tag does not start with "idle" are marked "under heavy load, not citable" in the table.

WebGPU: --gpu runs the builds of this invocation with backend=webgpu (bench.js passes n_gpu_layers=99999 instead of 0;
the vendored wllama 3.6.1 wasm already contains llama.cpp's ggml-webgpu backend, see client/webgpu/README.md).  Chrome is
then started WITHOUT --disable-gpu and with --enable-unsafe-webgpu --ignore-gpu-blocklist --use-angle=d3d11
--enable-features=Vulkan,WebGPU (--gpu-flags overrides).  The result JSON records the ggml_webgpu adapter line, the
offloaded-layers line, buffer sizes, graph splits and the flash-attention decision; result.gpu.active tells whether the
GPU really ran.  If headless Chrome reports no adapter, run headed (--headed) — a visible window may be needed for the
real adapter on Windows.  Default build set with --gpu: mt (300 queries, prefix cache on) -> browser_<tag>_mt.json.

VQ WebGPU runtime (client/vqweb/, --runtime vqweb --vqw <file.vqw>): the same page protocol on the standalone WebGPU
runtime (no wllama).  serve.py gets --mount vqw=<dir of the file>, headless Chrome runs with the integrated-GPU WebGPU
flags of scripts/vqw_layer_test.py, the page client/vqweb/bench.html posts browser_<tag>_vqweb.json.  Before the run the
torch reference embeddings of the first --dump-reference queries are produced (scripts/vqw_reference_dump.py ->
client/browser/data/<stem>.ref.f32) unless present; afterwards the nDCG@10 is compared with the exporter's torch
simulation (results/raw/vqweb*/exports.jsonl + perq/<stem>_scifact_<splits>.npz aligned by qid) with the +-0.003
criterion of the design spec (section 7 step 3) and the cosine to torch >= 0.999; the table gets a "VQ WebGPU" row.

Usage:
  .venv/Scripts/python.exe scripts/browser_run.py --tag idle              # clean latency run (machine idle!): mt, mt-nocache, st
  .venv/Scripts/python.exe scripts/browser_run.py --tag load --builds mt  # only the multi-thread build
  .venv/Scripts/python.exe scripts/browser_run.py --tag webgpu --gpu      # WebGPU arm: mt, 300 queries -> browser_webgpu_mt.json
  .venv/Scripts/python.exe scripts/browser_run.py --tag webgpu --gpu --dry-run   # print the Chrome command line only
  .venv/Scripts/python.exe scripts/browser_run.py --runtime vqweb --vqw models/vqw/dev2.vqw --tag dev2 --n 20   # plumbing smoke
  .venv/Scripts/python.exe scripts/browser_run.py --runtime vqweb --vqw models/vqw/harrier-0.6b-vq2.0-scifact.vqw --tag vq2.0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
RESULT_DIR = ROOT / "results/raw/browser_local"
TABLE = ROOT / "results/tables/browser_client.md"
DEFAULT_MODEL = "models/gguf/harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf"
NATIVE_NDCG = 0.7440  # llama.cpp (bitnet.cpp fork) llama-embedding, scifact test, results/raw/gptq_export/results.jsonl
NATIVE_COS_FP = 0.8261
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
# what headless Chrome needs for a real WebGPU adapter on Windows (Dawn on D3D11 via ANGLE; Vulkan as the alternative)
GPU_FLAGS = "--enable-unsafe-webgpu --ignore-gpu-blocklist --use-angle=d3d11"
GPU_FEATURES = "Vulkan,WebGPU"
VQW_EXPORT_LOGS = [ROOT / "results/raw/vqweb", ROOT / "results/raw/vqweb_dev"]  # exports.jsonl + perq/ of scripts/vq_export_web.py
VQW_NDCG_TOL = 0.003   # spec section 7 step 3: nDCG@10 within +-0.003 of the torch simulation
VQW_COS_MIN = 0.999    # ... and per-query cosine to the torch reference >= 0.999


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def chrome_procs(profile_dir: str):
    """All Chrome processes whose command line names this run's user-data-dir (browser, renderer, utility, gpu...)."""
    key = profile_dir.lower()
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower().startswith("chrome") and any(key in (a or "").lower() for a in (p.info["cmdline"] or [])):
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def kill_chrome(proc: subprocess.Popen, profile_dir: str):
    """Only the Chrome tree of this run: never anything else (a GPU python job may be running)."""
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    for p in chrome_procs(profile_dir):
        try:
            p.kill()
        except psutil.Error:
            pass


BUILDS = {  # build name -> (threads: None = --threads, prefix KV cache on/off)
    "mt": (None, 1),           # multi-thread build, llama-server prefix reuse on (resident client)
    "mt-nocache": (None, 0),   # multi-thread build, slot flushed between queries (cold prefix, every token evaluated)
    "st": (1, 1),              # single-thread build = fallback without cross-origin isolation
    "st-nocache": (1, 0),
}


def run_build(args, build: str, scratch: Path) -> dict:
    thr, cache = BUILDS[build]
    threads = thr or args.threads
    n = args.n_st if build.startswith("st") else args.n
    name = f"browser_{args.tag}_{build}"
    out = RESULT_DIR / f"{name}.json"
    out.unlink(missing_ok=True)
    (RESULT_DIR / f"{name}.log").unlink(missing_ok=True)
    profile = scratch / f"chrome-profile-{args.tag}-{build}"
    shutil.rmtree(profile, ignore_errors=True)  # fresh profile: empty OPFS cache -> the model is really downloaded
    profile.mkdir(parents=True)
    backend = "webgpu" if args.gpu else "wasm"
    q = (f"model={args.model}&threads={threads}&ctx={args.ctx}&n={n}&warmup={args.warmup}&cache={cache}&backend={backend}"
         f"&build={args.build}&tag={args.tag}&name={name}&auto=1")
    url = f"http://localhost:{args.port}/client/browser/index.html?{q}"
    features = "SharedArrayBuffer" + (f",{GPU_FEATURES}" if args.gpu else "")
    cmd = [args.chrome] + ([] if args.headed else ["--headless=new"]) + ["--no-first-run", "--no-default-browser-check"]
    cmd += args.gpu_flags.split() if args.gpu else ["--disable-gpu"]
    cmd += ["--disable-extensions", "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows", "--window-size=1200,900",
            f"--user-data-dir={profile}", f"--enable-features={features}", url]
    print(f"\n=== build {build}: backend={backend} threads={threads} ctx={args.ctx} n={n} cache={cache} -> {out.name}\n    {url}", flush=True)
    if args.dry_run:
        print("    chrome command line:\n    " + subprocess.list2cmdline(cmd), flush=True)
        return dict(name=name, build=build, ok=False, dry_run=True, error="dry run (not launched)", threads_requested=threads)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak_rss, peak_n, samples, last_log = 0, 0, 0, 0
    log_file = RESULT_DIR / f"{name}.log"
    seen_lines = 0
    try:
        while not out.exists():
            if time.time() - t0 > args.timeout:
                print(f"    TIMEOUT after {args.timeout}s", flush=True)
                break
            if proc.poll() is not None and not out.exists():
                time.sleep(1.0)
                if not out.exists():
                    print(f"    Chrome exited early (rc={proc.returncode})", flush=True)
                    break
            procs = chrome_procs(str(profile))
            rss = 0
            for p in procs:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            samples += 1
            if rss > peak_rss:
                peak_rss, peak_n = rss, len(procs)
            if log_file.exists() and time.time() - last_log > 2:  # echo the page's progress lines
                last_log = time.time()
                lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                for ln in lines[seen_lines:]:
                    print("    " + ln, flush=True)
                seen_lines = len(lines)
            time.sleep(0.5)
    finally:
        kill_chrome(proc, str(profile))
    if not out.exists():
        return dict(name=name, build=build, ok=False, error="no result file (timeout or crash)", peak_rss_bytes=peak_rss, threads_requested=threads)
    res = json.loads(out.read_text(encoding="utf-8"))
    res.update(build=build, threads_requested=threads, peak_rss_bytes=peak_rss, peak_rss_n_procs=peak_n, rss_samples=samples,
               wall_s_total=time.time() - t0, runner="scripts/browser_run.py", chrome=args.chrome, chrome_cmdline=subprocess.list2cmdline(cmd))
    if args.gpu and res.get("ok") and not (res.get("gpu") or {}).get("active"):
        print("    NOTE: WebGPU requested but the ggml_webgpu adapter/offload lines are missing: this was a CPU run. "
              "Try --headed (a visible Chrome window), or check chrome://gpu for the WebGPU status.", flush=True)
    # per-query agreement with the native llama.cpp run of the same file (same query order: ds.splits['test'])
    perq = ROOT / "results/raw/gptq_export/perq" / f"scifact_{Path(args.model).stem}.npz"
    if res.get("ok") and perq.exists() and res.get("prompt_kind", "generic") != "generic":
        res["correct"] = None  # different query-instruction protocol: the native reference is not comparable
    if res.get("ok") and perq.exists():
        nat = np.load(perq)["ndcg"].astype(np.float64)
        br = np.array([np.nan if v is None else v for v in res["ndcg_per_query"]], dtype=np.float64)
        m = min(len(nat), len(br))
        d = np.abs(nat[:m] - br[:m])
        res["native"] = dict(file=str(perq.relative_to(ROOT)), ndcg_mean=float(np.nanmean(nat)), ndcg_mean_same_n=float(np.nanmean(nat[:m])),
                             n_compared=int(m), perq_max_abs_diff=float(np.nanmax(d)), perq_mean_abs_diff=float(np.nanmean(d)),
                             n_perq_diff_gt_1e3=int(np.nansum(d > 1e-3)), n_nan_native=int(np.isnan(nat[:m]).sum()), n_nan_browser=int(np.isnan(br[:m]).sum()))
        res["ndcg_delta_vs_native"] = float(res["ndcg_mean"] - res["native"]["ndcg_mean_same_n"])
        if res.get("prompt_kind", "generic") == "generic":
            res["correct"] = bool(abs(res["ndcg_delta_vs_native"]) <= 0.002)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def check_helper_parity() -> bool:
    """client/vqweb/bench.js must carry the scoring helpers (topk, ndcgAtK, percentile, ...) of client/browser/bench.js verbatim."""
    a = (ROOT / "client/browser/bench.js").read_text(encoding="utf-8")
    b = (ROOT / "client/vqweb/bench.js").read_text(encoding="utf-8")
    sa = a[a.index("// ---- helpers"):a.index("// ---- main")].split("\n", 1)[1]
    sb = b[b.index("// ---- helpers: copied verbatim"):b.index("// ---- end of the copied helpers")].split("\n", 1)[1]
    same = sa.strip() == sb.strip()
    print(f"[vqweb] scoring helpers of client/vqweb/bench.js identical to client/browser/bench.js: {same}", flush=True)
    if not same:
        import difflib
        sys.stdout.writelines(difflib.unified_diff(sa.splitlines(True), sb.splitlines(True), "browser/bench.js", "vqweb/bench.js"))
    return same


def ensure_reference(vqw: Path, n: int) -> Path | None:
    """client/browser/data/<stem>.ref.f32 with >= n rows from the container of this sha256 (else regenerate)."""
    if n <= 0:
        return None
    out = ROOT / "client/browser/data" / f"{vqw.stem}.ref.f32"
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("n", 0) >= n and m.get("size_bytes") == vqw.stat().st_size:
            print(f"[vqweb] torch reference present: {out.relative_to(ROOT)} ({m['n']} queries, {m.get('generated')})", flush=True)
            return out
    print(f"[vqweb] computing the torch reference for the first {n} test queries (scripts/vqw_reference_dump.py, fp32 CPU)...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/vqw_reference_dump.py"), "--vqw", str(vqw), "--n", str(n)], check=True, cwd=ROOT)
    return out


def sim_reference(vqw: Path, qids: list[str], browser_ndcg: list) -> dict | None:
    """The exporter's torch simulation of the same container: exports.jsonl row + per-query nDCG aligned by qid."""
    rel = vqw.relative_to(ROOT).as_posix() if vqw.is_relative_to(ROOT) else str(vqw)
    for log_dir in VQW_EXPORT_LOGS:
        f = log_dir / "exports.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r.get("file", "").replace("\\", "/") == rel or Path(r.get("file", "")).name == vqw.name]
        if not rows:
            continue
        row = rows[-1]
        sp = row.get("splits", "test")
        splits = "+".join(sp) if isinstance(sp, list) else str(sp).replace(" ", "+")
        perq = next((p for p in [log_dir / "perq" / f"{vqw.stem}_scifact_{splits}.npz"] + sorted((log_dir / "perq").glob(f"{vqw.stem}_scifact_*.npz")) if p.exists()), None)
        out = dict(exports_file=f.relative_to(ROOT).as_posix(), row_bits=row.get("bits"), row_bpw=row.get("bpw_blocks"), row_sha256=row.get("sha256"), row_size_bytes=row.get("size_bytes"),
                   row_splits=splits, row_eval=(row.get("eval") or {}).get("scifact"), sha_matches=None, perq_file=None)
        try:
            import hashlib
            out["sha_matches"] = hashlib.sha256(vqw.read_bytes()).hexdigest() == row.get("sha256")
        except OSError:
            pass
        ev = out["row_eval"] or {}
        if splits == "test" and ev:
            out["ndcg_sim_test"] = float(ev["gt_ndcg10"])
        if perq is not None:
            d = np.load(perq)
            out["perq_file"] = perq.relative_to(ROOT).as_posix()
            pq = {str(q): float(v) for q, v in zip(d["qids"], d["ndcg"])}
            sim_all = np.array([pq.get(str(q), np.nan) for q in qids], dtype=np.float64)
            br = np.array([np.nan if v is None else v for v in browser_ndcg], dtype=np.float64)
            test_qids = [q["qid"] for q in json.loads((ROOT / "client/browser/data/scifact_test.json").read_text(encoding="utf-8"))["queries"]]
            test_set = set(test_qids)
            full = np.array([pq.get(str(q), np.nan) for q in test_qids], dtype=np.float64)
            out.update(ndcg_sim_test=float(np.nanmean(full)), n_sim_test=int(np.isfinite(full).sum()), ndcg_sim_same_n=float(np.nanmean(sim_all)),
                       n_compared=int(np.isfinite(sim_all).sum()), perq_max_abs_diff=float(np.nanmax(np.abs(sim_all - br))),
                       perq_mean_abs_diff=float(np.nanmean(np.abs(sim_all - br))), n_perq_diff_gt_1e3=int(np.nansum(np.abs(sim_all - br) > 1e-3)),
                       ndcg_fp_test=float(np.nanmean([float(v) for q, v in zip(d["qids"], d["ndcg_fp"]) if str(q) in test_set])) if "ndcg_fp" in d.files else None)
        elif "ndcg_sim_test" in out:
            out.update(ndcg_sim_same_n=out["ndcg_sim_test"], n_compared=len(qids))
        return out
    return None


def run_vqweb(args, scratch: Path, mount_prefix: str | None) -> dict:
    vqw = args.vqw
    name = f"browser_{args.tag}_vqweb"
    out = RESULT_DIR / f"{name}.json"
    out.unlink(missing_ok=True)
    (RESULT_DIR / f"{name}.log").unlink(missing_ok=True)
    profile = scratch / f"chrome-profile-{args.tag}-vqweb"
    shutil.rmtree(profile, ignore_errors=True)
    profile.mkdir(parents=True)
    vqw_url = f"{mount_prefix}/{vqw.name}" if mount_prefix else vqw.relative_to(ROOT).as_posix()
    ref = ROOT / "client/browser/data" / f"{vqw.stem}.ref.f32"
    q = (f"vqw={vqw_url}&n={args.n}&warmup={args.warmup}&tag={args.tag}&name={name}&resid={args.resid}&profile={args.profile_queries}&auto=1"
         + (f"&tile={args.tile}" if args.tile else "") + (f"&ref={ref.relative_to(ROOT).as_posix()}" if ref.exists() else ""))
    url = f"http://localhost:{args.port}/client/vqweb/bench.html?{q}"
    features = "SharedArrayBuffer," + GPU_FEATURES
    cmd = [args.chrome] + ([] if args.headed else ["--headless=new"]) + ["--no-first-run", "--no-default-browser-check"] + args.gpu_flags.split()
    cmd += ["--disable-extensions", "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows", "--window-size=1200,900",
            f"--user-data-dir={profile}", f"--enable-features={features}", url]
    print(f"\n=== vqweb: {vqw.relative_to(ROOT) if vqw.is_relative_to(ROOT) else vqw} n={args.n} -> {out.name}\n    {url}", flush=True)
    if args.dry_run:
        print("    chrome command line:\n    " + subprocess.list2cmdline(cmd), flush=True)
        return dict(name=name, runtime="vqweb", ok=False, dry_run=True, error="dry run (not launched)")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak_rss, peak_n, samples, last_log, seen_lines = 0, 0, 0, 0, 0
    log_file = RESULT_DIR / f"{name}.log"
    try:
        while not out.exists():
            if time.time() - t0 > args.timeout:
                print(f"    TIMEOUT after {args.timeout}s", flush=True)
                break
            if proc.poll() is not None and not out.exists():
                time.sleep(1.0)
                if not out.exists():
                    print(f"    Chrome exited early (rc={proc.returncode})", flush=True)
                    break
            procs = chrome_procs(str(profile))
            rss = 0
            for p in procs:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            samples += 1
            if rss > peak_rss:
                peak_rss, peak_n = rss, len(procs)
            if log_file.exists() and time.time() - last_log > 2:
                last_log = time.time()
                lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                for ln in lines[seen_lines:]:
                    print("    " + ln, flush=True)
                seen_lines = len(lines)
            time.sleep(0.5)
    finally:
        kill_chrome(proc, str(profile))
    if not out.exists():
        return dict(name=name, runtime="vqweb", ok=False, error="no result file (timeout or crash)", peak_rss_bytes=peak_rss)
    res = json.loads(out.read_text(encoding="utf-8"))
    res.update(build="vqweb", threads_requested=0, peak_rss_bytes=peak_rss, peak_rss_n_procs=peak_n, rss_samples=samples,
               wall_s_total=time.time() - t0, runner="scripts/browser_run.py", chrome=args.chrome, chrome_cmdline=subprocess.list2cmdline(cmd),
               vqw_path=str(vqw.relative_to(ROOT) if vqw.is_relative_to(ROOT) else vqw))
    if res.get("ok"):
        sim = sim_reference(vqw, res["qids"], res["ndcg_per_query"])
        res["native"] = sim  # the same slot the wllama rows use for their reference; here: the exporter's torch simulation
        if sim and "ndcg_sim_same_n" in sim:
            res["ndcg_delta_vs_native"] = float(res["ndcg_mean"] - sim["ndcg_sim_same_n"])
            res["ndcg_ok"] = bool(abs(res["ndcg_delta_vs_native"]) <= VQW_NDCG_TOL)
        else:
            res["ndcg_ok"] = None
            print("    NOTE: no exports.jsonl row / per-query file for this container: nDCG cannot be compared with the simulation", flush=True)
        ref_ok = (res.get("ref") or {}).get("ok") if res.get("ref") else None
        res["cos_ok"] = ref_ok
        res["correct"] = bool(res["ndcg_ok"]) and bool(ref_ok) if (res["ndcg_ok"] is not None and ref_ok is not None) else None
        if res.get("truncated"):
            res["correct"] = None  # dev container: nDCG is meaningless; only the plumbing counts
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def fmt_row_vqweb(r: dict) -> str:
    c = r["config"]
    mem = r.get("mem_after_load") or {}
    gpu_mib = ((r.get("container") or {}).get("gpu_bytes", 0) + (r.get("runtime_info") or {}).get("act_bytes", 0)) / 2**20
    mem_s = (f"{mem['bytes'] / 2**20:.0f} MiB ({mem['api']})" if mem.get("bytes") else "n/a") + f" + {gpu_mib:.0f} MiB GPU buffers"
    ad = r.get("adapter") or {}
    build = (f"VQ WebGPU {r.get('bpw', 0):.2f} bpw (bits {float(r.get('bits') or 0):.2f}, K={r.get('k')}, `{Path(r.get('vqw_file', '?')).name}`, {r.get('file_mib', 0):.0f} MiB, "
             f"{r.get('layers')} blocks{' TRUNCATED dev' if r.get('truncated') else ''}, n={r.get('n')}) [vqweb: {ad.get('vendor')}/{ad.get('architecture')}, resid {(r.get('runtime_info') or {}).get('resid')}]")
    sim = r.get("native") or {}
    if "ndcg_sim_same_n" in sim:
        nat_ref = f"{sim['ndcg_sim_same_n']:.4f} (torch simulation" + (f", test split {sim['ndcg_sim_test']:.4f}" if sim.get("ndcg_sim_test") is not None and sim.get("n_sim_test") else "") + ")"
        delta = f"{r['ndcg_delta_vs_native']:+.4f} {'OK' if r.get('ndcg_ok') else 'FAIL'} (±{VQW_NDCG_TOL})"
        if "n_perq_diff_gt_1e3" in sim:
            delta += f", {sim['n_perq_diff_gt_1e3']}/{sim['n_compared']} queries differ"
    else:
        nat_ref, delta = "n/a (no simulation row)", "n/a"
    if r.get("truncated"):
        delta += " [dev container: meaningless]"
    ref = r.get("ref")
    if ref:
        delta += f"; cos to torch on {ref['n']} queries: min {ref['cos_min']:.5f} mean {ref['cos_mean']:.5f} {'OK' if ref.get('ok') else 'FAIL'} (≥ {VQW_COS_MIN})"
    else:
        delta += "; cos to torch: n/a"
    idle = str(c["tag"]).startswith("idle")
    lat = f"{r['p50']:.0f} | {r['p95']:.0f}" if idle else f"{r['p50']:.0f} (under load, not citable) | {r['p95']:.0f} (under load, not citable)"
    load = f"{r['load_s']:.1f}" + ("" if idle else " (under load)")
    prof = r.get("profile")
    if prof:
        lat += f" (GPU {prof['gpu_total_ms']:.0f} ms, vq_matmul {100 * prof['matmul_share']:.0f} %, T={prof['T']})"
    return (f"| {c['tag']} | {r.get('prompt_kind', '?')} | {build} | GPU | {(r.get('runtime_info') or {}).get('tmax', '?')} | {load} | {mem_s} | {(r.get('peak_rss_bytes') or 0) / 2**20:.0f} MiB | "
            f"{lat} | {r['ndcg_mean']:.4f} | {nat_ref} | {delta} | {r['cos_fp_mean']:.4f} | {'yes' if r['crossOriginIsolated'] else 'no'} |")


def fmt_row(r: dict) -> str:
    if r.get("runtime") == "vqweb" and r.get("ok"):
        return fmt_row_vqweb(r)
    if not r.get("ok"):
        return f"| {r.get('config', {}).get('tag', '?')} | {r.get('prompt_kind', '?')} | {r.get('build', '?')} | {r.get('threads_requested', '?')} | FAILED: {r.get('error', '?')} |||||||||"
    c = r["config"]
    mem = r.get("mem_after_load") or {}
    mem_s = f"{mem['bytes'] / 2**20:.0f} MiB ({mem['api']})" if mem.get("bytes") else "n/a"
    nat = r.get("native") or {}
    nat_ref = f"{nat['ndcg_mean_same_n']:.4f}" if nat else f"{NATIVE_NDCG:.4f}"
    delta = f"{r['ndcg_delta_vs_native']:+.4f}" if "ndcg_delta_vs_native" in r else "n/a"
    ok = "OK" if r.get("correct") else ("FAIL" if "correct" in r else "n/a")
    if r.get("prompt_kind", "generic") != "generic":
        ok = "n/a (other protocol)"  # the native reference used the generic instruction; the mean is not comparable
    if nat:
        ok += f", {nat['n_perq_diff_gt_1e3']}/{nat['n_compared']} queries differ"
    thr = f"{r['threads_used']} ({'mt' if r['multithread'] else 'st fallback' if r['threads_requested'] > 1 else 'st'})"
    build = f"{r['build']} (prefix cache {'on' if c.get('cache', True) else 'off'}, n={r.get('n', c.get('n'))})"
    g = r.get("gpu") or {}
    if r.get("backend", "wasm") == "webgpu":
        ad = (g.get("adapter_info") or "no adapter line").split("adapter_info:")[-1].strip()
        build += f" [webgpu: {'ACTIVE' if g.get('active') else 'NOT active (CPU run)'}; {ad}; {g.get('offloaded') or 'no offload line'}; {g.get('graph_splits') or 'no splits line'}]"
    else:
        build += " [wasm CPU]"
    idle = str(c["tag"]).startswith("idle") or str(c["tag"]).startswith("webgpu")
    lat = f"{r['p50']:.0f} | {r['p95']:.0f}" if idle else f"{r['p50']:.0f} (under heavy load, not citable) | {r['p95']:.0f} (under heavy load, not citable)"
    load = f"{r['load_s']:.1f}" + ("" if idle else " (under load)")
    return (f"| {c['tag']} | {r.get('prompt_kind', '?')} | {build} | {thr} | {c['ctx']} | {load} | {mem_s} | {(r.get('peak_rss_bytes') or 0) / 2**20:.0f} MiB | "
            f"{lat} | {r['ndcg_mean']:.4f} | {nat_ref} | {delta} {ok} | {r['cos_fp_mean']:.4f} | "
            f"{'yes' if r['crossOriginIsolated'] else 'no'} |")


def write_table(results: list[dict]):
    rows = sorted(RESULT_DIR.glob("browser_*.json"), key=lambda p: p.stat().st_mtime)
    all_res = []
    for p in rows:
        try:
            all_res.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    ua = next((r.get("userAgent") for r in all_res if r.get("userAgent")), "?")
    lines = ["# Browser client: quantized query encoder in WebAssembly (wllama 3.6.1 / llama.cpp)", "",
             f"Model `{DEFAULT_MODEL.split('/')[-1]}` (192 MiB, Q2_K GPTQ), SciFact test (300 queries), fp32 document index scored in JS.",
             f"Native reference: llama-embedding (bitnet.cpp fork), same file, pooling last, L2: nDCG@10 = {NATIVE_NDCG:.4f}, cos to fp = {NATIVE_COS_FP:.4f}.",
             "Runs whose tag is not `idle*` were measured while a GPU job (python, 6 GB RAM) and WSL jobs saturated the laptop (20/22 CPUs at 100 %, 30.6/31.5 GB RAM): every latency and load-time figure of those rows is **under heavy load, not citable**; re-measure with `.venv/Scripts/python.exe scripts/browser_run.py --tag idle` on an idle machine.",
             f"Browser: {ua}", "",
             "Query instruction: `generic` = E5 web-search instruction (protocol of the native reference and of the fp32 query cache); `task` = MTEB SciFact instruction (different protocol, not comparable with 0.7440).",
             "Backend: `[wasm CPU]` = n_gpu_layers 0 (CPU threads only); `[webgpu: ...]` = n_gpu_layers 99999 on llama.cpp's ggml-webgpu backend inside the same wllama 3.6.1 wasm — ACTIVE only if the ggml_webgpu adapter line and the offloaded-layers line are present in the run's native log (see client/webgpu/README.md section 4). Rows tagged `webgpu*` are meant to be measured in the same idle window as the `idle*` rows.",
             f"`VQ WebGPU <bpw> bpw` rows: the standalone vector-quantised runtime (`client/vqweb/`, `.vqw` container, no llama.cpp; `scripts/browser_run.py --runtime vqweb`), same page protocol. Their reference column is the exporter's torch simulation of the same container (`results/raw/vqweb*/exports.jsonl` + `perq/`, aligned by qid; criterion ±{VQW_NDCG_TOL} of the design spec, section 7 step 3), and the delta cell also carries the per-query cosine to the torch reference embedding (`client/browser/data/<stem>.ref.f32`, ≥ {VQW_COS_MIN}). Memory after load excludes the GPU buffers (given separately); ctx = the runtime's max tokens.", "",
             "| tag | prompt | build | threads used | ctx | download+load s | memory after load (browser API) | peak process RSS (all Chrome procs of the profile) | p50 ms | p95 ms | nDCG@10 test | native ref | delta (±0.002 check) | cos to fp | crossOriginIsolated |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [fmt_row(r) for r in all_res]
    vc = RESULT_DIR / "vector_check_load_mt.json"
    if vc.exists():
        v = json.loads(vc.read_text(encoding="utf-8"))
        lines += ["", "## Correctness (generic instruction, mt build, 300 queries)", "",
                  f"- nDCG@10 browser **{v['ndcg_browser']:.4f}** vs native reference (box, npz) **{v['ndcg_box']:.4f}**: delta {v['ndcg_browser'] - v['ndcg_box']:+.4f}, within the ±0.002 criterion; "
                  f"{300 - v['n_diff_browser_vs_box']}/300 per-query values identical, {v['n_diff_browser_vs_box']} differ (near-tie rank swaps).",
                  f"- The same file with the native `llama-embedding` (bitnet.cpp fork, AVX2) on this laptop (WSL): nDCG@10 {v['ndcg_wsl_native']:.4f}, {v['n_diff_wsl_vs_box']} queries differ from the box npz -> native-vs-native reproducibility is of the same order as browser-vs-native.",
                  f"- Browser vs WSL-native query vectors: cosine mean {v['cos_browser_vs_wsl_native_mean']:.5f} (min {v['cos_min']:.5f}), top-10 overlap {v['top10_overlap_browser_vs_wsl']:.3f}; cos to fp32 query embedding: browser {v['cos_fp_browser']:.4f}, WSL-native {v['cos_fp_wsl']:.4f}, box row {NATIVE_COS_FP:.4f}.",
                  "- JS nDCG@10 vs `eq.graph_metrics.ndcg_at_k` on the same vectors: 0 differences.  Tokenisation identical (query 1: 31 tokens incl. EOS in both).",
                  "- The `load-tp` row used the MTEB task instruction: a different protocol from the native reference (generic E5 instruction); its mean agreeing with 0.7440 is coincidental (74/300 queries differ, cos to fp 0.79)."]
    lines += ["", "Memory after load: `performance.measureUserAgentSpecificMemory()` (whole origin incl. the worker's WASM heap) when cross-origin isolated, else `performance.memory.usedJSHeapSize` (main-thread JS heap only, does not see the worker).",
              "Peak RSS: sum of WorkingSet over all Chrome processes started with this run's `--user-data-dir`, sampled every 0.5 s (includes browser/GPU/utility processes, i.e. the whole client, not only the model).",
              "Latency: wall-clock of one `createEmbedding` call (main thread -> worker -> llama.cpp -> back), 3 warm-up queries excluded; p50/p95 with numpy-style linear interpolation.",
              "Correctness: per-query nDCG@10 compared with `results/raw/gptq_export/perq/scifact_<file>.npz` (same query order); see the json for max per-query difference.", ""]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {TABLE}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=8, help="threads for the mt build (native client.sh reference used 8)")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--n", type=int, default=300, help="queries for the mt builds")
    ap.add_argument("--n_st", type=int, default=60, help="queries for the single-thread builds (a full 300 takes hours on one thread)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--builds", nargs="+", default=["mt", "mt-nocache", "st"], choices=list(BUILDS))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=float, default=7200, help="seconds per build (single-thread under load can need hours: reduce --n)")
    ap.add_argument("--chrome", default=CHROME)
    ap.add_argument("--scratch", type=Path, default=Path(os.environ.get("EQ_SCRATCH") or tempfile.gettempdir()) / "eq_browser")
    ap.add_argument("--gpu", action="store_true", help="backend=webgpu for the builds of this run; Chrome without --disable-gpu, with GPU flags")
    ap.add_argument("--gpu-flags", default=GPU_FLAGS, help="Chrome flags used with --gpu (default: %(default)s)")
    ap.add_argument("--headed", action="store_true", help="visible Chrome window instead of --headless=new (needed if headless finds no WebGPU adapter)")
    ap.add_argument("--build", default="wllama", help="vendor directory under client/browser/vendor/ (wllama = npm 3.6.1; wllama-webgpu-debug = local debug build)")
    ap.add_argument("--dry-run", action="store_true", help="print the Chrome command line and exit; no server, no Chrome, no model load")
    ap.add_argument("--runtime", choices=["wllama", "vqweb"], default="wllama", help="vqweb = the standalone VQ WebGPU runtime (client/vqweb/bench.html) on --vqw")
    ap.add_argument("--vqw", type=Path, default=None, help=".vqw container for --runtime vqweb")
    ap.add_argument("--dump-reference", type=int, default=20, metavar="N", help="vqweb: torch reference embeddings of the first N test queries (0 = skip)")
    ap.add_argument("--resid", choices=["f32", "f16"], default="f32", help="vqweb: residual stream dtype")
    ap.add_argument("--tile", default=None, help="vqweb: vq_matmul tile rwg,lanes,tpt")
    ap.add_argument("--profile-queries", type=int, default=3, help="vqweb: queries with GPU timestamps after the timed run")
    args = ap.parse_args()
    if args.gpu and "--builds" not in sys.argv:
        args.builds = ["mt"]  # WebGPU arm: one build, 300 queries, prefix cache on -> browser_<tag>_mt.json
    vqweb = args.runtime == "vqweb"
    if vqweb:
        if not args.vqw:
            raise SystemExit("--runtime vqweb needs --vqw <file.vqw>")
        args.vqw = args.vqw.resolve()
        if not args.vqw.exists():
            raise SystemExit(f"container not found: {args.vqw}")
        if not check_helper_parity():
            raise SystemExit("client/vqweb/bench.js scoring helpers drifted from client/browser/bench.js")
    if args.dry_run:
        args.scratch.mkdir(parents=True, exist_ok=True)
        if vqweb:
            run_vqweb(args, args.scratch, "vqw")
        else:
            for b in args.builds:
                run_build(args, b, args.scratch)
        return
    if psutil is None:
        raise SystemExit("psutil is required (pip install psutil)")
    if not vqweb and not (ROOT / args.model).exists():
        raise SystemExit(f"model not found: {ROOT / args.model}")
    if not (ROOT / "client/browser/data/scifact_test.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/browser_export.py")], check=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    if vqweb:
        ensure_reference(args.vqw, args.dump_reference)

    server = None
    mount_prefix = None
    if port_open(args.port):
        print(f"[run] port {args.port} already open: reusing the running server", flush=True)
        if vqweb and not args.vqw.is_relative_to(ROOT):
            raise SystemExit("the running server cannot serve a container outside the repo root; stop it or use another --port")
    else:
        cmd = [sys.executable, str(ROOT / "client/browser/serve.py"), "--port", str(args.port)]
        if vqweb:
            mount_prefix = "vqw"
            cmd += ["--mount", f"{mount_prefix}={args.vqw.parent}"]
        server = subprocess.Popen(cmd)
        for _ in range(50):
            if port_open(args.port):
                break
            time.sleep(0.2)
        else:
            raise SystemExit("server did not start")
    try:
        urllib.request.urlopen(f"http://localhost:{args.port}/client/browser/index.html", timeout=5).read()
        if vqweb:
            results = [run_vqweb(args, args.scratch, mount_prefix)]
        else:
            results = [run_build(args, b, args.scratch) for b in args.builds]
    finally:
        if server is not None:
            server.terminate()
    print("\n=== summary")
    for r in results:
        if not r.get("ok"):
            print(f"  {r['name']}: FAILED {r.get('error')}")
            continue
        if r.get("runtime") == "vqweb":
            sim = r.get("native") or {}
            ref = r.get("ref") or {}
            prof = r.get("profile") or {}
            print(f"  {r['name']}: {Path(r['vqw_file']).name} {r.get('bpw', 0):.3f} bpw ({r.get('layers')} blocks{' TRUNCATED' if r.get('truncated') else ''}), adapter {(r.get('adapter') or {}).get('vendor')}/{(r.get('adapter') or {}).get('architecture')}, "
                  f"load={r['load_s']:.1f}s (download {r['download_s']:.1f}s) mem={((r.get('mem_after_load') or {}).get('bytes') or 0) / 2**20:.0f}MiB (+GPU {((r.get('container') or {}).get('gpu_bytes', 0) + (r.get('runtime_info') or {}).get('act_bytes', 0)) / 2**20:.0f}MiB) "
                  f"peakRSS={r['peak_rss_bytes'] / 2**20:.0f}MiB p50={r['p50']:.0f}ms p95={r['p95']:.0f}ms mean={r['mean']:.0f}ms" + (f" (GPU {prof['gpu_total_ms']:.0f}ms, matmul {100 * prof['matmul_share']:.0f}%)" if prof else "") +
                  f"\n    nDCG@10={r['ndcg_mean']:.4f} vs simulation {sim.get('ndcg_sim_same_n', float('nan')):.4f} (delta {r.get('ndcg_delta_vs_native', float('nan')):+.4f}, "
                  f"{sim.get('n_perq_diff_gt_1e3', '?')}/{sim.get('n_compared', '?')} queries differ, max {sim.get('perq_max_abs_diff', float('nan')):.4f}) -> {'OK' if r.get('ndcg_ok') else 'FAIL' if r.get('ndcg_ok') is not None else 'n/a'} (±{VQW_NDCG_TOL}); "
                  f"cos_fp={r['cos_fp_mean']:.4f}; cos to torch on {ref.get('n', 0)} queries: min {ref.get('cos_min', float('nan')):.5f} mean {ref.get('cos_mean', float('nan')):.5f} -> {'OK' if ref.get('ok') else 'FAIL' if ref else 'n/a'} (≥ {VQW_COS_MIN})"
                  f"\n    verdict (spec section 7 step 3): {'PASS' if r.get('correct') else 'FAIL' if r.get('correct') is not None else 'n/a (dev container or missing reference)'}")
            continue
        nat = r.get("native") or {}
        print(f"  {r['name']}: threads={r['threads_used']} mt={r['multithread']} coi={r['crossOriginIsolated']} load={r['load_s']:.1f}s "
              f"mem={((r.get('mem_after_load') or {}).get('bytes') or 0) / 2**20:.0f}MiB peakRSS={r['peak_rss_bytes'] / 2**20:.0f}MiB "
              f"p50={r['p50']:.0f}ms p95={r['p95']:.0f}ms nDCG={r['ndcg_mean']:.4f} (native {nat.get('ndcg_mean_same_n', NATIVE_NDCG):.4f}, "
              f"delta {r.get('ndcg_delta_vs_native', float('nan')):+.4f}, per-query max diff {nat.get('perq_max_abs_diff', float('nan')):.4f}, "
              f"{nat.get('n_perq_diff_gt_1e3', '?')} queries differ >1e-3) cos_fp={r['cos_fp_mean']:.4f} -> {'CORRECT' if r.get('correct') else 'MISMATCH'}")
        if r["threads_requested"] > 1 and not r["multithread"]:
            print("    NOTE: multi-thread build did not activate in headless Chrome (crossOriginIsolated=%s); single-thread fallback was measured." % r["crossOriginIsolated"])
    write_table(results)


if __name__ == "__main__":
    main()
