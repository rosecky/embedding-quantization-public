"""Verification helpers shared by the public recipe (scripts/quant_eval_queries.py, scripts/demo_export.py): the cached
teacher embeddings, relevance / self-document masks, and the native llama-embedding encoder with a chunk cache.

Consolidated 2026-09-09 from the research scripts (ablation_matched.load_emb, phase1_oracle.rel_matrix/self_mask/mask_self,
eval_student.l2n, vocab_runtime_eval.run_chunk/encode_runtime) so that the public export does not depend on them.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eq.teacher import emb_dir

SEP = "<#sep#>"


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def load_emb(dataset, key):
    """(doc_ids, corpus.npy, query_ids, queries.npy) of the cached teacher embeddings data/emb/<dataset>/<key>/."""
    p = emb_dir(dataset, key)
    ids = json.loads((p / "ids.json").read_text())
    return ids["doc_ids"], np.load(p / "corpus.npy"), ids["query_ids"], np.load(p / "queries.npy")


def rel_matrix(qids, doc_index, qrels, N):
    R = np.zeros((len(qids), N), dtype=np.int16)
    for i, q in enumerate(qids):
        for d, g in qrels.get(q, {}).items():
            j = doc_index.get(d)
            if j is not None and g > 0:
                R[i, j] = g
    return R


def self_mask(qids, doc_index, N):
    """Boolean (n_q, N) mask of the query's OWN document (same id in corpus and queries: ArguAna, where 1294/1401 test
    queries are verbatim corpus documents and never relevant to themselves). MTEB drops these hits
    (ignore_identical_ids); without it every system loses ~0.19 nDCG@10 on ArguAna. Empty for other datasets."""
    M = np.zeros((len(qids), N), dtype=bool)
    for i, q in enumerate(qids):
        j = doc_index.get(q)
        if j is not None:
            M[i, j] = True
    return M


def mask_self(S, M):
    """Scores with the query's own document removed from the ranking (no-op when M has no True)."""
    if M is None or not M.any():
        return S
    S = np.array(S, dtype=np.float32, copy=True)
    S[M] = -np.inf
    return S


def run_chunk(binary: Path, gguf: Path, texts: list[str], out_npy: Path, threads: int, ctx: int, pooling: str,
              log_path: Path) -> dict:
    """Encode one chunk of texts with llama-embedding into out_npy (atomic; a .lock lets parallel evals share the cache)."""
    if out_npy.exists():
        return dict(cached=True)
    lock = out_npy.with_suffix(".lock")
    try:  # several eval processes may share a chunk cache: first one takes the lock, others wait for the .npy
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        while not out_npy.exists():
            time.sleep(10)
        size = -1
        while out_npy.stat().st_size != size:  # wait until the writer is done (non-atomic writers)
            size = out_npy.stat().st_size
            time.sleep(2)
        return dict(cached=True)
    prompt_file = out_npy.with_suffix(".txt")
    with open(prompt_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(SEP.join(t.replace(SEP, " ") for t in texts))
    cmd = [str(binary), "-m", str(gguf), "-f", str(prompt_file), "--embd-separator", SEP, "--pooling", pooling,
           "--embd-normalize", "2", "--embd-output-format", "json", "-ngl", "0", "-t", str(threads),
           "-c", str(ctx), "-b", str(ctx), "--no-warmup"]
    t0 = time.perf_counter()
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    p = subprocess.run(cmd, capture_output=True, timeout=7200, env=env)
    dt = time.perf_counter() - t0
    out = p.stdout.decode("utf-8", errors="replace")
    err = p.stderr.decode("utf-8", errors="replace")
    log_path.write_text(err[-4000:], encoding="utf-8")
    if p.returncode != 0:
        lock.unlink(missing_ok=True)  # never leave a stale lock behind: a later process would wait for the .npy forever
        raise RuntimeError(f"llama-embedding failed rc={p.returncode} for {prompt_file}: {err[-1500:]}")
    doc = json.loads(out[out.index("{"): out.rindex("}") + 1])
    data = sorted(doc["data"], key=lambda d: d["index"])
    emb = np.array([d["embedding"] for d in data], dtype=np.float32)
    if emb.shape[0] != len(texts):
        raise RuntimeError(f"expected {len(texts)} embeddings, got {emb.shape[0]} for {prompt_file}")
    tmp = out_npy.with_name(out_npy.stem + ".tmp.npy")
    np.save(tmp, emb)
    os.replace(tmp, out_npy)  # atomic publish
    lock.unlink(missing_ok=True)
    prompt_file.unlink(missing_ok=True)
    m = re.search(r"prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens", err)
    return dict(cached=False, wall_s=dt, n_texts=len(texts),
                prompt_eval_ms=float(m.group(1)) if m else None, tokens=int(m.group(2)) if m else None)


def encode_runtime(binary, gguf, texts, work: Path, tag: str, threads, workers, chunk, ctx, pooling):
    """Encode texts with llama-embedding in chunks (cached under work/<tag>_NNNN.npy); returns (embeddings, stats).
    A stale <tag>_NNNN.lock from a crashed run makes the caller wait forever: delete it before re-running."""
    work.mkdir(parents=True, exist_ok=True)
    jobs = [(i, texts[s: s + chunk]) for i, s in enumerate(range(0, len(texts), chunk))]
    t0 = time.perf_counter()

    def one(job):
        i, tx = job
        r = run_chunk(binary, gguf, tx, work / f"{tag}_{i:04d}.npy", threads, ctx, pooling, work / f"{tag}_{i:04d}.log")
        done = sum(1 for j, _ in jobs if (work / f"{tag}_{j:04d}.npy").exists())
        status = "cached" if r.get("cached") else f"{r['wall_s']:.0f}s"
        print(f"    [{tag}] chunk {i + 1}/{len(jobs)} {status} "
              f"({done}/{len(jobs)} done, {time.perf_counter() - t0:.0f}s elapsed)", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        stats = list(ex.map(one, jobs))
    emb = np.concatenate([np.load(work / f"{tag}_{i:04d}.npy") for i, _ in jobs], axis=0)
    assert emb.shape[0] == len(texts)
    tok = [s["tokens"] for s in stats if s.get("tokens")]
    ms = [s["prompt_eval_ms"] for s in stats if s.get("prompt_eval_ms")]
    return emb, dict(n_chunks=len(jobs), chunk=chunk, threads=threads, workers=workers, ctx=ctx,
                     wall_s=time.perf_counter() - t0, runtime_tokens=int(sum(tok)) if tok else None,
                     runtime_tok_per_s_per_process=(sum(tok) / (sum(ms) / 1e3)) if tok and ms else None)
