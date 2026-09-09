"""Local dev server for the demo (stdlib only): serves demo/ as the site root with the same COOP/COEP headers that
Cloudflare Pages sends from demo/_headers, and mounts ../models at /models so that a local GGUF can be loaded with
`?model=/models/<file>.gguf`.

Correct MIME for .wasm/.js/.f16/.json/.gguf/.vqw/.wgsl, no caching.  No Range support (wllama streams the whole file).

The VQ WebGPU runtime (client/vqweb/: container.js, runtime.js, tokenizer.js, embed.js, kernels/, vendor/) is mounted
read-only at /vqweb/ so that the page can `import('./vqweb/runtime.js')` without a copy; a static host (Cloudflare Pages)
needs the directory copied to demo/vqweb/ instead (demo/README.md, section "M5 / VQ client").

Optional smoke endpoint (used by the headless smoke test only): with --smoke-out <file>, POST /smoke stores the JSON body
in that file (atomically); without the flag the endpoint answers 404 and the page ignores the failure.

Usage: .venv/Scripts/python.exe demo/serve.py [--port 8766] [--models <dir>] [--smoke-out <file>]
Then open http://localhost:8766/?model=/models/harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

DEMO = Path(__file__).resolve().parent
MODELS = DEMO.parent / "models" / "gguf"
VQWEB = DEMO.parent / "client" / "vqweb"   # mounted at /vqweb/ (the page's import base for the VQ runtime)
SMOKE_OUT: Path | None = None
MODELS_DIR: Path = MODELS
ES_BRIDGE = None  # demo/es_bridge.Bridge when started with --es (legal.html: kNN over the full Elasticsearch index + fp32 reference)


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
                      ".f16": "application/octet-stream", ".f32": "application/octet-stream", ".gguf": "application/octet-stream",
                      ".vqw": "application/octet-stream", ".wgsl": "text/plain; charset=utf-8",
                      ".md": "text/markdown; charset=utf-8", "": "application/octet-stream"}

    def translate_path(self, path: str) -> str:
        p = unquote(urlparse(path).path)
        for prefix, base in (("/models/", MODELS_DIR), ("/vqweb/", VQWEB)):
            if p.startswith(prefix):
                rel = p[len(prefix):].replace("\\", "/")
                target = (base / rel).resolve()
                if base.resolve() in target.parents or target == base.resolve():
                    return str(target)
                return str(base / "__forbidden__")
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/es/status":
            if ES_BRIDGE is None:
                self.send_error(404, "Elasticsearch bridge not enabled (start demo/serve.py --es)"); return
            try:
                self._json(ES_BRIDGE.status())
            except Exception as e:  # noqa: BLE001
                self.send_error(502, f"elasticsearch: {e}")
            return
        super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        if u.path in ("/es/knn", "/es/fp"):
            if ES_BRIDGE is None:
                self.send_error(404, "Elasticsearch bridge not enabled (start demo/serve.py --es)"); return
            try:
                q = json.loads(body.decode("utf-8"))
                kw = dict(k=int(q.get("k") or 10), facets=q.get("facets") or None, num_candidates=q.get("num_candidates"), since=q.get("since") or None)
                t = time.time()
                out = ES_BRIDGE.knn(q["vector"], **kw) if u.path == "/es/knn" else ES_BRIDGE.fp(str(q.get("text") or ""), **kw)
                print(f"[es {time.strftime('%H:%M:%S')}] {u.path} k={kw['k']} facets={kw['facets']} -> {len(out['hits'])} hits in {(time.time() - t) * 1000:.0f} ms", flush=True)
                self._json(out)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                self.send_error(502, f"{type(e).__name__}: {e}")
            return
        if u.path == "/smoke" and SMOKE_OUT is not None:
            try:
                doc = json.loads(body.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                self.send_error(400, f"bad json: {e}")
                return
            SMOKE_OUT.parent.mkdir(parents=True, exist_ok=True)
            tmp = SMOKE_OUT.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            os.replace(tmp, SMOKE_OUT)
            print(f"[serve] smoke report -> {SMOKE_OUT}", flush=True)
            self._ok("ok")
        elif u.path == "/log":
            print(f"[page {time.strftime('%H:%M:%S')}] {body.decode('utf-8', errors='replace').rstrip()}", flush=True)
            self._ok("ok")
        else:
            self.send_error(404)

    def _json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _ok(self, text: str):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # only non-200 lines
        if args and str(args[1] if len(args) > 1 else "") not in ("200", "304"):
            sys.stderr.write("[serve] " + fmt % args + "\n")


def main():
    global SMOKE_OUT, MODELS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--models", type=Path, default=MODELS, help="directory mounted at /models")
    ap.add_argument("--smoke-out", type=Path, default=None, help="enable POST /smoke and store the report here")
    ap.add_argument("--es", action="store_true", help="enable the Elasticsearch bridge for legal.html (/es/status, /es/knn, /es/fp); the API key is read "
                    "at run time from the fragmea deployment env file and stays on this server")
    ap.add_argument("--es-index", default="judikatura-v3-texts", help="index profile for the bridge: judikatura-v3-texts (production, 512-d) or nsoud-decision-segments-v1 (1024-d)")
    ap.add_argument("--es-preload", action="store_true", help="with --es: load the fp32 reference model at start instead of on the first /es/fp")
    args = ap.parse_args()
    MODELS_DIR = args.models
    SMOKE_OUT = args.smoke_out
    if args.es:
        global ES_BRIDGE
        sys.path.insert(0, str(DEMO))
        from es_bridge import Bridge
        ES_BRIDGE = Bridge(args.es_index)
        try:
            st = ES_BRIDGE.status()
            print(f"[serve] Elasticsearch bridge: {st['index']} ({st['count']} texts, {len(st['facets'])} kinds, {st['vector_field']} {st['dims']}-d) at {st['host']}", flush=True)
        except Exception as e:  # noqa: BLE001  (Tailscale down etc.: keep serving, the page shows the bridge error and retries on the next request)
            print(f"[serve] Elasticsearch bridge configured for {args.es_index} but not reachable now: {type(e).__name__}; /es/* will retry per request", flush=True)
        if args.es_preload:
            ES_BRIDGE.fp_embed("warm-up"); print("[serve] fp32 reference model loaded", flush=True)
    os.chdir(DEMO)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    print(f"[serve] http://localhost:{args.port}/  (root={DEMO}, /models -> {MODELS_DIR}, /vqweb -> {VQWEB}, COOP/COEP on)", flush=True)
    print(f"[serve] local model: http://localhost:{args.port}/?model=/models/<file>.gguf", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
