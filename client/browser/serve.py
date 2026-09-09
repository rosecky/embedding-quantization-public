"""Static server for the browser client (stdlib only).

Serves the repository root on http://localhost:8765 with the headers that make the page cross-origin isolated
(Cross-Origin-Opener-Policy: same-origin, Cross-Origin-Embedder-Policy: require-corp, Cross-Origin-Resource-Policy:
same-origin) -- SharedArrayBuffer and therefore the multi-thread WASM build need them.  Correct MIME for .wasm/.js/.f32,
no caching (every run re-downloads the model, that is what we measure).  No Range support (wllama streams the whole file).

POST /result  (JSON body)  -> results/raw/browser_local/<name>.json   (name = body.name or ?name=, sanitised)
POST /log     (text body)  -> appended to results/raw/browser_local/<name>.log (progress lines from the page)

Mounts (optional, backwards compatible: none by default): --mount vqw=<dir> serves <dir> under /vqw/ (the VQ WebGPU
bench, scripts/browser_run.py --runtime vqweb, points the page at /vqw/<file>.vqw so containers outside the repo work too).

Usage: .venv/Scripts/python.exe client/browser/serve.py [--port 8765] [--root <repo root>] [--mount vqw=models/vqw]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/raw/browser_local"
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.\-]+")


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript",
                      ".json": "application/json", ".f32": "application/octet-stream", ".gguf": "application/octet-stream",
                      ".vqw": "application/octet-stream", ".wgsl": "text/plain"}
    mounts: dict[str, Path] = {}  # url prefix ("vqw") -> directory, set from --mount

    def translate_path(self, path):
        u = urlparse(path)
        parts = u.path.split("/", 2)  # "", prefix, rest
        if len(parts) >= 2 and parts[1] in self.mounts:
            rest = parts[2] if len(parts) > 2 else ""
            base = self.mounts[parts[1]]
            target = (base / unquote(rest)).resolve() if rest else base
            if target == base or base in target.parents:  # no escape from the mounted directory
                return str(target)
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        q = parse_qs(u.query)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        if u.path == "/result":
            try:
                doc = json.loads(body.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                self.send_error(400, f"bad json: {e}")
                return
            name = SAFE_NAME.sub("_", str(doc.get("name") or (q.get("name") or ["browser_result"])[0]))
            out = RESULT_DIR / f"{name}.json"
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            tmp.replace(out)  # atomic publish: the runner polls for the final file
            self._ok(f"saved {out}")
            print(f"[serve] result -> {out}", flush=True)
        elif u.path == "/log":
            name = SAFE_NAME.sub("_", (q.get("name") or ["browser_result"])[0])
            line = body.decode("utf-8", errors="replace").rstrip("\n")
            with open(RESULT_DIR / f"{name}.log", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
            print(f"[page] {line}", flush=True)
            self._ok("ok")
        else:
            self.send_error(404)

    def _ok(self, text: str):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # keep the console readable: only non-200 and non-static lines
        if args and str(args[1] if len(args) > 1 else "") not in ("200", "304"):
            sys.stderr.write("[serve] " + fmt % args + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--mount", action="append", default=[], metavar="NAME=DIR", help="serve DIR under /NAME/ (repeatable)")
    args = ap.parse_args()
    for m in args.mount:
        name, _, d = m.partition("=")
        if not name or not d:
            raise SystemExit(f"--mount expects NAME=DIR, got {m!r}")
        Handler.mounts[name.strip("/")] = Path(d).resolve()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(args.root)))
    srv.daemon_threads = True
    print(f"[serve] http://localhost:{args.port}/client/browser/index.html  (root={args.root}, COOP/COEP on"
          + "".join(f", /{k}/ -> {v}" for k, v in Handler.mounts.items()) + ")", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
