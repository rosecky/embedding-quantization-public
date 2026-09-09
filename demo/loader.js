// Thinletter demo, shared loader: the DOM-free pieces used by app.js (corpus search in the tab) and legal.js (Czech
// legal corpus over the full Elasticsearch index). Nothing here touches the document; the pages own their elements.
//
//   fetchJSON / fetchBuf            same-origin fetch helpers
//   l2n, percentile, fmt*           maths and formatting (same code as client/browser/bench.js)
//   opfsRoot, opfsCachedFile        the browser's private origin file system (OPFS) where the assembled GGUF is kept
//   assembleFromManifest            byte-chunk manifest (model/<file>.chunks.json) -> one File in OPFS (or a Blob in memory)
//   probeWebGPU, parseGpuLog        the page's view of the WebGPU adapter; GPU evidence from llama.cpp's own log lines
//   newWllama, loadModelSource      wllama instance with the native log kept; download (or cache) + loadModel with timings
//   measureMemory                   performance.measureUserAgentSpecificMemory when cross-origin isolated
import { Wllama } from './vendor/wllama/index.js';

export { Wllama };
export const WLLAMA_VERSION = '3.6.1';
export const WASM_URL = new URL('./vendor/wllama/wllama.wasm', import.meta.url).href;

export const fmtMiB = (b) => `${(b / 2 ** 20).toFixed(b < 10 * 2 ** 20 ? 2 : 0)} MiB`;
export const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`);
export const fmtS = (s) => `${s.toFixed(1)} s`;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- maths ---------------------------------------------------------------------------------------------------
export function l2n(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const inv = s > 0 ? 1 / Math.sqrt(s) : 0;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}
export function percentile(a, p) { // numpy default (linear interpolation)
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  const h = (s.length - 1) * p, lo = Math.floor(h), hi = Math.ceil(h);
  return s[lo] + (s[hi] - s[lo]) * (h - lo);
}

// ---- fetch -----------------------------------------------------------------------------------------------------
export async function fetchJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`); return r.json(); }
export async function fetchBuf(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.arrayBuffer(); }

export async function measureMemory() {
  const out = { api: 'n/a', bytes: null };
  if (self.crossOriginIsolated && performance.measureUserAgentSpecificMemory) {
    try {
      const m = await performance.measureUserAgentSpecificMemory();
      return { api: 'measureUserAgentSpecificMemory', bytes: m.bytes };
    } catch (e) { console.warn('measureUserAgentSpecificMemory failed', e); }
  }
  try { if (performance.memory) return { api: 'performance.memory.usedJSHeapSize (main thread only)', bytes: performance.memory.usedJSHeapSize }; } catch (e) { /* ignore */ }
  return out;
}

// ---- model download: byte chunks -> one File in OPFS (or a Blob in memory) ------------------------------------
export async function opfsRoot() { try { return await navigator.storage.getDirectory(); } catch (e) { return null; } }
export async function opfsCachedFile(name, sizeBytes) { // the assembled file from an earlier visit, or null
  try { const root = await opfsRoot(); if (!root) return null; const f = await (await root.getFileHandle(name)).getFile(); return f.size === sizeBytes ? f : null; } catch (e) { return null; }
}
export async function assembleFromManifest(manifestUrl, onProgress) {
  const manifest = await fetchJSON(manifestUrl);
  const total = manifest.size_bytes, name = manifest.file;
  const root = await opfsRoot();
  if (root) { // cached copy from an earlier visit?
    try { const f = await (await root.getFileHandle(name)).getFile(); if (f.size === total) return { file: f, cached: true, persisted: true, manifest }; } catch (e) { /* not cached */ }
  }
  let fh = null, writable = null;
  if (root) { try { fh = await root.getFileHandle(name, { create: true }); writable = await fh.createWritable(); } catch (e) { console.warn('OPFS write unavailable, keeping the model in memory', e); writable = null; } }
  let loaded = 0;
  const fetchChunk = async (c) => {
    const r = await fetch(new URL(c.url, manifestUrl).href);
    if (!r.ok) throw new Error(`chunk ${c.url}: HTTP ${r.status}`);
    const reader = r.body.getReader(); const bufs = [];
    for (;;) { const { done, value } = await reader.read(); if (done) break; bufs.push(value); loaded += value.length; onProgress(loaded, total); }
    const blob = new Blob(bufs);
    if (blob.size !== c.bytes) throw new Error(`chunk ${c.url}: got ${blob.size} bytes, expected ${c.bytes}`);
    return blob;
  };
  const parts = [], pending = []; let next = 0;
  for (let i = 0; i < manifest.chunks.length; i++) { // up to 3 chunks in flight, written in order
    while (pending.length < 3 && next < manifest.chunks.length) pending.push(fetchChunk(manifest.chunks[next++]));
    const blob = await pending.shift();
    if (writable) await writable.write(blob); else parts.push(blob);
  }
  if (writable) {
    await writable.close();
    const f = await fh.getFile();
    if (f.size !== total) throw new Error(`assembled ${f.size} bytes, expected ${total}`);
    return { file: f, cached: false, persisted: true, manifest };
  }
  const blob = new Blob(parts);
  if (blob.size !== total) throw new Error(`assembled ${blob.size} bytes, expected ${total}`);
  return { file: blob, cached: false, persisted: false, manifest };
}
export async function clearModelCache(fileName, wllama) { // the OPFS copy and wllama's own cache (plain .gguf URLs)
  const root = await opfsRoot();
  if (root && fileName) { try { await root.removeEntry(fileName); } catch (e) { /* none */ } }
  try { await (wllama || new Wllama({ default: WASM_URL })).cacheManager.clear(); } catch (e) { /* none */ }
}

// ---- backend: WebGPU (llama.cpp ggml-webgpu inside the same wasm) or CPU WebAssembly ----------------------------
export async function probeWebGPU() { // the page's view of the adapter; the worker requests it again with default options
  const out = { available: false, reason: null, vendor: null, architecture: null, device: null, description: null, shader_f16: null };
  if (!navigator.gpu) { out.reason = 'navigator.gpu is not available in this browser'; return out; }
  try {
    const ad = await navigator.gpu.requestAdapter();
    if (!ad) { out.reason = 'requestAdapter() returned no adapter'; return out; }
    let info = null;
    try { info = ad.info || (ad.requestAdapterInfo ? await ad.requestAdapterInfo() : null); } catch (e) { /* no info */ }
    Object.assign(out, { vendor: info?.vendor || null, architecture: info?.architecture || null, device: info?.device || null, description: info?.description || null,
      shader_f16: ad.features.has('shader-f16') });
    if (!out.shader_f16) { out.reason = 'adapter has no shader-f16 (ggml-webgpu registers no device)'; return out; }
    out.available = true;
  } catch (e) { out.reason = `requestAdapter failed: ${e && e.message ? e.message : e}`; }
  return out;
}
export function adapterLabel(g) { return [g?.vendor, g?.architecture].filter(Boolean).join(' / ') || g?.description || 'unknown adapter'; }
export function pickBackend(requested, gpuProbe) { return requested === 'wasm' ? 'wasm' : requested === 'webgpu' ? 'webgpu' : (gpuProbe?.available ? 'webgpu' : 'wasm'); }
export function parseGpuLog(log, requested) { // GPU evidence from the llama.cpp INFO lines (client/webgpu/README.md section 4)
  const pick = (re) => log.filter((s) => re.test(s)).map((s) => s.trim());
  const g = { requested, adapter_info: pick(/ggml_webgpu: adapter_info/)[0] || null, offloaded: pick(/offloaded \d+\/\d+ layers to GPU/)[0] || null,
    graph_splits: pick(/graph splits/)[0] || null, model_buffers: pick(/model buffer size/), flash_attn: pick(/Flash Attention/)[0] || null,
    ggml_errors: pick(/ggml_webgpu: (Failed|Device|Error)/) };
  const m = /offloaded (\d+)\/(\d+) layers/.exec(g.offloaded || '');
  g.layers = m ? `${m[1]}/${m[2]}` : null;
  g.active = !!g.adapter_info && !!m && Number(m[1]) > 0;
  g.fallback = requested === 'webgpu' && !g.active;
  return g;
}

// ---- wllama instance + model load ------------------------------------------------------------------------------
export function newWllama(nativeLog) { // llama.cpp's own log lines are kept in nativeLog (first 600) for parseGpuLog
  const keep = (a) => { const t = a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' '); if (nativeLog.length < 600) nativeLog.push(t); };
  return new Wllama({ default: WASM_URL }, { suppressNativeLog: false, logger: { debug: (...a) => keep(a), log: (...a) => { keep(a); console.log(...a); }, warn: (...a) => console.warn(...a), error: (...a) => console.error(...a) } });
}
export const LOAD_OPTS = { n_ctx: 512, n_batch: 512, n_ubatch: 512, embeddings: true, pooling_type: 'last', n_parallel: 1 }; // plus n_threads, n_gpu_layers

/** Download (or take from the cache) and load one model into `wllama`.
 *  source: a plain (or llama.cpp-split) .gguf URL, which wllama downloads and caches itself, or a byte-chunk manifest
 *  (.chunks.json), assembled into OPFS and handed to wllama as one File.  onProgress(loaded, total) during the download,
 *  onStatus(text) for the phases, onCached(size) when no download was needed.  Returns the timings. */
export async function loadModelSource(wllama, source, loadOpts, { onProgress = () => {}, onStatus = () => {}, onCached = () => {}, backend = 'wasm' } = {}) {
  const t0 = performance.now(); let tDown = null, cached = false, persisted = null;
  const prog = (loaded, total) => { onProgress(loaded, total); if (loaded >= total && tDown == null) tDown = performance.now(); };
  onStatus(`fetching ${source.split('/').pop()} …`);
  if (/\.gguf(\?.*)?$/i.test(source)) { // a plain (or llama.cpp-split) GGUF URL: wllama downloads and caches it itself
    try { cached = (await wllama.modelManager.getModels()).some((m) => m.url === source); } catch (e) { /* ignore */ }
    if (cached) onStatus('model found in the browser cache, no download');
    await wllama.loadModelFromUrl(source, { ...loadOpts, useCache: true, progressCallback: ({ loaded, total }) => prog(loaded, total || loaded) });
    if (tDown == null) tDown = performance.now();
    persisted = true;
  } else { // byte-chunk manifest (default): assemble, then hand one File to wllama
    const a = await assembleFromManifest(source, prog);
    cached = a.cached; persisted = a.persisted;
    if (cached) { onStatus('model found in this browser\'s storage (OPFS), no download'); onCached(a.file.size); }
    tDown = performance.now();
    onStatus(backend === 'webgpu' ? 'starting the WebAssembly runtime and loading the weights onto the GPU (WebGPU) …' : 'starting the WebAssembly runtime and loading the weights …');
    await wllama.loadModel([a.file], loadOpts);
  }
  const t1 = performance.now();
  return { cached, persisted, download_s: cached ? 0 : (tDown - t0) / 1000, load_s: (t1 - tDown) / 1000 };
}
