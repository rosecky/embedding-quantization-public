// Thinletter demo: the query encoder of harrier-0.6b runs in the browser and searches the UNCHANGED document index of the
// original model (fp16 on disk, fp32 in memory).  Two kinds of client per corpus (M5):
//   runtime "wllama": the scalar Q2_K GPTQ GGUF, llama.cpp -> WebAssembly via wllama 3.6.1 (WebGPU through llama.cpp's own
//                     backend where the browser offers shader-f16, CPU WebAssembly otherwise);
//   runtime "vqweb":  the vector-quantised .vqw container, standalone WebGPU runtime (client/vqweb/, served at ./vqweb/:
//                     WGSL kernels, no WebAssembly; needs navigator.gpu with shader-f16).
//
// Protocol (identical to client/browser/bench.js, client/vqweb/bench.js and the native reference scripts/quant_eval_queries.py):
//   wllama: text = meta.prompt + query (generic E5 web-search instruction; the tokenizer adds EOS, no BOS: GGUF metadata);
//   vqweb:  the RAW query goes to VqwTokenizer.encode(), which applies the container header's prompt / add_eos and the
//           trimmed-vocabulary byte fallback (the header prompt is checked against meta.prompt at load);
//   pooling "last", L2-normalised embedding, cosine = dot product against unit-norm document rows,
//   nDCG@10 with linear gains, discount 1/log2(rank+1), IDCG from the sorted grades (eq.graph_metrics.ndcg_at_k).
//
// Corpora: data/index.json lists them; ds=<id> (URL parameter or the dropdown) selects data/<id>/meta.json, whose
// index is fp16 or row-scaled int8 shards (x = q * scale[row]) and whose documents come in JSON shards.
// Clients: meta.clients = [{id, label, runtime, model_url, size_bytes, bpw, ndcg_test, ndcg_browser, note, ...}] (written by
// demo/vq_clients.py; zero or more vqweb entries per corpus); client=<id> (URL parameter or the dropdown) selects one.
// Without `clients` (older export) the scalar client is synthesised from the top-level model_file / model_url fields.
// URL parameters: ds=<corpus id> (default index.json "default"), client=<client id> (default meta.default_client or the first
// entry), model=<url .gguf | .vqw | url .chunks.json> (overrides the selected client's model_url), threads=<n>, autoload=1,
// autoquery=<text>, verify_n=<n>, autoverify=1, smoke=1 (headless test: autoload + autoquery + verify_n queries,
// report to console as [smoke] lines and POSTed to ./smoke; nothing else changes),
// backend=auto|webgpu|wasm (wllama only; default auto: llama.cpp's WebGPU backend inside the same wasm when navigator.gpu
// gives an adapter with shader-f16, else CPU WebAssembly; n_gpu_layers 99999 vs 0, see client/webgpu/README.md).
// Shared, DOM-free pieces (fetch, maths, OPFS assembly, WebGPU probe, wllama load) live in loader.js: also used by legal.js.
import { Wllama, WLLAMA_VERSION, fmtMiB, fmtMs, fmtS, sleep, percentile, l2n, fetchJSON, fetchBuf, measureMemory, opfsRoot,
  clearModelCache, probeWebGPU, adapterLabel, parseGpuLog, newWllama, loadModelSource, assembleFromManifest, LOAD_OPTS } from './loader.js';

const params = new URLSearchParams(location.search);
const opt = {
  ds: params.get('ds'),
  client: params.get('client'),
  model: params.get('model'),
  threads: params.has('threads') ? Math.max(1, parseInt(params.get('threads'), 10) || 1) : null,
  autoload: params.get('autoload') === '1' || params.get('smoke') === '1',
  autoquery: params.get('autoquery'),
  autoverify: params.get('autoverify') === '1' || params.get('smoke') === '1',
  verifyN: params.has('verify_n') ? parseInt(params.get('verify_n'), 10) : (params.get('smoke') === '1' ? 20 : 300),
  smoke: params.get('smoke') === '1',
  backend: ['webgpu', 'wasm', 'auto'].includes(params.get('backend')) ? params.get('backend') : 'auto',
};
// the VQ runtime modules (client/vqweb/) as served by serve.py's /vqweb/ mount (or a copy at demo/vqweb/ on a static host);
// imported lazily, so the scalar client never depends on them
const VQWEB_BASE = new URL('./vqweb/', location.href);

const $ = (id) => document.getElementById(id);
const smokeLog = (msg, data) => {
  if (!opt.smoke) return;
  console.log(`[smoke] ${msg}${data !== undefined ? ' ' + JSON.stringify(data) : ''}`);
};
function status(msg, warn = false) {
  $('load-status').textContent = msg;
  $('load-status').className = warn ? 'warn-text' : 'muted';
  console.log(msg);
}
const signed = (x) => x.toFixed(4).replace(/^(?!-)/, '+');

// ---- maths (same code as client/browser/bench.js) -------------------------------------------------------------
function dot(a, aOff, b, bOff, d) {
  let s = 0;
  for (let i = 0; i < d; i++) s += a[aOff + i] * b[bOff + i];
  return s;
}
function topk(scores, k) { // indices sorted by descending score (ties: lower index first)
  const idx = new Int32Array(k).fill(-1);
  const val = new Float64Array(k).fill(-Infinity);
  for (let j = 0; j < scores.length; j++) {
    const s = scores[j];
    if (s <= val[k - 1]) continue;
    let p = k - 1;
    while (p > 0 && val[p - 1] < s) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; }
    val[p] = s; idx[p] = j;
  }
  return idx;
}
function ndcgAtK(top, relByIdx, k) {
  let dcg = 0;
  for (let i = 0; i < k; i++) dcg += (relByIdx.get(top[i]) || 0) / Math.log2(i + 2);
  const grades = [...relByIdx.values()].sort((a, b) => b - a).slice(0, k);
  let idcg = 0;
  for (let i = 0; i < grades.length; i++) idcg += grades[i] / Math.log2(i + 2);
  return idcg > 0 ? dcg / idcg : NaN;
}
function nanmean(a) { const v = a.filter((x) => !Number.isNaN(x)); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN; }
function f16ToF32(u16) {
  if (typeof Float16Array !== 'undefined') return new Float32Array(new Float16Array(u16.buffer, u16.byteOffset, u16.length));
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    table[h] = e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
  }
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = table[u16[i]];
  return out;
}

// ---- state ---------------------------------------------------------------------------------------------------
const S = {
  ds: null, dataDir: null, registry: null,
  meta: null, docs: null, corpus: null, dim: 0, nDocs: 0, docIndex: null, queries: null, queryByText: null,
  clients: [], client: null, ref: null,          // client = the selected entry of meta.clients; ref = its reference numbers (refFor)
  wllama: null, vq: null,                         // vq = {device, info, container, tokenizer, rt, ...} of the vqweb runtime
  loaded: false, loading: false, threads: null, multithread: null, libllama: null,
  backend: null, gpuProbe: null, gpu: null, nativeLog: [],
  timing: { download_s: null, load_s: null, cached: null, persisted: null }, memory: null, gpuBytes: null,
  encodeMs: [], searchMs: [], scores: null, verifying: false, abort: false, lastReport: null, modelSource: null,
};

// ---- data ----------------------------------------------------------------------------------------------------
function selectCorpus(id) { const u = new URL(location.href); u.searchParams.set('ds', id); u.searchParams.delete('client'); location.href = u.href; }
function clientsOf(m) { // meta.clients, or (older export without it) the scalar client from the top-level fields
  if (Array.isArray(m.clients) && m.clients.length) return m.clients;
  const typ = m.quantization?.type || 'scalar';
  return [{ id: 'scalar', label: `${typ} scalar, ${fmtMiB(m.size_bytes || 0)} (llama.cpp WASM/WebGPU)`, runtime: 'wllama', model_url: m.model_url, model_file: m.file,
    size_bytes: m.size_bytes, sha256: m.model_sha256, ndcg_test: m.reference?.native_ndcg10 ?? null, ndcg_browser: m.reference?.browser_ndcg10 ?? null, note: null }];
}
/** The reference numbers of a client for the verify table: the scalar client's are the native llama.cpp run of the same
 *  file (meta.reference), a VQ client's the torch simulation of the same file (clients[].ndcg_test) and our browser run. */
function refFor(c) {
  const ref = S.meta.reference || {};
  if (c.runtime === 'vqweb') {
    return { kind: 'simulation', label: 'Torch simulation, same file', test: c.ndcg_test ?? null, testPQ: c.ndcg_test_per_query || null, testLat: null,
      browser: c.ndcg_browser ?? null, browserPQ: c.ndcg_browser_per_query || null, browserNote: c.ndcg_browser_run ? `${adapterLabel(c.ndcg_browser_run.adapter)}, p50 ${Math.round(c.ndcg_browser_run.p50_ms)} ms under load` : 'not measured yet',
      fp16: ref.fp16_ndcg10 ?? null, sameNText: 'the torch simulation of this file on the same' };
  }
  return { kind: 'native', label: 'Native llama.cpp, same file', test: c.ndcg_test ?? ref.native_ndcg10 ?? null, testPQ: ref.native_ndcg10_per_query || null, testLat: ref.native_latency_ms ?? null,
    browser: c.ndcg_browser ?? ref.browser_ndcg10 ?? null, browserPQ: null, browserNote: 'under load, see README', fp16: ref.fp16_ndcg10 ?? null, sameNText: 'native llama.cpp on the same' };
}
function vqUnavailableReason() { return S.gpuProbe ? (S.gpuProbe.available ? null : (S.gpuProbe.reason || 'WebGPU not available')) : null; }
function renderClientSelect() {
  const sel = $('client-select');
  sel.replaceChildren(...S.clients.map((c) => {
    const o = document.createElement('option'); o.value = c.id;
    const why = c.runtime === 'vqweb' ? vqUnavailableReason() : null;
    o.textContent = c.label + (why ? ' — unavailable here' : ''); o.disabled = !!why; o.selected = c.id === S.client.id; o.title = c.note || '';
    return o;
  }));
  sel.disabled = S.clients.length < 2;
  const why = S.client.runtime === 'vqweb' ? vqUnavailableReason() : null;
  const note = $('client-note');
  if (why) { note.textContent = `The VQ client needs WebGPU with shader-f16 (standalone runtime, no CPU fallback): ${why}. Choose the scalar client.`; note.className = 'notice warn'; note.hidden = false; }
  else if (S.client.note) { note.textContent = S.client.note; note.className = 'muted small'; note.hidden = false; }
  else note.hidden = true;
}
function applyClient(c) { // everything on the page that depends on the selected client (before load)
  S.client = c; S.ref = refFor(c);
  const m = S.meta, r = S.ref;
  S.modelSource = new URL(opt.model || c.model_url, location.href).href;
  const sizeTxt = c.size_bytes ? fmtMiB(c.size_bytes) : '? MiB';
  document.title = `Thinletter demo: ${m.label} search with a ${sizeTxt} query encoder in your browser`;
  $('size-btn').textContent = sizeTxt; $('size-lead').textContent = sizeTxt; $('st-size').textContent = `${sizeTxt} (${c.model_file || c.model_url.split('/').pop()})`;
  if (c.runtime === 'vqweb') {
    $('lead-how').replaceChildren(`compressed by vector quantisation (GPTVQ-lite: 4-dimensional codebooks, ${c.bpw ? c.bpw.toFixed(2) : '?'} bits per weight, input-side rotation) into one `,
      Object.assign(document.createElement('span'), { id: 'size-lead', textContent: sizeTxt }), ' file, executed by a standalone WebGPU runtime (WGSL kernels, no WebAssembly) on your GPU');
    $('st-load-label').textContent = 'Load (GPU upload + tokenizer + pipelines)';
    $('ref-native-p').hidden = true;
    $('foot-runtime').textContent = 'client/vqweb (standalone WebGPU runtime, tokenizer @huggingface/tokenizers 0.2.0)';
  } else {
    $('lead-how').replaceChildren(`compressed by GPTQ onto llama.cpp's ${m.quantization?.type || 'K-quant'} grid into one `,
      Object.assign(document.createElement('span'), { id: 'size-lead', textContent: sizeTxt }), ' file, executed by llama.cpp compiled to WebAssembly, on your GPU through WebGPU where the browser offers it and on the CPU otherwise');
    $('st-load-label').textContent = 'Load (wasm + weights)';
    $('ref-native-p').hidden = false;
  }
  $('v-native-label').textContent = r.label;
  $('v-native').textContent = r.test?.toFixed(4) ?? '–';
  $('v-native-lat').textContent = r.testLat ? `${r.testLat} ms` : '–';
  $('v-browser').textContent = r.browser?.toFixed(4) ?? '–';
  $('v-browser-note').textContent = r.browserNote;
  $('v-fp16').textContent = r.fp16?.toFixed(4) ?? '–';
  $('v-browser-delta').textContent = r.browser != null && r.test != null ? signed(r.browser - r.test) : '–';
  $('v-fp16-delta').textContent = r.fp16 != null && r.test != null ? signed(r.fp16 - r.test) : '–';
  renderClientSelect();
  if (S.gpuProbe) renderBackendChoice();
}
function onClientChange(id) {
  const c = S.clients.find((x) => x.id === id);
  if (!c || c.id === S.client.id) return;
  const u = new URL(location.href); u.searchParams.set('client', id);
  if (S.loaded || S.loading) { location.href = u.href; return; }   // a loaded runtime is not swapped in place: reload with client=
  history.replaceState(null, '', u.href);
  applyClient(c);
  status(`index ready (${S.nDocs} documents). Click "Load model" to fetch the ${fmtMiB(c.size_bytes || 0)} query encoder.`);
  cachedHint();
}
async function cachedHint() {
  try { const root = await opfsRoot(); if (root && S.client.model_file) { const f = await (await root.getFileHandle(S.client.model_file)).getFile(); if (f.size === S.client.size_bytes) status($('load-status').textContent.replace(/\.?$/, '') + ' (a cached copy is in this browser: no download).'); } } catch (e) { /* not cached */ }
}
async function loadMeta() {
  S.registry = await fetchJSON('./data/index.json');
  const corpora = S.registry.corpora || [];
  S.ds = opt.ds || S.registry.default || (corpora[0] && corpora[0].id);
  if (!corpora.some((c) => c.id === S.ds)) throw new Error(`unknown corpus "${S.ds}" (known: ${corpora.map((c) => c.id).join(', ')})`);
  S.dataDir = `./data/${S.ds}/`;
  const sel = $('ds-select');
  sel.replaceChildren(...corpora.map((c) => { const o = document.createElement('option'); o.value = c.id; o.textContent = `${c.label} (${c.n_docs.toLocaleString('en-US')} docs, ${fmtMiB(c.size_bytes || 0)} client${c.clients && c.clients.length > 1 ? `, ${c.clients.length} clients` : ''})`; o.selected = c.id === S.ds; return o; }));
  sel.disabled = corpora.length < 2; sel.onchange = () => { if (sel.value !== S.ds) selectCorpus(sel.value); };
  S.meta = await fetchJSON(S.dataDir + 'meta.json');
  const m = S.meta, ref = m.reference || {};
  const nDocs = m.index.n_docs.toLocaleString('en-US'), nQ = String(m.n_queries);
  $('h1-ds').textContent = m.label; $('lead-index').textContent = `${nDocs} ${m.corpus_description || m.label + ' documents'}`;
  $('st-ndocs').textContent = nDocs; $('v-n').textContent = nQ; $('v-n2').textContent = nQ; $('v-n-btn').textContent = nQ; $('v-ds').textContent = m.label;
  $('foot-license').textContent = m.dataset_license || '?';
  if (m.dataset_url) { $('foot-data').href = m.dataset_url; $('foot-data').textContent = m.dataset_name || m.label; }
  else if (m.dataset_name) { $('foot-data').removeAttribute('href'); $('foot-data').textContent = m.dataset_name; }
  // per-teacher base model + licence (meta.json base_model_url / base_model_license) and the per-corpus notice (meta.json corpus_notice)
  if (m.base_model_url) { $('foot-base').href = m.base_model_url; $('foot-base').textContent = m.base_model_url.replace(/^https?:\/\/huggingface\.co\//, ''); }
  if (m.base_model_license) { $('foot-base-license').textContent = m.base_model_license.replace(/\s*\(.*\)\s*$/, ''); }
  if (m.corpus_notice) { $('foot-corpus').textContent = `Corpus: ${m.corpus_notice}`; $('foot-corpus').hidden = false; }
  if (m.model_short) { $('lead-model').textContent = m.model_short; }
  if (ref.native_latency_ms) { $('ref-native').textContent = `${ref.native_latency_ms} ms per query, ${ref.native_peak_rss_mib ?? '?'} MiB peak RSS`; }
  $('meta-date').textContent = m.date ? `Data exported ${m.date}.` : '';
  $('st-coi').textContent = self.crossOriginIsolated ? 'yes (multi-threaded wasm possible)' : 'no';
  $('warn-threads').hidden = !!self.crossOriginIsolated;
  try { $('libllama').textContent = Wllama.getLibllamaVersion(); } catch (e) { /* keep */ }
  // the client: client=<id>, else meta.default_client, else the first entry
  S.clients = clientsOf(m);
  const want = opt.client || m.default_client;
  const c = S.clients.find((x) => x.id === want) || S.clients[0];
  if (opt.client && c.id !== opt.client) console.warn(`unknown client "${opt.client}" (known: ${S.clients.map((x) => x.id).join(', ')}); using ${c.id}`);
  $('client-select').onchange = (e) => onClientChange(e.target.value);
  applyClient(c);
}
async function loadIndex() {
  const m = S.meta, D = S.dataDir, idx = m.index;
  const docFiles = m.docs_files ? m.docs_files.map((f) => f.url) : ['docs.json'];
  const idxFiles = idx.files ? idx.files.map((f) => f.url) : ['corpus.f16'];
  const [docParts, queries, bufs, scaleBuf] = await Promise.all([
    Promise.all(docFiles.map((f) => fetchJSON(D + f))), fetchJSON(D + 'test_queries.json'),
    Promise.all(idxFiles.map((f) => fetchBuf(D + f))), idx.dtype_on_disk === 'int8_rowscale' ? fetchBuf(D + idx.scale_file) : null,
  ]);
  const docs = docParts.flat();
  S.docs = docs; S.queries = queries; S.dim = idx.dim; S.nDocs = docs.length;
  const t = performance.now();
  const total = bufs.reduce((n, b) => n + b.byteLength, 0);
  if (idx.dtype_on_disk === 'int8_rowscale') { // x = q * scale[row]
    const q = new Int8Array(total); let o = 0; for (const b of bufs) { q.set(new Int8Array(b), o); o += b.byteLength; }
    const scale = new Float32Array(scaleBuf), d = S.dim, out = new Float32Array(q.length);
    for (let j = 0, k = 0; j < scale.length; j++) { const sc = scale[j]; for (let i = 0; i < d; i++, k++) out[k] = q[k] * sc; }
    S.corpus = out;
  } else {
    const u16 = new Uint16Array(total / 2); let o = 0; for (const b of bufs) { u16.set(new Uint16Array(b), o); o += b.byteLength / 2; }
    S.corpus = f16ToF32(u16);
  }
  if (S.corpus.length !== S.nDocs * S.dim) throw new Error(`index size mismatch: ${S.corpus.length} != ${S.nDocs}x${S.dim}`);
  S.scores = new Float64Array(S.nDocs);
  S.docIndex = new Map(docs.map((d, i) => [d.id, i]));
  S.queryByText = new Map(queries.map((q) => [q.text, q]));
  console.log(`index ${S.ds}: ${S.nDocs} docs x ${S.dim}, ${idx.dtype_on_disk} -> fp32 in ${(performance.now() - t).toFixed(0)} ms; ${queries.length} test queries`);
  // three example queries (deterministic pick)
  const ex = [7, 42, 133].filter((i) => i < queries.length).map((i) => queries[i]);
  $('examples').replaceChildren(...ex.map((q) => { const a = document.createElement('a'); a.href = '#'; a.textContent = `"${q.text}"`; a.onclick = (e) => { e.preventDefault(); $('q').value = q.text; if (S.loaded) search(q.text); }; return a; }));
}

async function clearCache() {
  await clearModelCache(S.client?.model_file, S.wllama || undefined);
  status('cached model removed; reload the page to download again');
}

// ---- backend: WebGPU (llama.cpp ggml-webgpu inside the same wasm) or CPU WebAssembly; vqweb: WebGPU only -------------
function backendUrl(b) { const u = new URL(location.href); u.searchParams.set('backend', b); return u.href; }
function renderBackendChoice() { // before load: what auto will pick, plus the link to reload with the other backend
  const g = S.gpuProbe, el = $('backend-choice');
  const avail = g?.available ? `WebGPU available: ${adapterLabel(g)} (shader-f16)` : `WebGPU not available: ${g?.reason || 'unknown'}`;
  if (S.client.runtime === 'vqweb') {
    el.replaceChildren(`${avail}. The VQ client runs on the GPU through the standalone vqweb runtime (no WebAssembly, no CPU fallback).`);
    $('btn-load').disabled = !g?.available;
    return g?.available ? 'webgpu' : null;
  }
  const pick = opt.backend === 'wasm' ? 'wasm' : opt.backend === 'webgpu' ? 'webgpu' : (g?.available ? 'webgpu' : 'wasm');
  const chosen = pick === 'webgpu' ? 'the GPU through WebGPU' : `CPU WebAssembly (${Math.min(8, navigator.hardwareConcurrency || 1)} threads)`;
  el.replaceChildren(`${avail}. Will run on ${chosen}${opt.backend === 'auto' ? ' (auto)' : ` (backend=${opt.backend})`}. `);
  const other = pick === 'webgpu' ? 'wasm' : 'webgpu';
  const a = document.createElement('a'); a.href = backendUrl(other); a.textContent = other === 'wasm' ? 'Use CPU WebAssembly instead' : 'Try WebGPU instead';
  el.append(a);
  $('btn-load').disabled = S.loaded || S.loading;
  return pick;
}
function renderBackendStats() {
  const g = S.gpu, el = $('st-backend');
  const tog = document.createElement('a'); tog.className = 'small';
  if (S.client.runtime === 'vqweb') {
    const v = S.vq, c = v.container;
    el.textContent = `WebGPU (vqweb standalone runtime): ${adapterLabel(v.info)}; ${c.model.layers} blocks on the GPU, K=${c.quant.k} (${c.quant.bpw_blocks?.toFixed(2)} bpw), ` +
      `upload ${fmtS(v.upload_s)} + tokenizer ${fmtS(v.tokenizer_s)} + pipelines ${fmtS(v.pipelines_s)}`;
    $('st-threads').textContent = 'n/a (GPU runtime; tokenizer and token table on the main thread)';
    $('warn-gpu-fallback').hidden = true;
    return;
  }
  if (g.active) {
    const ad = /vendor: ([^|]+)\|\s*architecture: ([^|]+)/.exec(g.adapter_info || '');
    const name = ad ? `${ad[1].trim()} / ${ad[2].trim()}` : adapterLabel(S.gpuProbe);
    const splits = /graph splits = (\d+)/.exec(g.graph_splits || '');
    el.textContent = `WebGPU: ${name}, ${g.layers} layers offloaded${splits ? `, graph splits ${splits[1]}` : ''}; CPU ${S.threads} thread${S.threads > 1 ? 's' : ''} for the rest `;
    tog.href = backendUrl('wasm'); tog.textContent = 'reload with CPU WebAssembly';
  } else {
    el.textContent = `${g.fallback ? 'WebGPU requested, fell back to CPU: ' : ''}WebAssembly, ${S.threads} thread${S.threads > 1 ? 's' : ''} (${S.multithread ? 'multi-threaded' : 'single-threaded'}) `;
    tog.href = backendUrl('webgpu'); tog.textContent = g.fallback ? 'retry WebGPU' : 'reload with WebGPU';
  }
  el.append(tog);
  $('warn-gpu-fallback').hidden = !g.fallback;
}

// ---- model load ----------------------------------------------------------------------------------------------
function progressUI() {
  const bar = $('progress-bar'), txt = $('progress-text'); $('progress-wrap').hidden = false;
  const t0 = performance.now(); let lastPaint = 0;
  return {
    onProgress: (loaded, total) => {
      const now = performance.now(); if (now - lastPaint < 100 && loaded < total) return; lastPaint = now;
      const dt = (now - t0) / 1000, speed = dt > 0 ? loaded / dt : 0;
      bar.style.width = `${(100 * loaded / total).toFixed(1)}%`;
      txt.textContent = `${fmtMiB(loaded)} / ${fmtMiB(total)}  (${(speed / 2 ** 20).toFixed(1)} MiB/s)`;
    },
    onCached: (size) => { bar.style.width = '100%'; txt.textContent = `${fmtMiB(size)} from cache`; },
  };
}
async function loadModel() {
  if (S.loaded || S.loading || !S.meta) return;
  const lowMem = navigator.deviceMemory && navigator.deviceMemory < 4;
  if (lowMem && S.client.runtime !== 'vqweb' && !opt.autoload && !window.confirm('This device reports less than 4 GB of memory; the model needs about 1.2 GB in this tab. Load anyway?')) return;
  S.loading = true;
  $('btn-load').disabled = true; $('client-select').disabled = true;
  if (!S.gpuProbe) S.gpuProbe = await probeWebGPU();
  try {
    if (S.client.runtime === 'vqweb') await loadVqweb(); else await loadWllama();
    S.loaded = true;
    $('q').disabled = false; $('btn-search').disabled = false; $('btn-verify').disabled = false; $('q').placeholder = S.meta.query_hint || 'Type a query';
    $('q').focus();
    $('st-memory').textContent = 'measuring …';
    S.memoryPromise = measureMemory().then((m) => {
      S.memory = m;
      const js = m.bytes != null ? `${fmtMiB(m.bytes)} (${m.api})` : 'n/a (API not available)';
      $('st-memory').textContent = S.client.runtime === 'vqweb' ? `${js} + ${fmtMiB(S.gpuBytes)} GPU buffers (weights ${fmtMiB(S.vq.container.gpuBytes)} + activations ${fmtMiB(S.vq.rt.actBytes)}; not visible to the tab's measurement)`
        : js + (S.gpu.active ? '; with WebGPU the weights live in GPU memory: the tab measures lower, the whole process uses more' : '');
    });
    smokeLog('loaded', { client: S.client.id, runtime: S.client.runtime, backend: S.backend, backend_requested: opt.backend, gpu: S.gpu, page_adapter: S.gpuProbe, threads: S.threads, multithread: S.multithread,
      crossOriginIsolated: !!self.crossOriginIsolated, cached: S.timing.cached, download_s: S.timing.download_s, load_s: S.timing.load_s, gpu_bytes: S.gpuBytes, vq: S.vq ? vqInfo() : null });
  } catch (e) {
    console.error(e);
    status(`load failed: ${e && e.message ? e.message : e}`, true);
    $('btn-load').disabled = false;
    smokeLog('load_failed', { client: S.client.id, error: String(e && e.message ? e.message : e) });
    throw e;
  } finally { S.loading = false; $('client-select').disabled = S.clients.length < 2; }
}
async function loadWllama() { // the scalar client: chunk manifest -> OPFS -> wllama (loader.loadModelSource); unchanged from before M5
  const threads = opt.threads || Math.min(8, navigator.hardwareConcurrency || 1);
  S.backend = opt.backend === 'wasm' ? 'wasm' : opt.backend === 'webgpu' ? 'webgpu' : (S.gpuProbe.available ? 'webgpu' : 'wasm');
  const ngl = S.backend === 'webgpu' ? 99999 : 0; // 0 = wllama stubs navigator.gpu in the worker (CPU); anything else lets ggml-webgpu request the adapter
  const wllama = newWllama(S.nativeLog);
  S.wllama = wllama;
  const loadOpts = { ...LOAD_OPTS, n_threads: threads, n_gpu_layers: ngl };
  console.log(`backend=${S.backend} (requested ${opt.backend}; page adapter: ${JSON.stringify(S.gpuProbe)}) n_gpu_layers=${ngl} threads=${threads}`);
  const ui = progressUI();
  const r = await loadModelSource(wllama, S.modelSource, loadOpts, { onProgress: ui.onProgress, onStatus: status, backend: S.backend, onCached: ui.onCached });
  const cached = r.cached;
  S.timing = { download_s: r.download_s, load_s: r.load_s, cached, persisted: r.persisted };
  S.threads = wllama.getNumThreads(); S.multithread = wllama.isMultithread();
  const info = wllama.getLoadedContextInfo();
  $('st-download').textContent = cached ? `0 s (cached${S.timing.persisted ? ', OPFS' : ''})` : `${fmtS(S.timing.download_s)}${S.timing.persisted ? ' (now cached for the next visit)' : ' (not persisted in this browser)'}`;
  $('st-load').textContent = fmtS(S.timing.load_s);
  $('st-threads').textContent = `${S.threads} (${S.multithread ? 'multi-threaded' : 'single-threaded'}, ${navigator.hardwareConcurrency || '?'} logical cores)`;
  $('warn-threads').hidden = !!S.multithread;
  S.gpu = parseGpuLog(S.nativeLog, S.backend);
  renderBackendStats();
  for (const l of S.nativeLog) if (/ggml_webgpu|offload|graph splits|buffer size|Flash Attention/.test(l)) console.log('llama.cpp: ' + l.trim());
  console.log(`loaded: n_ctx=${info.n_ctx} n_batch=${info.n_batch} n_embd=${info.n_embd} add_bos=${wllama.mustAddBosToken()} add_eos=${wllama.mustAddEosToken()} threads=${S.threads} mt=${S.multithread} gpu=${JSON.stringify({ active: S.gpu.active, adapter: S.gpu.adapter_info, layers: S.gpu.layers, splits: S.gpu.graph_splits })}`);
  if (S.gpu.fallback) console.warn('WebGPU requested but no ggml_webgpu adapter_info / offload line in the llama.cpp log: this is a CPU run');
  status('warming up …');
  await embedQuery('warm-up'); // untimed: primes the graph and the instruction-prefix KV cache
  status(`ready on ${S.gpu.active ? 'WebGPU' : `${S.threads} CPU thread${S.threads > 1 ? 's' : ''}`}${S.gpu.fallback ? ' (WebGPU requested, fell back to CPU)' : ''}, download ${cached ? 'from cache' : fmtS(S.timing.download_s)}, load ${fmtS(S.timing.load_s)}`);
}
async function loadVqweb() { // the VQ client: chunk manifest -> OPFS (same loader) -> container -> GPU upload -> tokenizer -> pipelines, as client/vqweb/bench.js
  const [{ parseVqw, fetchVqw, VqwContainer }, { VqwRuntime, requestDevice }, { loadVqwTokenizer }] = await Promise.all(
    ['container.js', 'runtime.js', 'tokenizer.js'].map((f) => import(new URL(f, VQWEB_BASE).href)));
  status('requesting a WebGPU device with shader-f16 …');
  const { device, info } = await requestDevice();   // powerPreference low-power (the integrated GPU where there is a choice), shader-f16 required
  device.addEventListener('uncapturederror', (e) => console.warn('WebGPU error:', e.error && e.error.message));
  S.backend = 'webgpu';
  const ui = progressUI();
  const name = S.modelSource.split('/').pop();
  const t0 = performance.now(); let buf, cached = false, persisted = null;
  if (/\.vqw(\?.*)?$/i.test(S.modelSource)) {   // a plain container URL (?model=...vqw): single fetch, nothing persisted
    status(`fetching ${name} …`);
    buf = await fetchVqw(S.modelSource, ui.onProgress);
  } else {                                        // byte-chunk manifest (default): assembled into OPFS by the shared loader, cached for the next visit
    status(`fetching ${name} …`);
    const a = await assembleFromManifest(S.modelSource, ui.onProgress);
    cached = a.cached; persisted = a.persisted;
    if (cached) { status('model found in this browser\'s storage (OPFS), no download'); ui.onCached(a.file.size); }
    buf = await a.file.arrayBuffer();
  }
  const tDown = performance.now();
  status('parsing the container and uploading the codebooks, indices and scales to the GPU …');
  const container = new VqwContainer(parseVqw(buf), device);
  const tUp = performance.now();
  status('loading the tokenizer …');
  const tokenizer = await loadVqwTokenizer(new URL(S.modelSource, location.href), container.header, container.tokens.ids);
  const tTok = performance.now();
  status('compiling the WebGPU pipelines …');
  const rt = await VqwRuntime.create(container, device);
  const t1 = performance.now();
  container.release(); buf = null;   // the file's ArrayBuffer is no longer referenced (token table copied out)
  S.vq = { device, info, container, tokenizer, rt, upload_s: (tUp - tDown) / 1000, tokenizer_s: (tTok - tUp) / 1000, pipelines_s: (t1 - tTok) / 1000 };
  S.gpuBytes = container.gpuBytes + rt.actBytes;
  S.timing = { download_s: cached ? 0 : (tDown - t0) / 1000, load_s: (t1 - tDown) / 1000, cached, persisted };
  S.threads = null; S.multithread = null;
  S.gpu = { requested: 'webgpu', active: true, fallback: false, runtime: 'vqweb', layers: `${container.model.layers}/${container.model.layers}`, graph_splits: null,
    adapter_info: `vendor: ${info.vendor} | architecture: ${info.architecture} | device: ${info.device || ''} | description: ${info.description || ''}`, page_adapter: info };
  const m = container.model;
  if (m.prompt !== S.meta.prompt) console.warn(`container prompt ${JSON.stringify(m.prompt)} differs from the corpus prompt ${JSON.stringify(S.meta.prompt)}: the tokenizer applies the container's`);
  if (m.truncated) console.warn('TRUNCATED dev container: nDCG is meaningless');
  console.log(`vqweb loaded: ${m.teacher} ${m.layers} blocks, K=${container.quant.k} (${container.quant.bits} b, ${container.quant.bpw_blocks?.toFixed(3)} bpw), ${fmtMiB(container.gpuBytes)} weights + ${fmtMiB(rt.actBytes)} activations on the GPU, ` +
    `table ${container.tokens.rows} rows, prompt ${JSON.stringify(m.prompt)} add_eos=${m.addEos} eos=${m.eosId}, adapter ${JSON.stringify(info)}`);
  $('st-download').textContent = cached ? `0 s (cached${persisted ? ', OPFS' : ''})` : `${fmtS(S.timing.download_s)}${persisted ? ' (now cached for the next visit)' : ' (not persisted in this browser)'}`;
  $('st-load').textContent = `${fmtS(S.timing.load_s)} (upload ${fmtS(S.vq.upload_s)}, tokenizer ${fmtS(S.vq.tokenizer_s)}, pipelines ${fmtS(S.vq.pipelines_s)})`;
  $('warn-threads').hidden = true;
  renderBackendStats();
  status('warming up …');
  await embedQuery('warm-up'); // untimed: first submit of the pipelines
  status(`ready on WebGPU (${adapterLabel(info)}, vqweb runtime), download ${cached ? 'from cache' : fmtS(S.timing.download_s)}, load ${fmtS(S.timing.load_s)}`);
}
function vqInfo() {
  const v = S.vq, c = v.container;
  return { adapter: v.info, layers: c.model.layers, k: c.quant.k, bits: c.quant.bits, bpw: c.quant.bpw_blocks, prompt: c.model.prompt, add_eos: c.model.addEos, prompt_match: c.model.prompt === S.meta.prompt,
    table_rows: c.tokens.rows, gpu_weight_bytes: c.gpuBytes, act_bytes: v.rt.actBytes, upload_s: v.upload_s, tokenizer_s: v.tokenizer_s, pipelines_s: v.pipelines_s, pipelines: v.rt.pipelines?.size ?? null, tmax: v.rt.tmax ?? null };
}

// ---- encode + search -----------------------------------------------------------------------------------------
/** query text WITHOUT the prompt -> L2-normalised Float32Array.  wllama: meta.prompt + text through llama.cpp;
 *  vqweb: VqwTokenizer.encode(text) applies the container header's prompt / add_eos, then the GPU forward pass. */
async function embedQuery(queryText) {
  if (S.client.runtime === 'vqweb') {
    const ids = S.vq.tokenizer.encode(queryText);
    const r = await S.vq.rt.embed(ids);
    return { vec: l2n(r.embedding), tokens: ids.length, gpu_ms: r.ms };
  }
  const r = await S.wllama.createEmbedding({ input: S.meta.prompt + queryText });
  return { vec: l2n(Float32Array.from(r.data[0].embedding)), tokens: r.usage?.prompt_tokens };
}
async function encodeAndRank(queryText) {
  const t0 = performance.now();
  const e = await embedQuery(queryText);
  const t1 = performance.now();
  const v = e.vec, corpus = S.corpus, d = S.dim, scores = S.scores;
  if (v.length !== d) throw new Error(`embedding dim ${v.length} != index dim ${d}`);
  for (let j = 0; j < S.nDocs; j++) scores[j] = dot(v, 0, corpus, j * d, d);
  const top = topk(scores, 10);
  const t2 = performance.now();
  return { top, scores: Array.from(top, (j) => scores[j]), encodeMs: t1 - t0, searchMs: t2 - t1, tokens: e.tokens, gpuMs: e.gpu_ms };
}
function relMap(q) {
  const relByIdx = new Map();
  if (q) for (const [d, g] of Object.entries(q.rel)) { const j = S.docIndex.get(d); if (j != null && g > 0) relByIdx.set(j, g); }
  return relByIdx;
}
function updateLatencyStats(r) {
  S.encodeMs.push(r.encodeMs); S.searchMs.push(r.searchMs);
  $('st-encode').textContent = `${fmtMs(r.encodeMs)} / ${fmtMs(percentile(S.encodeMs, 0.5))}`;
  $('st-search').textContent = `${fmtMs(r.searchMs)} / ${fmtMs(percentile(S.searchMs, 0.5))}`;
  $('st-count').textContent = String(S.encodeMs.length);
}
async function search(text) {
  text = (text || '').trim();
  if (!text || !S.loaded || S.verifying) return null;
  $('btn-search').disabled = true;
  try {
    const r = await encodeAndRank(text);
    updateLatencyStats(r);
    const q = S.queryByText.get(text) || null;
    const rel = relMap(q);
    renderResults(r, rel);
    const out = { query: text, qid: q ? q.qid : null, encode_ms: r.encodeMs, search_ms: r.searchMs, tokens: r.tokens, gpu_ms: r.gpuMs ?? null,
      top: Array.from(r.top, (j, i) => ({ id: S.docs[j].id, score: r.scores[i] })), ndcg10: q ? ndcgAtK(r.top, rel, 10) : null };
    smokeLog('search', out);
    return out;
  } finally { $('btn-search').disabled = false; }
}
function renderResults(r, rel) {
  const ol = $('results');
  ol.replaceChildren(...Array.from(r.top, (j, i) => {
    const d = S.docs[j];
    const li = document.createElement('li'); if (rel.has(j)) li.classList.add('rel');
    const head = document.createElement('div'); head.className = 'head';
    const rank = document.createElement('span'); rank.className = 'rank'; rank.textContent = `${i + 1}.`;
    const title = document.createElement('span'); title.className = 'title'; title.textContent = d.title || '(untitled)';
    const score = document.createElement('span'); score.className = 'score'; score.textContent = r.scores[i].toFixed(4);
    head.append(rank, title, score);
    const id = document.createElement('div'); id.className = 'docid'; id.textContent = `doc ${d.id}`;
    const abs = document.createElement('p'); abs.className = 'abstract'; abs.textContent = d.text; abs.hidden = true;
    head.addEventListener('click', () => { abs.hidden = !abs.hidden; });
    li.append(head, id, abs);
    return li;
  }));
}

// ---- verification: the 300 test queries -----------------------------------------------------------------------
async function verify(n) {
  if (!S.loaded || S.verifying) return null;
  n = Math.min(n || S.queries.length, S.queries.length);
  S.verifying = true; S.abort = false;
  $('btn-verify').disabled = true; $('btn-abort').hidden = false; $('btn-copy').hidden = true; $('verify-details').open = true;
  $('verify-progress-wrap').hidden = false; $('verify-table').hidden = false; $('verify-note').hidden = true;
  $('q').disabled = true; $('btn-search').disabled = true;
  const ref = S.ref, refPQ = ref.testPQ;   // the selected client's own reference (native llama.cpp, or the torch simulation of the VQ file), per query
  const lat = [], ndcg = [], perQuery = [];
  const tRun = performance.now();
  try {
    for (let i = 0; i < n; i++) {
      if (S.abort) break;
      const q = S.queries[i];
      const r = await encodeAndRank(q.text);
      const rel = relMap(q);
      const v = ndcgAtK(r.top, rel, 10);
      lat.push(r.encodeMs); ndcg.push(v);
      perQuery.push({ qid: q.qid, ndcg10: Number.isNaN(v) ? null : v, encode_ms: r.encodeMs, search_ms: r.searchMs, tokens: r.tokens, top1: S.docs[r.top[0]].id,
        native_ndcg10: refPQ ? refPQ[i] : undefined, browser_ndcg10: ref.browserPQ ? ref.browserPQ[i] : undefined });
      updateLatencyStats(r);
      $('verify-bar').style.width = `${(100 * (i + 1) / n).toFixed(1)}%`;
      $('v-ndcg').textContent = nanmean(ndcg).toFixed(4);
      $('v-delta').textContent = ref.test != null ? signed(nanmean(ndcg) - ref.test) : '–';
      $('v-p50').textContent = fmtMs(percentile(lat, 0.5)); $('v-p95').textContent = fmtMs(percentile(lat, 0.95));
      $('verify-status').textContent = `${i + 1} / ${n} queries, ${fmtS((performance.now() - tRun) / 1000)} elapsed`;
      if (i % 5 === 4) await sleep(0);
    }
  } finally {
    S.verifying = false; $('btn-verify').disabled = false; $('btn-abort').hidden = true; $('q').disabled = false; $('btn-search').disabled = false;
  }
  const done = perQuery.length, aborted = S.abort;
  const report = {
    demo: 'thinletter', dataset: S.ds, client: S.client.id, runtime: S.client.runtime, date: new Date().toISOString(),
    model_file: S.client.model_file, model_size_bytes: S.client.size_bytes, model_sha256: S.client.sha256 || null, bpw: S.client.bpw ?? null,
    prompt: S.client.runtime === 'vqweb' ? S.vq.container.model.prompt : S.meta.prompt, prompt_source: S.client.runtime === 'vqweb' ? 'container header' : 'meta.json',
    wllama_version: S.wllama ? WLLAMA_VERSION : null, libllama: (() => { try { return S.wllama ? Wllama.getLibllamaVersion() : null; } catch (e) { return null; } })(),
    userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, deviceMemory: navigator.deviceMemory ?? null,
    crossOriginIsolated: !!self.crossOriginIsolated, threads: S.threads, multithread: S.multithread,
    backend: S.backend, backend_requested: opt.backend, gpu: S.gpu, page_adapter: S.gpuProbe, vq: S.vq ? vqInfo() : null,
    download_s: S.timing.download_s, load_s: S.timing.load_s, model_cached: S.timing.cached, memory_after_load: S.memory, gpu_bytes: S.gpuBytes,
    n: done, aborted, run_s: (performance.now() - tRun) / 1000,
    reference_kind: ref.kind, // "native" = llama.cpp on the same GGUF; "simulation" = torch simulation of the same .vqw (the ndcg10_native* fields then hold that)
    ndcg10_mean: nanmean(ndcg), ndcg10_native: ref.test ?? null, ndcg10_browser_ours: ref.browser ?? null, ndcg10_fp16: ref.fp16 ?? null,
    ndcg10_native_same_n: refPQ ? nanmean(refPQ.slice(0, done).map((x) => (x == null ? NaN : x))) : null,
    ndcg10_browser_ours_same_n: ref.browserPQ ? nanmean(ref.browserPQ.slice(0, done).map((x) => (x == null ? NaN : x))) : null,
    latency_ms: { p50: percentile(lat, 0.5), p95: percentile(lat, 0.95), mean: lat.reduce((a, b) => a + b, 0) / Math.max(1, lat.length) },
    per_query: perQuery,
  };
  S.lastReport = report;
  const nat = report.ndcg10_native_same_n;
  const dq = refPQ ? perQuery.filter((p) => p.native_ndcg10 != null && p.ndcg10 != null && Math.abs(p.native_ndcg10 - p.ndcg10) > 1e-3).length : null;
  $('verify-status').textContent = `${aborted ? 'aborted after' : 'done:'} ${done} queries in ${fmtS(report.run_s)}`;
  $('verify-note').hidden = false;
  $('verify-note').textContent = `nDCG@10 over ${done} queries: ${report.ndcg10_mean.toFixed(4)}` +
    (nat != null ? ` (${ref.sameNText} ${done} queries: ${nat.toFixed(4)}${dq != null ? `, ${dq} queries differ by more than 0.001` : ''})` : '') +
    `. Encode latency p50 ${fmtMs(report.latency_ms.p50)}, p95 ${fmtMs(report.latency_ms.p95)} on ` +
    (S.client.runtime === 'vqweb' ? `WebGPU (${adapterLabel(S.vq.info)}, vqweb runtime)` :
      S.gpu?.active ? `WebGPU (${S.gpu.layers} layers offloaded)` : `${S.threads} CPU thread${S.threads > 1 ? 's' : ''}${self.crossOriginIsolated ? '' : ' (single-threaded fallback)'}${S.gpu?.fallback ? ', WebGPU requested but fell back to CPU' : ''}`) +
    (S.client.runtime === 'vqweb' ? '. Small differences from the simulation are expected: f16 matmul inputs on the GPU, and ties in the top-10.'
      : '. Small differences from the native number are expected: a different kernel order in WebAssembly or on the GPU, and ties in the top-10.');
  $('btn-copy').hidden = false;
  smokeLog('verify', { client: S.client.id, n: done, aborted, ndcg10_mean: report.ndcg10_mean, ndcg10_native_same_n: nat, reference_kind: ref.kind, p50: report.latency_ms.p50, p95: report.latency_ms.p95,
    per_query: perQuery.map((p) => ({ qid: p.qid, ndcg10: p.ndcg10, top1: p.top1 })) });
  return report;
}
async function copyReport() {
  if (!S.lastReport) return;
  const text = JSON.stringify(S.lastReport, null, 1);
  try { await navigator.clipboard.writeText(text); $('verify-status').textContent = 'report copied to the clipboard'; }
  catch (e) { const ta = document.createElement('textarea'); ta.value = text; ta.style.width = '100%'; ta.rows = 12; $('panel-verify').appendChild(ta); ta.select(); $('verify-status').textContent = 'clipboard unavailable: report shown below'; }
}

// ---- smoke test (headless) -------------------------------------------------------------------------------------
async function smoke() {
  const report = { ok: false, dataset: S.ds, client: S.client.id, runtime: S.client.runtime, started: new Date().toISOString(), crossOriginIsolated: !!self.crossOriginIsolated, userAgent: navigator.userAgent };
  try {
    await loadModel();
    report.load = { client: S.client.id, runtime: S.client.runtime, backend: S.backend, backend_requested: opt.backend, gpu: S.gpu, page_adapter: S.gpuProbe, threads: S.threads, multithread: S.multithread,
      download_s: S.timing.download_s, load_s: S.timing.load_s, cached: S.timing.cached, gpu_bytes: S.gpuBytes, vq: S.vq ? vqInfo() : null };
    report.native_log = S.nativeLog.filter((l) => /ggml_webgpu|offload|graph splits|buffer size|Flash Attention|Multithread/.test(l)).slice(0, 60);
    if (opt.autoquery) { $('q').value = opt.autoquery; report.search = await search(opt.autoquery); }
    report.verify = await verify(opt.verifyN);
    try { await S.memoryPromise; } catch (e) { /* measured or not */ }
    report.memory_after_load = S.memory;
    report.ok = true;
  } catch (e) { report.error = String(e && e.message ? e.message : e); report.stack = e && e.stack; }
  report.finished = new Date().toISOString();
  window.__smokeReport = report;
  smokeLog('report', { ok: report.ok, error: report.error || null, client: S.client.id, backend: S.backend, gpu_active: S.gpu?.active ?? null, n_results: report.search ? report.search.top.length : null, verify_n: report.verify ? report.verify.n : null, ndcg10_mean: report.verify ? report.verify.ndcg10_mean : null });
  try { await fetch(new URL('./smoke', location.href).href, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(report) }); } catch (e) { /* no endpoint: fine */ }
  smokeLog('done');
}

// ---- wiring --------------------------------------------------------------------------------------------------
(async function main() {
  try {
    await loadMeta();
    status('loading the document index …');
    await loadIndex();
    if (navigator.deviceMemory && navigator.deviceMemory < 4) $('warn-memory').hidden = false;
    S.gpuProbe = await probeWebGPU();
    renderClientSelect();          // VQ entries are disabled when the probe finds no WebGPU / shader-f16
    renderBackendChoice();
    status(`index ready (${S.nDocs} documents). Click "Load model" to fetch the ${fmtMiB(S.client.size_bytes || 0)} query encoder.`);
    await cachedHint();
    $('btn-load').addEventListener('click', () => loadModel().catch(() => {}));
    $('form-search').addEventListener('submit', (e) => { e.preventDefault(); search($('q').value).catch((err) => status(`search failed: ${err.message}`, true)); });
    $('btn-verify').addEventListener('click', () => verify(opt.verifyN).catch((err) => status(`verification failed: ${err.message}`, true)));
    $('btn-abort').addEventListener('click', () => { S.abort = true; });
    $('btn-copy').addEventListener('click', copyReport);
    const clr = document.createElement('a'); clr.href = '#'; clr.textContent = 'Remove the cached copy'; clr.className = 'small';
    clr.onclick = (e) => { e.preventDefault(); clearCache(); }; $('panel-load').querySelector('p.muted').append(' ', clr);
    if (opt.smoke) await smoke();
    else if (opt.autoload) { await loadModel(); if (opt.autoquery) { $('q').value = opt.autoquery; await search(opt.autoquery); } if (opt.autoverify) await verify(opt.verifyN); }
  } catch (e) {
    console.error(e);
    status(`startup failed: ${e && e.message ? e.message : e}`, true);
    smokeLog('startup_failed', { error: String(e && e.message ? e.message : e) });
  }
})();
