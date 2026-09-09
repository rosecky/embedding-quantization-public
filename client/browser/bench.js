// Browser client benchmark: quantized GGUF query encoder (llama.cpp -> WebAssembly via wllama 3.6.1),
// SciFact test queries encoded one at a time, scored against the fp32 document index in JS.
//
// Mirrors scripts/quant_eval_queries.py: prompt-prefixed query text, pooling "last", L2-normalised embedding,
// tokenizer adds EOS (GGUF metadata add_eos_token=true, add_bos_token=false; wllama tokenises with add_special=true
// exactly like llama-embedding), cosine scores, nDCG@10 with the semantics of eq.graph_metrics.ndcg_at_k.
// The wllama module is imported dynamically from ./vendor/<build>/ (default: the npm 3.6.1 package, which is built
// with llama.cpp's WebGPU backend; whether the GPU is used is decided at runtime by n_gpu_layers, see
// client/webgpu/README.md).
const WLLAMA_VERSION = '3.6.1';
const params = new URLSearchParams(location.search);
const cfg = {
  model: params.get('model') || 'models/gguf/harrier-0.6b-gptq-Q2_K-scifact_corpus_only-ps20000-t300k-ao-tabQ2_K.gguf',
  threads: params.has('threads') ? parseInt(params.get('threads'), 10) : (navigator.hardwareConcurrency || 1),
  ctx: parseInt(params.get('ctx') || '512', 10),
  n: parseInt(params.get('n') || '300', 10),
  warmup: parseInt(params.get('warmup') || '3', 10),
  tag: params.get('tag') || 'manual',
  // cache=1: llama-server prefix KV reuse stays on (a resident client re-uses the cached instruction prefix between
  // queries); cache=0: the single slot is flushed with an untimed dummy embedding after every query (cold prefix).
  cache: params.get('cache') !== '0',
  dataset: params.get('dataset') || 'scifact',
  split: params.get('split') || 'test',
  auto: params.get('auto') !== '0',
  // backend=wasm: n_gpu_layers 0 (CPU baseline; wllama stubs navigator.gpu in the worker) | webgpu: n_gpu_layers = ngl
  backend: params.get('backend') === 'webgpu' ? 'webgpu' : 'wasm',
  ngl: params.has('ngl') ? parseInt(params.get('ngl'), 10) : 99999,
  // fa=0 disables flash attention; otherwise llama.cpp's auto probe decides (recorded in result.gpu.flash_attn)
  flash_attn: params.get('fa') === '0' ? false : undefined,
  // vendor directory under client/browser/vendor/: wllama (npm 3.6.1) | wllama-webgpu | wllama-webgpu-debug (local builds)
  build: (params.get('build') || 'wllama').replace(/[^A-Za-z0-9_.-]/g, ''),
};
cfg.name = params.get('name') || `browser_${cfg.tag}_${cfg.threads === 1 ? 'st' : 'mt'}${cfg.cache ? '' : '-nocache'}${cfg.backend === 'webgpu' ? '-webgpu' : ''}`;
const ROOT = new URL('../../', location.href); // repo root as served by serve.py

const $ = (id) => document.getElementById(id);
const logEl = $('log');
function log(msg, cls) {
  const line = `[${(performance.now() / 1000).toFixed(1)}s] ${msg}`;
  const div = document.createElement('div');
  div.textContent = line;
  if (cls) div.className = cls;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  console.log(line);
  fetch(`${ROOT}log?name=${encodeURIComponent(cfg.name)}`, { method: 'POST', body: line }).catch(() => {});
}

$('cfg').textContent = JSON.stringify({ ...cfg, crossOriginIsolated: self.crossOriginIsolated,
  hardwareConcurrency: navigator.hardwareConcurrency }, null, 1);

// ---- helpers -------------------------------------------------------------------------------------------------
async function fetchF32(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Float32Array(await r.arrayBuffer());
}
function l2n(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const inv = s > 0 ? 1 / Math.sqrt(s) : 0;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}
function dot(a, aOff, b, bOff, d) {
  let s = 0;
  for (let i = 0; i < d; i++) s += a[aOff + i] * b[bOff + i];
  return s;
}
// top-k indices sorted by descending score (ties: lower index first)
function topk(scores, k) {
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
// eq.graph_metrics.ndcg_at_k: linear gains, discount 1/log2(rank+1), IDCG from the sorted grades; NaN if no relevant doc
function ndcgAtK(top, relByIdx, k) {
  let dcg = 0;
  for (let i = 0; i < k; i++) dcg += (relByIdx.get(top[i]) || 0) / Math.log2(i + 2);
  const grades = [...relByIdx.values()].sort((a, b) => b - a).slice(0, k);
  let idcg = 0;
  for (let i = 0; i < grades.length; i++) idcg += grades[i] / Math.log2(i + 2);
  return idcg > 0 ? dcg / idcg : NaN;
}
function nanmean(a) { const v = a.filter((x) => !Number.isNaN(x)); return v.reduce((s, x) => s + x, 0) / v.length; }
function percentile(a, p) { // numpy default (linear interpolation)
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  const h = (s.length - 1) * p, lo = Math.floor(h), hi = Math.ceil(h);
  return s[lo] + (s[hi] - s[lo]) * (h - lo);
}
async function measureMemory() {
  const out = { api: null, bytes: null, breakdown: null, js_heap_used: null };
  try { out.js_heap_used = performance.memory ? performance.memory.usedJSHeapSize : null; } catch (e) { /* ignore */ }
  if (performance.measureUserAgentSpecificMemory) {
    try {
      const m = await performance.measureUserAgentSpecificMemory();
      out.api = 'measureUserAgentSpecificMemory';
      out.bytes = m.bytes;
      const byType = {};
      for (const b of m.breakdown) { const t = (b.types || []).join('+') || 'unknown'; byType[t] = (byType[t] || 0) + b.bytes; }
      out.breakdown = byType;
      return out;
    } catch (e) { log(`measureUserAgentSpecificMemory failed: ${e.message}`, 'warn'); }
  }
  if (out.js_heap_used != null) { out.api = 'performance.memory.usedJSHeapSize'; out.bytes = out.js_heap_used; }
  return out;
}

// ---- main ---------------------------------------------------------------------------------------------------
async function run() {
  $('start').disabled = true;
  const result = { name: cfg.name, config: { ...cfg }, wllama_version: WLLAMA_VERSION, userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency, crossOriginIsolated: !!self.crossOriginIsolated,
    backend: cfg.backend, jspi: !!WebAssembly.Suspending, webgpu_available: !!navigator.gpu,
    started_iso: new Date().toISOString(), ok: false };
  try {
    const { Wllama } = await import(`./vendor/${cfg.build}/index.js`);
    result.libllama = Wllama.getLibllamaVersion();
    result.vendor_version = await fetch(`./vendor/${cfg.build}/VERSION.txt`).then((r) => (r.ok ? r.text() : null)).catch(() => null);
    log(`backend=${cfg.backend} build=${cfg.build} (${result.libllama}) jspi=${result.jspi} navigator.gpu=${result.webgpu_available}`);
    log(`crossOriginIsolated=${self.crossOriginIsolated} hardwareConcurrency=${navigator.hardwareConcurrency} threads requested=${cfg.threads}`);
    // 1. data (small; not part of the model timing)
    const dataUrl = `${ROOT}client/browser/data/`;
    const meta = await (await fetch(`${dataUrl}${cfg.dataset}_${cfg.split}.json`)).json();
    const corpus = await fetchF32(`${dataUrl}${cfg.dataset}_corpus.f32`);
    const qfp = await fetchF32(`${dataUrl}${cfg.dataset}_${cfg.split}_qfp.f32`);
    const dim = meta.dim, nDocs = meta.n_docs;
    if (corpus.length !== nDocs * dim || qfp.length !== meta.n_queries * dim) throw new Error('index size mismatch');
    const docIndex = new Map(meta.doc_ids.map((d, i) => [d, i]));
    result.prompt = meta.prompt; result.prompt_kind = meta.prompt_kind || 'unknown';
    log(`index: ${nDocs} docs x ${dim}, ${meta.n_queries} ${cfg.split} queries (prompt: ${JSON.stringify(meta.prompt)})`);

    // 2. model download + load
    const nativeLog = []; // llama.cpp INFO lines (model/KV/compute buffer sizes) arrive at debug level
    const keep = (a) => { const s = a.join(' '); if (nativeLog.length < 400) nativeLog.push(s); };
    const wllama = new Wllama({ default: new URL(`./vendor/${cfg.build}/wllama.wasm`, location.href).href },
      { suppressNativeLog: false, logger: { debug: (...a) => keep(a), log: (...a) => { keep(a); console.log(...a); }, warn: (...a) => log('wllama warn: ' + a.join(' '), 'warn'), error: (...a) => log('wllama error: ' + a.join(' '), 'warn') } });
    result.native_log = nativeLog;
    const modelUrl = new URL(cfg.model, ROOT).href;
    const ngl = cfg.backend === 'webgpu' ? cfg.ngl : 0;
    log(`loading ${modelUrl} (n_ctx=${cfg.ctx}, n_batch=${cfg.ctx}, pooling=last, embeddings=true, n_threads=${cfg.threads}, n_gpu_layers=${ngl}, flash_attn=${cfg.flash_attn === undefined ? 'auto' : cfg.flash_attn})`);
    const t0 = performance.now();
    let tDownload = null, modelBytes = null, lastPct = -1;
    await wllama.loadModelFromUrl(modelUrl, {
      n_ctx: cfg.ctx, n_batch: cfg.ctx, n_ubatch: cfg.ctx, n_threads: cfg.threads, n_gpu_layers: ngl, flash_attn: cfg.flash_attn,
      embeddings: true, pooling_type: 'last', n_parallel: 1, useCache: false,
      progressCallback: ({ loaded, total }) => {
        modelBytes = total;
        const pct = Math.floor((loaded / total) * 10) * 10;
        if (pct !== lastPct) { lastPct = pct; log(`download ${pct}% (${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MiB)`); }
        if (loaded >= total && tDownload == null) tDownload = performance.now();
      },
    });
    const t1 = performance.now();
    result.download_s = tDownload != null ? (tDownload - t0) / 1000 : null;
    result.load_s = (t1 - t0) / 1000;
    result.model_bytes = modelBytes;
    result.multithread = wllama.isMultithread();
    result.threads_used = wllama.getNumThreads();
    const info = wllama.getLoadedContextInfo();
    result.ctx_info = { n_ctx: info.n_ctx, n_batch: info.n_batch, n_ubatch: info.n_ubatch, n_embd: info.n_embd, n_layer: info.n_layer,
      token_bos: info.token_bos, token_eos: info.token_eos, add_bos_token: wllama.mustAddBosToken(), add_eos_token: wllama.mustAddEosToken(),
      general_name: info.metadata['general.name'], file_type: info.metadata['general.file_type'] };
    log(`loaded in ${result.load_s.toFixed(1)} s (download ${result.download_s?.toFixed(1)} s, ${(modelBytes / 2 ** 20).toFixed(1)} MiB); ` +
        `multithread=${result.multithread} threads=${result.threads_used}; n_ctx=${info.n_ctx} n_batch=${info.n_batch} n_ubatch=${info.n_ubatch} ` +
        `add_bos=${result.ctx_info.add_bos_token} add_eos=${result.ctx_info.add_eos_token} eos=${info.token_eos}`);
    if (cfg.threads > 1 && !result.multithread) log('multi-thread build NOT active (no SharedArrayBuffer / not cross-origin isolated): single-thread fallback', 'warn');
    for (const s of nativeLog) if (/buffer size|model size|KV self size|n_ctx_per_seq|type +f16|Multithread|ggml_webgpu|offload|graph splits|Flash Attention/.test(s)) log('llama.cpp: ' + s.trim());
    // GPU evidence parsed from the llama.cpp INFO lines (client/webgpu/README.md section 4): adapter_info absent => CPU run
    const pick = (re) => nativeLog.filter((s) => re.test(s)).map((s) => s.trim());
    result.gpu = {
      adapter_info: pick(/ggml_webgpu: adapter_info/)[0] || null,
      offloaded: pick(/offloaded \d+\/\d+ layers to GPU/)[0] || null,
      model_buffers: pick(/model buffer size/),
      compute_buffers: pick(/compute buffer size|KV buffer size/),
      graph_splits: pick(/graph splits/)[0] || null,
      flash_attn: pick(/Flash Attention/),
      unsupported_ops: pick(/ggml_webgpu op not supported/).slice(0, 50), // GGML_WEBGPU_DEBUG builds only
      ggml_errors: pick(/ggml_webgpu: (Failed|Device|Error)/),
      n_gpu_layers_requested: ngl,
    };
    result.gpu.active = !!result.gpu.adapter_info && /offloaded [1-9]\d*\/\d+ layers/.test(result.gpu.offloaded || '');
    if (cfg.backend === 'webgpu' && !result.gpu.adapter_info) log('WebGPU requested but no ggml_webgpu adapter_info line: this is a CPU run', 'warn');
    log(`gpu: ${JSON.stringify({ active: result.gpu.active, adapter: result.gpu.adapter_info, offloaded: result.gpu.offloaded, splits: result.gpu.graph_splits, fa: result.gpu.flash_attn })}`);
    try { // the page's own view of the adapter (same selection policy as the worker's requestAdapter with default options)
      const ad = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
      result.page_adapter = ad ? { vendor: ad.info?.vendor, architecture: ad.info?.architecture, device: ad.info?.device,
        description: ad.info?.description, shader_f16: ad.features.has('shader-f16'), subgroups: ad.features.has('subgroups'),
        maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } : null;
      log(`page adapter: ${JSON.stringify(result.page_adapter)}`);
    } catch (e) { result.page_adapter = { error: String(e) }; }
    result.mem_after_load = await measureMemory();
    log(`memory after load: ${result.mem_after_load.api} = ${result.mem_after_load.bytes != null ? (result.mem_after_load.bytes / 2 ** 20).toFixed(0) + ' MiB' : 'n/a'} ${JSON.stringify(result.mem_after_load.breakdown || {})}`);

    const embed = async (text) => {
      const r = await wllama.createEmbedding({ input: text });
      return { vec: l2n(Float32Array.from(r.data[0].embedding)), tokens: r.usage.prompt_tokens };
    };
    // 3. warm-up (excluded from the statistics); taken from the END of the list so that the first measured queries do
    //    not hit the slot's prompt cache with an identical prompt
    const flush = async () => { await wllama.createEmbedding({ input: 'x' }); }; // replaces the slot's KV with 2 tokens -> no common prefix
    for (let i = 0; i < Math.min(cfg.warmup, meta.queries.length); i++) {
      const q = meta.queries[meta.queries.length - 1 - i];
      const s = performance.now(); const e = await embed(q.text);
      log(`warm-up ${i + 1}/${cfg.warmup}: ${(performance.now() - s).toFixed(0)} ms, ${e.tokens} tokens, dim ${e.vec.length}`);
    }
    if (!cfg.cache) await flush();
    // 4. the measured run: one query at a time, wall-clock per query, scored in JS
    const n = Math.min(cfg.n, meta.queries.length);
    const lat = [], ndcg = [], cosfp = [], toks = [], qids = [], top1 = [], embs = [];
    const scores = new Float64Array(nDocs);
    const tRun = performance.now();
    for (let i = 0; i < n; i++) {
      const q = meta.queries[i];
      const s = performance.now();
      const e = await embed(q.text);
      lat.push(performance.now() - s);
      if (!cfg.cache) await flush(); // untimed: next query starts from an empty prefix cache
      toks.push(e.tokens); qids.push(q.qid);
      if (i === 0) result.first_embedding = Array.from(e.vec); // for a bit-level check against the native vector of query 1
      embs.push(Array.from(e.vec)); // all vectors (1.2 MB for 300 queries): vector-level comparison with the native run
      if (e.vec.length !== dim) throw new Error(`embedding dim ${e.vec.length} != ${dim}`);
      for (let j = 0; j < nDocs; j++) scores[j] = dot(e.vec, 0, corpus, j * dim, dim);
      const top = topk(scores, 10);
      const relByIdx = new Map();
      for (const [d, g] of Object.entries(q.rel)) { const j = docIndex.get(d); if (j != null && g > 0) relByIdx.set(j, g); }
      ndcg.push(ndcgAtK(top, relByIdx, 10));
      cosfp.push(dot(e.vec, 0, qfp, i * dim, dim));
      top1.push(meta.doc_ids[top[0]]);
      if ((i + 1) % 25 === 0 || i === n - 1) {
        log(`${i + 1}/${n}: p50 ${percentile(lat, 0.5).toFixed(0)} ms, running nDCG@10 ${nanmean(ndcg).toFixed(4)}, cos_fp ${nanmean(cosfp).toFixed(4)}`);
        await new Promise((r) => setTimeout(r, 0)); // let the log paint
      }
    }
    result.run_s = (performance.now() - tRun) / 1000;
    result.n = n;
    result.lat_ms = lat; result.p50 = percentile(lat, 0.5); result.p95 = percentile(lat, 0.95); result.mean = lat.reduce((a, b) => a + b, 0) / n;
    result.ndcg_mean = nanmean(ndcg); result.ndcg_per_query = ndcg.map((x) => (Number.isNaN(x) ? null : x));
    result.cos_fp_mean = nanmean(cosfp); result.cos_fp_per_query = cosfp;
    result.tokens_per_query = toks; result.qids = qids; result.top1_doc = top1; result.embeddings = embs;
    if (params.get('mem2') === '1') result.mem_after_run = await measureMemory(); // slow API (~1 min), off by default
    result.ok = true;
    log(`DONE n=${n}: load ${result.load_s.toFixed(1)} s, p50 ${result.p50.toFixed(0)} ms, p95 ${result.p95.toFixed(0)} ms, mean ${result.mean.toFixed(0)} ms, ` +
        `nDCG@10 ${result.ndcg_mean.toFixed(4)}, cos_fp ${result.cos_fp_mean.toFixed(4)}, threads ${result.threads_used} (mt=${result.multithread}, prefix cache=${cfg.cache}, backend=${cfg.backend}, gpu active=${result.gpu.active})`);
    $('summary').innerHTML = `<table><tr><th>metric</th><th>value</th></tr>
      <tr><td>download + load</td><td>${result.load_s.toFixed(1)} s</td></tr>
      <tr><td>memory after load (${result.mem_after_load.api})</td><td>${result.mem_after_load.bytes != null ? (result.mem_after_load.bytes / 2 ** 20).toFixed(0) + ' MiB' : 'n/a'}</td></tr>
      <tr><td>threads (multithread)</td><td>${result.threads_used} (${result.multithread})</td></tr>
      <tr><td>backend / GPU active</td><td>${cfg.backend} / ${result.gpu.active}${result.gpu.adapter_info ? ' — ' + result.gpu.adapter_info : ''}</td></tr>
      <tr><td>latency p50 / p95 / mean</td><td>${result.p50.toFixed(0)} / ${result.p95.toFixed(0)} / ${result.mean.toFixed(0)} ms</td></tr>
      <tr><td>nDCG@10 (${n} queries)</td><td>${result.ndcg_mean.toFixed(4)}</td></tr>
      <tr><td>cos to fp32 query embedding</td><td>${result.cos_fp_mean.toFixed(4)}</td></tr></table>`;
    try { await wllama.exit(); } catch (e) { /* ignore */ }
  } catch (e) {
    result.error = `${e && e.message ? e.message : e}`; result.stack = e && e.stack;
    log(`ERROR: ${result.error}`, 'warn');
    console.error(e);
  }
  result.finished_iso = new Date().toISOString();
  window.__benchResult = result;
  const r = await fetch(`${ROOT}result`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(result) }).catch((e) => ({ ok: false, statusText: e.message }));
  log(r.ok ? `result posted (${cfg.name}.json)` : `result POST failed: ${r.statusText}`, r.ok ? undefined : 'warn');
  $('status').textContent = result.ok ? 'finished' : 'failed';
  $('start').disabled = false;
}

$('start').disabled = false;
$('start').addEventListener('click', run);
if (cfg.auto) run();
