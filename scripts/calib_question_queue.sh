#!/bin/bash
# Calibration question (product decision): does corpus / synthetic-query calibration add anything over GENERIC text on
# the deployment grids?  Missing arms for the two models and corpora we have, all GPTQ act-order per-sample, 300k-token
# budget, token table Q2_K (identical settings to scripts/legal_queue.sh and scripts/scidocs_queue.sh):
#   (a) jina-v5-small / legal-cs: Q2_K generic_wikitext, Q2_K generic_cs, Q3_K generic_wikitext, Q3_K generic_cs,
#       Q3_K legal-cs_corpus_only   (existing: Q2_K corpus, Q2_K synth, Q3_K synth in models/gguf, reused, never recomputed)
#   (b) harrier-0.6b / scidocs: Q2_K generic_wikitext            (existing: Q2_K corpus, Q2_K synth)
#   (c) harrier-0.6b / arguana + nfcorpus: Q3_K generic_wikitext (one file serves both), Q3_K arguana synthetic queries
#       (leak-clean set arguana_synth_noleak: the 645 test-query documents are excluded before query selection, see
#       scripts/arguana_clean_queue.sh), Q3_K nfcorpus_synth_only
# Evaluation: scripts/quant_eval_queries.py, upstream Windows llama-embedding.exe, 8 threads, 1 worker, query side
# quantised against the fp32 index of the teacher; ONE call per dataset with every relevant GGUF (existing + new) so all
# rows of a dataset come from the same runtime session (cached chunks of the existing files are reused as they are).
# ctx: 512 for legal-cs (as results/raw/legal_local); 1024 for the harrier datasets (as results/raw/scidocs_local, and
# 32 ArguAna test queries exceed 512 tokens with the E5 prompt -> llama-embedding refuses inputs longer than the batch).
# Strictly sequential, one GPU job at a time, every step tolerant, nothing is killed here.  Every export is skipped when
# its GGUF already exists in models/gguf (another session may have produced it).
# Pre-registered in research/lab_log.md ("2026-09-08 — pre-registrace: přidává kalibrace na korpusu něco proti generickému textu?").
# Log: results/raw/calibq_local/queue.log (+ one log per step); ends with CALIBQ_DONE.
#
# launch (Git Bash, from the repo root, detached):
#   nohup bash scripts/calib_question_queue.sh > results/raw/calibq_local/queue.nohup 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH"   # a bash.exe started by Start-Process has no Git usr/bin on PATH
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
OUT=results/raw/calibq_local
LOG=$OUT/queue.log
BIN=third_party/llama.cpp/build-clang/bin/llama-embedding.exe
GG=models/gguf
SUF=ps20000-t300k-ao-tabQ2_K          # name suffix scripts/gptq_export_gguf.py derives from the flags below
GPTQ="--per_sample --n_seq 20000 --calib_tokens 300000 --act_order --table_type Q2_K --out $OUT/gptq_export"
ARG_SYNTH=arguana_synth_noleak        # leak-clean synthetic queries for ArguAna (arguana_synth_only contains queries of leaked test documents)
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
step() {  # step <name> <command...>: stdout+stderr -> $OUT/<name>.log, OK/FAILED line in queue.log, never aborts the queue
  local name=$1; shift
  log "=== $name: $*"
  local t=$(date +%s)
  "$@" >> "$OUT/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "--- $name OK ($(( $(date +%s) - t )) s)"; else log "!!! $name FAILED rc=$rc ($(( $(date +%s) - t )) s), see $OUT/$name.log"; echo "FAILED $name"; fi
  return $rc
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '; }
gguf_of() { echo "$GG/$1-gptq-$2-$3-$SUF.gguf"; }   # gguf_of <teacher> <TYPE> <calib>
export_arm() {  # export_arm <teacher> <TYPE> <calib> <datasets...>: GPU; skipped when the GGUF or the calibration set is missing/present
  local teacher=$1 type=$2 calib=$3; shift 3
  local f; f=$(gguf_of "$teacher" "$type" "$calib")
  if [ -f "$f" ]; then log "skip export $teacher $type $calib: $f exists"; return 0; fi
  if [ ! -f "data/calib/$calib.txt" ]; then log "!!! export $teacher $type $calib skipped: data/calib/$calib.txt missing"; return 1; fi
  step "gptq_${teacher}_${type}_${calib}" env PYTHONPATH=third_party/llama.cpp/gguf-py $PY scripts/gptq_export_gguf.py \
      --teacher "$teacher" --type "$type" $GPTQ --calib "$calib" --datasets "$@" --splits test || echo FAILED
}
eval_arm() {  # eval_arm <name> <teacher> <dataset> <ctx> <gguf...>: CPU; every existing file of the list in one runtime session
  local name=$1 teacher=$2 ds=$3 ctx=$4; shift 4
  local files=(); local g
  for g in "$@"; do [ -f "$g" ] && files+=("$g") || log "eval $name: missing $g (arm will show as –)"; done
  if [ ${#files[@]} -eq 0 ]; then log "!!! eval $name skipped: no GGUF file"; return 1; fi
  step "eval_$name" $PY scripts/quant_eval_queries.py --teacher "$teacher" --datasets "$ds" --splits test --bin "$BIN" \
      --threads 8 --workers 1 --ctx "$ctx" --out "$OUT" --ggufs "${files[@]}" || echo FAILED
}

log "CALIBQ queue start (pid $$)"
# soft guard: give a busy GPU up to 30 min to drain before the first GPU job (the steps then fail on their own checks)
n=0
while [ "$(gpu_used)" != "" ] && [ "$(gpu_used)" -gt 800 ] && [ $n -lt 30 ]; do
  [ $n -eq 0 ] && log "GPU still busy ($(gpu_used) MiB used): waiting up to 30 min"
  sleep 60; n=$((n + 1))
done
log "GPU memory used at start: $(gpu_used) MiB"

# ---- (a) jina-v5-small / legal-cs: five exports (~7 min each), then one native evaluation over all eight clients ----
J=jina-v5-small
export_arm $J Q2_K generic_wikitext    legal-cs
export_arm $J Q2_K generic_cs          legal-cs
export_arm $J Q3_K generic_wikitext    legal-cs
export_arm $J Q3_K generic_cs          legal-cs
export_arm $J Q3_K legal-cs_corpus_only legal-cs
eval_arm legal $J legal-cs 512 \
  "$(gguf_of $J Q2_K generic_wikitext)" "$(gguf_of $J Q2_K generic_cs)" "$(gguf_of $J Q2_K legal-cs_corpus_only)" "$(gguf_of $J Q2_K legal-cs_synth_only)" \
  "$(gguf_of $J Q3_K generic_wikitext)" "$(gguf_of $J Q3_K generic_cs)" "$(gguf_of $J Q3_K legal-cs_corpus_only)" "$(gguf_of $J Q3_K legal-cs_synth_only)"

# ---- (b) harrier-0.6b / scidocs: the generic Q2_K arm, evaluated together with the two existing clients ----
H=harrier-0.6b
export_arm $H Q2_K generic_wikitext scidocs scifact
eval_arm scidocs $H scidocs 1024 \
  "$(gguf_of $H Q2_K generic_wikitext)" "$(gguf_of $H Q2_K scidocs_corpus_only)" "$(gguf_of $H Q2_K scidocs_synth_only)"

# ---- (c) harrier-0.6b / arguana + nfcorpus on Q3_K: generic (one file for both) vs synthetic queries per corpus ----
export_arm $H Q3_K generic_wikitext arguana nfcorpus
export_arm $H Q3_K $ARG_SYNTH        arguana
export_arm $H Q3_K nfcorpus_synth_only nfcorpus
eval_arm arguana  $H arguana  1024 "$(gguf_of $H Q3_K generic_wikitext)" "$(gguf_of $H Q3_K $ARG_SYNTH)"
eval_arm nfcorpus $H nfcorpus 1024 "$(gguf_of $H Q3_K generic_wikitext)" "$(gguf_of $H Q3_K nfcorpus_synth_only)"

# ---- table: results/tables/calib_question.md (existing rows of legal_local / scidocs_local are merged in) ----
step table $PY scripts/calib_question_table.py || echo FAILED
log "CALIBQ_DONE"
