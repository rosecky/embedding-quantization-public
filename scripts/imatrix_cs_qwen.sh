#!/usr/bin/env bash
# E3/E4 CPU arms (pre-registered 2026-09-08 evening): llama-quantize --imatrix on Qwen3-Embedding-0.6B f16 with a Czech
# (generic_cs) and an English (generic_wikitext) importance matrix; Q4_K_M / IQ3_XXS / Q3_K --pure. Native eval on legal-cs
# (Czech arms) here; the English arms are evaluated by scripts/release_gpu_queue2.sh once the qwen3-0.6b caches exist.
#   bash scripts/imatrix_cs_qwen.sh all      (steps: imatrix | quantize | eval | all; existing outputs are skipped)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=$ROOT/third_party/llama.cpp/build-clang/bin
F16=$ROOT/models/gguf/qwen3-embedding-0.6b-f16.gguf
OUT=$ROOT/models/gguf
RES=$ROOT/results/raw/release_local; mkdir -p "$RES"
T=${THREADS:-8}
CHUNKS=585

imx_path() { echo "$RES/imatrix_qwen3_$1.gguf"; }
imatrix() {
  local calib=$1 dst; dst=$(imx_path "$calib")
  if [ -f "$dst" ]; then echo "[imatrix] $dst exists"; return; fi
  echo "[imatrix] qwen3 $calib: -c 512 --chunks $CHUNKS $T threads"
  "$BIN/llama-imatrix.exe" -m "$F16" -f "$ROOT/data/calib/$calib.txt" -c 512 -b 2048 -ub 512 --chunks "$CHUNKS" -t "$T" \
      --no-ppl -lv 1 -ofreq 10 -o "$dst" > "$RES/imatrix_qwen3_$calib.log" 2>&1 || { tail -n 20 "$RES/imatrix_qwen3_$calib.log"; exit 1; }
}
quant() {  # quant <ftype> <calib> [--pure]
  local ft=$1 calib=$2 pure=${3:-} tag dst
  tag="$ft${pure:+pure}-$calib-tabQ2_K"; dst="$OUT/qwen3-0.6b-imx-$tag.gguf"
  if [ -f "$dst" ]; then echo "[quantize] $dst exists"; return; fi
  "$BIN/llama-quantize.exe" --imatrix "$(imx_path "$calib")" ${pure:+--pure} --token-embedding-type q2_k "$F16" "$dst.part" "$ft" "$T" \
     > "$RES/quantize_qwen3_$tag.log" 2>&1 || { tail -n 20 "$RES/quantize_qwen3_$tag.log"; exit 1; }
  mv "$dst.part" "$dst"
  printf "[quantize] %-60s %7.1f MiB\n" "$(basename "$dst")" "$(stat -c %s "$dst" | awk '{print $1/1048576}')"
}
quantize() {
  quant Q4_K_M generic_cs; quant IQ3_XXS generic_cs; quant Q3_K generic_cs --pure
  quant Q4_K_M generic_wikitext; quant IQ3_XXS generic_wikitext; quant Q3_K generic_wikitext --pure
}
evaluate() {
  "$ROOT/.venv/Scripts/python.exe" "$ROOT/scripts/quant_eval_queries.py" --teacher qwen3-0.6b --datasets legal-cs --splits test \
      --bin "$BIN/llama-embedding.exe" --threads "$T" --workers 1 --ctx 512 --out "$RES" \
      --ggufs "$OUT"/qwen3-0.6b-imx-Q4_K_M-generic_cs-tabQ2_K.gguf "$OUT"/qwen3-0.6b-imx-IQ3_XXS-generic_cs-tabQ2_K.gguf "$OUT"/qwen3-0.6b-imx-Q3_Kpure-generic_cs-tabQ2_K.gguf 2>&1 | tee -a "$RES/e4_imx_eval.log"
}
for cmd in "$@"; do
  case "$cmd" in
    imatrix) imatrix generic_cs; imatrix generic_wikitext ;;
    quantize) quantize ;;
    eval) evaluate ;;
    all) imatrix generic_cs; imatrix generic_wikitext; quantize; evaluate ;;
    *) echo "unknown step: $cmd" >&2; exit 2 ;;
  esac
done
echo IMATRIX_CS_QWEN_DONE
