#!/usr/bin/env bash
# llama-quantize --imatrix baseline for the Czech branch (jina-v5-small, legal-cs index unchanged), pre-registered in
# research/lab_log.md 2026-09-08 ("chybějící baseline české větve"). Windows git-bash, build-clang binaries, CPU only.
#   bash scripts/imatrix_cs_legal.sh all      (steps: imatrix | quantize | eval | all; every step skips existing outputs)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=$ROOT/third_party/llama.cpp/build-clang/bin
F16=$ROOT/models/gguf/jina-v5-small-f16.gguf
OUT=$ROOT/models/gguf
RES=$ROOT/results/raw/imatrix_cs_local; mkdir -p "$RES"
T=${THREADS:-8}
CHUNKS=585   # x 512 = 299 520 tokens = the GPTQ generic_cs budget (ps20000-t300k)

imx_path() { echo "$RES/imatrix_$1.gguf"; }

imatrix() {  # imatrix <calib> [chunks]
  local calib=$1 chunks=${2:-} dst; dst=$(imx_path "$calib")
  if [ -f "$dst" ]; then echo "[imatrix] $dst exists"; return; fi
  echo "[imatrix] $calib: -c 512 ${chunks:+--chunks $chunks} $T threads"
  "$BIN/llama-imatrix.exe" -m "$F16" -f "$ROOT/data/calib/$calib.txt" -c 512 -b 2048 -ub 512 ${chunks:+--chunks $chunks} -t "$T" \
      --no-ppl -lv 1 -ofreq 10 -o "$dst" > "$RES/imatrix_$calib.log" 2>&1 || { tail -n 20 "$RES/imatrix_$calib.log"; exit 1; }
  grep -aE "computing over|per pass|stored collected|save" "$RES/imatrix_$calib.log" | tail -n 2 || true
}

quant() {  # quant <ftype> <calib|noimatrix> [--pure]
  local ft=$1 calib=$2 pure=${3:-} tag imx=() dst
  tag="$ft${pure:+pure}-$calib-tabQ2_K"; dst="$OUT/jina-v5-small-imx-$tag.gguf"
  if [ -f "$dst" ]; then echo "[quantize] $dst exists"; return; fi
  [ "$calib" = noimatrix ] || imx=(--imatrix "$(imx_path "$calib")")
  "$BIN/llama-quantize.exe" "${imx[@]}" ${pure:+--pure} --token-embedding-type q2_k "$F16" "$dst.part" "$ft" "$T" \
     > "$RES/quantize_$tag.log" 2>&1 || { tail -n 20 "$RES/quantize_$tag.log"; exit 1; }
  mv "$dst.part" "$dst"
  printf "[quantize] %-60s %7.1f MiB\n" "$(basename "$dst")" "$(stat -c %s "$dst" | awk '{print $1/1048576}')"
}

quantize() {
  quant Q3_K   generic_cs                 # A  K-quant mixture (attn_v/ffn_down/... upgraded)
  quant Q3_K   generic_cs --pure          # B  exactly our client's format
  quant IQ2_M  generic_cs                 # C
  quant IQ3_XXS generic_cs                # D
  quant Q3_K   noimatrix --pure           # E  round-to-nearest control
  quant IQ2_M  legal-cs_synth_only        # F
  quant Q3_K   legal-cs_synth_only --pure # G
}

evaluate() {
  local files=()
  for f in "$OUT"/jina-v5-small-imx-*.gguf; do files+=("$f"); done
  "$ROOT/.venv/Scripts/python.exe" "$ROOT/scripts/quant_eval_queries.py" --teacher jina-v5-small --datasets legal-cs --splits test \
      --bin "$BIN/llama-embedding.exe" --threads "$T" --workers 1 --ctx 512 --out "$RES" --ggufs "${files[@]}" 2>&1 | tee -a "$RES/eval_legal.log"
}

for cmd in "$@"; do
  case "$cmd" in
    imatrix) imatrix generic_cs "$CHUNKS"; imatrix legal-cs_synth_only ;;
    quantize) quantize ;;
    eval) evaluate ;;
    all) imatrix generic_cs "$CHUNKS"; imatrix legal-cs_synth_only; quantize; evaluate ;;
    *) echo "unknown step: $cmd" >&2; exit 2 ;;
  esac
done
echo IMATRIX_CS_LEGAL_DONE
