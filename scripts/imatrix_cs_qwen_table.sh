#!/usr/bin/env bash
# E5 (pre-registered 2026-09-08 20:35): is the Q2_K token table what breaks Qwen3-Embedding-0.6B on Czech? llama-quantize
# with the generic_cs imatrix, token table q8_0 vs q2_k, Q3_K mixture and Q4_K_M; native eval on legal-cs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=$ROOT/third_party/llama.cpp/build-clang/bin
F16=$ROOT/models/gguf/qwen3-embedding-0.6b-f16.gguf
OUT=$ROOT/models/gguf; RES=$ROOT/results/raw/release_local; T=${THREADS:-8}
IMX=$RES/imatrix_qwen3_generic_cs.gguf
quant() {  # quant <ftype> <table>
  local ft=$1 tab=$2; local dst="$OUT/qwen3-0.6b-imx-$ft-generic_cs-tab$tab.gguf"
  if [ -f "$dst" ]; then echo "[quantize] exists $dst"; return; fi
  "$BIN/llama-quantize.exe" --imatrix "$IMX" --token-embedding-type "$tab" "$F16" "$dst.part" "$ft" "$T" > "$RES/quantize_qwen3_$ft-generic_cs-tab$tab.log" 2>&1 || { tail -20 "$RES/quantize_qwen3_$ft-generic_cs-tab$tab.log"; exit 1; }
  mv "$dst.part" "$dst"; printf "[quantize] %-60s %7.1f MiB\n" "$(basename "$dst")" "$(stat -c %s "$dst" | awk '{print $1/1048576}')"
}
quant Q3_K q8_0; quant Q4_K_M q8_0; quant Q3_K q2_k
"$ROOT/.venv/Scripts/python.exe" "$ROOT/scripts/quant_eval_queries.py" --teacher qwen3-0.6b --datasets legal-cs --splits test \
    --bin "$BIN/llama-embedding.exe" --threads "$T" --workers 1 --ctx 512 --out "$RES" \
    --ggufs "$OUT/qwen3-0.6b-imx-Q3_K-generic_cs-tabq8_0.gguf" "$OUT/qwen3-0.6b-imx-Q4_K_M-generic_cs-tabq8_0.gguf" "$OUT/qwen3-0.6b-imx-Q3_K-generic_cs-tabq2_k.gguf" 2>&1 | tee -a "$RES/e5_table_eval.log"
echo E5_DONE
