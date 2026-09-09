#!/usr/bin/env bash
# E5 arm (d): Q4_K_M + q4_0 token table (pre-registered 2026-09-08 20:52).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); BIN=$ROOT/third_party/llama.cpp/build-clang/bin
F16=$ROOT/models/gguf/qwen3-embedding-0.6b-f16.gguf; OUT=$ROOT/models/gguf; RES=$ROOT/results/raw/release_local; T=8
dst="$OUT/qwen3-0.6b-imx-Q4_K_M-generic_cs-tabq4_0.gguf"
[ -f "$dst" ] || { "$BIN/llama-quantize.exe" --imatrix "$RES/imatrix_qwen3_generic_cs.gguf" --token-embedding-type q4_0 "$F16" "$dst.part" Q4_K_M $T > "$RES/quantize_qwen3_Q4_K_M-generic_cs-tabq4_0.log" 2>&1; mv "$dst.part" "$dst"; }
ls -l "$dst"
"$ROOT/.venv/Scripts/python.exe" "$ROOT/scripts/quant_eval_queries.py" --teacher qwen3-0.6b --datasets legal-cs --splits test --bin "$BIN/llama-embedding.exe" --threads $T --workers 1 --ctx 512 --out "$RES" --ggufs "$dst" 2>&1 | tee -a "$RES/e5_table_eval.log"
echo E5D_DONE
