#!/usr/bin/env bash
# E8: Qwen3-Embedding-0.6B Q4_K_M + q4_0 token table with the English imatrix, evaluated on the four English corpora.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
BIN=third_party/llama.cpp/build-clang/bin; PY=.venv/Scripts/python.exe; OUT=results/raw/release_local
dst=models/gguf/qwen3-0.6b-imx-Q4_K_M-generic_wikitext-tabq4_0.gguf
[ -f "$dst" ] || { "$BIN/llama-quantize.exe" --imatrix $OUT/imatrix_qwen3_generic_wikitext.gguf --token-embedding-type q4_0 models/gguf/qwen3-embedding-0.6b-f16.gguf "$dst.part" Q4_K_M 8 > $OUT/quantize_qwen3_Q4_K_M-generic_wikitext-tabq4_0.log 2>&1 && mv "$dst.part" "$dst"; }
ls -l "$dst"
$PY scripts/quant_eval_queries.py --teacher qwen3-0.6b --datasets scifact nfcorpus arguana scidocs --splits test --bin $BIN/llama-embedding.exe --threads 8 --workers 1 --ctx 1024 --out $OUT --ggufs "$dst" 2>&1 | tee -a $OUT/e8_eval_en.log
echo E8_DONE
