#!/usr/bin/env bash
# Windows-side (git-bash) twin of scripts/imatrix_compare.sh: collects the importance matrices and quantizes with the
# Windows build of the SAME llama.cpp revision (third_party/llama.cpp/build-clang, scripts/build_llamacpp_upstream_clang.cmd
# + `ninja -C build-clang llama-imatrix`). Used on 2026-09-07 because the WSL VM on this laptop is torn down
# (Wsl/Service/E_UNEXPECTED) whenever the host runs out of memory, which no multi-hour f16 pass survived; the
# quantizers are deterministic reference code, so the files are byte-for-byte what the WSL build would write.
# The measurements (quality / RSS / latency) still run through the WSL bitnet.cpp llama-embedding = the client runtime;
# scripts/imatrix_compare.sh copies the files produced here into /home/fragmea/eq when they exist.
#
#   bash scripts/imatrix_compare_win.sh imatrix quantize        (from the repo root, Windows git-bash)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=$ROOT/third_party/llama.cpp/build-clang/bin
F16=$ROOT/models/gguf/harrier-0.6b-f16.gguf
OUT=$ROOT/models/gguf
RES=$ROOT/results/raw/imatrix_local; mkdir -p "$RES"
T=${THREADS:-12}
CHUNKS=${CHUNKS:-585}   # x 512 tokens = 299 520 tokens = the client's calibration budget (585 windows of 512)

imx_path() { echo "$RES/imatrix_$1_c512x$CHUNKS.gguf"; }

imatrix() {
  for calib in "$@"; do
    local dst; dst=$(imx_path "$calib")
    if [ -f "$dst" ]; then echo "[imatrix] $dst exists"; continue; fi
    echo "[imatrix] $calib: -c 512 --chunks $CHUNKS ($((CHUNKS*512)) tokens), $T threads, 4 chunks per batch"
    "$BIN/llama-imatrix.exe" -m "$F16" -f "$ROOT/data/calib/$calib.txt" -c 512 -b 2048 -ub 512 --chunks "$CHUNKS" -t "$T" \
        --no-ppl -lv 1 -ofreq 10 -o "$dst" > "$RES/imatrix_$calib.log" 2>&1 || { tail -n 20 "$RES/imatrix_$calib.log"; exit 1; }
    grep -aE "computing over|per pass|stored collected" "$RES/imatrix_$calib.log" | tail -n 2
  done
}

quant() {  # quant <ftype> <calib|noimatrix> [--pure]
  local ft=$1 calib=$2 pure=${3:-} tag imx=() dst
  tag="$ft${pure:+pure}-$calib-tabQ2_K"; dst="$OUT/harrier-0.6b-imx-$tag.gguf"
  if [ -f "$dst" ]; then echo "[quantize] $dst exists"; return; fi
  [ "$calib" = noimatrix ] || imx=(--imatrix "$(imx_path "$calib")")
  "$BIN/llama-quantize.exe" "${imx[@]}" ${pure:+--pure} --token-embedding-type q2_k "$F16" "$dst.part" "$ft" "$T" \
     > "$RES/quantize_$tag.log" 2>&1 || { tail -n 20 "$RES/quantize_$tag.log"; exit 1; }
  mv "$dst.part" "$dst"
  printf "[quantize] %-60s %7.1f MiB\n" "$(basename "$dst")" "$(stat -c %s "$dst" | awk '{print $1/1048576}')"
}

quantize() {
  quant IQ2_M generic_wikitext            # A
  quant IQ2_M scifact_corpus_only         # B
  quant Q2_K  scifact_corpus_only         # C  (llama.cpp "Q2_K" ftype = K-quant MIXTURE: attn_v/ffn_down/attn_output get Q3_K/Q4_K)
  quant Q2_K  generic_wikitext            # D
  quant Q2_K  noimatrix                   # E  round-to-nearest on the same grid
  quant Q2_K  scifact_corpus_only --pure  # C' every block linear Q2_K = exactly our client's format
  quant Q2_K  generic_wikitext    --pure  # D'
  quant Q2_K  noimatrix           --pure  # E'
  quant IQ2_XS scifact_corpus_only        # F (optional, smaller)
}

for cmd in "$@"; do
  case "$cmd" in
    imatrix) imatrix scifact_corpus_only generic_wikitext ;;
    imatrix-*) imatrix "${cmd#imatrix-}" ;;
    quantize) quantize ;;
    quantize-scifact) quant IQ2_M scifact_corpus_only; quant Q2_K scifact_corpus_only; quant Q2_K scifact_corpus_only --pure; quant IQ2_XS scifact_corpus_only ;;
    quantize-generic) quant IQ2_M generic_wikitext; quant Q2_K generic_wikitext; quant Q2_K generic_wikitext --pure ;;
    quantize-noimatrix) quant Q2_K noimatrix; quant Q2_K noimatrix --pure ;;
    quantize-nfcorpus) quant IQ2_M nfcorpus_corpus_only; quant Q2_K nfcorpus_corpus_only ;;
    *) echo "unknown step: $cmd" >&2; exit 2 ;;
  esac
done
echo IMATRIX_COMPARE_WIN_DONE
