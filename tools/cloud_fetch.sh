#!/usr/bin/env bash
# One-shot model fetch for the CLOUD verification machine.
#
# Run this on the cloud box (not locally -- local egress measured ~0.6 MB/s).
# Put files straight into ComfyUI/models so no copying is needed:
#
#   COMFY_MODELS_DIR=/mnt/workspace/ComfyUI/models bash tools/cloud_fetch.sh
#
# Override the repo path with H3_REPO if you use a different mirror.
set -euo pipefail

REPO="${H3_REPO:-Comfy-Org/MiniMax-H3}"
OUT="${COMFY_MODELS_DIR:-$PWD/ComfyUI/models}"
MS="https://www.modelscope.cn/models/${REPO}/resolve/master"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FULL="diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
FULL_SIZE=34038894550
PRUNED="diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
PRUNED_SIZE=20970379616
# The test LoRA is a local custom file, not published in the Comfy-Org repo.
DMAD_LORA="dmad_minimax_h3_8step_full_critic_comfyui.safetensors"
DMAD_SIZE=1956196016

echo "models dir : $OUT"
echo "repo       : $REPO (ModelScope primary / hf-mirror fallback)"
mkdir -p "$OUT/diffusion_models" "$OUT/loras"

fetch_one() {
  local rel="$1" expected="$2" model_key="$3"
  local dst="$OUT/$rel"
  local dir; dir="$(dirname "$dst")"
  if [[ -f "$dst" ]] && [[ "$(stat -c%s "$dst" 2>/dev/null || echo 0)" == "$expected" ]]; then
    echo "OK  already complete: $dst"
    return 0
  fi
  echo "GET $rel  ($(( expected / 1024 / 1024 )) MiB)"
  local url="$MS/$rel"
  if command -v aria2c >/dev/null 2>&1; then
    # Fastest on a well-connected cloud host: 16 connections, resumable.
    aria2c --continue=true --max-connection-per-server=16 --split=16 \
      --min-split-size=16M --file-allocation=none --console-log-level=warn \
      --dir="$dir" --out="$(basename "$dst")" "$url"
  else
    # No extra dependency: uses the bundled IPv4-pinned chunked downloader.
    # Resumable, writes directly to $dir, verifies total size itself.
    echo "aria2c not found -> using bundled Python downloader"
    python3 "$HERE/fetch_model.py" --model "$model_key" \
      --connections 32 --out "$dir"
  fi
  local got; got="$(stat -c%s "$dst" 2>/dev/null || echo 0)"
  if [[ "$got" != "$expected" ]]; then
    echo "SIZE MISMATCH: $dst got $got expected $expected" >&2
    return 1
  fi
  echo "OK  $dst"
}

fetch_one "$FULL"   "$FULL_SIZE"   "ref2va_full_int8"
fetch_one "$PRUNED" "$PRUNED_SIZE" "ref2va_pruned_int8"

if [[ -f "$OUT/loras/$DMAD_LORA" ]] && \
   [[ "$(stat -c%s "$OUT/loras/$DMAD_LORA" 2>/dev/null || echo 0)" == "$DMAD_SIZE" ]]; then
  echo "OK  DMAD 8-step LoRA already in place"
elif [[ -n "${H3_LOCAL_LORA:-}" ]] && [[ -f "$H3_LOCAL_LORA" ]]; then
  cp "$H3_LOCAL_LORA" "$OUT/loras/$DMAD_LORA"
  echo "OK  copied LoRA from $H3_LOCAL_LORA"
else
  echo "ACTION NEEDED: upload D:/模型整合/loras/$DMAD_LORA"
  echo "            -> $OUT/loras/"
  echo "   or rerun with: H3_LOCAL_LORA=/path/to/$DMAD_LORA bash tools/cloud_fetch.sh"
fi

echo ""
echo "Done. Now:"
echo "  1) copy ComfyUI-H3-ExactAdaLN/ into ComfyUI/custom_nodes/ and restart"
echo "  2) make sure $DMAD_LORA is in models/loras/"
echo "  3) open workflows/01_bake_exact_adaln.json, queue once"
echo "     -> Bake produces an in-memory table, Save writes output/h3_adaln/*.safetensors"
echo "     -> copy that file anywhere to share it"
echo "  4) open workflows/02_verify_full_vs_pruned.json"
echo "     -> use the Load node's upload button (or put the file in input/),"
echo "        then pick it in the dropdown; refresh the page if it is not listed"
echo "     -> queue, read the H3 Latent Compare report"
