#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHUNK_MB="${CHUNK_MB:-256}"
OUT_DIR="${OUT_DIR:-checkpoints/openai}"

python -m color_finetune.download_openai \
  --model all \
  --out_dir "$OUT_DIR" \
  --chunk_mb "$CHUNK_MB"
