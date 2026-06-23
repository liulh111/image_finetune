#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

K="${K:-4}"
N="${N:-4}"
SCALES="${SCALES:-0,0.25,0.5,1,1.5,2,2.5,3,5,10}"
METHOD="${METHOD:-ddpm}"
STEPS="${STEPS:-250}"
DDIM_ETA="${DDIM_ETA:-0.0}"
OUT_DIR="${OUT_DIR:-runs/behavior_samples}"

python -m color_finetune.sample_behavior \
  --k "$K" \
  --n "$N" \
  --guidance_scales "$SCALES" \
  --sample_method "$METHOD" \
  --steps "$STEPS" \
  --ddim_eta "$DDIM_ETA" \
  --out_dir "$OUT_DIR"
