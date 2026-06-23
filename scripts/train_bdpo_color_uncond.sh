#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPUS="${GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-500000}"
ETA="${ETA:-0.02}"
DATA_DIR="${DATA_DIR:-Data/train}"
COLOR="${COLOR:-red}"
REVERSE_SAMPLES="${REVERSE_SAMPLES:-10}"
ACTOR_LR="${ACTOR_LR:-1e-5}"
VALUE_LR="${VALUE_LR:-3e-4}"
SAMPLE_EVERY="${SAMPLE_EVERY:-0}"
SAMPLE_K="${SAMPLE_K:-4}"
SAMPLE_METHOD="${SAMPLE_METHOD:-ddpm}"
SAMPLE_STEPS="${SAMPLE_STEPS:-250}"
SAMPLE_DDIM_ETA="${SAMPLE_DDIM_ETA:-0.0}"
SAMPLE_GUIDANCE_LEVELS="${SAMPLE_GUIDANCE_LEVELS:-0,0.25,0.5,1,1.5,2,2.5,3,5,10}"

torchrun --standalone --nproc_per_node="$GPUS" \
  -m color_finetune.train_bdpo \
  --model_path checkpoints/openai/256x256_diffusion_uncond.pt \
  --data_dir "$DATA_DIR" \
  --color "$COLOR" \
  --eta "$ETA" \
  --actor_lr "$ACTOR_LR" \
  --value_lr "$VALUE_LR" \
  --reverse_samples "$REVERSE_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --sample_every "$SAMPLE_EVERY" \
  --sample_k "$SAMPLE_K" \
  --sample_method "$SAMPLE_METHOD" \
  --sample_steps "$SAMPLE_STEPS" \
  --sample_ddim_eta "$SAMPLE_DDIM_ETA" \
  --sample_guidance_levels "$SAMPLE_GUIDANCE_LEVELS" \
  --out_dir "runs/bdpo_uncond_${COLOR}"
