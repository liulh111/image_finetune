# CEP/BDPO Image Color Fine-Tuning

This repository compares CEP-style energy guidance and a BDPO-style residual
diffusion actor on the ImageNet 256 color-guidance experiment.

The shared objective is

```text
max_pi E[Q(x)] - eta * KL(pi || pi_behavior)
```

so the target sampler distribution is

```text
pi*(x) proportional to pi_behavior(x) * exp(Q(x) / eta)
```

CEP writes the same coefficient as `1 / beta`. The paper uses `beta = 50` for
energy-guided image synthesis, so this code uses

```text
eta = 1 / 50 = 0.02
```

by default. This is the value that keeps the color experiment numerically
aligned with the original CEP hyperparameter.

## Repository Layout

```text
papers/
  cep/                 CEP paper TeX source.
  bdpo/                BDPO paper TeX source.

images/                Original CEP image code, based on OpenAI guided-diffusion.
flow-rl/               Original BDPO implementation.

color_finetune/        Standalone PyTorch implementation for this experiment.
scripts/               Launch scripts for model download, behavior sampling,
                       CEP training, and BDPO training.
environment.yml        Conda environment for the standalone implementation.
```

`images/` and `flow-rl/` are kept as reference implementations. The runnable
experiment code lives in `color_finetune/`.

## Alignment Audit Summary

The intended interpretation of the current code is:

```text
CEP is the reference baseline. It should be able to reproduce the CEP
energy-guided ImageNet 256 color experiment, modulo stochastic training
variance and the deliberate one-color-per-run simplification.

BDPO is the experimental method. Its behavior prior, reward, eta, sampler,
guidance-scale sweep, and evaluation grids are aligned with CEP so the result
can be used to judge whether BDPO works in this setting.
```

Reference files used for alignment:

```text
papers/cep/6-image.tex
papers/cep/0-appendix.tex
images/scripts/classifier_train.py
images/scripts/classifier_sample_color.py
images/scripts/color_exp.sh
papers/bdpo/text/4.methods.tex
flow-rl/flowrl/agent/offline/bdpo/bdpo.py
flow-rl/scripts/d4rl/bdpo.sh
```

## What Is Being Matched

### Behavior Diffusion

The behavior model is the frozen OpenAI ImageNet 256 guided-diffusion prior:

```text
checkpoints/openai/256x256_diffusion_uncond.pt
```

The architecture and diffusion schedule match the CEP image script:

```text
image_size=256
class_cond=False for the default experiment
learn_sigma=True
noise_schedule=linear
diffusion_steps=1000
timestep_respacing=250 for default DDPM sampling
num_channels=256
num_head_channels=64
num_res_blocks=2
resblock_updown=True
use_scale_shift_norm=True
```

The conditional checkpoint is also supported, but the default scripts run the
unconditional color experiment.

### Color Reward

The reward is `Q(x) = -E(x)`, where CEP defines

```text
E(x) = mean circular hue distance to target hue + low-saturation penalty
```

The target hues are:

```text
red   = 0
green = 2*pi/3
blue  = 4*pi/3
```

The low-saturation penalty follows the CEP appendix: if average saturation is
below `0.1`, add `3 * (0.1 - mean_saturation)` to the energy. Each training run
targets one color, selected by `--color red|green|blue`.

### CEP Implementation

CEP uses the normalized contrastive objective:

```text
target_i = softmax(Q(x0_i) / eta) over the batch
pred_i   = softmax(f_phi(x_t_i, t_i)) over the batch
loss     = -sum_i target_i * log pred_i
```

`f_phi` is trained as the scaled log weight `Q_t / eta`. Sampling uses
`grad_x f_phi(x_t, t)` directly, then multiplies it by the guidance scale `s`.
This matches the original guided-diffusion classifier guidance style, where the
classifier output already contains the temperature scale.

By default the CEP energy model uses the OpenAI guided-diffusion
`EncoderUNetModel` classifier architecture used in the paper:

```text
classifier_width=128
classifier_depth=2
classifier_attention_resolutions=32,16,8
classifier_pool=attention
classifier_resblock_updown=True
classifier_use_scale_shift_norm=True
```

The default CEP hyperparameters are the image-experiment values:

```text
steps=500000
global batch size=256 on 8 GPUs, via --batch_size 32 per GPU
lr=3e-4
weight_decay=0
eta=0.02
```

### BDPO Implementation

The image BDPO variant follows the BDPO lower-level diffusion MDP:

```text
V_0(x_0) = Q(x_0)
V_t(x_t) = E[V_{t-1}(x_{t-1})] - eta * KL(p_actor(x_{t-1}|x_t) || p_behavior(x_{t-1}|x_t))
```

The actor objective is the same one-step improvement target:

```text
maximize E[V_{t-1}(x_{t-1})] - eta * one_step_KL
```

The original BDPO tricks for offline RL stability are intentionally removed
here:

```text
no Q ensemble
no lower-confidence-bound target
no target networks
no environment-level TD critic
```

The behavior model is frozen. The actor is a residual epsilon adapter initialized
at zero, so the unscaled actor is initially exactly the behavior prior and the
trained actor represents the learned actor-behavior local policy shift. The
one-step KL is reduced by summing over image dimensions by default
(`--kl_reduce sum`), which corresponds to the actual Gaussian KL rather than a
per-pixel mean.

`--reverse_samples` defaults to `10`, matching flow-rl's `num_samples=10`. The
implementation evaluates those reverse samples in one vectorized forward pass;
this is appropriate for the intended 8xA100 run. Lower it if memory is tight.

BDPO defaults are chosen to keep the comparison close to CEP:

```text
same frozen OpenAI ImageNet 256 unconditional behavior prior
same Q(x), eta=0.02, data loader, and color target
same 250-step DDPM evaluation sampler
same guidance scale grid
weight_decay=0, matching flow-rl's Adam usage
reverse_samples=10, matching flow-rl's num_samples=10
```

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate cep-bdpo-color
```

Download OpenAI checkpoints:

```bash
scripts/download_models.sh
```

The downloader writes and verifies:

```text
checkpoints/openai/256x256_diffusion_uncond.pt
checkpoints/openai/256x256_diffusion.pt
```

Set `CHUNK_MB=128` if the connection needs smaller download chunks.

## Data

Training reads ImageNet-style image files from `--data_dir`. The default scripts
expect:

```text
Data/train
```

For unconditional runs, labels are ignored. For conditional runs, labels are
inferred from the filename prefix before `_`, matching OpenAI's ImageNet loader
convention.

## Run CEP

Default unconditional red run:

```bash
COLOR=red scripts/train_cep_color_uncond.sh
```

Useful overrides:

```bash
COLOR=green SAMPLE_EVERY=5000 scripts/train_cep_color_uncond.sh
COLOR=blue DATA_DIR=/path/to/imagenet/train scripts/train_cep_color_uncond.sh
```

Equivalent command:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m color_finetune.train_cep \
  --model_path checkpoints/openai/256x256_diffusion_uncond.pt \
  --data_dir Data/train \
  --color red \
  --eta 0.02 \
  --batch_size 32 \
  --steps 500000 \
  --lr 3e-4 \
  --weight_decay 0 \
  --energy_arch openai_classifier \
  --out_dir runs/cep_uncond_red
```

## Run BDPO

Default unconditional red run:

```bash
COLOR=red scripts/train_bdpo_color_uncond.sh
```

Equivalent command:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m color_finetune.train_bdpo \
  --model_path checkpoints/openai/256x256_diffusion_uncond.pt \
  --data_dir Data/train \
  --color red \
  --eta 0.02 \
  --kl_reduce sum \
  --reverse_samples 10 \
  --batch_size 1 \
  --steps 500000 \
  --weight_decay 0 \
  --out_dir runs/bdpo_uncond_red
```

## Evaluation Grids

Set `SAMPLE_EVERY` or `--sample_every` to write training-time grids under
`runs/.../samples/`.

The default evaluation sampler matches CEP Appendix Figures 9 and 10:

```text
method=ddpm
steps=250
guidance scales = 0,0.25,0.5,1,1.5,2,2.5,3,5,10
```

Each row uses the same initial `x_T`, the same class label when applicable, and
the same DDPM reverse-noise sequence across all guidance scales. This mirrors
the fixed-seed visual comparison in the CEP appendix.

For deterministic DDIM-style debug grids:

```bash
SAMPLE_METHOD=ddim SAMPLE_STEPS=25 SAMPLE_DDIM_ETA=0 SAMPLE_EVERY=5000 \
  scripts/train_cep_color_uncond.sh
```

In CEP grids, `s` scales the energy-guided behavior mean shift. In BDPO grids,
`s` scales the actor-behavior reverse-transition residual:

```text
s=0: behavior reverse transition
s=1: trained actor reverse transition
s>1: extrapolated actor-behavior residual
```

For DDPM grids this is implemented explicitly as
`mean_behavior + s * (mean_actor - mean_behavior)`, so BDPO is evaluated in the
same reverse-mean-guidance form as CEP.

CEP DDPM grids use OpenAI guided-diffusion's classifier-guidance update:

```text
mean_guided = mean_behavior + s * variance * grad_x f_phi(x_t, t)
```

The same initial `x_T` and reverse noise are reused for every column in both CEP
and BDPO grids, so changes across a row are attributable to the guidance scale.

## Behavior Samples

To inspect the frozen behavior prior before training:

```bash
K=4 N=8 scripts/sample_behavior.sh
```

This saves one grid per guidance scale in `runs/behavior_samples/`.

## Checkpoints

CEP checkpoints contain:

```text
model
optimizer
step
config
```

BDPO checkpoints contain:

```text
actor
value
actor_optimizer
value_optimizer
step
config
```

The `config` records the important alignment fields, including `eta`,
`energy_arch`, `kl_reduce`, `reverse_samples`, sampler settings, and color.

## Intentional Differences From The Papers

CEP:

```text
one target color per run
eta notation instead of beta notation
standalone PyTorch implementation instead of editing images/
```

BDPO:

```text
analytic image reward replaces environment Q learning
no offline-RL ensembles, LCB, or target networks
frozen OpenAI behavior prior instead of pretraining behavior in this repo
residual image actor instead of a full actor diffusion model initialized from behavior
```

These are deliberate experiment choices. The core distributions and update
targets remain aligned with `max E[Q] - eta KL`.

## Remote Smoke Tests

The local development machine used for this audit did not have `torch`
installed, so only syntax/static checks were run locally. On the 8xA100 machine,
run these before launching long jobs:

```bash
python -m compileall color_finetune

STEPS=1 SAMPLE_EVERY=1 SAMPLE_K=1 GPUS=1 \
  scripts/train_cep_color_uncond.sh

STEPS=1 SAMPLE_EVERY=1 SAMPLE_K=1 GPUS=1 REVERSE_SAMPLES=1 \
  scripts/train_bdpo_color_uncond.sh
```

The expected outcome is that each smoke test writes one checkpoint and one grid
under `runs/.../samples/` without shape, checkpoint-loading, or sampler errors.
