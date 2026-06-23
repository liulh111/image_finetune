# CEP/BDPO Color Fine-Tuning

This package is independent from `images/` and `flow-rl/`. It only contains the
code needed for 256x256 OpenAI guided-diffusion color fine-tuning.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate cep-bdpo-color
```

The environment is pinned to PyTorch 2.3.1 + CUDA 12.1, which is suitable for
8-card A100 servers.

## Model Download

On a new server, run one script from the repository root:

```bash
scripts/download_models.sh
```

It downloads both checkpoints in chunks and verifies MD5:

```text
checkpoints/openai/256x256_diffusion_uncond.pt
checkpoints/openai/256x256_diffusion.pt
```

Set `CHUNK_MB=128` or another value if the VPN needs smaller chunks:

```bash
CHUNK_MB=128 scripts/download_models.sh
```

If the proxy slows down large downloads, disable it before launching the script:

```bash
p_off
scripts/download_models.sh
```

If the server must use the proxy, enable it and use smaller chunks:

```bash
p_on
CHUNK_MB=128 scripts/download_models.sh
```

## What Was Removed

`Cache behavior samples` meant pre-generating images from the behavior diffusion
prior into an `.npz` file and training from that cache. It saves repeated
sampling time, but adds an extra data-generation stage and changes the workflow.
It is removed. Training reads ImageNet-style image files directly from
`--data_dir`, matching the CEP image experiment setup.

`amp` meant automatic mixed precision. It is removed from the user interface to
avoid another execution mode. The code runs in regular precision; CEP defaults
to the paper's 8-GPU global batch size, while BDPO keeps a smaller per-GPU
batch because it evaluates actor, behavior, and value networks in each update.

## Objective

Both methods use the same target distribution:

```text
pi*(x) proportional to pi_behavior(x) * exp(Q(x) / eta)
```

Equivalently, they optimize:

```text
max E[Q(x)] - eta * KL(pi_theta || pi_behavior)
```

`Q(x)` is the differentiable hue reward at `t=0`, so BDPO does not learn a
separate `t=0` value network. `V_0(x)` is evaluated directly by the analytic
reward. CEP uses `beta=50` in the paper, so this code sets `eta=1/beta=0.02`
by default.

CEP's energy network and BDPO's value network both use the OpenAI
classifier-style `EncoderUNetModel`. BDPO's actor is an OpenAI U-Net-style
residual epsilon adapter with the same width, block count, channel multipliers,
attention resolutions, head width, up/down residual blocks, and scale-shift
normalization as the CEP classifier backbone, but with a decoder outputting
pixel-level epsilon residuals.

## 8-GPU Training

All training commands support:

```bash
torchrun --standalone --nproc_per_node=8 -m color_finetune.train_...
```

`--batch_size` is per GPU.

Training `x0` is always loaded from `--data_dir`. For unconditional diffusion,
labels are ignored. For conditional diffusion, labels are inferred exactly like
the CEP/OpenAI ImageNet loader: recursively list image files, take the filename
prefix before `_` such as `n02033041`, sort the unique prefixes, and use that
position as the class index.

Conditional training writes `label_info.json` in `--out_dir`, containing the
exact `class_to_idx` mapping used for that run.

Each run trains one target color. Choose it with `--color red`, `--color green`,
or `--color blue`. To run all three colors, launch three separate jobs.

### CEP, Unconditional Prior, One Color

```bash
COLOR=red
scripts/train_cep_color_uncond.sh
```

Equivalent explicit command:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m color_finetune.train_cep \
  --model_path checkpoints/openai/256x256_diffusion_uncond.pt \
  --data_dir Data/train \
  --color red \
  --eta 0.02 \
  --batch_size 32 \
  --steps 500000 \
  --weight_decay 0 \
  --sample_every 1000 \
  --sample_k 4 \
  --sample_method ddpm \
  --sample_steps 250 \
  --out_dir runs/cep_uncond_red
```

### BDPO, Unconditional Prior, One Color

```bash
COLOR=red REVERSE_SAMPLES=10
scripts/train_bdpo_color_uncond.sh
```

Equivalent explicit command:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m color_finetune.train_bdpo \
  --model_path checkpoints/openai/256x256_diffusion_uncond.pt \
  --data_dir Data/train \
  --color red \
  --eta 0.02 \
  --actor_lr 1e-5 \
  --value_lr 3e-4 \
  --reverse_samples 10 \
  --batch_size 1 \
  --steps 500000 \
  --weight_decay 0 \
  --sample_every 1000 \
  --sample_k 4 \
  --sample_method ddpm \
  --sample_steps 250 \
  --out_dir runs/bdpo_uncond_red
```

`--reverse_samples N` controls BDPO's one-step reverse target averaging. It
samples `N` versions of `x_{t-1}` from the actor reverse Gaussian and averages
their value/reward before subtracting the same one-step KL penalty.

### Conditional Prior

Add `--class_cond` and use the conditional checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m color_finetune.train_cep \
  --model_path checkpoints/openai/256x256_diffusion.pt \
  --data_dir Data/train \
  --class_cond \
  --color red \
  --eta 0.02 \
  --batch_size 32 \
  --steps 500000 \
  --weight_decay 0 \
  --out_dir runs/cep_cond_red
```

For conditional experiments, use a standard ImageNet-style `--data_dir` so the
sorted filename prefixes match the pretrained model's class order. A single
class directory can still be used for quick loader/debug smoke tests, but its
label mapping is directory-local.

## Sampling

### CEP Sampling Rules

The CEP paper follows OpenAI guided-diffusion for image sampling. For ImageNet
256 color guidance, the relevant rules are:

- The diffusion prior is frozen. Color control is applied by guidance during
  reverse sampling, not by sampling an already fine-tuned prior without
  guidance.
- The default sampler is OpenAI's respaced ancestral sampler:
  `timestep_respacing 250`, equivalent here to `--sample_method ddpm
  --sample_steps 250`.
- DDIM examples use `timestep_respacing ddim25 --use_ddim True`, equivalent
  here to `--sample_method ddim --sample_steps 25 --sample_ddim_eta 0`.
- CEP sweeps guidance scales `0,0.25,0.5,1,1.5,2,2.5,3,5,10`, which is the
  default grid here.
- For the color experiment, CEP uses one trained energy guidance model per
  target color and changes the sampler guidance scale `s` at inference time.

### Training-Time Guidance Grid

Set `--sample_every M` to periodically save a full guidance grid during
training. The default sampler is CEP-style `ddpm/250`. The default levels are
`s=0,0.25,0.5,1,1.5,2,2.5,3,5,10`, so the saved image has `K` rows and 10
columns:

```text
row i: same x_T, same reverse noise, and same label; columns sweep s
```

For CEP, `s` is the classifier/energy guidance scale. For BDPO, `s` scales the
actor-behavior reverse-transition residual: `s=0` is the behavior reverse
transition and `s=1` is the trained actor reverse transition. In DDPM grids this
is implemented as `mean_behavior + s * (mean_actor - mean_behavior)`, matching
CEP's reverse-mean-guidance form. `ddpm/250` matches CEP's default sampler. When
you want the same content to stay more visibly comparable across guidance
levels, use `--sample_method ddim --sample_steps 25 --sample_ddim_eta 0`.

The training scripts expose this through environment variables:

```bash
COLOR=red SAMPLE_EVERY=1000 SAMPLE_K=4 \
scripts/train_cep_color_uncond.sh
```

Files are written under `runs/.../samples/` with `K`, method, steps, and levels
encoded in the filename.

### Behavior Prior Samples

This can be run before training. It loads the OpenAI unconditional and
conditional behavior priors, samples `K x N` images, and saves one grid per
guidance scale. Each row uses one ImageNet class label and contains `N` samples
from that same class. The default scales are
`0,0.25,0.5,1,1.5,2,2.5,3,5,10`; `s=0` is exactly the unconditional prior,
`s=1` is exactly the conditional prior, and larger values use
classifier-free-style extrapolation from the two separate priors.

```bash
K=4 N=8 SCALES=0,0.25,0.5,1,1.5,2,2.5,3,5,10 METHOD=ddpm STEPS=250 scripts/sample_behavior.sh
```

Equivalent explicit command:

```bash
python -m color_finetune.sample_behavior \
  --k 4 \
  --n 8 \
  --guidance_scales 0,0.25,0.5,1,1.5,2,2.5,3,5,10 \
  --sample_method ddpm \
  --steps 250 \
  --out_dir runs/behavior_samples
```

Output filenames include the guidance scale, `K`, `N`, sample method, step
count, and DDIM eta. No JSON sidecar is written.

Standalone sampling from trained CEP/BDPO checkpoints is intentionally not
implemented yet; training-time evaluation grids are written when
`--sample_every` is positive.
