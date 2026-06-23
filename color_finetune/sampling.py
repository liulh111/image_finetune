import os
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from .common import make_imagenet_labels, unwrap_model
from .rewards import COLOR_TO_INDEX


DEFAULT_GUIDANCE_LEVELS = (0.0, 1.0, 2.0, 3.0, 10.0)


def add_training_sample_args(parser):
    parser.add_argument("--sample_every", type=int, default=0)
    parser.add_argument("--sample_k", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=250)
    parser.add_argument("--sample_method", choices=("ddim", "ddpm"), default="ddpm")
    parser.add_argument("--sample_ddim_eta", type=float, default=0.0)
    parser.add_argument(
        "--sample_guidance_levels",
        default=",".join(f"{x:g}" for x in DEFAULT_GUIDANCE_LEVELS),
    )


def parse_guidance_levels(spec):
    levels = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not levels:
        raise ValueError("at least one guidance level is required")
    return levels


def level_token(levels):
    parts = []
    for level in levels:
        text = f"{level:g}".replace("-", "m").replace(".", "p")
        parts.append(text)
    return "-".join(parts)


class BehaviorGuidanceModel(nn.Module):
    """Classifier-free-style guidance from separate unconditional/conditional priors."""

    def __init__(self, uncond_model, cond_model):
        super().__init__()
        self.uncond_model = uncond_model
        self.cond_model = cond_model

    def forward(self, x, timesteps, y=None, guidance_scale=1.0):
        scale = float(guidance_scale)
        if scale == 0.0:
            return self.uncond_model(x, timesteps)
        if y is None:
            raise ValueError("class labels are required when guidance_scale != 0")

        cond = self.cond_model(x, timesteps, y)
        if scale == 1.0:
            return cond

        uncond = self.uncond_model(x, timesteps)
        c = x.shape[1]
        if cond.shape[1] == c * 2 and uncond.shape[1] == c * 2:
            eps_uncond, var_uncond = torch.split(uncond, c, dim=1)
            eps_cond, var_cond = torch.split(cond, c, dim=1)
            eps = eps_uncond + scale * (eps_cond - eps_uncond)
            if scale <= 0.0:
                var = var_uncond
            elif scale >= 1.0:
                var = var_cond
            else:
                var = var_uncond + scale * (var_cond - var_uncond)
            return torch.cat([eps, var], dim=1)

        return uncond + scale * (cond - uncond)


def make_color_labels(batch_size, color, device):
    return torch.full(
        (batch_size,), COLOR_TO_INDEX[color], device=device, dtype=torch.long
    )


def build_cep_cond_fn(energy, eta, color_y, y):
    def cond_fn(x, t, _kwargs):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            value = energy(x_in, t, color_y, y)
            grad = torch.autograd.grad((value / eta).sum(), x_in)[0]
        return grad

    return cond_fn


def run_sample_loop(
    diffusion,
    model,
    shape,
    device,
    sample_method,
    steps,
    ddim_eta=0.0,
    model_kwargs=None,
    cond_fn=None,
    guidance_scale=1.0,
    noise=None,
    progress=False,
):
    if sample_method == "ddim":
        sample_diffusion = diffusion.respaced(f"ddim{steps}")
        return sample_diffusion.ddim_sample_loop(
            model,
            shape,
            device=device,
            steps=sample_diffusion.num_timesteps,
            eta=ddim_eta,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            guidance_scale=guidance_scale,
            noise=noise,
            progress=progress,
        )
    if sample_method == "ddpm":
        sample_diffusion = diffusion.respaced(str(steps))
        return sample_diffusion.p_sample_loop(
            model,
            shape,
            device=device,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            guidance_scale=guidance_scale,
            noise=noise,
            progress=progress,
        )
    raise ValueError(f"unknown sample method: {sample_method}")


def save_grid(samples, path, nrow):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image((samples.clamp(-1, 1) + 1) * 0.5, path, nrow=nrow)
    return path


def _sample_labels(k, class_cond, device):
    return make_imagenet_labels(k, class_cond, device)


def save_training_guidance_grid(
    *,
    kind,
    diffusion,
    model,
    behavior_model=None,
    out_dir,
    step,
    color,
    class_cond,
    device,
    k,
    sample_method,
    steps,
    ddim_eta,
    guidance_levels,
    eta=None,
    progress=False,
):
    model = unwrap_model(model)
    if behavior_model is not None:
        behavior_model = unwrap_model(behavior_model)
    was_training = model.training
    model.eval()
    if behavior_model is not None:
        behavior_was_training = behavior_model.training
        behavior_model.eval()
    else:
        behavior_was_training = False
    labels = _sample_labels(k, class_cond, device)
    color_y = make_color_labels(k, color, device)
    noise = torch.randn(k, 3, 256, 256, device=device)
    outputs = []
    for level in guidance_levels:
        if kind == "cep":
            if eta is None:
                raise ValueError("eta is required for CEP guidance sampling")
            if behavior_model is None:
                raise ValueError("CEP sampling requires the behavior model")
            kwargs = {"y": labels} if class_cond else {}
            cond_fn = (
                None
                if float(level) == 0.0
                else build_cep_cond_fn(model, eta, color_y, labels)
            )
            sample_model = behavior_model
            kwargs_model = kwargs
        elif kind == "bdpo":
            kwargs = {"color_y": color_y, "adapter_scale": float(level)}
            if class_cond:
                kwargs["y"] = labels
            cond_fn = None
            sample_model = model
            kwargs_model = kwargs
        else:
            raise ValueError(f"unknown training sample kind: {kind}")

        outputs.append(
            run_sample_loop(
                diffusion,
                sample_model,
                (k, 3, 256, 256),
                device,
                sample_method,
                steps,
                ddim_eta=ddim_eta,
                model_kwargs=kwargs_model,
                cond_fn=cond_fn,
                guidance_scale=float(level),
                noise=noise,
                progress=progress,
            )
        )
    if was_training:
        model.train()
    if behavior_was_training:
        behavior_model.train()

    grid = torch.stack(outputs, dim=1).reshape(k * len(guidance_levels), 3, 256, 256)
    sample_dir = Path(out_dir) / "samples"
    name = (
        f"{kind}_step_{step:07d}_color_{color}_K{k}_{sample_method}"
        f"_steps{steps}_levels_{level_token(guidance_levels)}.png"
    )
    return save_grid(grid, str(sample_dir / name), nrow=len(guidance_levels))


def _fork_rng_devices(device):
    if device.type != "cuda":
        return []
    if device.index is None:
        return [torch.cuda.current_device()]
    return [device.index]


def save_behavior_guidance_grids(
    *,
    diffusion,
    uncond_model,
    cond_model,
    out_dir,
    device,
    k,
    n,
    sample_method,
    steps,
    ddim_eta,
    guidance_scales,
    sample_seed=0,
    progress=False,
):
    model = BehaviorGuidanceModel(uncond_model, cond_model)
    was_training = model.training
    model.eval()

    row_labels = make_imagenet_labels(k, True, device)
    labels = row_labels.repeat_interleave(n)
    shape = (k * n, 3, 256, 256)
    noise = torch.randn(*shape, device=device)
    paths = []

    rng_devices = _fork_rng_devices(device)
    for scale in guidance_scales:
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(int(sample_seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(sample_seed))
            samples = run_sample_loop(
                diffusion,
                model,
                shape,
                device,
                sample_method,
                steps,
                ddim_eta=ddim_eta,
                model_kwargs={"y": labels, "guidance_scale": float(scale)},
                noise=noise,
                progress=progress,
            )
        name = (
            f"behavior_scale_{level_token([scale])}_K{k}_N{n}_{sample_method}"
            f"_steps{steps}_eta{ddim_eta:g}.png"
        )
        paths.append(save_grid(samples, str(Path(out_dir) / name), nrow=n))

    if was_training:
        model.train()
    return paths
