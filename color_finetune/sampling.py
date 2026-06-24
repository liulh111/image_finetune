import os
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from .common import make_imagenet_labels, unwrap_model
from .diffusion import _extract_into_tensor
from .rewards import COLOR_TO_INDEX


DEFAULT_GUIDANCE_LEVELS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0)


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


def parse_step_list(spec):
    steps = [int(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not steps:
        raise ValueError("at least one step count is required")
    if any(step <= 0 for step in steps):
        raise ValueError("step counts must be positive")
    return steps


def level_token(levels):
    parts = []
    for level in levels:
        text = f"{level:g}".replace("-", "m").replace(".", "p")
        parts.append(text)
    return "-".join(parts)


def make_color_labels(batch_size, color, device):
    return torch.full(
        (batch_size,), COLOR_TO_INDEX[color], device=device, dtype=torch.long
    )


def build_cep_cond_fn(energy, color_y, y):
    def cond_fn(x, t, _kwargs):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            value = energy(x_in, t, color_y, y)
            grad = torch.autograd.grad(value.sum(), x_in)[0]
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


def _bdpo_reverse_stats(diffusion, actor, x, t, behavior_kwargs, actor_kwargs):
    behavior = actor.behavior_model
    behavior_stats = diffusion.p_mean_variance(
        behavior,
        x,
        t,
        model_kwargs=behavior_kwargs,
        clip_denoised=True,
    )
    actor_stats = diffusion.p_mean_variance(
        actor,
        x,
        t,
        model_kwargs={**actor_kwargs, "adapter_scale": 1.0},
        clip_denoised=True,
    )
    return behavior_stats, actor_stats


@torch.no_grad()
def run_bdpo_residual_sample_loop(
    diffusion,
    actor,
    shape,
    device,
    sample_method,
    steps,
    ddim_eta=0.0,
    behavior_kwargs=None,
    actor_kwargs=None,
    guidance_scale=1.0,
    noise=None,
    progress=False,
):
    if not hasattr(actor, "behavior_model"):
        raise ValueError("BDPO residual sampling requires an OpenAIResidualEpsAdapter")
    behavior_kwargs = behavior_kwargs or {}
    actor_kwargs = actor_kwargs or {}
    x = torch.randn(*shape, device=device) if noise is None else noise.to(device)
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"noise shape {tuple(x.shape)} does not match {tuple(shape)}")

    scale = float(guidance_scale)
    if sample_method == "ddpm":
        sample_diffusion = diffusion.respaced(str(steps))
        iterator = range(sample_diffusion.num_timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator)
        for i in iterator:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            behavior_stats, actor_stats = _bdpo_reverse_stats(
                sample_diffusion, actor, x, t, behavior_kwargs, actor_kwargs
            )
            mean = behavior_stats["mean"] + scale * (
                actor_stats["mean"] - behavior_stats["mean"]
            )
            step_noise = torch.randn_like(x)
            nonzero = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
            x = (
                mean
                + nonzero
                * torch.exp(0.5 * behavior_stats["log_variance"])
                * step_noise
            )
        return x

    if sample_method == "ddim":
        sample_diffusion = diffusion.respaced(f"ddim{steps}")
        times = np.linspace(
            0,
            sample_diffusion.num_timesteps - 1,
            sample_diffusion.num_timesteps,
            dtype=np.int64,
        )
        pairs = list(zip(times[::-1], np.append(times[:-1][::-1], -1)))
        if progress:
            from tqdm import tqdm

            pairs = tqdm(pairs)
        for i, prev_i in pairs:
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            behavior_stats, actor_stats = _bdpo_reverse_stats(
                sample_diffusion, actor, x, t, behavior_kwargs, actor_kwargs
            )
            eps = behavior_stats["eps"] + scale * (
                actor_stats["eps"] - behavior_stats["eps"]
            )
            pred_xstart = sample_diffusion.predict_xstart_from_eps(x, t, eps).clamp(
                -1, 1
            )
            alpha = _extract_into_tensor(sample_diffusion.alphas_cumprod, t, x.shape)
            if prev_i < 0:
                alpha_prev = torch.ones_like(alpha)
            else:
                prev_t = torch.full(
                    (shape[0],), int(prev_i), device=device, dtype=torch.long
                )
                alpha_prev = _extract_into_tensor(
                    sample_diffusion.alphas_cumprod, prev_t, x.shape
                )
            sigma = (
                ddim_eta
                * torch.sqrt((1 - alpha_prev) / (1 - alpha))
                * torch.sqrt(1 - alpha / alpha_prev)
            )
            step_noise = torch.randn_like(x)
            mean_pred = (
                pred_xstart * torch.sqrt(alpha_prev)
                + torch.sqrt((1 - alpha_prev - sigma**2).clamp_min(0)) * eps
            )
            nonzero = 0.0 if prev_i < 0 else 1.0
            x = mean_pred + nonzero * sigma * step_noise
        return x

    raise ValueError(f"unknown sample method: {sample_method}")


def save_grid(samples, path, nrow):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image((samples.clamp(-1, 1) + 1) * 0.5, path, nrow=nrow)
    return path


def _sample_labels(k, class_cond, device):
    return make_imagenet_labels(k, class_cond, device)


def _capture_rng_state(device):
    state = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state, device):
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


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
    reverse_rng_state = _capture_rng_state(device)
    outputs = []
    for level in guidance_levels:
        _restore_rng_state(reverse_rng_state, device)
        if kind == "cep":
            if behavior_model is None:
                raise ValueError("CEP sampling requires the behavior model")
            kwargs = {"y": labels} if class_cond else {}
            cond_fn = (
                None
                if float(level) == 0.0
                else build_cep_cond_fn(model, color_y, labels)
            )
            sample_model = behavior_model
            kwargs_model = kwargs
        elif kind == "bdpo":
            behavior_kwargs = {"y": labels} if class_cond else {}
            actor_kwargs = {"color_y": color_y}
            if class_cond:
                actor_kwargs["y"] = labels
            outputs.append(
                run_bdpo_residual_sample_loop(
                    diffusion,
                    model,
                    (k, 3, 256, 256),
                    device,
                    sample_method,
                    steps,
                    ddim_eta=ddim_eta,
                    behavior_kwargs=behavior_kwargs,
                    actor_kwargs=actor_kwargs,
                    guidance_scale=float(level),
                    noise=noise,
                    progress=progress,
                )
            )
            continue
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


def save_behavior_prior_grids(
    *,
    diffusion,
    model,
    model_kind,
    out_dir,
    device,
    k,
    n,
    sample_method,
    steps,
    ddim_eta,
    sample_seed=0,
    progress=False,
):
    if model_kind not in {"uncond", "cond"}:
        raise ValueError("model_kind must be 'uncond' or 'cond'")
    was_training = model.training
    model.eval()

    labels = None
    if model_kind == "cond":
        row_labels = make_imagenet_labels(k, True, device)
        labels = row_labels.repeat_interleave(n)
    shape = (k * n, 3, 256, 256)
    noise = torch.randn(*shape, device=device)
    paths = []

    rng_devices = _fork_rng_devices(device)
    for step_count in steps:
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
                int(step_count),
                ddim_eta=ddim_eta,
                model_kwargs={"y": labels} if labels is not None else None,
                noise=noise,
                progress=progress,
            )
        name = (
            f"behavior_{model_kind}_K{k}_N{n}_{sample_method}"
            f"_steps{step_count}_eta{ddim_eta:g}.png"
        )
        paths.append(save_grid(samples, str(Path(out_dir) / name), nrow=n))

    if was_training:
        model.train()
    return paths
