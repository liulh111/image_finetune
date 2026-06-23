import math

import numpy as np
import torch


def mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


def sum_flat(x):
    return x.sum(dim=tuple(range(1, x.ndim)))


def get_named_beta_schedule(name, steps):
    if name == "linear":
        scale = 1000 / steps
        return np.linspace(scale * 0.0001, scale * 0.02, steps, dtype=np.float64)
    raise ValueError(f"unknown beta schedule: {name}")


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for stride in range(1, num_timesteps):
                if len(range(0, num_timesteps, stride)) == desired_count:
                    return set(range(0, num_timesteps, stride))
            raise ValueError(
                f"cannot create exactly {desired_count} DDIM steps from "
                f"{num_timesteps} diffusion steps"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        start_idx += size
    return set(all_steps)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if not torch.is_tensor(arr):
        arr = torch.from_numpy(arr)
    res = arr.to(device=timesteps.device, dtype=torch.float32)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    def __init__(self, betas):
        betas = np.array(betas, dtype=np.float64)
        if (betas <= 0).any() or (betas > 1).any():
            raise ValueError("betas must be in (0, 1]")
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    @classmethod
    def openai_256(cls):
        return cls(get_named_beta_schedule("linear", 1000))

    def respaced(self, section_counts):
        return SpacedGaussianDiffusion(
            use_timesteps=space_timesteps(self.num_timesteps, section_counts),
            betas=self.betas,
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        log_var = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def p_mean_variance_from_output(self, model_output, x, t, clip_denoised=True):
        c = x.shape[1]
        if model_output.shape[1] != c * 2:
            raise ValueError("expected learned-sigma output with 2 * input channels")
        eps, var_values = torch.split(model_output, c, dim=1)
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
        frac = (var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        pred_xstart = self.predict_xstart_from_eps(x, t, eps)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "eps": eps,
        }

    def p_mean_variance(self, model, x, t, model_kwargs=None, clip_denoised=True):
        if model_kwargs is None:
            model_kwargs = {}
        out = model(x, t, **model_kwargs)
        return self.p_mean_variance_from_output(out, x, t, clip_denoised=clip_denoised)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None, scale=1.0):
        gradient = cond_fn(x, t, model_kwargs or {})
        return p_mean_var["mean"] + p_mean_var["variance"] * gradient * scale

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None, scale=1.0):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = p_mean_var["eps"] - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, model_kwargs or {}
        ) * scale
        pred_xstart = self.predict_xstart_from_eps(x, t, eps)
        mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        out = dict(p_mean_var)
        out["eps"] = eps
        out["pred_xstart"] = pred_xstart
        out["mean"] = mean
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        model_kwargs=None,
        clip_denoised=True,
        cond_fn=None,
        guidance_scale=1.0,
    ):
        out = self.p_mean_variance(model, x, t, model_kwargs, clip_denoised)
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs, scale=guidance_scale
            )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        out["sample"] = sample
        return out

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        device,
        model_kwargs=None,
        clip_denoised=True,
        cond_fn=None,
        guidance_scale=1.0,
        progress=False,
        noise=None,
    ):
        x = torch.randn(*shape, device=device) if noise is None else noise.to(device)
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"noise shape {tuple(x.shape)} does not match {tuple(shape)}")
        iterator = range(self.num_timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator)
        for i in iterator:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(
                model,
                x,
                t,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
                cond_fn=cond_fn,
                guidance_scale=guidance_scale,
            )["sample"]
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model,
        shape,
        device,
        steps=50,
        eta=0.0,
        model_kwargs=None,
        clip_denoised=True,
        cond_fn=None,
        guidance_scale=1.0,
        progress=False,
        noise=None,
    ):
        x = torch.randn(*shape, device=device) if noise is None else noise.to(device)
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"noise shape {tuple(x.shape)} does not match {tuple(shape)}")
        times = np.linspace(0, self.num_timesteps - 1, steps, dtype=np.int64)
        pairs = list(zip(times[::-1], np.append(times[:-1][::-1], -1)))
        if progress:
            from tqdm import tqdm

            pairs = tqdm(pairs)
        for i, prev_i in pairs:
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            out = self.p_mean_variance(model, x, t, model_kwargs, clip_denoised)
            if cond_fn is not None:
                out = self.condition_score(
                    cond_fn, out, x, t, model_kwargs, scale=guidance_scale
                )
            eps = out["eps"]
            pred_xstart = out["pred_xstart"]
            alpha = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            if prev_i < 0:
                alpha_prev = torch.ones_like(alpha)
            else:
                prev_t = torch.full((shape[0],), int(prev_i), device=device, dtype=torch.long)
                alpha_prev = _extract_into_tensor(self.alphas_cumprod, prev_t, x.shape)
            sigma = (
                eta
                * torch.sqrt((1 - alpha_prev) / (1 - alpha))
                * torch.sqrt(1 - alpha / alpha_prev)
            )
            noise = torch.randn_like(x)
            mean_pred = (
                pred_xstart * torch.sqrt(alpha_prev)
                + torch.sqrt((1 - alpha_prev - sigma**2).clamp_min(0)) * eps
            )
            nonzero = 0.0 if prev_i < 0 else 1.0
            x = mean_pred + nonzero * sigma * noise
        return x


class SpacedGaussianDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, betas):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)

        base_diffusion = GaussianDiffusion(betas)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        super().__init__(np.array(new_betas))

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_timesteps(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_timesteps(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_timesteps(cond_fn), *args, **kwargs)

    def _wrap_timesteps(self, fn):
        if isinstance(fn, _WrappedTimesteps):
            return fn
        return _WrappedTimesteps(fn, self.timestep_map)


class _WrappedTimesteps:
    def __init__(self, fn, timestep_map):
        self.fn = fn
        self.timestep_map = timestep_map

    def __call__(self, x, timesteps, *args, **kwargs):
        map_tensor = torch.tensor(
            self.timestep_map, device=timesteps.device, dtype=timesteps.dtype
        )
        original_timesteps = map_tensor[timesteps]
        return self.fn(x, original_timesteps, *args, **kwargs)


def reverse_step_sample(diffusion, mean, log_variance, t):
    noise = torch.randn_like(mean)
    nonzero_mask = (t != 0).float().view(-1, *([1] * (mean.ndim - 1)))
    return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise


def diagonal_gaussian_kl_mean(mean_a, mean_b, log_variance, reduce="mean"):
    kl_per_dim = (mean_a - mean_b).pow(2) / (2 * torch.exp(log_variance).clamp_min(1e-20))
    if reduce == "sum":
        return sum_flat(kl_per_dim)
    if reduce == "mean":
        return mean_flat(kl_per_dim)
    raise ValueError(f"unknown KL reduction: {reduce}")
