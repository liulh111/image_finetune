from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .common import (
    build_run_config,
    cleanup_distributed,
    create_arg_parser,
    create_diffusion,
    init_distributed,
    load_behavior_model,
    rank0_print,
    unwrap_model,
)
from .data import make_image_batch_iterator, save_checkpoint, save_label_info
from .diffusion import diagonal_gaussian_kl_mean, reverse_step_sample
from .models import OpenAIColorScalarNet, OpenAIResidualEpsAdapter
from .rewards import color_index, color_reward
from .sampling import (
    add_training_sample_args,
    parse_guidance_levels,
    save_training_guidance_grid,
)


def previous_value(value_net, x_prev, t, color_y, y):
    prev_t = (t - 1).clamp_min(0)
    analytic = color_reward(x_prev, color_y)
    learned = value_net(x_prev, prev_t, color_y, y)
    use_analytic = prev_t == 0
    return torch.where(use_analytic, analytic, learned)


def average_previous_value(
    value_net, diffusion, mean, log_variance, t, color_y, y, samples
):
    if samples < 1:
        raise ValueError("--reverse_samples must be >= 1")
    if samples == 1:
        x_prev = reverse_step_sample(diffusion, mean, log_variance, t)
        return previous_value(value_net, x_prev, t, color_y, y)

    bsz = mean.shape[0]
    noise = torch.randn((samples, *mean.shape), device=mean.device, dtype=mean.dtype)
    nonzero = (t != 0).float().view(1, bsz, *([1] * (mean.ndim - 1)))
    std = torch.exp(0.5 * log_variance).unsqueeze(0)
    x_prev = mean.unsqueeze(0) + nonzero * std * noise
    flat_x_prev = x_prev.reshape(samples * bsz, *mean.shape[1:])
    flat_t = t.repeat(samples)
    flat_color_y = color_y.repeat(samples)
    flat_y = None if y is None else y.repeat(samples)
    values = previous_value(value_net, flat_x_prev, flat_t, flat_color_y, flat_y)
    return values.view(samples, bsz).mean(dim=0)


def actor_and_behavior_stats(diffusion, actor, behavior, x_t, t, y, color_y):
    actor_out = actor(x_t, t, y=y, color_y=color_y)
    actor_stats = diffusion.p_mean_variance_from_output(
        actor_out, x_t, t, clip_denoised=True
    )
    with torch.no_grad():
        behavior_out = behavior(x_t, t, y)
        behavior_stats = diffusion.p_mean_variance_from_output(
            behavior_out, x_t, t, clip_denoised=True
        )
    return actor_stats, behavior_stats


def trainable_actor_state(actor):
    actor = unwrap_model(actor)
    return {
        k: v
        for k, v in actor.state_dict().items()
        if not k.startswith("behavior_model.")
    }


def main():
    parser = create_arg_parser("Train BDPO residual actor for color fine-tuning.")
    parser.add_argument("--out_dir", default="runs/bdpo_color")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU batch size.")
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--value_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--kl_reduce", choices=("mean", "sum"), default="sum")
    parser.add_argument("--actor_update_interval", type=int, default=1)
    parser.add_argument(
        "--reverse_samples",
        type=int,
        default=10,
        help="Number of x_{t-1} reverse samples used to average the value target.",
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=20)
    add_training_sample_args(parser)
    args = parser.parse_args()
    args.actor_arch = "openai_residual_unet"
    args.value_arch = "openai_classifier"

    ctx = init_distributed(args.device)
    torch.manual_seed(args.seed + ctx.rank)
    device = ctx.device
    diffusion = create_diffusion()
    target_color = color_index(args.color)
    sample_guidance_levels = parse_guidance_levels(args.sample_guidance_levels)

    behavior = load_behavior_model(args.model_path, args.class_cond, device=device)
    data_iter = make_image_batch_iterator(
        args.data_dir,
        args.batch_size,
        device,
        class_cond=args.class_cond,
        rank=ctx.rank,
        world_size=ctx.world_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    actor = OpenAIResidualEpsAdapter(
        behavior,
        class_cond=args.class_cond,
    ).to(device)
    value = OpenAIColorScalarNet().to(device)
    if ctx.distributed:
        ddp_kwargs = (
            {"device_ids": [ctx.local_rank], "output_device": ctx.local_rank}
            if device.type == "cuda"
            else {}
        )
        actor = DDP(actor, **ddp_kwargs)
        value = DDP(value, **ddp_kwargs)
    actor_opt = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad],
        lr=args.actor_lr,
        weight_decay=args.weight_decay,
    )
    value_opt = torch.optim.AdamW(
        value.parameters(), lr=args.value_lr, weight_decay=args.weight_decay
    )

    out_dir = Path(args.out_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.class_cond:
            save_label_info(out_dir / "label_info.json", args.data_dir)
    if ctx.distributed:
        dist.barrier()
    pbar = tqdm(range(1, args.steps + 1), disable=not ctx.is_main)
    metrics = {"value_loss": None, "actor_loss": None, "kl": None}

    for step in pbar:
        x0, y = next(data_iter)
        color_y = torch.full(
            (args.batch_size,), target_color, device=device, dtype=torch.long
        )

        # Diffusion value evaluation: V_0 is the analytic color reward.
        t_v = torch.randint(1, diffusion.num_timesteps, (args.batch_size,), device=device)
        x_t_v = diffusion.q_sample(x0, t_v)
        with torch.no_grad():
            a_stats, b_stats = actor_and_behavior_stats(
                diffusion, actor, behavior, x_t_v, t_v, y, color_y
            )
            kl = diagonal_gaussian_kl_mean(
                a_stats["mean"], b_stats["mean"], a_stats["log_variance"], reduce=args.kl_reduce
            )
            target_v = (
                average_previous_value(
                    value,
                    diffusion,
                    a_stats["mean"],
                    a_stats["log_variance"],
                    t_v,
                    color_y,
                    y,
                    args.reverse_samples,
                )
                - args.eta * kl
            )
        pred_v = value(x_t_v, t_v, color_y, y)
        value_loss = F.mse_loss(pred_v, target_v)
        value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        value_opt.step()

        actor_loss = None
        kl_for_log = kl.detach()
        if step % args.actor_update_interval == 0:
            t_a = torch.randint(1, diffusion.num_timesteps, (args.batch_size,), device=device)
            x_t_a = diffusion.q_sample(x0.detach(), t_a)
            a_stats, b_stats = actor_and_behavior_stats(
                diffusion, actor, behavior, x_t_a, t_a, y, color_y
            )
            kl_a = diagonal_gaussian_kl_mean(
                a_stats["mean"], b_stats["mean"], a_stats["log_variance"], reduce=args.kl_reduce
            )
            v_prev = average_previous_value(
                value,
                diffusion,
                a_stats["mean"],
                a_stats["log_variance"],
                t_a,
                color_y,
                y,
                args.reverse_samples,
            )
            objective = v_prev - args.eta * kl_a
            actor_loss = -objective.mean()
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in actor.parameters() if p.requires_grad], 1.0
            )
            actor_opt.step()
            kl_for_log = kl_a.detach()

        metrics["value_loss"] = float(value_loss.detach().cpu())
        metrics["kl"] = float(kl_for_log.mean().cpu())
        if actor_loss is not None:
            metrics["actor_loss"] = float(actor_loss.detach().cpu())
        if ctx.is_main and step % args.log_every == 0:
            pbar.set_description(
                f"v={metrics['value_loss']:.4f} "
                f"a={metrics['actor_loss'] if metrics['actor_loss'] is not None else 0:.4f} "
                f"kl={metrics['kl']:.4f}"
            )
        if ctx.is_main and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(
                str(out_dir / f"bdpo_step_{step}.pt"),
                actor=trainable_actor_state(actor),
                value=unwrap_model(value).state_dict(),
                actor_optimizer=actor_opt.state_dict(),
                value_optimizer=value_opt.state_dict(),
                step=step,
                config={
                    **build_run_config(args),
                    "kl_reduce": args.kl_reduce,
                    "reverse_samples": args.reverse_samples,
                },
            )
        if args.sample_every > 0 and (step % args.sample_every == 0 or step == args.steps):
            if ctx.distributed:
                dist.barrier()
            if ctx.is_main:
                rng_devices = []
                if device.type == "cuda":
                    cuda_index = (
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    )
                    rng_devices = [cuda_index]
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(args.seed + step)
                    save_training_guidance_grid(
                        kind="bdpo",
                        diffusion=diffusion,
                        model=actor,
                        out_dir=str(out_dir),
                        step=step,
                        color=args.color,
                        class_cond=args.class_cond,
                        device=device,
                        k=args.sample_k,
                        sample_method=args.sample_method,
                        steps=args.sample_steps,
                        ddim_eta=args.sample_ddim_eta,
                        guidance_levels=sample_guidance_levels,
                    )
            if ctx.distributed:
                dist.barrier()

    rank0_print(ctx, f"finished BDPO training, metrics={metrics}")
    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
