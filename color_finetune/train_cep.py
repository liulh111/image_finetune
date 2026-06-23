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
from .models import OpenAIColorScalarNet
from .rewards import color_index, color_reward
from .sampling import (
    add_training_sample_args,
    parse_guidance_levels,
    save_training_guidance_grid,
)


def contrastive_color_loss(pred, reward, eta):
    # The energy model predicts the scaled log weight Q_t / eta. CEP's target
    # label is therefore the self-normalized exp(Q / eta) weight.
    target = F.softmax(reward / eta, dim=0).detach()
    log_prob = F.log_softmax(pred, dim=0)
    return -(target * log_prob).sum()


def main():
    parser = create_arg_parser(
        "Train CEP color energy guidance.", require_model_path=False
    )
    parser.add_argument("--out_dir", default="runs/cep_color")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--same_t_batch", action="store_true")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    add_training_sample_args(parser)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("CEP contrastive training requires --batch_size >= 2 per GPU")
    if args.sample_every > 0 and not args.model_path:
        raise ValueError("--model_path is required when --sample_every > 0")
    args.energy_arch = "openai_classifier"

    ctx = init_distributed(args.device)
    torch.manual_seed(args.seed + ctx.rank)
    device = ctx.device
    diffusion = create_diffusion()
    target_color = color_index(args.color)
    sample_guidance_levels = parse_guidance_levels(args.sample_guidance_levels)
    sample_behavior = None
    if args.sample_every > 0 and ctx.is_main:
        sample_behavior = load_behavior_model(args.model_path, args.class_cond, device=device)

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
    energy = OpenAIColorScalarNet().to(device)
    if ctx.distributed:
        ddp_kwargs = (
            {"device_ids": [ctx.local_rank], "output_device": ctx.local_rank}
            if device.type == "cuda"
            else {}
        )
        energy = DDP(energy, **ddp_kwargs)
    opt = torch.optim.AdamW(
        energy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    out_dir = Path(args.out_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.class_cond:
            save_label_info(out_dir / "label_info.json", args.data_dir)
    if ctx.distributed:
        dist.barrier()
    pbar = tqdm(range(1, args.steps + 1), disable=not ctx.is_main)
    last_loss = None
    for step in pbar:
        x0, y = next(data_iter)
        color_y = torch.full(
            (args.batch_size,), target_color, device=device, dtype=torch.long
        )
        if args.same_t_batch:
            t = torch.full(
                (args.batch_size,),
                torch.randint(0, diffusion.num_timesteps, (), device=device),
                device=device,
                dtype=torch.long,
            )
        else:
            t = torch.randint(0, diffusion.num_timesteps, (args.batch_size,), device=device)
        x_t = diffusion.q_sample(x0, t)
        reward = color_reward(x0, color_y)
        pred = energy(x_t, t, color_y, y)
        loss = contrastive_color_loss(pred, reward, args.eta)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
        if ctx.is_main and step % args.log_every == 0:
            pbar.set_description(f"loss={last_loss:.4f}")
        if ctx.is_main and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(
                str(out_dir / f"cep_step_{step}.pt"),
                model=unwrap_model(energy).state_dict(),
                optimizer=opt.state_dict(),
                step=step,
                config=build_run_config(args),
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
                        kind="cep",
                        diffusion=diffusion,
                        model=energy,
                        behavior_model=sample_behavior,
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
                        eta=args.eta,
                    )
            if ctx.distributed:
                dist.barrier()

    rank0_print(ctx, f"finished CEP training, last_loss={last_loss}")
    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
