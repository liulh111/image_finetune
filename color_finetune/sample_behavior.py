import argparse
from pathlib import Path

import torch

from .common import create_diffusion, load_behavior_model, resolve_device
from .sampling import (
    DEFAULT_GUIDANCE_LEVELS,
    parse_guidance_levels,
    save_behavior_guidance_grids,
)


def create_parser():
    parser = argparse.ArgumentParser("Sample OpenAI behavior diffusion priors.")
    parser.add_argument(
        "--uncond_model_path",
        default="checkpoints/openai/256x256_diffusion_uncond.pt",
    )
    parser.add_argument(
        "--cond_model_path",
        default="checkpoints/openai/256x256_diffusion.pt",
    )
    parser.add_argument("--out_dir", default="runs/behavior_samples")
    parser.add_argument("--k", type=int, default=4, help="Number of grid rows.")
    parser.add_argument("--n", type=int, default=4, help="Images per row.")
    parser.add_argument(
        "--guidance_scales",
        default=",".join(f"{x:g}" for x in DEFAULT_GUIDANCE_LEVELS),
        help="Comma-separated scales. 0 is unconditional; 1 is conditional.",
    )
    parser.add_argument("--sample_method", choices=("ddim", "ddpm"), default="ddpm")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.k <= 0 or args.n <= 0:
        raise ValueError("--k and --n must be positive")
    guidance_scales = parse_guidance_levels(args.guidance_scales)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    diffusion = create_diffusion()

    uncond_model = load_behavior_model(
        args.uncond_model_path, class_cond=False, device=device
    )
    cond_model = load_behavior_model(args.cond_model_path, class_cond=True, device=device)
    with torch.no_grad():
        paths = save_behavior_guidance_grids(
            diffusion=diffusion,
            uncond_model=uncond_model,
            cond_model=cond_model,
            out_dir=args.out_dir,
            device=device,
            k=args.k,
            n=args.n,
            sample_method=args.sample_method,
            steps=args.steps,
            ddim_eta=args.ddim_eta,
            guidance_scales=guidance_scales,
            sample_seed=args.seed + 1,
            progress=True,
        )
    for scale, path in zip(guidance_scales, paths):
        print(f"saved behavior samples at scale {scale:g} to {path}")


if __name__ == "__main__":
    main()
