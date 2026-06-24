import argparse
from pathlib import Path

import torch

from .common import create_diffusion, load_behavior_model, resolve_device
from .sampling import (
    parse_step_list,
    save_behavior_prior_grids,
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
        "--model",
        choices=("uncond", "cond"),
        default="uncond",
        help="Behavior prior to sample.",
    )
    parser.add_argument("--sample_method", choices=("ddim", "ddpm"), default="ddpm")
    parser.add_argument(
        "--steps",
        default="250",
        help="Comma-separated sampler step counts, e.g. 25,50,100,250.",
    )
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.k <= 0 or args.n <= 0:
        raise ValueError("--k and --n must be positive")
    steps = parse_step_list(args.steps)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    diffusion = create_diffusion()

    model_path = (
        args.uncond_model_path if args.model == "uncond" else args.cond_model_path
    )
    behavior_model = load_behavior_model(
        model_path, class_cond=args.model == "cond", device=device
    )
    with torch.no_grad():
        paths = save_behavior_prior_grids(
            diffusion=diffusion,
            model=behavior_model,
            model_kind=args.model,
            out_dir=args.out_dir,
            device=device,
            k=args.k,
            n=args.n,
            sample_method=args.sample_method,
            steps=steps,
            ddim_eta=args.ddim_eta,
            sample_seed=args.seed + 1,
            progress=True,
        )
    for step_count, path in zip(steps, paths):
        print(f"saved {args.model} behavior samples at {step_count} steps to {path}")


if __name__ == "__main__":
    main()
