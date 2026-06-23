import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .diffusion import GaussianDiffusion
from .guided_unet import create_openai_256_unet
from .rewards import COLOR_NAMES


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def add_prior_args(parser, require_model_path=True):
    parser.add_argument("--model_path", required=require_model_path)
    parser.add_argument("--class_cond", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--color", choices=COLOR_NAMES, default="red")
    parser.add_argument("--seed", type=int, default=0)


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def init_distributed(requested_device="cuda"):
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return DistributedContext(
            distributed=True,
            rank=dist.get_rank(),
            local_rank=local_rank,
            world_size=dist.get_world_size(),
            device=device,
        )

    return DistributedContext(
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=resolve_device(requested_device),
    )


def cleanup_distributed(ctx):
    if ctx.distributed:
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def rank0_print(ctx, *args, **kwargs):
    if ctx.is_main:
        print(*args, **kwargs)


def make_imagenet_labels(batch_size, class_cond, device):
    if not class_cond:
        return None
    return torch.randint(0, 1000, (batch_size,), device=device)


def load_behavior_model(model_path, class_cond, device):
    model = create_openai_256_unet(class_cond=class_cond)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_run_config(args):
    return {
        "model_path": args.model_path,
        "data_dir": getattr(args, "data_dir", None),
        "class_cond": bool(args.class_cond),
        "color": args.color,
        "eta": getattr(args, "eta", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "energy_arch": getattr(args, "energy_arch", None),
        "actor_arch": getattr(args, "actor_arch", None),
        "value_arch": getattr(args, "value_arch", None),
        "kl_reduce": getattr(args, "kl_reduce", None),
        "reverse_samples": getattr(args, "reverse_samples", None),
        "sample_every": getattr(args, "sample_every", None),
        "sample_k": getattr(args, "sample_k", None),
        "sample_steps": getattr(args, "sample_steps", None),
        "sample_method": getattr(args, "sample_method", None),
        "sample_ddim_eta": getattr(args, "sample_ddim_eta", None),
        "sample_guidance_levels": getattr(args, "sample_guidance_levels", None),
    }


def create_arg_parser(description, require_model_path=True):
    parser = argparse.ArgumentParser(description=description)
    add_prior_args(parser, require_model_path=require_model_path)
    return parser


def create_diffusion():
    return GaussianDiffusion.openai_256()
