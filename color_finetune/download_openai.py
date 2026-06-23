import argparse
import base64
import hashlib
import os
import subprocess
from pathlib import Path


MODELS = {
    "256x256_diffusion": {
        "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt",
        "size": 2215479544,
        "content_md5": "2U8w+xr7L+ghwkvvIENxfg==",
    },
    "256x256_diffusion_uncond": {
        "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
        "size": 2211383297,
        "content_md5": "/Z3SM1uHNtUh3grtVL2Qyg==",
    },
}


def run(cmd):
    subprocess.run(cmd, check=True)


def sha256(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def md5_base64(path, block=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(block)
            if not data:
                break
            h.update(data)
    return base64.b64encode(h.digest()).decode("ascii")


def download_one(model_name, out_dir, chunk_mb, force=False):
    spec = MODELS[model_name]
    out_dir = Path(out_dir)
    parts_dir = out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{model_name}.pt"
    if target.exists() and not force:
        if target.stat().st_size != spec["size"]:
            raise RuntimeError(f"{target} has bad size; rerun with --force")
        actual_md5 = md5_base64(target)
        if actual_md5 != spec["content_md5"]:
            raise RuntimeError(f"{target} has bad MD5; rerun with --force")
        print(f"{target} already exists and passed MD5")
        return str(target)

    chunk = chunk_mb * 1024 * 1024
    size = spec["size"]
    start = 0
    part_paths = []
    idx = 0
    while start < size:
        end = min(start + chunk - 1, size - 1)
        part = parts_dir / f"{model_name}.part{idx:02d}"
        expected = end - start + 1
        if not part.exists() or part.stat().st_size != expected or force:
            print(f"download part {idx}: bytes={start}-{end}")
            run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "8",
                    "--retry-delay",
                    "2",
                    "--range",
                    f"{start}-{end}",
                    "--output",
                    str(part),
                    spec["url"],
                ]
            )
        part_paths.append(part)
        start = end + 1
        idx += 1

    tmp = target.with_suffix(".pt.tmp")
    with open(tmp, "wb") as w:
        for part in part_paths:
            with open(part, "rb") as r:
                while True:
                    data = r.read(1024 * 1024)
                    if not data:
                        break
                    w.write(data)
    os.replace(tmp, target)
    actual = target.stat().st_size
    if actual != size:
        raise RuntimeError(f"bad final size: {actual} != {size}")
    actual_md5 = md5_base64(target)
    if actual_md5 != spec["content_md5"]:
        raise RuntimeError(f"bad MD5: {actual_md5} != {spec['content_md5']}")
    print(f"wrote {target}")
    print(f"md5_base64={actual_md5}")
    print(f"sha256={sha256(target)}")
    return str(target)


def main():
    parser = argparse.ArgumentParser(description="Chunked OpenAI checkpoint downloader.")
    parser.add_argument("--model", choices=["all", *MODELS.keys()], default="all")
    parser.add_argument("--out_dir", default="checkpoints/openai")
    parser.add_argument("--chunk_mb", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        download_one(name, Path(args.out_dir), args.chunk_mb, force=args.force)


if __name__ == "__main__":
    main()
