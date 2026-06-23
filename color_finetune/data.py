import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_image_files(data_dir):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"no image files found under: {data_dir}")
    return files


def _image_class_name(path):
    return path.name.split("_")[0]


def infer_class_to_idx(data_dir):
    class_names = [_image_class_name(p) for p in _list_image_files(data_dir)]
    return {name: idx for idx, name in enumerate(sorted(set(class_names)))}


def save_label_info(path, data_dir):
    payload = {
        "label_rule": "filename_prefix_sorted",
        "class_to_idx": infer_class_to_idx(data_dir),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _center_crop_arr(pil_image, image_size):
    while min(pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def _random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = int(np.ceil(image_size / max_crop_frac))
    max_smaller_dim_size = int(np.ceil(image_size / min_crop_frac))
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    while min(pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = smaller_dim_size / min(pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class ImageNetStyleDataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_size=256,
        class_cond=False,
        random_crop=True,
        random_flip=True,
    ):
        self.image_paths = _list_image_files(data_dir)
        self.image_size = image_size
        self.class_cond = class_cond
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.classes = None
        self.class_to_idx = None
        if class_cond:
            self.class_to_idx = infer_class_to_idx(data_dir)
            class_names = [_image_class_name(p) for p in self.image_paths]
            self.classes = [self.class_to_idx[name] for name in class_names]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as pil_image:
            pil_image = pil_image.convert("RGB")
            if self.random_crop:
                arr = _random_crop_arr(pil_image, self.image_size)
            else:
                arr = _center_crop_arr(pil_image, self.image_size)
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = (arr.astype(np.float32) / 127.5 - 1.0).copy()
        image = torch.from_numpy(np.transpose(arr, [2, 0, 1]))
        if not self.class_cond:
            return image, torch.tensor(-1, dtype=torch.long)
        label = int(self.classes[idx])
        return image, torch.tensor(label, dtype=torch.long)


def make_image_batch_iterator(
    data_dir,
    batch_size,
    device,
    class_cond=False,
    rank=0,
    world_size=1,
    seed=0,
    num_workers=2,
    random_crop=True,
):
    dataset = ImageNetStyleDataset(
        data_dir,
        class_cond=class_cond,
        random_crop=random_crop,
    )
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x0, labels in loader:
            x0 = x0.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True) if class_cond else None
            yield x0, y
        epoch += 1


def save_samples(x, out_dir, prefix, nrow=4):
    os.makedirs(out_dir, exist_ok=True)
    grid_path = Path(out_dir) / f"{prefix}_grid.png"
    save_image((x.clamp(-1, 1) + 1) * 0.5, grid_path, nrow=nrow)
    return str(grid_path)


def save_checkpoint(path, **payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
