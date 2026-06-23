import math

import torch


COLOR_NAMES = ("red", "green", "blue")
COLOR_TO_INDEX = {name: i for i, name in enumerate(COLOR_NAMES)}
COLOR_ANGLES = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])


def color_index(color):
    return COLOR_TO_INDEX[color]


def rgb_to_hsv(x):
    rgb = ((x + 1.0) * 0.5).clamp(0, 1)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = torch.max(rgb, dim=1).values
    minc = torch.min(rgb, dim=1).values
    delta = maxc - minc
    safe_delta = delta.clamp_min(1e-6)

    hue_r = ((g - b) / safe_delta) % 6.0
    hue_g = ((b - r) / safe_delta) + 2.0
    hue_b = ((r - g) / safe_delta) + 4.0
    hue = torch.where(maxc == r, hue_r, torch.where(maxc == g, hue_g, hue_b))
    hue = torch.where(delta < 1e-6, torch.zeros_like(hue), hue)
    hue = hue * (math.pi / 3.0)

    saturation = torch.where(maxc < 1e-6, torch.zeros_like(maxc), delta / maxc.clamp_min(1e-6))
    value = maxc
    return hue, saturation, value


def angular_distance(hue, target):
    return torch.atan2(torch.sin(hue - target), torch.cos(hue - target)).abs()


def color_reward(x, color_y):
    """Differentiable color reward Q(x): higher is closer to the target hue.

    This follows the CEP color experiment's HSV energy shape: hue distance is
    minimized, and low-saturation gray/white images receive an extra penalty.
    """

    hue, saturation, _ = rgb_to_hsv(x)
    targets = COLOR_ANGLES.to(device=x.device, dtype=x.dtype)[color_y]
    while targets.ndim < hue.ndim:
        targets = targets[..., None]
    dist = angular_distance(hue, targets) / (2.0 * math.pi)
    mean_sat = saturation.mean(dim=tuple(range(1, saturation.ndim)))
    low_saturation_penalty = 3.0 * torch.relu(0.1 - mean_sat)
    return -(
        dist.mean(dim=tuple(range(1, dist.ndim))) + low_saturation_penalty
    )
