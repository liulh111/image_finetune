import torch
from torch import nn

from .guided_unet import SiLU, timestep_embedding, zero_module


class SmallResBlock(nn.Module):
    def __init__(self, channels, emb_channels):
        super().__init__()
        groups = min(32, channels)
        self.in_norm = nn.GroupNorm(groups, channels)
        self.in_act = SiLU()
        self.in_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.emb = nn.Sequential(SiLU(), nn.Linear(emb_channels, channels))
        self.out_norm = nn.GroupNorm(groups, channels)
        self.out_act = SiLU()
        self.out_conv = zero_module(nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x, emb):
        h = self.in_conv(self.in_act(self.in_norm(x.float()).type(x.dtype)))
        e = self.emb(emb).type(h.dtype)
        while e.ndim < h.ndim:
            e = e[..., None]
        h = h + e
        h = self.out_conv(self.out_act(self.out_norm(h.float()).type(h.dtype)))
        return x + h


class ColorConditionedBackbone(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        emb_channels=256,
        color_count=3,
        class_cond=False,
        num_classes=1000,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.emb_channels = emb_channels
        self.class_cond = class_cond
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_channels),
            SiLU(),
            nn.Linear(emb_channels, emb_channels),
        )
        self.color_emb = nn.Embedding(color_count, emb_channels)
        if class_cond:
            self.label_emb = nn.Embedding(num_classes, emb_channels)

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        chans = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        blocks = []
        ch = base_channels
        for out_ch in chans:
            if out_ch != ch:
                blocks.append(nn.Conv2d(ch, out_ch, 3, padding=1))
                ch = out_ch
            blocks.append(SmallResBlock(ch, emb_channels))
            blocks.append(SmallResBlock(ch, emb_channels))
            blocks.append(nn.AvgPool2d(2))
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = ch

    def embedding(self, t, color_y, y=None):
        emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        emb = emb + self.color_emb(color_y)
        if self.class_cond:
            if y is None:
                raise ValueError("class labels are required for class-conditional network")
            emb = emb + self.label_emb(y)
        return emb

    def forward_features(self, x, t, color_y, y=None):
        emb = self.embedding(t, color_y, y)
        h = self.in_conv(x)
        for block in self.blocks:
            if isinstance(block, SmallResBlock):
                h = block(h, emb)
            else:
                h = block(h)
        return h, emb


class ColorScalarNet(nn.Module):
    def __init__(
        self,
        base_channels=64,
        emb_channels=256,
        color_count=3,
        class_cond=False,
        num_classes=1000,
    ):
        super().__init__()
        self.backbone = ColorConditionedBackbone(
            in_channels=3,
            base_channels=base_channels,
            emb_channels=emb_channels,
            color_count=color_count,
            class_cond=class_cond,
            num_classes=num_classes,
        )
        ch = self.backbone.out_channels
        self.head = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch),
            SiLU(),
            nn.Linear(ch, 1),
        )

    def forward(self, x, t, color_y, y=None):
        h, _ = self.backbone.forward_features(x, t, color_y, y)
        return self.head(h).squeeze(-1)


class ResidualEpsAdapter(nn.Module):
    def __init__(
        self,
        behavior_model,
        base_channels=48,
        emb_channels=192,
        color_count=3,
        class_cond=False,
        num_classes=1000,
        train_variance=False,
    ):
        super().__init__()
        self.behavior_model = behavior_model
        for p in self.behavior_model.parameters():
            p.requires_grad_(False)
        self.train_variance = train_variance
        self.adapter = ColorConditionedBackbone(
            in_channels=3,
            base_channels=base_channels,
            emb_channels=emb_channels,
            color_count=color_count,
            class_cond=class_cond,
            num_classes=num_classes,
        )
        ch = self.adapter.out_channels
        out_channels = 6 if train_variance else 3
        self.delta_head = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            SiLU(),
            nn.Upsample(scale_factor=16, mode="nearest"),
            nn.Conv2d(ch, base_channels, 3, padding=1),
            SiLU(),
            zero_module(nn.Conv2d(base_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None, color_y=None, adapter_scale=1.0):
        if color_y is None:
            raise ValueError("color_y is required for residual adapter")
        with torch.no_grad():
            base = self.behavior_model(x, timesteps, y)
        h, _ = self.adapter.forward_features(x, timesteps, color_y, y)
        delta = self.delta_head(h).type(base.dtype)
        if self.train_variance:
            return base + adapter_scale * delta
        eps, var = torch.chunk(base, 2, dim=1)
        return torch.cat([eps + adapter_scale * delta, var], dim=1)
