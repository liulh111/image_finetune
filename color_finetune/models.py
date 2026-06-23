import torch
from torch import nn

from .guided_unet import create_openai_256_classifier, create_openai_256_residual_unet


class OpenAIColorScalarNet(nn.Module):
    """OpenAI classifier-style scalar model used by CEP energy and BDPO value."""

    def __init__(self):
        super().__init__()
        self.net = create_openai_256_classifier(out_channels=1)

    def forward(self, x, t, color_y=None, y=None):
        del color_y, y
        return self.net(x, t).squeeze(-1)


class OpenAIResidualEpsAdapter(nn.Module):
    """OpenAI U-Net residual actor aligned with the CEP classifier backbone."""

    def __init__(self, behavior_model, class_cond=False, train_variance=False):
        super().__init__()
        self.behavior_model = behavior_model
        for p in self.behavior_model.parameters():
            p.requires_grad_(False)
        self.class_cond = class_cond
        self.train_variance = train_variance
        out_channels = 6 if train_variance else 3
        self.delta_model = create_openai_256_residual_unet(
            out_channels=out_channels,
            class_cond=class_cond,
        )

    def forward(self, x, timesteps, y=None, color_y=None, adapter_scale=1.0):
        del color_y
        with torch.no_grad():
            base = self.behavior_model(x, timesteps, y)
        delta_y = y if self.class_cond else None
        delta = self.delta_model(x, timesteps, delta_y).type(base.dtype)
        if self.train_variance:
            return base + adapter_scale * delta
        eps, var = torch.chunk(base, 2, dim=1)
        return torch.cat([eps + adapter_scale * delta, var], dim=1)
