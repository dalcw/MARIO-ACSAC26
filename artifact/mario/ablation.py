"""Inference modules for the single-stage CelebA ablation variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from .defenses import ChannelDecoder, ChannelSelfAttn, ClientResNet18, ServerResNet18


class ChannelObfuscationOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ChannelSelfAttn(width=28, height=28, embed_dim=28 * 28)

    def forward(self, smashed_data: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = smashed_data.shape
        return self.encoder(smashed_data).reshape(batch, channels, height, width)


class LatentDecompositionOnly(nn.Module):
    def __init__(self):
        super().__init__()
        dimension = 28 * 28
        self.linear_z = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(inplace=True))
        self.linear_z_pub = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(inplace=True))
        self.info_expand_priv = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(inplace=True))
        self.priv_decoder = ChannelDecoder(width=28, height=28, embed_dim=dimension)

    def forward(self, smashed_data: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = smashed_data.shape
        flattened = torch.flatten(smashed_data, start_dim=2)
        latent = self.linear_z(flattened)
        public = self.linear_z_pub(latent)
        return public.reshape(batch, channels, height, width)


class VariationalSamplingOnly(nn.Module):
    def __init__(self):
        super().__init__()
        dimension = 28 * 28
        self.linear_mu = nn.Linear(dimension, dimension)
        self.linear_logvar = nn.Linear(dimension, dimension)

    def forward(self, smashed_data: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = smashed_data.shape
        flattened = torch.flatten(smashed_data, start_dim=2)
        mean = self.linear_mu(flattened)
        log_variance = self.linear_logvar(flattened)
        sampled = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return sampled.reshape(batch, channels, height, width)


ABLATION_ADAPTERS = {
    "channel_obfuscation_only": ChannelObfuscationOnly,
    "latent_decomposition_only": LatentDecompositionOnly,
    "variational_sampling_only": VariationalSamplingOnly,
}


class SingleStageAblation(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in ABLATION_ADAPTERS:
            raise ValueError(f"Unknown ablation variant: {variant}")
        self.client_model = ClientResNet18(input_size=224)
        self.adapter_model = ABLATION_ADAPTERS[variant]()
        self.server_model = ServerResNet18(num_classes=10)

    def load_checkpoint(self, payload: dict) -> None:
        self.client_model.load_state_dict(payload["client_model"])
        self.adapter_model.load_state_dict(payload["adapter_model"])
        self.server_model.load_state_dict(payload["server_model"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.adapter_model(self.client_model(images))
