"""Latent decomposition and deep image probes used in Figure 11."""

from __future__ import annotations

import torch
import torch.nn as nn

from .defenses import Adapter


class DeepConvImageDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        image_size: int,
        hidden_dim: int = 4096,
        base_channels: int = 384,
        extra_conv_blocks: int = 3,
    ):
        super().__init__()
        self.start_size = 4 if image_size == 32 else 7
        self.fc = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, base_channels * self.start_size * self.start_size),
            nn.BatchNorm1d(base_channels * self.start_size * self.start_size),
            nn.ReLU(inplace=True),
        )

        channels = [base_channels, 128, 64, 32]
        if image_size != 32:
            channels.extend([16, 8])
        layers = []
        in_channels = channels[0]
        for out_channels in channels[1:]:
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            for _ in range(extra_conv_blocks):
                layers.extend(
                    [
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
            in_channels = out_channels
        layers.extend([nn.Conv2d(in_channels, 3, kernel_size=3, padding=1), nn.Sigmoid()])
        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        feature = self.fc(latent)
        feature = feature.view(latent.shape[0], -1, self.start_size, self.start_size)
        return self.decoder(feature)


class MarioLatentDecomposition(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter(channel=128, width=16, height=16, embed_dim=128, dropout=0.1)
        self.z_dim = self.adapter.z_dim
        self.private_heads = nn.ModuleList()
        self.pub_image_decoder = DeepConvImageDecoder(self.z_dim, 32)
        self.priv_image_decoder = DeepConvImageDecoder(self.z_dim, 32)

    def forward(self, smashed: torch.Tensor) -> dict[str, torch.Tensor]:
        x_pub, x_priv, mean, log_variance, z_pub, z_priv = self.adapter(smashed)
        return {
            "z_pub": z_pub,
            "z_priv": z_priv.flatten(1),
            "z_mu": mean,
            "z_logvar": log_variance,
            "x_pub_feature": x_pub,
            "x_priv_feature": x_priv,
            "x_pub_rec": self.pub_image_decoder(z_pub),
            "x_priv_rec": self.priv_image_decoder(z_priv.flatten(1)),
        }

    def latents(self, smashed: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = smashed.shape[0]
        encoded = self.adapter.encoder(smashed)
        bottleneck = self.adapter.info_bottleneck(encoded)
        latent = self.adapter.linear_z(torch.flatten(bottleneck, start_dim=1))
        public = self.adapter.linear_z_pub(latent)
        private = latent - public
        return {"z_pub": public, "z_priv": private.reshape(batch_size, -1)}
