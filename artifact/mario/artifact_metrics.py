"""Strict metric implementation for reproducing reported reconstruction results."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_image_psnr(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(reconstruction, target, reduction="none").flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


class ReconstructionMetrics:
    def __init__(self, device: torch.device, lpips_net: str = "alex"):
        try:
            import lpips
            from torchmetrics.image import StructuralSimilarityIndexMeasure
        except ImportError as exc:
            raise RuntimeError(
                "Install lpips and torchmetrics before running reconstruction evaluation"
            ) from exc

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = lpips.LPIPS(net=lpips_net).to(device).eval()
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, reconstruction: torch.Tensor, target: torch.Tensor) -> None:
        reconstruction = reconstruction.clamp(0, 1)
        target = target.clamp(0, 1)
        batch_size = target.shape[0]

        self.psnr_sum += per_image_psnr(reconstruction, target).sum().item()
        self.ssim_sum += self.ssim(reconstruction, target).item() * batch_size
        target_lpips = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
        reconstruction_lpips = (
            reconstruction.repeat(1, 3, 1, 1) if reconstruction.shape[1] == 1 else reconstruction
        )
        target_lpips = (target_lpips * 2 - 1).clamp(-1, 1)
        reconstruction_lpips = (reconstruction_lpips * 2 - 1).clamp(-1, 1)
        self.lpips_sum += self.lpips(reconstruction_lpips, target_lpips).view(-1).sum().item()
        self.count += batch_size

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise RuntimeError("No samples were evaluated")
        return {
            "psnr": self.psnr_sum / self.count,
            "ssim": self.ssim_sum / self.count,
            "lpips": self.lpips_sum / self.count,
        }
