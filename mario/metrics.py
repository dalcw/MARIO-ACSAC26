import math

import torch
import torch.nn.functional as F


def psnr(x_hat, x, data_range=1.0):
    mse = F.mse_loss(x_hat, x, reduction="none").flatten(1).mean(dim=1)
    return 10.0 * torch.log10((data_range ** 2) / (mse + 1e-12))


def ssim_simple(x_hat, x, data_range=1.0):
    """
    Lightweight global SSIM fallback. If torchmetrics is installed, run.py uses it instead.
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    dims = (1, 2, 3)
    mu_x = x.mean(dim=dims)
    mu_y = x_hat.mean(dim=dims)
    sigma_x = ((x - mu_x[:, None, None, None]) ** 2).mean(dim=dims)
    sigma_y = ((x_hat - mu_y[:, None, None, None]) ** 2).mean(dim=dims)
    sigma_xy = ((x - mu_x[:, None, None, None]) * (x_hat - mu_y[:, None, None, None])).mean(dim=dims)
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-12)
    return score


def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


class MetricAccumulator:
    def __init__(self, device):
        self.device = device
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0
        self.n = 0
        self.lpips_available = False
        self.ssim_metric = None

        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        except Exception:
            self.ssim_metric = None

        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
            self.lpips_available = True
        except Exception:
            self.lpips_fn = None

    def update(self, x_hat, x):
        x = x.clamp(0, 1)
        x_hat = x_hat.clamp(0, 1)
        b = x.size(0)

        batch_psnr = psnr(x_hat, x).sum().item()
        self.psnr_sum += batch_psnr
        if self.ssim_metric is not None:
            batch_ssim = self.ssim_metric(x_hat, x).item() * b
        else:
            batch_ssim = ssim_simple(x_hat, x).sum().item()
        self.ssim_sum += batch_ssim

        if self.lpips_fn is not None:
            x_lp = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
            xhat_lp = x_hat.repeat(1, 3, 1, 1) if x_hat.shape[1] == 1 else x_hat
            x_lp = (x_lp * 2 - 1).clamp(-1, 1)
            xhat_lp = (xhat_lp * 2 - 1).clamp(-1, 1)
            batch_lpips = self.lpips_fn(xhat_lp, x_lp).view(-1).sum().item()
            self.lpips_sum += batch_lpips
        else:
            batch_lpips = math.nan
            self.lpips_sum += math.nan

        self.n += b
        return {
            "psnr": batch_psnr / b,
            "ssim": batch_ssim / b,
            "lpips": batch_lpips / b if self.lpips_available else math.nan,
        }

    def compute(self):
        if self.n == 0:
            return {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
        lpips_value = self.lpips_sum / self.n if self.lpips_available else math.nan
        return {
            "psnr": self.psnr_sum / self.n,
            "ssim": self.ssim_sum / self.n,
            "lpips": lpips_value,
        }
