# Implementation Notes

## Reconstruction Metrics

- PSNR and SSIM use an image data range of `[0, 1]`.
- SSIM is computed with torchmetrics.
- LPIPS is computed with `lpips==0.1.4` using the AlexNet backbone.

## Checkpoint Loading

PyTorch checkpoints are loaded with `weights_only=True`.
