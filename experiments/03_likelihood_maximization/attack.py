#!/usr/bin/env python3
"""Likelihood-maximization reconstruction attack used for Figure 6."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mario.artifact_data import build_dataset, input_size
from mario.defenses import RepresentationExposureModel
from mario.evaluation import (
    DISPLAY_NAMES,
    METHODS,
    load_torch,
    seed_everything,
    select_device,
    write_csv,
)
from mario.evaluation import print_expected_comparison
from mario.metrics import MetricAccumulator, total_variation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "celeba", "nih_chest_xray"), default="cifar10")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "release")
    parser.add_argument("--output", type=Path, default=ROOT / "generated_results" / "03_likelihood_maximization")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--l2-weight", type=float, default=1e-5)
    parser.add_argument("--init", choices=("noise", "gray", "target_jitter"), default="noise")
    parser.add_argument("--save-images", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-cifar10", action="store_true")
    return parser.parse_args()


def checkpoint_name(method: str) -> str:
    return "mario.pt" if method == "our" else f"{method}.pt"


def initialize(target: torch.Tensor, mode: str) -> torch.Tensor:
    epsilon = 1e-4
    if mode == "noise":
        initial = torch.rand_like(target).clamp(epsilon, 1 - epsilon)
    elif mode == "gray":
        initial = torch.full_like(target, 0.5)
    else:
        initial = (target + 0.05 * torch.randn_like(target)).clamp(epsilon, 1 - epsilon)
    return torch.logit(initial).detach().requires_grad_(True)


def reconstruct(model, target_x, target_z, args):
    raw = initialize(target_x, args.init)
    optimizer = torch.optim.Adam([raw], lr=args.lr)
    best = None
    best_loss = None

    for _ in range(args.steps):
        candidate = torch.sigmoid(raw)
        candidate_z = model(candidate)
        feature_loss = F.mse_loss(candidate_z, target_z)
        prior = args.tv_weight * total_variation(candidate)
        prior = prior + args.l2_weight * ((candidate - 0.5) ** 2).mean()
        loss = feature_loss + prior
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        value = loss.detach()
        if best_loss is None or value < best_loss:
            best_loss = value
            best = candidate.detach().clone()
    return best.clamp(0, 1), best_loss.item()


def evaluate_method(args: argparse.Namespace, method: str, device: torch.device):
    dataset = build_dataset(
        args.dataset,
        args.data_root,
        "test",
        download=args.download_cifar10 and args.dataset == "cifar10",
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = RepresentationExposureModel(method, input_size(args.dataset)).to(device)
    checkpoint = args.checkpoint_root / args.dataset / "defenses" / checkpoint_name(method)
    model.load_checkpoint(load_torch(checkpoint, device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metrics = MetricAccumulator(device)
    if metrics.ssim_metric is None or not metrics.lpips_available:
        raise RuntimeError("torchmetrics and AlexNet LPIPS are required for this experiment")

    display_name = DISPLAY_NAMES[method]
    output_name = "MARIO" if method == "our" else method
    method_output = args.output / args.dataset / output_name
    image_output = method_output / "reconstruction_images"
    image_output.mkdir(parents=True, exist_ok=True)
    batch_rows = []
    sample_offset = 0

    for batch in tqdm(loader, desc=f"{args.dataset}/{display_name} LMA"):
        images = batch[0].to(device)
        with torch.no_grad():
            target_z = model(images).detach()
        reconstruction, optimization_loss = reconstruct(model, images, target_z, args)
        batch_metrics = metrics.update(reconstruction, images)
        batch_rows.append(
            {
                "start_index": sample_offset,
                "batch_size": images.shape[0],
                "optimization_loss": optimization_loss,
                **batch_metrics,
            }
        )

        remaining = args.save_images - sample_offset
        if remaining > 0:
            count = min(remaining, images.shape[0])
            for index in range(count):
                sample_index = sample_offset + index
                save_image(images[index].cpu(), image_output / f"{sample_index}_original.png")
                save_image(reconstruction[index].cpu(), image_output / f"{sample_index}_reconstruction.png")
            grid = make_grid(
                torch.cat([images[:count].cpu(), reconstruction[:count].cpu()]),
                nrow=count,
            )
            save_image(grid, image_output / f"{sample_offset}_original_then_reconstruction.png")
        sample_offset += images.shape[0]

    summary = {
        "dataset": args.dataset,
        "method": display_name,
        "samples": metrics.n,
        "steps": args.steps,
        "learning_rate": args.lr,
        "tv_weight": args.tv_weight,
        "l2_weight": args.l2_weight,
        **metrics.compute(),
    }
    write_csv(method_output / "batch_metrics.csv", batch_rows)
    write_csv(method_output / "metrics.csv", [summary])
    del model, metrics
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    rows = []
    for method in args.methods:
        seed_everything(args.seed)
        row = evaluate_method(args, method, device)
        rows.append(row)
        print(row, flush=True)
        write_csv(args.output / args.dataset / "summary.csv", rows)

    print_expected_comparison(
        rows,
        ROOT / "expected_results" / "03_likelihood_maximization.csv",
        key_fields=("dataset", "method"),
        metric_fields=("psnr", "ssim", "lpips"),
    )
    print(f"\033[94mResults saved to: {(args.output / args.dataset).resolve()}\033[0m", flush=True)


if __name__ == "__main__":
    main()
