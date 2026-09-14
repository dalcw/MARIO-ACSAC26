#!/usr/bin/env python3
"""Evaluate task accuracy and supervised reconstruction for each defense."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from mario.artifact_data import DATASET_NAMES, build_loader, default_batch_size, input_size
from mario.artifact_metrics import ReconstructionMetrics
from mario.attackers import SmashedReconstructionAttacker, SmashedReconstructionAttacker224
from mario.defenses import RepresentationExposureModel
from mario.evaluation import DISPLAY_NAMES, METHODS, load_torch, seed_everything, select_device, write_csv
from mario.evaluation import print_expected_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=DATASET_NAMES, dest="datasets")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "release")
    parser.add_argument("--attacker-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_results" / "01_main_privacy_utility")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the complete test set")
    parser.add_argument("--qualitative-samples", type=int, default=8)
    parser.add_argument("--lpips-net", choices=("alex",), default="alex")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-cifar10", action="store_true")
    return parser.parse_args()


def unpack_batch(batch):
    return batch[0], batch[1]


def checkpoint_name(method: str) -> str:
    return "mario.pt" if method == "our" else f"{method}.pt"


def attacker_name(method: str) -> str:
    return "our_attacker.pt" if method == "our" else f"{method}_attacker.pt"


def evaluate_method(args: argparse.Namespace, dataset_name: str, method: str, device: torch.device):
    size = input_size(dataset_name)
    batch_size = args.batch_size or default_batch_size(dataset_name)
    if args.batch_size is None and size == 224:
        batch_size = 16 if method == "disco" else 32
    loader = build_loader(
        dataset_name,
        args.data_root,
        "test",
        batch_size,
        args.workers,
        download=args.download_cifar10 and dataset_name == "cifar10",
        shuffle=False,
    )

    defense_path = args.checkpoint_root / dataset_name / "defenses" / checkpoint_name(method)
    attacker_root = args.attacker_root or args.checkpoint_root
    attacker_path = attacker_root / dataset_name / "reconstruction_attackers" / attacker_name(method)
    model = RepresentationExposureModel(method, size).to(device)
    model.load_checkpoint(load_torch(defense_path, device))
    model.eval()

    attacker_class = SmashedReconstructionAttacker if size == 32 or method == "disco" else SmashedReconstructionAttacker224
    attacker = attacker_class().to(device)
    attacker.load_state_dict(load_torch(attacker_path, device))
    attacker.eval()

    metrics = ReconstructionMetrics(device, lpips_net=args.lpips_net)
    correct = 0
    evaluated = 0
    originals = []
    reconstructions = []

    with torch.inference_mode():
        progress = tqdm(loader, desc=f"{dataset_name}/{DISPLAY_NAMES[method]}")
        for batch in progress:
            images, labels = unpack_batch(batch)
            if args.max_samples:
                remaining = args.max_samples - evaluated
                if remaining <= 0:
                    break
                images = images[:remaining]
                labels = labels[:remaining]

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            exposed = model(images)
            logits = model.server_model(exposed)
            reconstructed = attacker(exposed)

            correct += logits.argmax(dim=1).eq(labels).sum().item()
            evaluated += images.shape[0]
            metrics.update(reconstructed, images)

            needed = args.qualitative_samples - sum(item.shape[0] for item in originals)
            if needed > 0:
                originals.append(images[:needed].cpu())
                reconstructions.append(reconstructed[:needed].clamp(0, 1).cpu())
            progress.set_postfix(samples=evaluated, accuracy=f"{correct / evaluated:.4f}")

    values = metrics.compute()
    figure_dir = args.output / "figures" / dataset_name
    figure_dir.mkdir(parents=True, exist_ok=True)
    if originals:
        original_grid = torch.cat(originals)
        reconstruction_grid = torch.cat(reconstructions)
        save_image(
            torch.cat([original_grid, reconstruction_grid]),
            figure_dir / f"{method}_reconstruction.png",
            nrow=original_grid.shape[0],
            padding=1,
        )

    del model, attacker, metrics
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "dataset": dataset_name,
        "method": DISPLAY_NAMES[method],
        "samples": evaluated,
        "accuracy": correct / evaluated,
        **values,
    }


def main() -> None:
    args = parse_args()
    datasets_to_run = args.datasets or ["cifar10"]
    device = select_device(args.device)
    seed_everything(args.seed)
    print(f"device={device}; datasets={datasets_to_run}; methods={args.methods}")

    rows = []
    for dataset_name in datasets_to_run:
        for method in args.methods:
            seed_everything(args.seed)
            row = evaluate_method(args, dataset_name, method, device)
            rows.append(row)
            print(row, flush=True)
            write_csv(args.output / "metrics.csv", rows)

    print_expected_comparison(
        rows,
        REPOSITORY_ROOT / "claims" / "claim1" / "expected" / "metrics.csv",
        key_fields=("dataset", "method"),
        metric_fields=("accuracy", "psnr", "ssim", "lpips"),
    )
    print(f"\033[94mResults saved to: {args.output.resolve()}\033[0m", flush=True)


if __name__ == "__main__":
    main()
