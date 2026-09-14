#!/usr/bin/env python3
"""Optionally train property-inference attackers against fixed models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mario.artifact_data import build_dataset, default_batch_size, input_size, property_names
from mario.attackers import SmashedBinaryClassifier
from mario.defenses import RepresentationExposureModel
from mario.evaluation import DISPLAY_NAMES, METHODS, load_torch, seed_everything, select_device, write_csv


PROPERTY_DATASETS = ("celeba", "nih_chest_xray")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=PROPERTY_DATASETS, dest="datasets")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "release")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "generated_results" / "02_property_inference" / "scratch_checkpoints",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def checkpoint_name(method: str) -> str:
    return "mario.pt" if method == "our" else f"{method}.pt"


def attacker_name(method: str) -> str:
    return "our_attacker.pt" if method == "our" else f"{method}_attacker.pt"


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def report_path(path: Path) -> str:
    absolute_path = path.resolve()
    try:
        return absolute_path.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute_path.as_posix()


def train_one(args: argparse.Namespace, dataset_name: str, method: str, device: torch.device):
    batch_size = args.batch_size or default_batch_size(dataset_name)
    if args.batch_size is None and method == "disco":
        batch_size = 32
    dataset = build_dataset(dataset_name, args.data_root, "train")
    if args.max_train_samples > 0:
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = RepresentationExposureModel(method, input_size(dataset_name)).to(device)
    defense_path = args.checkpoint_root / dataset_name / "defenses" / checkpoint_name(method)
    model.load_checkpoint(load_torch(defense_path, device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    attackers = [SmashedBinaryClassifier().to(device), SmashedBinaryClassifier().to(device)]
    optimizer = torch.optim.Adam(
        [parameter for attacker in attackers for parameter in attacker.parameters()],
        lr=args.learning_rate,
    )
    final_losses = [0.0, 0.0]

    for epoch in range(args.epochs):
        for attacker in attackers:
            attacker.train()
        loss_sums = [0.0, 0.0]
        sample_count = 0
        progress = tqdm(loader, desc=f"train property/{dataset_name}/{DISPLAY_NAMES[method]} {epoch + 1}/{args.epochs}")
        for images, _, properties in progress:
            images = images.to(device, non_blocking=True)
            properties = properties.to(device, non_blocking=True)
            with torch.no_grad():
                exposed = model(images)
            losses = [
                F.cross_entropy(attacker(exposed), properties[:, index])
                for index, attacker in enumerate(attackers)
            ]
            optimizer.zero_grad(set_to_none=True)
            sum(losses).backward()
            optimizer.step()

            for index, loss in enumerate(losses):
                loss_sums[index] += loss.item() * images.shape[0]
            sample_count += images.shape[0]
            progress.set_postfix(
                loss_1=f"{loss_sums[0] / sample_count:.4f}",
                loss_2=f"{loss_sums[1] / sample_count:.4f}",
            )
        final_losses = [value / sample_count for value in loss_sums]

    output_path = args.output_root / dataset_name / "property_attackers" / attacker_name(method)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {f"attacker{index + 1}": cpu_state_dict(attacker) for index, attacker in enumerate(attackers)},
        output_path,
    )
    names = property_names(dataset_name)
    del model, attackers, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "method": DISPLAY_NAMES[method],
        "train_samples": len(dataset),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "loss": "cross_entropy",
        f"{names[0]}_final_train_loss": final_losses[0],
        f"{names[1]}_final_train_loss": final_losses[1],
        "checkpoint": report_path(output_path),
    }


def main() -> None:
    args = parse_args()
    datasets_to_run = args.datasets or ["celeba"]
    device = select_device(args.device)
    rows = []
    for dataset_name in datasets_to_run:
        for method in args.methods:
            seed_everything(args.seed)
            row = train_one(args, dataset_name, method, device)
            rows.append(row)
            print(row, flush=True)
            write_csv(args.output_root / "training_summary.csv", rows)


if __name__ == "__main__":
    main()
