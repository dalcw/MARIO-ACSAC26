#!/usr/bin/env python3
"""Optionally train supervised reconstruction attackers against fixed models."""

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

from mario.artifact_data import DATASET_NAMES, build_dataset, default_batch_size, input_size
from mario.attackers import SmashedReconstructionAttacker, SmashedReconstructionAttacker224
from mario.defenses import RepresentationExposureModel
from mario.evaluation import DISPLAY_NAMES, METHODS, load_torch, seed_everything, select_device, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=DATASET_NAMES, dest="datasets")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "release")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "generated_results" / "01_main_privacy_utility" / "scratch_checkpoints",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, help="Defaults to 30 for CIFAR-10 and 10 otherwise")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-cifar10", action="store_true")
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
    size = input_size(dataset_name)
    batch_size = args.batch_size or default_batch_size(dataset_name)
    if args.batch_size is None and size == 224 and method == "disco":
        batch_size = 32
    epochs = args.epochs or (30 if dataset_name == "cifar10" else 10)

    dataset = build_dataset(
        dataset_name,
        args.data_root,
        "train",
        download=args.download_cifar10 and dataset_name == "cifar10",
    )
    if args.max_train_samples > 0:
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = RepresentationExposureModel(method, size).to(device)
    defense_path = args.checkpoint_root / dataset_name / "defenses" / checkpoint_name(method)
    model.load_checkpoint(load_torch(defense_path, device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    attacker_class = (
        SmashedReconstructionAttacker
        if size == 32 or method == "disco"
        else SmashedReconstructionAttacker224
    )
    attacker = attacker_class().to(device)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=args.learning_rate)
    final_loss = 0.0

    for epoch in range(epochs):
        attacker.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(loader, desc=f"train recon/{dataset_name}/{DISPLAY_NAMES[method]} {epoch + 1}/{epochs}")
        for batch in progress:
            images = batch[0].to(device, non_blocking=True)
            with torch.no_grad():
                exposed = model(images)
            reconstruction = attacker(exposed)
            loss = F.mse_loss(reconstruction, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * images.shape[0]
            sample_count += images.shape[0]
            progress.set_postfix(loss=f"{loss_sum / sample_count:.5f}")
        final_loss = loss_sum / sample_count

    output_path = args.output_root / dataset_name / "reconstruction_attackers" / attacker_name(method)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state_dict(attacker), output_path)
    del model, attacker, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "method": DISPLAY_NAMES[method],
        "train_samples": len(dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "loss": "mse",
        "final_train_loss": final_loss,
        "checkpoint": report_path(output_path),
    }


def main() -> None:
    args = parse_args()
    datasets_to_run = args.datasets or ["cifar10"]
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
