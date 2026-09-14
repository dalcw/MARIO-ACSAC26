#!/usr/bin/env python3
"""Optionally train ablation attackers while every evaluated model stays fixed."""

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

from mario.ablation import SingleStageAblation
from mario.artifact_data import build_dataset
from mario.attackers import SmashedBinaryClassifier, SmashedReconstructionAttacker224
from mario.defenses import RepresentationExposureModel
from mario.evaluation import load_torch, seed_everything, select_device, write_csv


VARIANTS = (
    "vanilla",
    "channel_obfuscation_only",
    "latent_decomposition_only",
    "variational_sampling_only",
    "all",
)

DISPLAY_VARIANTS = {
    "vanilla": "Vanilla",
    "channel_obfuscation_only": "Channel obfuscation only",
    "latent_decomposition_only": "Latent decomposition only",
    "variational_sampling_only": "Variational sampling only",
    "all": "MARIO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "release")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "generated_results" / "04_ablation" / "scratch_checkpoints",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_fixed_model(args: argparse.Namespace, variant: str, device: torch.device):
    if variant in {"vanilla", "all"}:
        method = "vanilla" if variant == "vanilla" else "our"
        model_name = "vanilla.pt" if variant == "vanilla" else "mario.pt"
        model = RepresentationExposureModel(method, input_size=224).to(device)
        payload = load_torch(args.checkpoint_root / "celeba" / "defenses" / model_name, device)
    else:
        model = SingleStageAblation(variant).to(device)
        payload = load_torch(args.checkpoint_root / "ablation" / variant / "model.pt", device)
    model.load_checkpoint(payload)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def output_paths(root: Path, variant: str) -> tuple[Path, Path]:
    if variant in {"vanilla", "all"}:
        name = "vanilla_attacker.pt" if variant == "vanilla" else "our_attacker.pt"
        return (
            root / "celeba" / "reconstruction_attackers" / name,
            root / "celeba" / "property_attackers" / name,
        )
    variant_root = root / "ablation" / variant
    return variant_root / "reconstruction_attacker.pt", variant_root / "property_attacker.pt"


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def report_path(path: Path) -> str:
    absolute_path = path.resolve()
    try:
        return absolute_path.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute_path.as_posix()


def train_one(args: argparse.Namespace, variant: str, device: torch.device):
    dataset = build_dataset("celeba", args.data_root, "train")
    if args.max_train_samples > 0:
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = load_fixed_model(args, variant, device)
    reconstruction_attacker = SmashedReconstructionAttacker224().to(device)
    property_attacker = SmashedBinaryClassifier().to(device)
    optimizer = torch.optim.Adam(
        list(reconstruction_attacker.parameters()) + list(property_attacker.parameters()),
        lr=args.learning_rate,
    )
    final_reconstruction_loss = 0.0
    final_property_loss = 0.0

    for epoch in range(args.epochs):
        reconstruction_attacker.train()
        property_attacker.train()
        reconstruction_sum = 0.0
        property_sum = 0.0
        sample_count = 0
        progress = tqdm(loader, desc=f"train ablation attackers/{DISPLAY_VARIANTS[variant]} {epoch + 1}/{args.epochs}")
        for images, _, properties in progress:
            images = images.to(device, non_blocking=True)
            properties = properties.to(device, non_blocking=True)
            with torch.no_grad():
                exposed = model(images)
            reconstruction_loss = F.mse_loss(reconstruction_attacker(exposed), images)
            property_loss = F.cross_entropy(property_attacker(exposed), properties[:, 0])
            optimizer.zero_grad(set_to_none=True)
            (reconstruction_loss + property_loss).backward()
            optimizer.step()

            reconstruction_sum += reconstruction_loss.item() * images.shape[0]
            property_sum += property_loss.item() * images.shape[0]
            sample_count += images.shape[0]
            progress.set_postfix(
                recon=f"{reconstruction_sum / sample_count:.5f}",
                property=f"{property_sum / sample_count:.5f}",
            )
        final_reconstruction_loss = reconstruction_sum / sample_count
        final_property_loss = property_sum / sample_count

    reconstruction_path, property_path = output_paths(args.output_root, variant)
    reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
    property_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state_dict(reconstruction_attacker), reconstruction_path)
    torch.save({"attacker1": cpu_state_dict(property_attacker)}, property_path)
    del model, reconstruction_attacker, property_attacker, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": DISPLAY_VARIANTS[variant],
        "train_samples": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "reconstruction_loss": "mse",
        "property_loss": "cross_entropy",
        "final_reconstruction_train_loss": final_reconstruction_loss,
        "final_property_train_loss": final_property_loss,
        "reconstruction_checkpoint": report_path(reconstruction_path),
        "property_checkpoint": report_path(property_path),
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    rows = []
    for variant in args.variants:
        seed_everything(args.seed)
        row = train_one(args, variant, device)
        rows.append(row)
        print(row, flush=True)
        write_csv(args.output_root / "training_summary.csv", rows)


if __name__ == "__main__":
    main()
