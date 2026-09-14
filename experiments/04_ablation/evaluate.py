#!/usr/bin/env python3
"""Evaluate the CelebA single-stage ablation reported in Table 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mario.ablation import SingleStageAblation
from mario.artifact_data import build_loader
from mario.artifact_metrics import ReconstructionMetrics
from mario.attackers import SmashedBinaryClassifier, SmashedReconstructionAttacker224
from mario.defenses import RepresentationExposureModel
from mario.evaluation import load_torch, seed_everything, select_device, write_csv
from mario.evaluation import print_expected_comparison


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
    parser.add_argument("--attacker-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_results" / "04_ablation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates all 39,829 test images")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_variant(args: argparse.Namespace, variant: str, device: torch.device):
    attacker_root = args.attacker_root or args.checkpoint_root
    if variant in {"vanilla", "all"}:
        method = "vanilla" if variant == "vanilla" else "our"
        model_name = "vanilla.pt" if variant == "vanilla" else "mario.pt"
        model = RepresentationExposureModel(method, input_size=224).to(device)
        model.load_checkpoint(
            load_torch(args.checkpoint_root / "celeba" / "defenses" / model_name, device)
        )
        attacker_name = "vanilla_attacker.pt" if variant == "vanilla" else "our_attacker.pt"
        reconstruction_path = (
            attacker_root / "celeba" / "reconstruction_attackers" / attacker_name
        )
        property_path = attacker_root / "celeba" / "property_attackers" / attacker_name
        property_payload = load_torch(property_path, device)["attacker1"]
    else:
        model = SingleStageAblation(variant).to(device)
        variant_root = args.checkpoint_root / "ablation" / variant
        model.load_checkpoint(load_torch(variant_root / "model.pt", device))
        attacker_variant_root = attacker_root / "ablation" / variant
        reconstruction_path = attacker_variant_root / "reconstruction_attacker.pt"
        property_payload = load_torch(attacker_variant_root / "property_attacker.pt", device)
        property_payload = property_payload["attacker1"]

    reconstruction_attacker = SmashedReconstructionAttacker224().to(device)
    reconstruction_attacker.load_state_dict(load_torch(reconstruction_path, device))
    property_attacker = SmashedBinaryClassifier().to(device)
    property_attacker.load_state_dict(property_payload)
    model.eval()
    reconstruction_attacker.eval()
    property_attacker.eval()
    return model, reconstruction_attacker, property_attacker


def evaluate_variant(args: argparse.Namespace, variant: str, device: torch.device):
    loader = build_loader(
        "celeba", args.data_root, "test", args.batch_size, args.workers, shuffle=False
    )
    model, reconstruction_attacker, property_attacker = load_variant(args, variant, device)
    metrics = ReconstructionMetrics(device, lpips_net="alex")
    task_correct = 0
    property_correct = 0
    evaluated = 0

    with torch.inference_mode():
        progress = tqdm(loader, desc=f"CelebA ablation/{DISPLAY_VARIANTS[variant]}")
        for images, labels, properties in progress:
            if args.max_samples:
                remaining = args.max_samples - evaluated
                if remaining <= 0:
                    break
                images = images[:remaining]
                labels = labels[:remaining]
                properties = properties[:remaining]

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            properties = properties.to(device, non_blocking=True)
            exposed = model(images)
            logits = model.server_model(exposed)
            reconstruction = reconstruction_attacker(exposed)
            property_prediction = property_attacker(exposed).argmax(dim=1)

            task_correct += logits.argmax(dim=1).eq(labels).sum().item()
            property_correct += property_prediction.eq(properties[:, 0]).sum().item()
            evaluated += images.shape[0]
            metrics.update(reconstruction, images)
            progress.set_postfix(samples=evaluated)

    reconstruction_values = metrics.compute()
    result = {
        "variant": DISPLAY_VARIANTS[variant],
        "samples": evaluated,
        "task_accuracy": task_correct / evaluated,
        "reconstruction_lpips": reconstruction_values["lpips"],
        "property_inference_accuracy": property_correct / evaluated,
        "reconstruction_psnr": reconstruction_values["psnr"],
        "reconstruction_ssim": reconstruction_values["ssim"],
    }
    del model, reconstruction_attacker, property_attacker, metrics
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    rows = []
    for variant in args.variants:
        seed_everything(args.seed)
        row = evaluate_variant(args, variant, device)
        rows.append(row)
        print(row, flush=True)
        write_csv(args.output / "metrics.csv", rows)

    print_expected_comparison(
        rows,
        ROOT / "expected_results" / "04_ablation.csv",
        key_fields=("variant",),
        metric_fields=(
            "task_accuracy",
            "reconstruction_lpips",
            "property_inference_accuracy",
        ),
    )
    print(f"\033[94mResults saved to: {args.output.resolve()}\033[0m", flush=True)


if __name__ == "__main__":
    main()
