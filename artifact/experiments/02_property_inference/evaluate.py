#!/usr/bin/env python3
"""Evaluate pretrained property-inference attackers on exposed representations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from mario.artifact_data import build_loader, input_size, property_names
from mario.attackers import SmashedBinaryClassifier
from mario.defenses import RepresentationExposureModel
from mario.evaluation import DISPLAY_NAMES, METHODS, load_torch, seed_everything, select_device, write_csv
from mario.evaluation import print_expected_comparison


PROPERTY_DATASETS = ("celeba", "nih_chest_xray")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=PROPERTY_DATASETS, dest="datasets")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "release")
    parser.add_argument("--attacker-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_results" / "02_property_inference")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the complete test set")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def checkpoint_name(method: str) -> str:
    return "mario.pt" if method == "our" else f"{method}.pt"


def attacker_name(method: str) -> str:
    return "our_attacker.pt" if method == "our" else f"{method}_attacker.pt"


def evaluate_method(args: argparse.Namespace, dataset_name: str, method: str, device: torch.device):
    batch_size = args.batch_size or 16
    loader = build_loader(
        dataset_name,
        args.data_root,
        "test",
        batch_size,
        args.workers,
        shuffle=False,
    )
    defense_path = args.checkpoint_root / dataset_name / "defenses" / checkpoint_name(method)
    attacker_root = args.attacker_root or args.checkpoint_root
    attacker_path = attacker_root / dataset_name / "property_attackers" / attacker_name(method)

    model = RepresentationExposureModel(method, input_size(dataset_name)).to(device)
    model.load_checkpoint(load_torch(defense_path, device))
    model.eval()

    attacker_payload = load_torch(attacker_path, device)
    attackers = [SmashedBinaryClassifier().to(device), SmashedBinaryClassifier().to(device)]
    for index, attacker in enumerate(attackers, start=1):
        attacker.load_state_dict(attacker_payload[f"attacker{index}"])
        attacker.eval()

    correct = [0, 0]
    evaluated = 0
    with torch.inference_mode():
        progress = tqdm(loader, desc=f"{dataset_name}/{DISPLAY_NAMES[method]}")
        for images, _, properties in progress:
            if args.max_samples:
                remaining = args.max_samples - evaluated
                if remaining <= 0:
                    break
                images = images[:remaining]
                properties = properties[:remaining]

            images = images.to(device, non_blocking=True)
            properties = properties.to(device, non_blocking=True)
            exposed = model(images)
            for index, attacker in enumerate(attackers):
                prediction = attacker(exposed).argmax(dim=1)
                correct[index] += prediction.eq(properties[:, index]).sum().item()
            evaluated += images.shape[0]
            progress.set_postfix(samples=evaluated)

    names = property_names(dataset_name)
    result = {
        "dataset": dataset_name,
        "method": DISPLAY_NAMES[method],
        "samples": evaluated,
        "property_1": names[0],
        "property_1_accuracy": correct[0] / evaluated,
        "property_2": names[1],
        "property_2_accuracy": correct[1] / evaluated,
    }
    del model, attackers, attacker_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    datasets_to_run = args.datasets or ["celeba"]
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
        REPOSITORY_ROOT / "claims" / "claim2" / "expected" / "metrics.csv",
        key_fields=("dataset", "method"),
        metric_fields=("property_1_accuracy", "property_2_accuracy"),
    )
    print(f"\033[94mResults saved to: {args.output.resolve()}\033[0m", flush=True)


if __name__ == "__main__":
    main()
