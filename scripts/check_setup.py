#!/usr/bin/env python3
"""Validate the artifact environment, layout, and core CIFAR-10 checkpoints."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mario.attackers import SmashedReconstructionAttacker
from mario.defenses import RepresentationExposureModel
from mario.evaluation import load_torch


REQUIRED_FILES = (
    "experiments/01_main_privacy_utility/run.sh",
    "experiments/02_property_inference/run.sh",
    "experiments/03_likelihood_maximization/run.sh",
    "experiments/04_ablation/run.sh",
    "experiments/05_latent_decomposition/run.sh",
    "expected_results/01_main_privacy_utility.csv",
    "expected_results/02_property_inference.csv",
    "expected_results/03_likelihood_maximization.csv",
    "expected_results/04_ablation.csv",
    "expected_results/05_latent_decomposition.csv",
)


def main() -> None:
    print("Python:", sys.version.split()[0])
    for package in ("torch", "torchvision", "numpy", "pandas", "torchmetrics", "lpips"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing artifact files:\n" + "\n".join(missing))

    mario_path = ROOT / "checkpoints" / "release" / "cifar10" / "defenses" / "mario.pt"
    attacker_path = (
        ROOT
        / "checkpoints"
        / "release"
        / "cifar10"
        / "reconstruction_attackers"
        / "our_attacker.pt"
    )
    model = RepresentationExposureModel("our", input_size=32)
    model.load_checkpoint(load_torch(mario_path))
    attacker = SmashedReconstructionAttacker()
    attacker.load_state_dict(load_torch(attacker_path))
    model.eval()
    attacker.eval()

    with torch.inference_mode():
        exposed = model(torch.zeros(1, 3, 32, 32))
        reconstruction = attacker(exposed)
        logits = model.server_model(exposed)
    assert exposed.shape == (1, 128, 16, 16)
    assert reconstruction.shape == (1, 3, 32, 32)
    assert logits.shape == (1, 10)
    print("MARIO exposed representation:", tuple(exposed.shape))
    print("Reconstruction output:", tuple(reconstruction.shape))
    print("Task logits:", tuple(logits.shape))
    print("Setup check passed.")


if __name__ == "__main__":
    main()
