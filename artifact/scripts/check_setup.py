#!/usr/bin/env python3
"""Validate the artifact environment, layout, and core CIFAR-10 checkpoints."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import torch


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ARTIFACT_ROOT.parent
sys.path.insert(0, str(ARTIFACT_ROOT))

from mario.attackers import SmashedReconstructionAttacker
from mario.defenses import RepresentationExposureModel
from mario.evaluation import load_torch


REPOSITORY_FILES = (
    "README.md",
    "LICENSE.md",
    "USE.md",
    "infrastructure/constraints",
    "infrastructure/access",
    *(f"claims/claim{index}/claim.md" for index in range(1, 6)),
    *(f"claims/claim{index}/run.sh" for index in range(1, 6)),
    *(f"claims/claim{index}/expected/metrics.csv" for index in range(1, 6)),
)

ARTIFACT_FILES = (
    "experiments/01_main_privacy_utility/run.sh",
    "experiments/02_property_inference/run.sh",
    "experiments/03_likelihood_maximization/run.sh",
    "experiments/04_ablation/run.sh",
    "experiments/05_latent_decomposition/run.sh",
)


def main() -> None:
    print("Python:", sys.version.split()[0])
    for package in ("torch", "torchvision", "numpy", "pandas", "torchmetrics", "lpips"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    missing = [
        relative
        for relative in REPOSITORY_FILES
        if not (REPOSITORY_ROOT / relative).is_file()
    ]
    missing.extend(
        f"artifact/{relative}"
        for relative in ARTIFACT_FILES
        if not (ARTIFACT_ROOT / relative).is_file()
    )
    if missing:
        raise FileNotFoundError("Missing artifact files:\n" + "\n".join(missing))

    mario_path = ARTIFACT_ROOT / "checkpoints" / "release" / "cifar10" / "defenses" / "mario.pt"
    attacker_path = (
        ARTIFACT_ROOT
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
