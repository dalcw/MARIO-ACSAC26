#!/usr/bin/env python3
"""Extract only forward-reachable tensors from historical ablation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "channel_obfuscation_only": ("C5", ("encoder.",)),
    "latent_decomposition_only": (
        "C6",
        ("linear_z.", "linear_z_pub.", "info_expand_priv.", "priv_decoder."),
    ),
    "variational_sampling_only": ("C7", ("linear_mu.", "linear_logvar.")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the historical 08_ablation_study directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoints" / "release" / "ablation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = []

    for variant, (source_name, prefixes) in VARIANTS.items():
        source_dir = args.source.resolve() / source_name / "parameters"
        source_model = source_dir / "model.pt"
        print(f"Preparing {variant} from {source_model}", flush=True)
        payload = torch.load(source_model, map_location="cpu", weights_only=True, mmap=True)
        adapter = {
            key: value
            for key, value in payload["adapter_model"].items()
            if key.startswith(prefixes)
        }
        release_payload = {
            "variant": variant,
            "client_model": payload["client_model"],
            "adapter_model": adapter,
            "server_model": payload["server_model"],
        }
        variant_output = output / variant
        variant_output.mkdir(parents=True, exist_ok=True)
        model_output = variant_output / "model.pt"
        torch.save(release_payload, model_output)
        reloaded = torch.load(model_output, map_location="cpu", weights_only=True)
        for group in ("client_model", "adapter_model", "server_model"):
            if list(release_payload[group]) != list(reloaded[group]):
                raise AssertionError(f"{variant}/{group}: keys changed during serialization")
            for key in release_payload[group]:
                if not torch.equal(release_payload[group][key], reloaded[group][key]):
                    raise AssertionError(f"{variant}/{group}/{key}: tensor changed")

        shutil.copy2(source_dir / "recon_attack.pt", variant_output / "reconstruction_attacker.pt")
        shutil.copy2(source_dir / "property_attack.pt", variant_output / "property_attacker.pt")
        del payload, release_payload, reloaded

        for filename in ("model.pt", "reconstruction_attacker.pt", "property_attacker.pt"):
            path = variant_output / filename
            entries.append(
                {
                    "variant": variant,
                    "path": path.relative_to(output).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "note": "Unused, forward-unreachable tensors were removed from historical ablation checkpoints.",
                "files": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
