#!/usr/bin/env python3
"""Remove local Path metadata from the trusted historical latent checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path, PosixPath

import torch


CONFIG_KEYS = (
    "batch_size",
    "max_train_samples",
    "max_test_samples",
    "epochs",
    "lr",
    "weight_decay",
    "embed_dim",
    "dropout",
    "decoder_hidden_dim",
    "decoder_base_channels",
    "recon_extra_blocks",
    "task_weight",
    "priv_attr_weight",
    "priv_recon_weight",
    "pub_recon_suppress_weight",
    "decor_weight",
    "var_weight",
    "kl_weight",
    "pub_feature_align_weight",
    "recon_probe_epochs",
    "recon_probe_lr",
    "recon_probe_weight_decay",
    "recon_decoder_hidden_dim",
    "recon_decoder_base_channels",
    "seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with torch.serialization.safe_globals([PosixPath]):
        source = torch.load(args.input, map_location="cpu", weights_only=True, mmap=True)
    release = {
        "model_state_dict": source["model_state_dict"],
        "meta": source["meta"],
        "z_dim": source["z_dim"],
        "config": {key: source["args"][key] for key in CONFIG_KEYS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    if list(release["model_state_dict"]) != list(reloaded["model_state_dict"]):
        raise AssertionError("State-dict keys changed")
    for key, tensor in release["model_state_dict"].items():
        if not torch.equal(tensor, reloaded["model_state_dict"][key]):
            raise AssertionError(f"Tensor changed: {key}")
    print(f"Wrote safe weights-only checkpoint: {args.output}")


if __name__ == "__main__":
    main()
