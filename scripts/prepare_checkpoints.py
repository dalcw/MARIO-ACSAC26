#!/usr/bin/env python3
"""Create inference-only release checkpoints from locally staged checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "checkpoints" / "raw"
DEFAULT_OUTPUT = ROOT / "checkpoints" / "release"
MANIFEST_NAME = "manifest.json"
REMOVED_KEYS = {"optimizer_state_dict"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip training-only optimizer state and verify every retained tensor."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset directory to process (repeatable). Defaults to every staged dataset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_training_state(payload: Any) -> tuple[Any, list[str]]:
    if not isinstance(payload, Mapping):
        return payload, []

    removed = [key for key in payload if key in REMOVED_KEYS]
    if not removed:
        return payload, []

    stripped = type(payload)((key, value) for key, value in payload.items() if key not in REMOVED_KEYS)
    return stripped, removed


def assert_equal(expected: Any, actual: Any, path: str = "checkpoint") -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: expected a tensor, got {type(actual).__name__}")
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(f"{path}: tensor metadata differs")
        if not torch.equal(expected, actual):
            raise AssertionError(f"{path}: tensor values differ")
        return

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or list(expected.keys()) != list(actual.keys()):
            raise AssertionError(f"{path}: mapping keys differ")
        for key in expected:
            assert_equal(expected[key], actual[key], f"{path}.{key}")
        return

    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if not isinstance(actual, Sequence) or len(expected) != len(actual):
            raise AssertionError(f"{path}: sequence differs")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            assert_equal(expected_item, actual_item, f"{path}[{index}]")
        return

    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def tensor_stats(value: Any) -> tuple[int, int]:
    tensors = 0
    parameters = 0
    if isinstance(value, torch.Tensor):
        return 1, value.numel()
    if isinstance(value, Mapping):
        for item in value.values():
            item_tensors, item_parameters = tensor_stats(item)
            tensors += item_tensors
            parameters += item_parameters
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            item_tensors, item_parameters = tensor_stats(item)
            tensors += item_tensors
            parameters += item_parameters
    return tensors, parameters


def selected_files(source: Path, datasets: list[str] | None) -> list[Path]:
    roots = [source / name for name in datasets] if datasets else [path for path in source.iterdir() if path.is_dir()]
    missing = [path for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing staged dataset directories: " + ", ".join(map(str, missing)))
    return sorted(path for root in roots for path in root.rglob("*.pt") if ".ipynb_checkpoints" not in path.parts)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint source does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)

    files = selected_files(source, args.dataset)
    if not files:
        raise FileNotFoundError(f"No .pt files found under {source}")

    manifest_entries = []
    for index, source_path in enumerate(files, start=1):
        relative = source_path.relative_to(source)
        output_path = output / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] keep {relative}")
        else:
            print(f"[{index}/{len(files)}] prepare {relative}", flush=True)
            payload = torch.load(source_path, map_location="cpu", weights_only=True)
            release_payload, removed = strip_training_state(payload)
            torch.save(release_payload, output_path)

            reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
            assert_equal(release_payload, reloaded)
            del reloaded
            tensors, parameters = tensor_stats(release_payload)
            del payload, release_payload

            print(f"  verified; removed={removed or 'none'}", flush=True)

        release_payload = torch.load(output_path, map_location="cpu", weights_only=True)
        tensors, parameters = tensor_stats(release_payload)
        top_level_keys = list(release_payload.keys()) if isinstance(release_payload, Mapping) else []
        del release_payload

        manifest_entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(output_path),
                "size_bytes": output_path.stat().st_size,
                "tensor_count": tensors,
                "tensor_elements": parameters,
                "top_level_keys": top_level_keys,
            }
        )

    manifest = {
        "format_version": 1,
        "description": "Inference-only MARIO artifact checkpoints.",
        "source_policy": "Only optimizer_state_dict is removed; all retained values are verified after serialization.",
        "files": manifest_entries,
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
