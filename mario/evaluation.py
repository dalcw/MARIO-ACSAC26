"""Shared evaluation helpers used by the artifact experiments."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


METHODS = ("vanilla", "noise", "nopeek", "r3elu", "disco", "our")
DISPLAY_NAMES = {
    "vanilla": "Vanilla",
    "noise": "Noise",
    "nopeek": "NoPeek",
    "r3elu": "R3eLU",
    "disco": "DISCO",
    "our": "MARIO",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def select_device(name: str) -> torch.device:
    if name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda" and (limit_gb := os.environ.get("MARIO_GPU_MEMORY_LIMIT_GB")):
        requested_bytes = float(limit_gb) * 1024**3
        if requested_bytes <= 0:
            raise ValueError("MARIO_GPU_MEMORY_LIMIT_GB must be greater than zero")
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        fraction = min(requested_bytes / total_bytes, 1.0)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        print(
            f"PyTorch GPU memory limit: {requested_bytes / 1024**3:.2f} GiB "
            f"of {total_bytes / 1024**3:.2f} GiB",
            flush=True,
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch(path: Path, device: torch.device | str = "cpu") -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}\n"
            "Download the artifact checkpoints as described in README.md."
        )
    return torch.load(path, map_location=device, weights_only=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_expected_comparison(
    observed_rows: list[dict[str, Any]],
    expected_path: Path,
    key_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> None:
    if not expected_path.is_file():
        print(f"Expected results not found: {expected_path}", flush=True)
        return

    with expected_path.open(newline="", encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))

    def key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[field]).strip().casefold() for field in key_fields)

    expected_by_key = {key(row): row for row in expected_rows}
    table_rows: list[tuple[str, str, str, str, str]] = []
    for observed in observed_rows:
        entry = " / ".join(str(observed[field]) for field in key_fields)
        expected = expected_by_key.get(key(observed))
        if expected is None:
            table_rows.append((entry, "-", "not found", "-", "-"))
            continue
        for metric in metric_fields:
            expected_value = float(expected[metric])
            observed_value = float(observed[metric])
            table_rows.append(
                (
                    entry,
                    metric,
                    f"{expected_value:.6f}",
                    f"{observed_value:.6f}",
                    f"{observed_value - expected_value:+.6f}",
                )
            )

    headers = ("Entry", "Metric", "Expected", "Observed", "Difference")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))

    lines = [
        "",
        "Expected vs. observed results (Difference = Observed - Expected)",
        format_row(headers),
        "-+-".join("-" * width for width in widths),
        *(format_row(row) for row in table_rows),
    ]
    print("\033[94m" + "\n".join(lines) + "\033[0m", flush=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
