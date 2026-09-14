#!/usr/bin/env python3
"""Verify every downloaded release checkpoint against its JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root", type=Path, default=ROOT / "checkpoints" / "release"
    )
    return parser.parse_args()


def verify_manifest(manifest_path: Path, file_root: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for entry in manifest["files"]:
        path = file_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint listed by {manifest_path}: {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")
        print(f"OK {path.relative_to(args.release_root)}")
        checked += 1
    return checked


if __name__ == "__main__":
    args = parse_args()
    root = args.release_root.resolve()
    count = 0
    count += verify_manifest(root / "manifest.json", root)
    count += verify_manifest(root / "ablation" / "manifest.json", root / "ablation")
    count += verify_manifest(
        root / "latent_decomposition" / "manifest.json", root / "latent_decomposition"
    )
    print(f"Verified {count} checkpoint files.")
