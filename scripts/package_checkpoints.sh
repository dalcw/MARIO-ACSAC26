#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT/checkpoint_archives}"
RELEASE_DIR="$ROOT/checkpoints/release"

mkdir -p "$OUTPUT_DIR"

tar -czf "$OUTPUT_DIR/mario-cifar10-checkpoints.tar.gz" \
  -C "$RELEASE_DIR" cifar10 latent_decomposition
tar -czf "$OUTPUT_DIR/mario-celeba-checkpoints.tar.gz" \
  -C "$RELEASE_DIR" celeba ablation
tar -czf "$OUTPUT_DIR/mario-nih-checkpoints.tar.gz" \
  -C "$RELEASE_DIR" nih_chest_xray

(
  cd "$OUTPUT_DIR"
  sha256sum \
    mario-cifar10-checkpoints.tar.gz \
    mario-celeba-checkpoints.tar.gz \
    mario-nih-checkpoints.tar.gz \
    > SHA256SUMS
)

printf 'Checkpoint archives and SHA256SUMS written to %s\n' "$OUTPUT_DIR"
