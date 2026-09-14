#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python experiments/05_latent_decomposition/evaluate.py \
  --download-cifar10 \
  --epochs 50 \
  --max-train-samples 2000 \
  --max-test-samples 500 \
  "$@"
