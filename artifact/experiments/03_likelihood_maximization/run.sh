#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python experiments/03_likelihood_maximization/attack.py \
  --dataset cifar10 \
  --download-cifar10 \
  --steps 2000 \
  --max-samples 16 \
  "$@"
