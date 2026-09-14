#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python experiments/02_property_inference/evaluate.py \
  --dataset celeba \
  --batch-size 8 \
  "$@"
