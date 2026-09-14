#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python experiments/01_main_privacy_utility/train_attackers.py \
  --dataset cifar10 \
  --download-cifar10 \
  "$@"
