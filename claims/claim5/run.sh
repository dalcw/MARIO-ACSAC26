#!/usr/bin/env bash
set -euo pipefail

CLAIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CLAIM_DIR/../.." && pwd)"
exec "$ROOT/artifact/experiments/05_latent_decomposition/run.sh" \
  --output "$CLAIM_DIR/results" \
  "$@"
