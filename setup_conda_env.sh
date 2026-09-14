#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mario}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  conda env create -n "$ENV_NAME" -f environment.yml
fi

conda run -n "$ENV_NAME" python -m ipykernel install \
  --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
conda run -n "$ENV_NAME" python -c \
  'import torch, torchvision, numpy, pandas, torchmetrics, lpips; print("environment ready"); print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cuda available", torch.cuda.is_available())'

printf '\nEnvironment setup complete. Activate it with:\n  conda activate %s\n' "$ENV_NAME"
