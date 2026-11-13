#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv/bin/python ]]; then
    uv venv --python "$PYTHON_BIN" .venv
  fi
  uv pip install --python .venv/bin/python -e '.[dev]'
else
  if [[ ! -x .venv/bin/python ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e '.[dev]'
fi

mkdir -p outputs/{checkpoints,figures,logs,submissions,dogs_vs_cats,cifar10,class_imbalance}
.venv/bin/python scripts/verify_dataset.py --data-dir . \
  --output "outputs/logs/dataset_verification_$(date +%Y%m%d_%H%M%S).json"
.venv/bin/python -m pytest -q
printf '\nSetup complete. In VS Code select: %s/.venv/bin/python\n' "$ROOT"

