#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "Error: python3 not found. Create .venv or install Python 3." >&2
  exit 1
fi

echo "Using Python: $($PYTHON --version)"
echo "Working directory: $ROOT_DIR"

if [[ "${1:-}" == "--install" ]]; then
  "$PYTHON" -m pip install -r requirements.txt
  shift
fi

if [[ "${1:-}" == "--clean-figures" ]]; then
  echo "Removing existing generated figures..."
  rm -f figures/*.png
  shift
fi

if [[ "${1:-}" == "--figures" ]]; then
  shift
  set -- --from 13 "$@"
fi

"$PYTHON" scripts/run_pipeline.py "$@"

echo
echo "Pipeline complete."
