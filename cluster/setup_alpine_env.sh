#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"

cd "$PROJECT_ROOT"

python3 --version
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "Environment ready at: $PROJECT_ROOT/.venv"
