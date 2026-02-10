#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.template .env
  echo "Created .env from .env.template. Now edit .env and add your NVIDIA_API_KEY."
fi

echo "Done."
echo "Try: nat info components | grep travel"
