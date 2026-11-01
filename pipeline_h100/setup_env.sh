#!/usr/bin/env bash
set -euo pipefail
# Reproduce the locally verified software versions; no model downloads here.
# Usage: bash pipeline_h100/setup_env.sh /path/to/python3.10 /path/to/new/venv
BASE_PYTHON="${1:?Usage: setup_env.sh /path/to/python3.10 /path/to/new/venv}"
ENV_DIR="${2:?Set the new virtual environment directory}"
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then "$BASE_PYTHON" -m venv "$ENV_DIR"; fi
"$ENV_DIR/bin/python" -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
"$ENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements.txt"
"$ENV_DIR/bin/python" -c 'import torch, transformers; print("torch:", torch.__version__, "transformers:", transformers.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count())'
printf 'Set PYTHON_BIN="%s/bin/python" in pipeline_h100/config.sh\n' "$ENV_DIR"
