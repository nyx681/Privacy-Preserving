#!/usr/bin/env bash
# 可选：在已有的 Python 3.10/3.11 独立环境中安装，不创建或切换 conda 环境。
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config.sh"
[[ -x "$PYTHON_BIN" ]] || { echo "先填写 PYTHON_BIN" >&2; exit 2; }
mkdir -p -- "$RUN_TEMP_DIR" "$PIP_CACHE_DIR" "$PYTHON_CACHE_DIR"
"$PYTHON_BIN" -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_WHEEL_INDEX"
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
"$PYTHON_BIN" -m pip check
