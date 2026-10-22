#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config.sh"
source "$CODE_DIR/launch_common.sh"
launch_experiment image "${1:-all}"
