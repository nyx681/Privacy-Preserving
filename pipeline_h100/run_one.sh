#!/usr/bin/env bash
set -euo pipefail
PIPELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$PIPELINE_DIR/.." && pwd)"
CONFIG_SH="${H100_CONFIG_SH:-$PIPELINE_DIR/config.sh}"
args=()
while (($#)); do
  case "$1" in
    --config) CONFIG_SH="$2"; shift 2 ;;
    -h|--help)
      cat <<'HELP'
Usage: bash pipeline_h100/run_one.sh --image IMAGE_PATH_OR_ID --gpu 0 [--detach]
  --config FILE       Server-specific paths (default: pipeline_h100/config.sh)
  --gpu N             Physical GPU index; one image uses one GPU
  --detach            Run in background; print supervisor PID and log directory
  --check-only        Validate inputs/processor on CPU; no training
  --smoke-only        One real image update + one real text update (engineering check)
  --output-root DIR   Separate output location (optional)
  --list              List registered image IDs without running training
Resume: rerun the same command, with the same data/settings/output; GPU may change.
HELP
      exit 0 ;;
    *) args+=("$1"); shift ;;
  esac
done
[[ -f "$CONFIG_SH" ]] || { echo "Missing config: $CONFIG_SH" >&2; exit 2; }
source "$CONFIG_SH"
: "${PYTHON_BIN:?Set PYTHON_BIN in config.sh}"
: "${MODEL_DIR:?Set MODEL_DIR in config.sh}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT in config.sh}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${CPU_THREADS:-4}" MKL_NUM_THREADS="${CPU_THREADS:-4}"
export HF_HOME="${CACHE_ROOT:?Set CACHE_ROOT}/huggingface"
export TMPDIR="$CACHE_ROOT/tmp" PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
mkdir -p -- "$TMPDIR"
exec "$PYTHON_BIN" "$PIPELINE_DIR/launch_one.py" "${args[@]}"
