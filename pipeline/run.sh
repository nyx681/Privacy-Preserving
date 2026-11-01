#!/usr/bin/env bash
set -euo pipefail
PIPELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$PIPELINE_DIR/.." && pwd)"
# Required server-specific paths; set via environment or a local sourced .sh file.
: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${BASIS_IMAGE:?Set BASIS_IMAGE}"
: "${NORMAL_IMAGE_DIR:?Set NORMAL_IMAGE_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/card_pipeline}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-0}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${CPU_THREADS:-4}" MKL_NUM_THREADS="${CPU_THREADS:-4}"
exec "$PYTHON_BIN" "$PIPELINE_DIR/run_pipeline.py" \
  --model "$MODEL_DIR" --basis "$BASIS_IMAGE" --normal-dir "$NORMAL_IMAGE_DIR" \
  --normal-pools "${NORMAL_POOLS_FILE:-$REPO_DIR/data/image/negative_pools.json}" \
  --model-identity "${MODEL_IDENTITY_FILE:-$REPO_DIR/data/model_identity.json}" \
  --registry "${REGISTRY_FILE:-$PIPELINE_DIR/registry.json}" --output "$OUTPUT_ROOT" "$@"
