#!/usr/bin/env bash
# 80-image dataset. Inherit model/python paths from a supplied base config.
# Set MOSAIC80_BASE_CONFIG to your machine-specific configuration when needed.
source "${MOSAIC80_BASE_CONFIG:-$REPO_DIR/pipeline_h100/config.sh}"
export IMAGE_DIR="$REPO_DIR/assets/mosaic_50"
export QUESTIONS_DIR="$REPO_DIR/data/mosaic_80"
export DATASET_MANIFEST="$QUESTIONS_DIR/ready_manifest.json"
export REFUSAL_TARGET="Sorry, I can't help with that."
# Use a separate output root: the data labels/paths differ from historical runs.
export OUTPUT_ROOT="$REPO_DIR/outputs/mosaic_80"
export CACHE_ROOT="$OUTPUT_ROOT/.cache"
export GPU_LOCK_DIR="$OUTPUT_ROOT/.gpu_locks"
