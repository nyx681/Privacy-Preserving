#!/usr/bin/env bash
# Edit paths for the H100 server. REPO_DIR is set by run_one.sh.
export PYTHON_BIN="/path/to/conda/envs/privacy/bin/python"
export MODEL_DIR="/path/to/Qwen2.5-VL-7B-Instruct"
# These inputs are included when build_bundle.py creates the portable package.
export BASIS_IMAGE="$REPO_DIR/pipeline_h100/inputs/basis_best.png"
export NORMAL_IMAGE_DIR="$REPO_DIR/pipeline_h100/inputs/normals"
export IMAGE_DIR="$REPO_DIR/assets/mosaic_50"
export QUESTIONS_DIR="$REPO_DIR/data/mosaic_50"
export DATASET_MANIFEST="$QUESTIONS_DIR/ready_manifest.json"
export NORMAL_POOLS_FILE="$REPO_DIR/data/image/negative_pools.json"
export MODEL_IDENTITY_FILE="$REPO_DIR/data/model_identity.json"
export OUTPUT_ROOT="$REPO_DIR/outputs/h100"
export CACHE_ROOT="$OUTPUT_ROOT/cache"
export GPU_LOCK_DIR="$OUTPUT_ROOT/.gpu_locks"
export CPU_THREADS=4
export STEPS=2000
export SEARCH_STEPS=200
export EVAL_EVERY=100
export CHECKPOINT_EVERY=50
export SEED=42
export MAX_NEW_TOKENS=64
