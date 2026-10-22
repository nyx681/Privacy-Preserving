#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config.sh"
mode="${1:-all}"
arrangement="${2:-sequential}"
case "$arrangement" in
    sequential)
        bash "$CODE_DIR/run_image_key.sh" "$mode"
        bash "$CODE_DIR/run_text_key.sh" "$mode"
        ;;
    parallel)
        "$PYTHON_BIN" - "$IMAGE_PARALLEL_GPUS" "$TEXT_PARALLEL_GPUS" "$PARALLEL_NPROC" <<'PY'
import sys
a,b=(s.split(',') for s in sys.argv[1:3]);n=int(sys.argv[3])
assert len(a)==len(b)==n and len(set(a+b))==2*n,'Parallel GPU groups must be distinct, non-overlapping and match PARALLEL_NPROC'
PY
        GPU_LIST="$IMAGE_PARALLEL_GPUS" NPROC_PER_NODE="$PARALLEL_NPROC" bash "$CODE_DIR/run_image_key.sh" "$mode" &
        image_pid=$!
        GPU_LIST="$TEXT_PARALLEL_GPUS" NPROC_PER_NODE="$PARALLEL_NPROC" bash "$CODE_DIR/run_text_key.sh" "$mode" &
        text_pid=$!
        image_code=0; text_code=0
        wait "$image_pid" || image_code=$?
        wait "$text_pid" || text_code=$?
        [[ "$image_code" == 0 && "$text_code" == 0 ]]
        ;;
    *) echo "第二个参数应为 sequential 或 parallel" >&2; exit 2;;
esac
