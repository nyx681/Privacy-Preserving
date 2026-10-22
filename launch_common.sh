#!/usr/bin/env bash
# 由两个实验入口 source，路径只来自 config.sh。
set -euo pipefail
launch_experiment() {
    local experiment="$1" mode="${2:-all}" data_dir out_dir log_dir resume
    case "$mode" in check|smoke|train|evaluate|all) ;; *) echo "模式应为 check / smoke / train / evaluate / all" >&2; return 2;; esac
    [[ -x "$PYTHON_BIN" ]] || { echo "请在 config.sh 填写可执行的 PYTHON_BIN：$PYTHON_BIN" >&2; return 2; }
    [[ -d "$MODEL_DIR" ]] || { echo "请在 config.sh 填写 MODEL_DIR：$MODEL_DIR" >&2; return 2; }
    if [[ "$experiment" == image ]]; then
        data_dir="$IMAGE_DATA_DIR"; out_dir="$IMAGE_OUTPUT_DIR"; log_dir="$IMAGE_LOG_DIR"; resume="$IMAGE_RESUME"
    else
        data_dir="$TEXT_DATA_DIR"; out_dir="$TEXT_OUTPUT_DIR"; log_dir="$TEXT_LOG_DIR"; resume="$TEXT_RESUME"
    fi
    mkdir -p -- "$out_dir" "$log_dir" "$HF_CACHE_DIR" "$TORCH_CACHE_DIR" "$PIP_CACHE_DIR" "$PYTHON_CACHE_DIR" "$RUN_TEMP_DIR"
    local -a common=(--experiment "$experiment" --model-dir "$MODEL_DIR" --model-identity "$MODEL_IDENTITY_FILE"
        --data-dir "$data_dir" --protected-image "$PROTECTED_IMAGE" --output-dir "$out_dir" --cpu-threads "$CPU_THREADS")
    if [[ "$experiment" == image ]]; then common+=(--key-image "$IMAGE_KEY_INIT" --normal-dir "$NORMAL_IMAGE_DIR"); fi
    "$PYTHON_BIN" -u "$CODE_DIR/run.py" "${common[@]}" --mode check 2>&1 | tee -a "$log_dir/check.log"
    [[ "$mode" != check ]] || return 0
    # 检查 GPU 数量和 BF16 支持，不设置空闲显存门槛、不占用其他作业的 GPU。
    "$PYTHON_BIN" - "$NPROC_PER_NODE" <<'PY'
import sys,torch
n=int(sys.argv[1])
assert n>0 and torch.cuda.device_count()>=n, f'Need {n} visible GPUs; found {torch.cuda.device_count()}'
for i in range(n):
    with torch.cuda.device(i):
        assert torch.cuda.is_bf16_supported(), f'GPU {i} lacks BF16 support'
print('Visible GPUs:',[(i,torch.cuda.get_device_name(i)) for i in range(n)])
PY
    local -a stages
    if [[ "$mode" == all ]]; then stages=(smoke train); else stages=("$mode"); fi
    local stage
    for stage in "${stages[@]}"; do
        local -a extra=()
        if [[ "$stage" != smoke && -n "$resume" ]]; then extra+=(--resume "$resume"); fi
        "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
            --max_restarts=0 "$CODE_DIR/run.py" "${common[@]}" --mode "$stage" "${extra[@]}" \
            2>&1 | tee -a "$log_dir/${stage}.log"
        if [[ "$stage" == smoke && ! -f "$out_dir/smoke/smoke_passed.json" ]]; then
            echo "预检未生成通过标记，停止。" >&2; return 1
        fi
    done
}
