#!/usr/bin/env bash
# 所有服务器相关路径集中在这里。可直接修改，也可通过同名环境变量覆盖。
# 路径可以含空格或中文。模型与图片必须与随包记录的内容一致。
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

# ===== 请在新服务器填写这两项 =====
PYTHON_BIN="${PYTHON_BIN:-/path/to/conda/env/bin/python}"
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen2.5-VL-7B-Instruct}"

# ===== 代码、数据、初始化图片（已含 student_id.png；其余图片自行复制或填写路径） =====
CODE_DIR="${CODE_DIR:-${PACKAGE_ROOT}}"
MODEL_IDENTITY_FILE="${MODEL_IDENTITY_FILE:-${PACKAGE_ROOT}/data/model_identity.json}"
IMAGE_DATA_DIR="${IMAGE_DATA_DIR:-${PACKAGE_ROOT}/data/image}"
TEXT_DATA_DIR="${TEXT_DATA_DIR:-${PACKAGE_ROOT}/data/text}"
PROTECTED_IMAGE="${PROTECTED_IMAGE:-${PACKAGE_ROOT}/assets/student_id.png}"
IMAGE_KEY_INIT="${IMAGE_KEY_INIT:-${PACKAGE_ROOT}/assets/card_020_basis_080.png}"
NORMAL_IMAGE_DIR="${NORMAL_IMAGE_DIR:-${PACKAGE_ROOT}/assets/normals}"

# ===== 结果、日志、缓存和临时目录（可放到大容量磁盘） =====
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs}"
IMAGE_OUTPUT_DIR="${IMAGE_OUTPUT_DIR:-${OUTPUT_ROOT}/image_key}"
TEXT_OUTPUT_DIR="${TEXT_OUTPUT_DIR:-${OUTPUT_ROOT}/text_key}"
IMAGE_LOG_DIR="${IMAGE_LOG_DIR:-${IMAGE_OUTPUT_DIR}/logs}"
TEXT_LOG_DIR="${TEXT_LOG_DIR:-${TEXT_OUTPUT_DIR}/logs}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_ROOT}/cache}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${CACHE_DIR}/huggingface}"
TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-${CACHE_DIR}/torch}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_DIR}/pip}"
PYTHON_CACHE_DIR="${PYTHON_CACHE_DIR:-${CACHE_DIR}/pycache}"
RUN_TEMP_DIR="${RUN_TEMP_DIR:-${OUTPUT_ROOT}/tmp}"

# 空字符串表示自动恢复各自输出目录 run/checkpoint_latest.pt。
# 仅支持本迁移包生成的断点，旧单卡脚本的断点格式不同。
IMAGE_RESUME="${IMAGE_RESUME:-}"
TEXT_RESUME="${TEXT_RESUME:-}"

# ===== 单机 8 卡；也支持单独设置 4 卡或 1 卡做相同实验 =====
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CPU_THREADS="${CPU_THREADS:-4}"
# run_both.sh parallel 使用以下两个互不重叠的分组。
IMAGE_PARALLEL_GPUS="${IMAGE_PARALLEL_GPUS:-0,1,2,3}"
TEXT_PARALLEL_GPUS="${TEXT_PARALLEL_GPUS:-4,5,6,7}"
PARALLEL_NPROC="${PARALLEL_NPROC:-4}"

# ===== 可选环境安装参数；不会随训练命令自动安装 =====
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCH_WHEEL_INDEX="${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu124}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PACKAGE_ROOT}/requirements.txt}"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME="$HF_CACHE_DIR" TORCH_HOME="$TORCH_CACHE_DIR" PIP_CACHE_DIR
export PYTHONPYCACHEPREFIX="$PYTHON_CACHE_DIR" TMPDIR="$RUN_TEMP_DIR"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 不硬编码网卡、IB 或 NCCL 拓扑；可按服务器配置在启动前设置 NCCL_* 环境变量。
