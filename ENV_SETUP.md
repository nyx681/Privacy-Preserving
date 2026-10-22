# 8 卡 H20 环境配置命令

适用于 Linux x86_64、单机 8 张 H20。图片密钥与自然语言密钥实验共用一个环境。
以下命令在目标服务器的 Bash 终端中分段执行；遇到报错先处理，不继续运行后续步骤。
服务器需已安装 Git、Conda（Miniconda / Anaconda / Miniforge 均可）及 NVIDIA 驱动。

## 1. 检查服务器并下载代码

```bash
git --version
conda --version
nvidia-smi
nvidia-smi -L

# 在希望存放代码的目录执行；已有仓库则直接进入该目录，不重复 clone。
git clone https://github.com/nyx681/Privacy-Preserving.git
cd Privacy-Preserving
```

`nvidia-smi -L` 应列出 8 张 H20。若 `nvidia-smi` 不可用，需先由服务器管理员处理驱动。

## 2. 创建独立 Python 环境

```bash
# 初始化当前 Bash 中的 conda activate 功能。
source "$(conda info --base)/etc/profile.d/conda.sh"

# 只需创建一次；已有同名环境时不要重复创建。
conda create -n privacy-h20 python=3.11 pip -y
conda activate privacy-h20

python --version
python -c 'import sys; print(sys.executable)'
```

记下最后一行输出的 Python 绝对路径，下一步填入 `PYTHON_BIN`。

## 3. 填写路径并准备模型、图片

用编辑器打开仓库根目录的 `config.sh`，例如：

```bash
vi config.sh
```

修改对应配置项的默认值；以下只是填写示例，必须换成新服务器上的实际路径：

```bash
PYTHON_BIN="${PYTHON_BIN:-/实际的conda环境路径/bin/python}"
MODEL_DIR="${MODEL_DIR:-/实际的模型目录/Qwen2.5-VL-7B-Instruct}"
PROTECTED_IMAGE="${PROTECTED_IMAGE:-/实际的图片目录/student_id.png}"
IMAGE_KEY_INIT="${IMAGE_KEY_INIT:-/实际的图片目录/card_020_basis_080.png}"
NORMAL_IMAGE_DIR="${NORMAL_IMAGE_DIR:-/实际的图片目录/normals}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/实际的结果目录/privacy_h20}"
```

仓库已包含 `assets/student_id.png`，自然语言密钥实验的图片路径可保留默认值。
图片密钥实验所需的另外两项也复制到 `assets/` 后，三个图片路径均可保留默认值。具体文件清单见
[assets/README.md](assets/README.md)。模型需复制原实验完整的 Qwen2.5-VL-7B-Instruct
目录；仓库未包含模型权重、融合密钥图和正常负例图片，运行时也不会联网下载模型。
保留 `GPU_LIST=0,1,2,3,4,5,6,7`、`NPROC_PER_NODE=8` 的默认设置。
缓存和临时目录默认位于 `OUTPUT_ROOT` 下，也可在 `config.sh` 中单独修改。

## 4. 安装依赖

在仓库根目录执行：

```bash
bash install_env.sh
```

该脚本使用 `config.sh` 中的 Python，默认执行的核心安装命令如下。
这是脚本内容的说明，上一条成功后无需重复安装：

```bash
source ./config.sh
mkdir -p -- "$RUN_TEMP_DIR" "$PIP_CACHE_DIR" "$PYTHON_CACHE_DIR"
"$PYTHON_BIN" -m pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
"$PYTHON_BIN" -m pip check
```

默认使用 PyTorch 2.5.1 / torchvision 0.20.1 的 CUDA 12.4 构建，其余依赖版本由
`requirements.txt` 固定。该组合见 [PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/#v251)。
服务器驱动需兼容该 CUDA 构建。若管理员要求使用其他构建，在安装前统一修改
`config.sh` 的 `TORCH_VERSION`、`TORCHVISION_VERSION` 和 `TORCH_WHEEL_INDEX`。
本项目不需要另行安装 FlashAttention、DeepSpeed、bitsandbytes 或 qwen-vl-utils。

## 5. 检查依赖、8 张 GPU、BF16 和 NCCL

```bash
source ./config.sh
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" - "$NPROC_PER_NODE" <<'PY'
import sys
import torch
import torchvision
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA 不可用，请检查驱动及 PyTorch 构建"
assert torch.distributed.is_nccl_available(), "当前 PyTorch 不支持 NCCL"
n = int(sys.argv[1])
assert n == 8, "本节检查按 8 卡配置运行"
assert torch.cuda.device_count() >= n, "可见 GPU 少于 8 张，请检查 GPU_LIST 和作业分配"
for i in range(n):
    with torch.cuda.device(i):
        assert torch.cuda.is_bf16_supported(), f"GPU {i} 不支持 BF16"
        x = torch.ones(1, device=f"cuda:{i}", dtype=torch.bfloat16)
        assert (x + x).item() == 2
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}; BF16 OK")
print("基础环境检查通过；实际多卡通信和模型检查见下一步。")
PY
```

此处仅做小张量检查，不加载模型。NCCL 可用不等于实际多卡通信已经通过，下一步
的 `smoke` 会使用完整模型验证梯度汇总及生成。

## 6. 检查输入并做一步多卡预检

```bash
# CPU 检查路径、文件哈希、问题、分词器及图像处理配置。
bash run_image_key.sh check
bash run_text_key.sh check

# 依次使用配置的 8 张卡加载模型并做一步预检；不启动 2000 步训练。
bash run_image_key.sh smoke
bash run_text_key.sh smoke
```

两个 `smoke` 都成功、各自生成 `smoke/smoke_passed.json` 后，目标服务器环境才算
通过本实验的运行检查。当前仓库的 CPU 验证记录不代替新服务器的 H20 实测。
若文件哈希不匹配，检查是否复制了原实验输入和完整模型，不要关闭校验。

## 7. 环境就绪后启动正式训练

已在第 6 步通过两个预检时，可以依次训练两个实验，每个实验使用全部 8 张卡：

```bash
bash run_both.sh train sequential
```

若尚未运行预检，使用下面的入口自动执行检查、预检和训练；与上一条二选一：

```bash
bash run_both.sh all sequential
```

正式训练建议在服务器已有的 tmux / screen 会话中运行。脚本在前台执行，日志自动
写入 `config.sh` 指定目录。完整的独立运行、4+4 并行及恢复方法见 [README.md](README.md)。
