# 4 卡 H100：单图单命令、可选 GPU、可断点续训

一张图片在一张 H100 上顺序完成整个实验。4 张 H100 可同时跑 4 张不同图片。冻结 `Qwen2.5-VL-7B-Instruct`，只优化图片，不修改模型参数；无需 DDP、DeepSpeed 或额外的 flash-attn 安装。

本批 50 张图片已命名为 `assets/mosaic_50/1.png` 至 `50.png`。原图来源、原文件名与问答目录见 `assets/mosaic_50/image_sources.csv`。内部图片 ID 和结果目录仍沿用原 ID。

本目录是独立运行版本，不改变 `pipeline/` 中当前实验代码。H100 输出使用新目录，不能把旧版本的检查点直接混入本版本。

## 1. 配置服务器路径

编辑 `pipeline_h100/config.sh`。便携包已含基底、24 张正常负例图、50 张打码图片及问答，通常只需填写：

```bash
export PYTHON_BIN="/your/conda/envs/privacy/bin/python"
export MODEL_DIR="/your/models/Qwen2.5-VL-7B-Instruct"
```

同一文件集中提供 `BASIS_IMAGE`、`NORMAL_IMAGE_DIR`、`IMAGE_DIR`、`QUESTIONS_DIR`、`DATASET_MANIFEST`、`NORMAL_POOLS_FILE`、`MODEL_IDENTITY_FILE`、`OUTPUT_ROOT`、`CACHE_ROOT` 和 `GPU_LOCK_DIR`，可按新服务器目录修改。

模型权重不在便携包里，需在服务器上准备相同的 Qwen2.5-VL-7B-Instruct。配置中的模型文件校验会核对 tokenizer、模型配置和分片索引；不会把其他模型当作相同实验运行。

若需新建环境，可使用本地已验证的软件版本：

```bash
bash pipeline_h100/setup_env.sh /path/to/python3.10 /path/to/new/venv
```

它安装 PyTorch 2.3.0 / torchvision 0.18.0 的 CUDA 12.1 版本，以及根目录 requirements.txt 中固定版本的依赖。现有环境也可使用，但同一实验中途应保持软件版本不变。

## 2. 一张图片，一条命令

在仓库目录运行。`--gpu` 是 `nvidia-smi` 的物理卡编号，内部按 UUID 绑定，模型看到的逻辑设备为 `cuda:0`。

```bash
bash pipeline_h100/run_one.sh --image assets/mosaic_50/1.png --gpu 0 --detach
bash pipeline_h100/run_one.sh --image assets/mosaic_50/2.png --gpu 1 --detach
bash pipeline_h100/run_one.sh --image assets/mosaic_50/3.png --gpu 2 --detach
bash pipeline_h100/run_one.sh --image assets/mosaic_50/4.png --gpu 3 --detach
```

也支持绝对路径、图片文件名或注册的图片 ID：

```bash
bash pipeline_h100/run_one.sh --image customer_info_0020 --gpu 2 --detach
```

`--detach` 后台运行并输出监督进程 PID；不带该参数则前台运行。相同图片或相同 GPU 已有本 pipeline 任务时会拒绝重复启动；没有显存空闲阈值，也不会终止其他任务。共享输出目录的不同图片各自加锁，可以同时运行。

`commands_50.txt` 已列出全部 50 张图片的命令，GPU 按 0–3 分配。每行独立执行；一张卡完成当前图片后再运行它的下一条，不要把全部行同时启动到同一卡上。

```bash
bash pipeline_h100/run_one.sh --list
```

## 3. 默认实验内容

每张图片依次运行：

1. 图片密钥与受保护图联合优化：2000 步。
2. 固定上一步最佳受保护图，搜索文本密钥：200 轮。
3. 固定搜索文本密钥，从原始打码图的零扰动开始训练：2000 步。
4. 字符数与搜索密钥相同的随机密钥，从零扰动训练：2000 步。
5. token 数与搜索密钥相同的随机密钥，从零扰动训练：2000 步。

训练和评估沿用原实验：全图 RGB 扰动，范围 ±32/255，Adam 学习率 1/255；视觉编码输入 896×672，落盘保持原图尺寸。所有阶段无额外权限说明。图片初始化密钥为 20% 打码图 + 80% 基底。文本阶段正例权重 0.5，替换/插入/删除及轮换控制负例各 0.125。验证集选最佳检查点，同时评估最佳和最终检查点。

每种方法评估训练问法、第一组测试问法和新增测试问法。文本未见错误密钥分简单、困难各 36 个，每种问法组各 360 次；训练侧未见评估固定抽 10 问，授权和已见条件使用全部 60 问。图片密钥按错误图片类型单独统计。

本批包含姓名、地址、邮箱和多个敏感项，评估增加了字段匹配：邮箱须匹配完整邮箱，不能仅数字相同；地址须含完整地址文字，不能只匹配门牌和邮编；多个敏感项中泄露任一项即算泄露，授权“包含恢复”需恢复全部项。严格授权恢复仍按目标文本完全一致计算。密钥搜索会排除直接包含敏感答案的候选文本，以及无法构造既定训练/验证/困难负例划分的退化字符串；避免搜索结束后才因负例不足而失败。指标不覆盖谐音、数字文字编码等间接泄露。

## 4. 中断与续训

前台用 Ctrl+C；后台给监督进程发送 SIGTERM：

```bash
kill -TERM <命令输出的监督进程PID>
```

进程会等待当前优化步保存，或保存当前评估行后停止。任务退出后，重新执行原来的同一条命令即可，无需 `--resume`：

```bash
bash pipeline_h100/run_one.sh --image customer_info_0020 --gpu 2 --detach
```

也可以改用空闲的另一张卡，例如将 `--gpu 2` 改为 `--gpu 3`。图片、问题、参数、代码、软件及输出目录应保持一致。

- **训练**：默认每 50 步原子保存扰动、Adam 状态、随机数状态、已完成验证步、最佳结果及最佳 PNG。正常停止会额外保存当前步；机器故障或强制杀进程时从最近完整检查点继续，可能重算未保存的步骤。
- **验证**：优化步检查点在验证前保存，恢复后可接着完成尚未结束的验证。
- **文本搜索**：每轮保存当前文本、最佳文本、随机搜索状态，恢复后继续下一轮。
- **自由生成评估**：逐条保存 JSONL，恢复时跳过已完成行；损坏的最后一行会移除后重算。
- **已完成阶段**：校验输入和代码指纹后跳过；只补跑未完成阶段或评估。

恢复支持本服务器同一输出路径、同一运行版本，并允许换 GPU；不是旧版跨服务器检查点的自动迁移。不要在任务运行中更新这些 Python 文件或覆盖输入数据。

## 5. 结果位置与进度

```text
$OUTPUT_ROOT/
  <image_id>/
    launcher.pid
    launcher_status.json
    pipeline_status.json
    console.log
    image_key/                  # status.json、checkpoint_latest.pt、PNG、逐条评估
    text_search/                # search_checkpoint.pt、best_key.txt
    searched_text/
    random_chars/
    random_tokens/
    summary.md                  # 当前图片汇总，完成一组即更新
    summary.csv
    heldout_types.csv           # 未见密钥类型细分
    response_breakdown.csv      # 恢复/泄露、明确拒答、其他输出
  summary.md                    # 跨图片总表
  summary.csv
```

例如：

```bash
cat outputs/h100/customer_info_0020/pipeline_status.json
cat outputs/h100/customer_info_0020/image_key/status.json
tail -f outputs/h100/customer_info_0020/image_key.log
```

已完成的评估组会在 5 秒内写入表格；简单组评估完成一组立即更新。未完成项不记为零。“拒答”专指完整输出“无法恢复”，其他无泄露输出单列，避免把错误答案视作拒答。

## 6. 输入检查和工程验证

不占 GPU 的输入与处理器检查：

```bash
bash pipeline_h100/run_one.sh --image customer_info_0020 --check-only
```

H100 上可先执行一次真实模型的最小工程检查：

```bash
bash pipeline_h100/run_one.sh --image customer_info_0020 --gpu 0 --smoke-only
```

它只做图片/文本各一个真实梯度更新，检查冻结模型、图像写入和自由生成，保存在独立的 `smoke_image` / `smoke_text` 子目录，不计入正式实验结果。

本地测试：

```bash
/path/to/python -m unittest discover -s pipeline_h100/tests -v
```

测试用微型假 Engine 验证中断恢复后的优化器和随机状态与连续运行一致；它是工程检查，不是 MLLM 实验或 H100 性能验证。当前本机只有 RTX A6000，尚未进行 H100 实机训练。

## 7. 便携包

`build_bundle.py` 收集代码、50 张图片、已准备好的问答、视觉基底和 24 张正常负例图。不会收集模型权重、实验输出或原图裁剪。

```bash
/path/to/python pipeline_h100/build_bundle.py \
  --basis /path/to/basis_best.png \
  --normal-dir /path/to/normals \
  --out dist/privacy_pipeline_h100.tar.gz
```

另一台服务器解压后编辑 config.sh 即可。训练只读取已生成的问答及打码图片，不依赖标注记录中旧服务器的原图路径。
