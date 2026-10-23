# Privacy-Preserving：图片密钥与自然语言密钥实验

这是两个已有实验的迁移版本。问题、数据划分、2000 步采样顺序、随机种子、CE 权重、图片尺寸、扰动约束及检查点选择规则沿用原配置。模型保持冻结，没有 LoRA、模型微调或新增模型模块。

## 1. 修改路径

克隆本仓库后，编辑 **config.sh**。GitHub 版本包含代码、固定问题、采样日程和原始打码学生证 `assets/student_id.png`，不包含模型权重或训练输出。自然语言密钥实验所需图片已齐全；图片密钥实验还需按 [assets/README.md](assets/README.md) 复制融合密钥图和正常负例图片，或者在 config.sh 中指定它们的位置。必须填写：

```bash
PYTHON_BIN="/你的环境/bin/python"
MODEL_DIR="/你的模型目录/Qwen2.5-VL-7B-Instruct"
```

模型必须是与原实验相同的 Qwen2.5-VL-7B-Instruct，而不是 Qwen3、量化版或其他参数规模。本包验证模型配置、分词器、图像处理配置与分片索引的指纹及分片是否存在；不对所有权重文件做完整哈希。请复制原模型目录。

其他可配置路径都在同一个 SH 文件中：

| 变量 | 用途 |
| --- | --- |
| CODE_DIR | Python 代码及运行脚本所在目录 |
| MODEL_IDENTITY_FILE | 随包提供的原模型元数据指纹 |
| PROTECTED_IMAGE | 原始打码学生证，仓库已包含 assets/student_id.png，可保留默认路径 |
| IMAGE_KEY_INIT | 图片密钥初始化，自行复制到 assets/card_020_basis_080.png，或填写图片密钥初始化路径 |
| NORMAL_IMAGE_DIR | 24 张正常负例图片目录 |
| IMAGE_DATA_DIR / TEXT_DATA_DIR | 各自的问题、采样日程、条件和协议目录 |
| IMAGE_OUTPUT_DIR / TEXT_OUTPUT_DIR | 两个实验的结果目录 |
| IMAGE_LOG_DIR / TEXT_LOG_DIR | 日志目录 |
| CACHE_DIR 及 HF/Torch/Pip/Python 缓存变量 | 缓存目录 |
| RUN_TEMP_DIR | 临时文件目录 |
| IMAGE_RESUME / TEXT_RESUME | 可选的本包断点路径，空值表示自动查找输出目录断点 |
| REQUIREMENTS_FILE | Python 依赖清单 |

默认图片与数据路径相对于仓库目录；数据和打码学生证已提交，融合密钥图及正常负例图片需要自行复制或填写路径。输出建议放到容量充足的磁盘。Python 文件中没有旧服务器的绝对路径；内部 JSON 和结果文件名是固定文件格式，由上述目录控制位置。路径可以包含中文和空格。

输入文件有哈希校验；单张输入图片可重命名或更换目录，正常图片目录内保留清单中的文件名，但不能用其他图片、重采样的数据或其他密钥冒充同一个实验。使用的是原始打码图和原始 20% 学生证 + 80% 基底融合图，**不是之前已经训练好的 protected_best/key_best**。

可以先将本地迁移包中的整个 assets 目录复制到本仓库根目录；.gitignore 仅允许纳入指定的 student_id.png，其他图片仍默认排除。

## 2. 环境

新服务器从创建 Conda 环境到多卡预检的完整命令见 **[ENV_SETUP.md](ENV_SETUP.md)**。

使用独立的 Python 3.10 或 3.11 环境。代码不依赖 FlashAttention、DeepSpeed、bitsandbytes 或 qwen-vl-utils，沿用原生 SDPA 和已验证的 Qwen 图像处理。

若需要安装依赖，在填写 PYTHON_BIN 后手动执行：

```bash
bash install_env.sh
```

默认安装 PyTorch 2.5.1 / torchvision 0.20.1 的 CUDA 12.4 构建，以及 requirements.txt 中固定版本。可以在 config.sh 里调整 PyTorch 构建以适配服务器驱动。已有适配 H20 的环境可以直接使用，不会随运行脚本自动安装或升级。

原服务器使用 PyTorch 2.3.0；迁移后硬件、CUDA 或框架版本改变可能造成浮点数值及优化轨迹变化。因此，本包保持实验定义，不承诺逐位一致或复现同样的百分比。实际运行记录框架版本和 GPU 型号。安装命令参考 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

## 3. 启动

进入包目录后，先检查路径、数据、分词器与图像处理，不加载模型权重、不优化：

```bash
bash run_image_key.sh check
bash run_text_key.sh check
```

分别使用全部 8 张卡运行：

```bash
bash run_image_key.sh all
bash run_text_key.sh all
```

依次运行两个实验，每个实验使用同一组 8 张卡：

```bash
bash run_both.sh all sequential
```

或者同时运行两个实验，各分配 4 张卡，总共 8 张卡：

```bash
bash run_both.sh all parallel
```

并行分组在 config.sh 的 IMAGE_PARALLEL_GPUS、TEXT_PARALLEL_GPUS、PARALLEL_NPROC 中设置。默认 0–3 跑图片密钥，4–7 跑自然语言密钥。两个输出目录必须不同。请在 tmux 等会话中运行；脚本本身在前台运行，日志同时写入配置的日志目录。

支持的模式：

| 模式 | 行为 |
| --- | --- |
| check | CPU 路径、哈希、采样和分词/图像处理检查 |
| smoke | 检查后运行独立的一步多卡预检，包含实际模型的并行/串行梯度对照 |
| train | 检查后开始/恢复 2000 步训练，并完成最终评估 |
| all（默认） | check → smoke → train，预检失败即停止 |
| evaluate | 从本包 2000 步断点重新执行最佳/最终图片评估，不更新参数 |

`smoke` 的输出在各实验输出目录的 smoke 子目录，正式实验在 run 子目录；正式实验从零扰动开始，不沿用预检更新。恢复正式实验时仍读取自己的断点。

## 4. 实验设置不变

| 设置 | 图片密钥 | 自然语言密钥 |
| --- | --- | --- |
| 输入 | 受保护学生证 + 对应图片条件 + 问题 | 受保护学生证 + 密钥文本 + 问题 |
| 待优化 | 学生证、图片密钥的两个全图 RGB 扰动 | 学生证的一个全图 RGB 扰动 |
| 输出图片 | 2592×1944 | 2592×1944 |
| 模型每张图输入 | 896×672 | 896×672 |
| 优化器 | Adam，学习率 1/255 | 相同 |
| 扰动约束 | 每张图 ±32/255，零扰动初始化 | 相同 |
| 步数 | 2000 | 2000 |
| 每步问题数 | 1 | 1 |
| 每步条件 | 1 个正例 + 5 个负例 | 1 个正例 + 4 个轮换负例 |
| 正例总权重 | 0.5 | 0.5 |
| 负例权重 | 每个 0.1 | 每个 0.125 |
| 损失 | 回答及 EOS 的平均 token CE | 相同；未加入 Margin/难例挖掘 |
| 问题 | 60 条训练 / 2 条历史开发 / 10 条历史测试 | 相同问题文本，保留各自原始 ID |
| 权限说明 | 不加 | 不加 |

图片密钥负例仍是正常图、黑/灰/白空白图、随机 RGB 噪声图、只有学生证、只有密钥图。正常图固定 16/4/4 划分；训练/开发/最终评估使用原来的噪声种子。图片密钥的基底 a 固定，不重新训练。

自然语言密钥正文完全固定：

```text
星期三清晨，林舟在北门把七枚铜币放进蓝色木盒。
```

保持 17 种训练负例和 12 种未训练密钥变体。每个问题的负例覆盖沿用已生成日程，正负总权重不会因负例总数增加而改变。不会去掉空格或规范化密钥标点。

## 5. 多卡如何保持一次更新不变

每张卡加载同一冻结模型的副本。一次全局更新仍然只使用日程中的一个问题：按条件索引把它的 6 个或 5 个条件分配给不同 rank，各自计算加权图片梯度；以 **SUM** 汇总，**不除以卡数**。只有 rank 0 执行一次 Adam 和扰动裁剪，再广播更新后的扰动。

不使用会隐式平均梯度的 DDP，也不随卡数增大学习率或问题 batch。8 个 rank 下训练最多 6/5 个 rank 有条件计算，其余参与同步。评估病例分散到全部 8 个 rank，每个病例只生成一次，再按原顺序合并。4+4 同时运行也是同一数学目标。

本包的 CPU 多进程测试用于验证梯度求和、空闲 rank、一次 Adam 更新和评估病例覆盖。目标服务器的 `smoke` 还会对实际 Qwen 比较多卡汇总与 rank 0 顺序计算的梯度，随后检查 PNG 保存/重载和自由生成。这些检查是工程验证，不是实验结果。

`smoke` 的梯度对照使用数学 SDPA 后端，避免 BF16 融合注意力的重复反向计算波动干扰对照，仍保留相对误差小于 0.001 的阈值。正式训练和评估沿用原来的自动 SDPA 后端选择；因此预检用于检查汇总、更新和生成流程，不保证正式训练后端逐位确定。实际后端设置记入各阶段的 environment.json。

## 6. 结果与恢复

每个实验输出目录下：

- preflight.json：CPU 检查。
- smoke/smoke_passed.json：实际 GPU 预检通过记录。
- run/status.json、training_log.jsonl：进度、每个条件的 CE 和梯度范数。
- run/checkpoint_latest.pt：每 50 步及第 1 步保存的原子断点，含优化器及最佳 PNG。
- run/protected_best.png、protected_final.png；图片密钥另有 key_best.png、key_final.png。
- run/development_*：开发评估；每 100 步及第 0/1 步执行。
- run/best_train_questions*、best_legacy_test_questions* 及 final_*：最终逐条自由生成与汇总。
- 自然语言实验另有 best_heldout_key_edits* / final_heldout_key_edits*。
- run/completed.json：训练与全部评估完成标记。

检查点仍按开发集正负平衡精确匹配率选取，并列取更低开发 CE。图片实验的五类负例各占验证目标的 0.1；正常图/空白图/噪声的病例数不同不会改变类型权重。最终测试不参与选择。

中断后再次运行同一训练命令即可从本包断点恢复。数据、代码或依赖版本不兼容会停止恢复；路径和卡数可以变化。旧单卡脚本的 checkpoint 格式不同，不能直接当作本包断点使用。实验已完成时再次 train 不会重新训练；重新开始请更改输出目录。

历史测试问法已在之前实验中分析过；12 个额外错误密钥只用于本轮最终评估。跨服务器、卡数和浮点内核的差别需要记录，不能直接假定恢复率仍为旧实验的 100%。
