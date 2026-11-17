# 新版AMK：训练、评估与密钥重建

这里的AMK均指加入错误密钥训练的扩展版本。代码覆盖Qwen3.5-4B、LLaVA-Vicuna-7B和InternVL3.5-4B，以及各模型的best评估和公开算法密钥重建测试。本分支只包含代码、协议配置和说明；数据、模型权重、保护图、检查点和实验结果由本地提供。

## 方法与目录

| 模型 | 训练/评估实现 | 模型图像输入 |
|---|---|---|
| Qwen3.5-4B | `pipeline_amk_fullres896` | 原尺寸扰动后缩放到896×672 |
| LLaVA-Vicuna-7B | `pipeline_amk_llava_wrongkeys` | 原尺寸扰动后缩放到448×336，再进行原生anyres处理 |
| InternVL3.5-4B | `pipeline_amk_internvl35` | 原尺寸扰动后进行原生448×448动态分块，最多12块，多块时加缩略图 |

模型权重始终冻结，只优化原图同尺寸的扰动。每张2000步PGD，扰动上限32/255，步长1/255，seed=42。正确密钥条件权重0.5；单字符替换、插入、删除及一个轮换条件各0.125。轮换条件顺序为无密钥、空密钥、随机密钥、截断密钥。Qwen和InternVL逐条件累积梯度后进行一次更新；LLaVA联合计算五个条件的梯度。

CLIP密钥分支使用224×224输入、4×4块，每块再缩放至224×224；归一化图像特征与原始词嵌入矩阵计算内积，逐块取最大值词元并解码拼接。密钥随保护图动态变化；VLM提示直接拼接问题和文本密钥。

`pipeline_amk_shared`保存训练中实际使用的必要共享依赖；Qwen兼容辅助模块也已放入代码目录。三个模型的训练和评测实现与原发布提交`c2bcd5a`一致。

## 环境与本地输入

当前验证环境是Python 3.12、PyTorch 2.13.0+cu132、torchvision 0.28.0+cu132和transformers 5.14.1。先安装与机器CUDA环境匹配的PyTorch/torchvision，再安装本目录`requirements.txt`。

模型检查点为`Qwen/Qwen3.5-4B`、`llava-hf/llava-v1.6-vicuna-7b-hf`、`OpenGVLab/InternVL3_5-4B-HF`。InternVL固定版本为`6bd4487402110ef9889ba50eb7aefeb302526fed`。初始化时会按`model_checksums.json`校验模型与CLIP；该文件只含公开模型的元数据、权重字节数和哈希，不含数据集记录。

需要已有的本地打码图与英文问答文件，划分为每图60条训练、2条验证、10条原有测试、10条新增测试。初始化只绑定现有数据，不翻译、不重新生成问题。常用清单是`data/mosaic_50/ready_manifest.json`和`data/mosaic_80/ready_manifest.json`；前者优先绑定1–50号，后者补足51–63号。若本地清单位置不同，显式传入其路径。清单的图片及问题路径相对于`--dataset-root`解析。

```bash
python -m pip install -r pipeline_amk_release/requirements.txt
export AMK_CLIP_DIR=/path/to/clip-vit-large-patch14
export AMK_QWEN_MODEL=/path/to/Qwen3.5-4B
export AMK_LLAVA_MODEL=/path/to/llava-v1.6-vicuna-7b-hf
export AMK_INTERNVL_MODEL=/path/to/InternVL3_5-4B-HF
export AMK_DATASET_ROOT=/path/to/local/input-tree
```

## 从新目录准备实验

默认40张为1–8、13、15、18、20、22、24、25、26、27、31、32、33、34、36、40、43、45、46、47、51–63。也可用`--images 1`或逗号及区间指定子集。保持正式实验2000步及原始数据配置不变。

```bash
python pipeline_amk_release/prepare.py \
  --family qwen --model "$AMK_QWEN_MODEL" --clip "$AMK_CLIP_DIR" \
  --dataset-root "$AMK_DATASET_ROOT" \
  --manifest data/mosaic_50/ready_manifest.json data/mosaic_80/ready_manifest.json \
  --output outputs/amk_qwen35_wrongkeys_fullres896

python pipeline_amk_release/prepare.py \
  --family llava --model "$AMK_LLAVA_MODEL" --clip "$AMK_CLIP_DIR" \
  --dataset-root "$AMK_DATASET_ROOT" \
  --manifest data/mosaic_50/ready_manifest.json data/mosaic_80/ready_manifest.json \
  --output outputs/amk_llava_wrongkeys_mosaic50

python pipeline_amk_release/prepare.py \
  --family internvl --model "$AMK_INTERNVL_MODEL" --clip "$AMK_CLIP_DIR" \
  --dataset-root "$AMK_DATASET_ROOT" \
  --manifest data/mosaic_50/ready_manifest.json data/mosaic_80/ready_manifest.json \
  --output outputs/amk_internvl35_wrongkeys_selected40
```

每个输出目录生成本地配置、问题绑定、模型身份、代码指纹、best评估策略和`commands.sh`。`bash outputs/<实验目录>/commands.sh`会按清单逐图运行；失败立即停止，重跑时已完成图片自动跳过。也可直接运行某张：

```bash
python pipeline_amk_fullres896/run.py --output outputs/amk_qwen35_wrongkeys_fullres896 --image 1
python pipeline_amk_llava_wrongkeys/run.py --output outputs/amk_llava_wrongkeys_mosaic50 --image 1
python pipeline_amk_internvl35/run.py --output outputs/amk_internvl35_wrongkeys_selected40 --image 1
```

环境或代码指纹改变时会拒绝继续旧检查点。发布代码调整了依赖路径，应在新输出目录初始化；正在运行的原实验继续使用原工作目录和原指纹。

## InternVL三进程并行

对默认完整40张清单，可以在三个终端分别运行下面命令。调度器共用逐图锁，不会重复领取同一图片；模型加载先获取全局启动锁，并检查20 GiB可用显存。全部图片完成后自动退出。

```bash
python pipeline_amk_internvl35/parallel_controller.py --output outputs/amk_internvl35_wrongkeys_selected40 --worker primary --min-free-gib 20
python pipeline_amk_internvl35/parallel_controller.py --output outputs/amk_internvl35_wrongkeys_selected40 --worker secondary --min-free-gib 20
python pipeline_amk_internvl35/parallel_controller.py --output outputs/amk_internvl35_wrongkeys_selected40 --worker tertiary --min-free-gib 20
```

不要同时给同一输出目录运行`commands.sh`和另一套调度器。

## 验证、评估与输出

每100步使用2条验证问题的正确/无密钥条件选择best；每50步保存可恢复检查点。完整评估只运行best，共3520条：训练问法1740条、原有测试问法890条、新增测试问法890条。困难组为各12个单字符替换、插入、删除密钥；训练问法固定抽10条，与两类测试问法各10条分别组合成360条。简单组为各12个随机串、无关短语和大幅多字符修改。

错误密钥池保留原评测口径，围绕2000步最终密钥生成，再用于best评估；困难密钥不保证只与best密钥相差一个字符。困难和简单组均排除与实际训练错误密钥的字符串及分词重合。标准错误密钥取自实际训练记录。

每图主要文件为`protected_best.png`、`best_key.json`、`checkpoint_latest.pt`、`status.json`、`summary.md`与原始回答JSONL。`protected_final.png`保留最终保护图用于构造错误密钥池，不执行完整final评估。模型输入与回答使用原有英文配置，汇总报告使用中文。泄露与恢复按原字段归一化字符串匹配统计；严格恢复要求回答去除首尾空白后等于目标全文。

## 密钥重建与严格恢复测试

```bash
python pipeline_amk_analysis/key_recovery_experiment.py \
  --output outputs/amk_key_reconstruction/fresh_recovery
```

默认对上述标准输出目录中已完整完成的best保护图固定取样，也兼容原Qwen的51–57及58–63分批目录。不存在的实验目录会跳过。运行顺序是：只根据保护PNG及公开CLIP规则重建并保存全部密钥；再将重建密钥与每图10条原有测试问题输入对应VLM，实际生成新回答；最后读取正确密钥和敏感答案进行评分。攻击阶段不读取保存的正确密钥、原始未保护图、扰动张量或目标答案。

主要报告两个指标：密钥重建率（整个解码字符串完全一致，以保护图为单位）与严格恢复率（回答去除首尾空白后与敏感目标全文相等，以全部图像—问题对为单位）。即使密钥重建失败，也保留对应问题在恢复率分母中。包含恢复与任意敏感项泄露保留为可审计的辅助字段，不替代严格恢复率。原始回答、图像哈希、生成配置及评分细节均保存在本地输出目录。条件为同一未修改PNG、已知公开CLIP检查点及完整密钥派生规则；每图进行一次确定性重建。
