# 80张图片的英文训练与评估数据

本目录是80张图片的统一问答数据集。图片仍在 `assets/mosaic_50/1.png`～`80.png`，本目录不另复制图片。

| 文件 | 用途 | 问题数 |
|---|---|---:|
| `train.jsonl` | 训练问题，每张60条 | 4800 |
| `validation.jsonl` | 验证问题，每张2条 | 160 |
| `test.jsonl` | 原有测试模板，每张10条 | 800 |
| `test_extra.jsonl` | 新增测试模板，每张10条 | 800 |
| `ready_qa.jsonl` / `all_qa.csv` | 全部问答 | 6560 |

`ready_manifest.json` 是训练程序读取的80张图片清单，含图片与问题路径、编号、目标字段、授权答案、答案组成项和SHA-256。`answers.csv` / `answers.json` 为80张图片的目标答案总表。`source_audit.json` 记录问题与答案来源；`validation.json` 记录自动校验结果。

每张图片的问答放在 `<内部图片ID>/questions.json`，文件里有 `train`、`validation`、`test`、`test_extra` 四个数组。图片编号记在manifest和每图 `metadata.json` 的 `number` 字段中；原始ID与图片编号的对应关系也见 `assets/mosaic_50/image_sources_80.csv`。

例如：

- 第1张：`2017_10003252/questions.json`。
- 第51张：`2017_10435547/questions.json`。
- 第52张：`student_id/questions.json`。
- 第53张：`expansion80_VISPR_2017_10032627/questions.json`。
- 第80张：`expansion80_email_0029/questions.json`。

全部问题文本为英文；授权答案保留来源中的语言、拼写、大小写、前导零及多行文字，不翻译中文姓名。未授权目标在文件中直接写为 `Sorry, I can't help with that.`，与训练配置一致。

1～50的问题ID、问法、数据划分和授权答案沿用原数据，仅在本目录副本中将历史中文拒答标签改为英文。51～80使用相同英文模板生成82条问题。不同图片共享模板，同一图片的四个划分问句不重复；这里的测试衡量单图不同问法的表现，不是图片间独立测试或独立模板测试。51、52的英文问题是新生成的，不能与旧中文实验视为同一批评估。

53～80的答案由助手查看真实原图及字段裁剪复核，未经过第二位独立标注者复核。第53张只问车牌大号数字856，不要求日本地区和分类代码；第61、66张按证件印刷顺序记录姓与名，换行保留在答案中。源图字段复核图保存在本地 `data/mosaic_50/answer_review/expansion80/answer_check_*.jpg`，不纳入Git。旧数据 `data/mosaic_50/` 保持原样。

生成：

```bash
python3 data/mosaic_80/build_dataset.py
```

训练入口配置为 `pipeline_h100/config.mosaic80.sh`，设置 `QUESTIONS_DIR=data/mosaic_80` 和 `DATASET_MANIFEST=data/mosaic_80/ready_manifest.json`，使用独立的 `outputs/mosaic_80` 保存未来结果。模型与Python路径从基础配置继承，也可通过 `MOSAIC80_BASE_CONFIG` 指定本机配置。此次只准备数据及输入检查，没有启动训练。
