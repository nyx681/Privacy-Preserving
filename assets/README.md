# 本地图片准备

GitHub 仓库已包含自然语言密钥实验所需的 `student_id.png`。图片密钥实验另外需要从原本地实验资料或迁移压缩包复制融合密钥图和正常负例照片：

- `student_id.png`（已包含）：原始打码学生证，2592×1944；对应原来的“学号1.png”。
- `card_020_basis_080.png`（需自行复制）：学生证 20% + 基底 a 80% 的初始化融合图。
- `normals/`（需自行复制）：24 张正常负例照片，文件名与 `data/image/negative_pools.json` 的 `filename` 一致。

可以直接复制本地迁移包的整个 `assets/` 目录，也可以编辑根目录 `config.sh` 中的 `PROTECTED_IMAGE`、`IMAGE_KEY_INIT`、`NORMAL_IMAGE_DIR`。自然语言密钥实验只需要学生证原图，图片密钥实验需要三项。

请使用初始化图片，而不是已训练的 protected_best.png 或 key_best.png。输入图像哈希写在两个实验的 protocol.json 中，正常照片哈希写在 negative_pools.json 中，用于保持实验内容一致。

模型目录也需要自行准备，设置 MODEL_DIR。除明确纳入的 `assets/student_id.png` 外，其他图片、权重、断点与输出仍由 .gitignore 排除。
