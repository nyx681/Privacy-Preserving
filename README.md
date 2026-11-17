# AMK 代码版

仅包含加入错误密钥训练的 AMK 实现：Qwen3.5-4B、LLaVA-Vicuna-7B、InternVL3.5-4B 的训练与 best 评测，以及公开规则密钥重建与严格恢复测试。

运行步骤见 [pipeline_amk_release/README.md](pipeline_amk_release/README.md)。模型、CLIP 权重、打码图与英文问题文件需要在本地准备；本分支不携带这些文件及任何实验输出。

```bash
git clone --branch amk-code-only --single-branch --depth 1 https://github.com/nyx681/Privacy-Preserving.git AMK
cd AMK
```

训练和评测核心代码来自原发布提交 `c2bcd5afd04c1871c4ebfae3818ad585ffa0202c`，移除了旧批次专用调度脚本。本分支使用独立初始提交，不继承原仓库图片和 Git 历史。
