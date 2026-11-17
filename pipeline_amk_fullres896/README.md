# Qwen3.5-4B：新版AMK

在原图尺寸上优化同尺寸扰动，保存原尺寸保护图，再以896×672输入Qwen3.5-4B。每图2000步，正确密钥权重0.5，四个错误/缺失密钥条件各0.125；逐条件累积梯度后进行一次PGD更新。CLIP密钥分支为224×224、4×4块。

环境、准备和运行命令见[统一运行说明](../pipeline_amk_release/README.md)。从现有本地英文数据生成配置时使用统一prepare.py；本目录prepare_selected.py、manifest_batch.py与priority_queue.py保留原运行批次的历史调度接口，需要相应的已有配置与引用批次。
