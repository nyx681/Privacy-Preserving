# LLaVA-Vicuna-7B：新版AMK

在原图尺寸上优化扰动，再以448×336预画布进入LLaVA原生anyres处理。每图2000步；每步对正确密钥及四个错误/缺失密钥条件联合计算梯度，权重分别为0.5和各0.125。每图只完整评估best。

环境、数据准备和运行命令见[统一运行说明](../pipeline_amk_release/README.md)。
