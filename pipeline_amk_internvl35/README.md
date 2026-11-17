# InternVL3.5-4B：新版AMK

使用官方OpenGVLab/InternVL3_5-4B-HF，版本6bd4487402110ef9889ba50eb7aefeb302526fed。在原图尺寸上优化扰动，再经原生448×448动态分块（1–12块，多块附加缩略图）。五条件逐个累积梯度，每步一次PGD更新；只评估best。

环境、数据准备和运行命令见[统一运行说明](../pipeline_amk_release/README.md)。
