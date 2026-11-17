# InternVL3.5-4B：新版AMK

使用官方OpenGVLab/InternVL3_5-4B-HF，版本6bd4487402110ef9889ba50eb7aefeb302526fed。在原图尺寸上优化扰动，再经原生448×448动态分块（1–12块，多块附加缩略图）。五条件逐个累积梯度，每步一次PGD更新；只评估best。

环境与初始化见[统一运行说明](../pipeline_amk_release/README.md)。parallel_controller.py支持primary、secondary、tertiary三个调度进程，共用逐图锁和模型加载锁；命令显式使用--min-free-gib 20。start_third_after_qwen.py保留等待Qwen完整完成后自动启动第三个InternVL进程的接口。controller.py/prepare.py保留原下载预检与Qwen批次绑定逻辑，需要相应历史输入；新目录由统一prepare.py直接初始化。
