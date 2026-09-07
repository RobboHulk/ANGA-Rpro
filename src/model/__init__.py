"""
================================================================================
model 包：模型模块
================================================================================
对外暴露：
    - ANGA              : 论文提出的 ANGA 主模型（锚点引导的梯度对齐框架）
    - ViltModel         : 从 HuggingFace transformers 复制的 ViLT 骨干网络
    - ViltImageProcessor: ViLT 图像预处理器
================================================================================
"""
from .ANGA import ANGA
from .vilt import ViltModel, ViltImageProcessor