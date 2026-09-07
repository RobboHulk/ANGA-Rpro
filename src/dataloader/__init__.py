"""
================================================================================
dataloader 包：数据集加载模块
================================================================================
对外暴露三个数据集的 Dataset 类：
    - Food101Dataset    : Food101 食物图像分类
    - HatememesDataset  : HateMemes 仇恨梗图二分类
    - MMIMDbDataset     : MM-IMDb 电影类型分类

这些 Dataset 统一继承 torch.utils.data.Dataset，负责：
    读取预处理好的 .pkl、模拟缺失模态、加载检索到的记忆库特征。
================================================================================
"""
from .food101_dataset import Food101Dataset
from .hatememes_dataset import HatememesDataset
from .mmimdb_dataset import MMIMDbDataset