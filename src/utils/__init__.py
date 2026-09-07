"""
================================================================================
utils 包：工具函数模块
================================================================================
对外暴露训练所需的各类工具：
    - Trainer            : 训练器（课程学习 + 梯度对齐）
    - seed_init / generate_missing_table / init_data_hatememes : 数据与随机种子
    - MemoryBankGenerator : 记忆库生成器
    - MCR                 : 多通道检索器
    - EarlyStopping       : 早停
    - load_model / get_dataset / Collator / get_evaluator : 工厂函数
    - compute_loss        : 交叉熵损失
================================================================================
"""
from .trainer import Trainer
from .core_tools import seed_init, generate_missing_table, init_data_hatememes, init_data_mmimdb, init_data_food101, MemoryBankGenerator, MCR, EarlyStopping, load_model, get_dataset, Collator, get_evaluator, compute_loss