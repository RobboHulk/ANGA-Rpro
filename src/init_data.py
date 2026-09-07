"""
================================================================================
数据集初始化脚本（Data Preprocessing Entry）
================================================================================
本文件是整个 ANGA 框架的【数据预处理】入口，负责把原始数据集处理成
训练所需的全部中间产物。处理流程分为三个相互独立的阶段（下文用 ①②③ 标注），
需要【按顺序、逐段取消注释】运行：

    ① 生成原始样本文件：把 jsonl/csv 等原始标注文件转换为
       train.pkl / valid.pkl / test.pkl（pandas DataFrame 的序列化格式）。

    ② 构建记忆库（Memory Bank）：用冻结的预训练 ViLT 模型对每个样本的
       文本和图像分别编码，并把得到的特征向量保存为 .npy 文件。
       这些特征后续用于"缺失模态的重建"（论文中的 instance retrieval）。

    ③ 构建检索列表（Retrieval List）：用预训练的 CLIP 模型计算样本间
       相似度，为每个样本找出与其最相似的 Top-K 个邻居样本（分 i2i 图-图、
       t2t 文-文两条检索通道），并连同邻居标签一起写回 .pkl 文件。

说明：三段代码当前都处于【注释状态】，因为它们是"一次性的数据准备步骤"，
运行一次生成中间文件后即可长期复用，无需在每次训练前重复执行。

各阶段产物的字段含义（最终 .pkl 的列）：
    - item_id           : 样本唯一编号
    - text              : 原始文本内容
    - label             : 样本标签
    - i2i_id_list       : 图像-图像检索得到的相似样本 id 列表（Top-K）
    - t2t_id_list       : 文本-文本检索得到的相似样本 id 列表（Top-K）
    - i2i_label_list    : i2i 检索到的邻居样本的标签列表
    - t2t_label_list    : t2t 检索到的邻居样本的标签列表
================================================================================
"""
import sys, os
from utils import (
    init_data_hatememes,
    init_data_mmimdb,
    init_data_food101,
    MemoryBankGenerator,
    MCR)
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import itertools

def main():
    # 关闭 pandas 未来版本中"静默类型降级"的告警提示，保持输出干净
    pd.set_option('future.no_silent_downcasting', True)

    # =========================================================================
    # ① 阶段一：生成 train/valid/test.pkl 文件
    # -------------------------------------------------------------------------
    # 以 HateMemes 数据集为例：把原始 jsonl 文件读取并转换成 DataFrame，
    # 最终得到形状为 (8500/500/1000, 4) 的 DataFrame，列名：
    # item_id, img, label, text。
    # 其中 train/valid/test 分别对应 8500/500/1000 条样本。
    # 运行方式：取消下面三行注释，运行一次，然后恢复注释。
    # =========================================================================
    # 按数据集选择其一（三个数据集的产物列均为 item_id/img/label/text）：
    # init_data_hatememes()
    # init_data_mmimdb()
    # init_data_food101()
    # dataset = 'hatememes'  # 'hatememes' | 'mmimdb' | 'food101'
    # df_train = pd.read_pickle(f'dataset/{dataset}/train.pkl')
    # df_valid = pd.read_pickle(f'dataset/{dataset}/valid.pkl')
    # df_test = pd.read_pickle(f'dataset/{dataset}/test.pkl')
    # print(df_train.shape, df_valid.shape, df_test.shape)
    # sys.exit(0)

    # =========================================================================
    # ② 阶段二：构建记忆库（Memory Bank）
    # -------------------------------------------------------------------------
    # MemoryBankGenerator 使用【冻结参数的预训练 ViLT 模型】对每个样本的
    # 文本（token 序列）和图像（patch 序列）分别做前向，提取最后一层
    # embedding 之后的表征（尚未经过 Transformer encoder），并保存为 .npy。
    #
    # 为什么需要记忆库？
    #   论文核心思路之一是：当某个样本缺失了某一模态（如缺文本）时，利用
    #   "检索到的相似样本的对应模态特征"来重建缺失信息。记忆库就是预先
    #   计算好的、可供检索和填充的特征仓库。
    #
    # 运行方式：取消注释，运行一次。下面的 image_count/text_count 用于
    # 校验生成的特征文件数量是否正确（应等于样本总数）。
    # =========================================================================
    # memory_bank_generator = MemoryBankGenerator()
    # memory_bank_generator.run()
    # image_count = sum(len(files) for _, _, files in os.walk("/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/image"))
    # text_count= sum(len(files) for _, _, files in os.walk("/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/text"))
    # print(image_count, text_count)
    # sys.exit(0)

    # =========================================================================
    # ③ 阶段三：构建检索列表（Multi-Channel Retriever, MCR）
    # -------------------------------------------------------------------------
    # MCR 使用预训练的 CLIP 模型，分别计算：
    #   - i2i：图像特征之间的相似度（image-to-image retrieval）
    #   - t2t：文本特征之间的相似度（text-to-text retrieval）
    # 为每个样本找出 Top-K 个最相似的邻居，并记录它们的 id 与标签。
    # 检索结果写回 train/valid/test.pkl，使 DataFrame 从 4 列扩展到 12 列
    # （新增 i2i_id_list / i2i_sims_list / i2i_label_list、
    #       t2t_id_list / t2t_sims_list / t2t_label_list 等）。
    #
    # 运行方式：取消注释，运行一次。
    # =========================================================================
    # mcr = MCR()
    # mcr.run()
    # df_train_ = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/train.pkl')
    # df_valid_ = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/valid.pkl')
    # df_test_ = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/test.pkl')
    # print(df_train_.shape, df_valid_.shape, df_test_.shape)
    # sys.exit(0)


if  __name__ == '__main__':
    main()