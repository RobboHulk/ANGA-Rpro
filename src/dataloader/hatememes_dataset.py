"""
================================================================================
HateMemes 数据集加载器
================================================================================
HateMemes 是 Facebook 发布的仇恨梗图（meme）二分类数据集：给定一张梗图
（图像 + 图中文字），判断其是否包含仇恨内容（label ∈ {0, 1}）。

本 Dataset 的核心职责与 Food101Dataset 完全一致，只是数据路径、图像格式
（.png）与类别数（2 类）不同：
    1. 读取 {split}.pkl；
    2. 依据 missing_mask 模拟缺失模态（缺失文本时用占位字符串填充，
       并借助检索到的相似样本的记忆库特征重建缺失模态）；
    3. 返回训练所需字段（含样本 id，用于课程学习阶段的可靠样本筛选）。

缺失掩码语义（与 train.py 一致）：
    - single（Text/Image 单模态缺失）：0 缺失，1 不缺失
    - both（双模态缺失）：0 缺失文本，1 缺失图像，2 都不缺失
================================================================================
"""
import sys

from PIL import  Image
import numpy as np
import torch.utils.data
import os
import pandas as pd
from PIL import Image

class HatememesDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_text_len,  missing_type, missing_rate, k, **kargs):
        super().__init__()
        # 1️⃣ 读取预处理好的数据划分（split ∈ {train, valid, test}）
        dataframe = pd.read_pickle(os.path.join('./dataset/hatememes', f'{split}.pkl'))

        # 依据缺失类型选择对应的缺失掩码表（single / both）
        if missing_type == "Image" or missing_type == "Text":
            missing_table = pd.read_pickle('./dataset/missing_table/single/hatememes/missing_table.pkl')
        elif missing_type == "Both":
            missing_table = pd.read_pickle('./dataset/missing_table/both/hatememes/missing_table.pkl')

        # 按 item_id 合并缺失掩码；合并后 DataFrame 形状约为 (8500/1000, 13)
        dataframe = pd.merge(dataframe, missing_table, on='item_id')

        # ---- 各列转成列表 ----
        self.k = k
        self.missing_type = missing_type
        self.max_text_len = max_text_len
        self.id_list = dataframe['item_id'].tolist()
        self.text_list = dataframe['text'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.i2i_list = dataframe['i2i_id_list'].tolist()      # 图像→图像检索邻居 id
        self.t2t_list = dataframe['t2t_id_list'].tolist()      # 文本→文本检索邻居 id
        self.i2i_r_l_list_list = dataframe['i2i_label_list'].tolist()
        self.t2t_r_l_list_list = dataframe['t2t_label_list'].tolist()
        self.missing_mask_list = dataframe[f'missing_mask_{int(10 * missing_rate)}'].tolist()

    def __getitem__(self, index):
        k = self.k
        text = self.text_list[index]
        # 打开梗图图像并统一为 RGB
        image = Image.open(fr'./dataset/hatememes/image/{self.id_list[index]}.png').convert("RGB")
        r_t_list = []   # 检索到的文本特征列表
        r_i_list = []   # 检索到的图像特征列表

        # 2️⃣ 根据缺失类型与缺失掩码，分情况处理（语义与 Food101Dataset 相同）
        # 情况 1：缺文本（single Text，mask==0）
        if self.missing_type == "Text" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024      # 占位文本填充
            i2i_list = self.i2i_list[index]

            # 取出前 top-k 个与该样本图像相似度最高的 item_id，并提取对应的图像特征和文本特征
            for i in i2i_list[:k]:
                r_i = np.load(fr'./dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'./dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())

            # 提取全部标签
            r_l_list = self.i2i_r_l_list_list[index]

        # 情况 2：缺图像（single Image，mask==0）
        elif self.missing_type == "Image" and self.missing_mask_list[index] == 0:
            t2t_list = self.t2t_list[index]

            for i in t2t_list[:k]:
                r_t = np.load(fr'./dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'./dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]

        # 情况 3：单模态缺失但不缺失（mask==1）
        elif self.missing_type == "Text" or self.missing_type == "Image" and self.missing_mask_list[index] == 1:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'./dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())

            for i in t2t_list[:k]:
                r_t = np.load(fr'./dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]

        # 情况 4：Both 且缺失文本（mask==0）
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024
            i2i_list = self.i2i_list[index]
            for i in i2i_list[:k]:
                r_i = np.load(fr'./dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'./dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]

        # 情况 5：Both 且缺失图像（mask==1）
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 1:
            t2t_list = self.t2t_list[index]
            for i in t2t_list[:k]:
                r_t = np.load(fr'./dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'./dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]

        # 情况 6：Both 且两模态都完整（mask==2）
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 2:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'./dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())

            for i in t2t_list[:k]:
                r_t = np.load(fr'./dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())

            r_l_list = self.i2i_r_l_list_list[index]

        return {
            "image": image,                                         # 图像特征（全）
            "text": text,                                           # 文本特征（全）若缺失则填充
            "label": self.label_list[index],                        # 标签（全）
            "r_t_list": r_t_list,                                   # 检索到的文本（Top-K）---若缺失，和下面成对，否则是全局的不一定成对
            "r_i_list": r_i_list,                                   # 检索到的图像（Top-K）---若缺失，和上面成对，否则是全局的不一定成对
            "missing_mask": self.missing_mask_list[index],          # 0：缺失；1：不缺失
            "r_l_list": r_l_list[:k],                               # 检索到 Top-K 个实例的对应标签
            "id": self.id_list[index]                               # 样本全局 id（供课程学习阶段筛选用）
        }
    def __len__(self):
        return len(self.text_list)