"""
================================================================================
MM-IMDb 数据集加载器
================================================================================
MM-IMDb 是多模态电影类型分类数据集：每部电影包含一张海报（图像）与一段
剧情简介（文本），需要预测该电影所属的 23 个类型之一（多标签→此处按
单标签处理）。

本 Dataset 的核心职责与 Food101Dataset 完全一致，仅数据路径、图像格式
（.jpeg）与类别数（23 类）不同：
    1. 读取 {split}.pkl；
    2. 依据 missing_mask 模拟缺失模态（缺失文本时用占位字符串填充，
       并借助检索到的相似样本的记忆库特征重建缺失模态）；
    3. 返回训练所需字段。

缺失掩码语义（与 train.py 一致）：
    - single（Text/Image 单模态缺失）：0 缺失，1 不缺失
    - both（双模态缺失）：0 缺失文本，1 缺失图像，2 都不缺失
================================================================================
"""
from PIL import  Image
import numpy as np
import torch.utils.data
import os
import pandas as pd
from PIL import Image


class MMIMDbDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_text_len,  missing_type, missing_rate, k, **kargs):
        super().__init__()
        # 读取预处理好的数据划分
        dataframe = pd.read_pickle(os.path.join('dataset/mmimdb', f'{split}.pkl'))

        # 依据缺失类型选择对应的缺失掩码表
        if missing_type == "Image" or missing_type == "Text":
            missing_table = pd.read_pickle('dataset/missing_table/single/mmimdb/missing_table.pkl')
        elif missing_type == "Both":
            missing_table = pd.read_pickle('dataset/missing_table/both/mmimdb/missing_table.pkl')

        # 按 item_id 合并缺失掩码
        dataframe = pd.merge(dataframe, missing_table, on='item_id')

        # ---- 各列转成列表 ----
        self.k = k
        self.missing_type = missing_type
        self.max_text_len = max_text_len
        self.id_list = dataframe['item_id'].tolist()
        self.text_list = dataframe['text'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.i2i_list = dataframe['i2i_id_list'].tolist()
        self.t2t_list = dataframe['t2t_id_list'].tolist()
        self.i2i_r_l_list_list = dataframe['i2i_label_list'].tolist()
        self.t2t_r_l_list_list = dataframe['t2t_label_list'].tolist()
        self.missing_mask_list = dataframe[f'missing_mask_{int(10 * missing_rate)}'].tolist()

    def __getitem__(self, index):
        k = self.k
        text = self.text_list[index]
        # 打开电影海报并统一为 RGB
        image = Image.open(fr'dataset/mmimdb/image/{self.id_list[index]}.jpeg').convert("RGB")
        r_t_list = []   # 检索到的文本特征列表
        r_i_list = []   # 检索到的图像特征列表

        # ---- 情况 1：缺文本（single Text，mask==0）----
        if self.missing_type == "Text" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024
            i2i_list = self.i2i_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/mmimdb/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'dataset/memory_bank/mmimdb/text/{i}.npy')
                r_t_list.append(r_t.tolist())

            r_l_list = self.i2i_r_l_list_list[index]

        # ---- 情况 2：缺图像（single Image，mask==0）----
        elif self.missing_type == "Image" and self.missing_mask_list[index] == 0:
            t2t_list = self.t2t_list[index]

            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/mmimdb/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'dataset/memory_bank/mmimdb/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]

        # ---- 情况 3：单模态缺失但不缺失（mask==1）----
        elif self.missing_type == "Text" or self.missing_type == "Image" and self.missing_mask_list[index] == 1:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/mmimdb/image/{i}.npy')
                r_i_list.append(r_i.tolist())

            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/mmimdb/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]

        # ---- 情况 4：Both 且缺失文本（mask==0）----
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024
            i2i_list = self.i2i_list[index]
            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/mmimdb/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'dataset/memory_bank/mmimdb/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]

        # ---- 情况 5：Both 且缺失图像（mask==1）----
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 1:
            t2t_list = self.t2t_list[index]
            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/mmimdb/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'dataset/memory_bank/mmimdb/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]

        # ---- 情况 6：Both 且两模态都完整（mask==2）----
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 2:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/mmimdb/image/{i}.npy')
                r_i_list.append(r_i.tolist())

            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/mmimdb/text/{i}.npy')
                r_t_list.append(r_t.tolist())

            r_l_list = self.i2i_r_l_list_list[index]

        return {
            "image": image,
            "text": text,
            "label": self.label_list[index],
            "r_t_list": r_t_list,
            "r_i_list": r_i_list,
            "missing_mask": self.missing_mask_list[index],
            "r_l_list": r_l_list
        }
    def __len__(self):
        return len(self.text_list)