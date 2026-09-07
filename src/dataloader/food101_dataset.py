"""
================================================================================
Food101 数据集加载器
================================================================================
Food101 是一个包含 101 类食物的图像分类数据集。在本框架中它被构造成
"多模态"任务：每张食物图像配一段文本描述（text），模型需要根据图文对
进行 101 分类。

本 Dataset 的核心职责：
    1. 读取预处理好的 {split}.pkl（含 text/label/检索列表等列）；
    2. 根据 missing_mask（缺失掩码）模拟"缺失模态"场景：
        - 若某样本被标记为缺失文本，则把文本替换为一段无意义占位字符串
          （"I love deep learning" 重复 1024 次，用于构造一个会被模型忽略
           的伪文本）；
        - 缺失的模态信息将借助【检索到的相似样本】的记忆库特征来重建；
    3. 返回训练所需的原始字段（图像、文本、标签、检索特征、缺失掩码等）。

缺失掩码语义（与 train.py 一致）：
    - single（Text/Image 单模态缺失）：0 缺失，1 不缺失
    - both（双模态缺失）：0 缺失文本，1 缺失图像，2 都不缺失
================================================================================
"""
from PIL import  Image
import numpy as np
import torch.utils.data
import pandas as pd
import os

from PIL import Image

class Food101Dataset(torch.utils.data.Dataset):
    def __init__(self, split, max_text_len,  missing_type, missing_rate, k, **kargs):
        super().__init__()
        # 读取预处理好的数据集划分（split ∈ {train, valid, test}）
        dataframe = pd.read_pickle(os.path.join('dataset/food101', f'{split}.pkl'))

        # 根据缺失类型读取对应的"缺失掩码表"
        #   - 单模态缺失（Image 或 Text）→ 读取 single/ 目录
        #   - 双模态缺失（Both）         → 读取 both/ 目录
        if missing_type == "Image" or missing_type == "Text":
            missing_table = pd.read_pickle('dataset/missing_table/single/food101/missing_table.pkl')
        elif missing_type == "Both":
            missing_table = pd.read_pickle('dataset/missing_table/both/food101/missing_table.pkl')

        # 按 item_id 合并：把每个样本的缺失掩码合并进主 DataFrame
        dataframe = pd.merge(dataframe, missing_table, on='item_id')

        # ---- 将 DataFrame 各列转成 Python 列表，方便按索引取用 ----
        self.k = k                                          # 检索邻居数（Top-K）
        self.missing_type = missing_type                    # 缺失类型
        self.max_text_len = max_text_len                    # 文本最大长度
        self.id_list = dataframe['item_id'].tolist()        # 样本 id 列表
        self.text_list = dataframe['text'].tolist()         # 文本内容列表
        self.label_list = dataframe['label'].tolist()       # 标签列表
        self.i2i_list = dataframe['i2i_id_list'].tolist()   # 图像→图像检索到的相似样本 id 列表
        self.t2t_list = dataframe['t2t_id_list'].tolist()   # 文本→文本检索到的相似样本 id 列表
        self.i2i_r_l_list_list = dataframe['i2i_label_list'].tolist()  # i2i 邻居的标签
        self.t2t_r_l_list_list = dataframe['t2t_label_list'].tolist()  # t2t 邻居的标签
        # 缺失掩码：列名形如 missing_mask_7（7 = int(10 * 0.7)，即 70% 缺失率）
        self.missing_mask_list = dataframe[f'missing_mask_{int(10 * missing_rate)}'].tolist()

    def __getitem__(self, index):
        k = self.k
        text = self.text_list[index]
        # 打开原始图像并统一转为 RGB 三通道
        image = Image.open(fr'dataset/food101/image/{self.id_list[index]}.jpg').convert("RGB")
        r_t_list = []   # 检索到的文本特征列表（Top-K 个，每个形状 [seq_len, hidden]）
        r_i_list = []   # 检索到的图像特征列表（Top-K 个，每个形状 [n_patch, hidden]）

        # ---------------------------------------------------------------------
        # 情况 1：单模态缺失 = Text，且该样本【缺失文本】(mask == 0)
        #   做法：文本用无意义占位符填充；同时用 i2i（图-图）检索到的相似样本，
        #   把它们的图像特征与文本特征都取出来，供 MMG 重建缺失的文本模态。
        # ---------------------------------------------------------------------
        if self.missing_type == "Text" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024     # 用占位文本"填充"缺失文本
            i2i_list = self.i2i_list[index]          # 该样本的 i2i 相似样本 id 列表

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/food101/image/{i}.npy')
                r_i_list.append(r_i.tolist())        # 检索到的图像特征
                r_t = np.load(fr'dataset/memory_bank/food101/text/{i}.npy')
                r_t_list.append(r_t.tolist())        # 检索到的文本特征（用于重建缺失文本）

            r_l_list = self.i2i_r_l_list_list[index]  # 检索邻居的标签

        # ---------------------------------------------------------------------
        # 情况 2：单模态缺失 = Image，且该样本【缺失图像】(mask == 0)
        #   做法：用 t2t（文-文）检索到的相似样本的特征来重建缺失的图像模态。
        # ---------------------------------------------------------------------
        elif self.missing_type == "Image" and self.missing_mask_list[index] == 0:
            t2t_list = self.t2t_list[index]

            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/food101/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'dataset/memory_bank/food101/image/{i}.npy')
                r_i_list.append(r_i.tolist())        # 检索到的图像特征（用于重建缺失图像）
            r_l_list = self.t2t_r_l_list_list[index]

        # ---------------------------------------------------------------------
        # 情况 3：单模态缺失（Text 或 Image），且该样本【不缺失】(mask == 1)
        #   做法：样本完整，只需额外取出 i2i 与 t2t 检索特征（供动态提示 CAP 使用）。
        # ---------------------------------------------------------------------
        elif self.missing_type == "Text" or self.missing_type == "Image" and self.missing_mask_list[index] == 1:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/food101/image/{i}.npy')
                r_i_list.append(r_i.tolist())

            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/food101/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]

        # ---------------------------------------------------------------------
        # 情况 4：双模态缺失 = Both，且该样本【缺失文本】(mask == 0)
        # ---------------------------------------------------------------------
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024
            i2i_list = self.i2i_list[index]
            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/food101/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'dataset/memory_bank/food101/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]

        # ---------------------------------------------------------------------
        # 情况 5：双模态缺失 = Both，且该样本【缺失图像】(mask == 1)
        # ---------------------------------------------------------------------
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 1:
            t2t_list = self.t2t_list[index]
            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/food101/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'dataset/memory_bank/food101/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]

        # ---------------------------------------------------------------------
        # 情况 6：双模态缺失 = Both，且该样本【两模态都完整】(mask == 2)
        # ---------------------------------------------------------------------
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 2:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/food101/image/{i}.npy')
                r_i_list.append(r_i.tolist())

            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/food101/text/{i}.npy')
                r_t_list.append(r_t.tolist())

            r_l_list = self.i2i_r_l_list_list[index]

        # ---- 返回一个样本的全部字段（随后由 Collator 统一编码成张量）----
        return {
            "image": image,                        # 原始图像（PIL）
            "text": text,                          # 文本（若缺失则为占位符）
            "label": self.label_list[index],       # 标签
            "r_t_list": r_t_list,                  # 检索到的文本特征（Top-K）
            "r_i_list": r_i_list,                  # 检索到的图像特征（Top-K）
            "missing_mask": self.missing_mask_list[index],  # 缺失掩码
            "r_l_list": r_l_list                   # 检索邻居的标签
        }
    def __len__(self):
        return len(self.text_list)