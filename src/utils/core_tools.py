"""
================================================================================
核心工具函数集（Core Tools）
================================================================================
本文件是 ANGA 框架的"工具箱"，提供数据准备、模型加载、损失计算、
评估指标、早停等训练所需的各类辅助功能。主要包含：

    - init_data_hatememes       : 把 HateMemes 原始 jsonl 转成 .pkl
    - MemoryBankGenerator       : 用冻结 ViLT 预计算每个样本的图文特征（记忆库）
    - MCR                       : 多通道检索器，用 CLIP 计算样本间相似度，生成检索列表
    - generate_missing_table    : 生成"缺失掩码表"（模拟缺失模态）
    - resize_image              : 图像尺寸调整
    - load_model / get_dataset / get_collator / get_evaluator : 各种工厂函数
    - Collator                  : 把一个 batch 的原始样本编码成张量
    - EarlyStopping             : 早停机制
    - seed_init                 : 固定随机种子
    - print_init_msg            : 打印训练配置
    - get_optim                 : 构建优化器与学习率调度器（warmup + 线性衰减）
    - compute_loss              : 交叉熵损失
    - HatememesMetric           : 评测指标（AUROC + ACC）

注：文件中大量路径硬编码为作者实验机器上的绝对路径
（./...），实际使用时需要按需修改。
================================================================================
"""
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, AUROC
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
import random
from datetime import datetime
import numpy as np
import torch
from model import ANGA as Model
from model import ViltModel, ViltImageProcessor
import importlib
from PIL import Image
import pandas as pd
from transformers import BertTokenizer
import json
from tqdm import tqdm
from transformers import AutoModel,AutoProcessor
# abbreviation: MCR: Multi-Channel Retriever（多通道检索器）
# from torchvision import transforms

def init_data_hatememes():
    """
    阶段①：把 HateMemes 的原始 jsonl 标注文件转换为 .pkl。

    处理过程：
        - 依次读取 train.jsonl / dev.jsonl / test_seen.jsonl（逐行 json）
        - 把列名 'id' 重命名为 'item_id'
        - 把 item_id 统一格式化为 5 位数字（如 42 → '00042'），与图像文件名一致
        - 将 test_seen 重命名为 test、dev 重命名为 valid（与训练代码约定一致）
        - 保存为 train.pkl / valid.pkl / test.pkl
    """
    root = 'dataset/hatememes'
    for split in tqdm(['train', 'dev', 'test_seen']):
        data = pd.read_json(f'{root}/meta_data/{split}.jsonl', lines=True)
        data.rename(columns={'id': 'item_id'}, inplace=True)
        # 把 id 格式化为 5 位字符串（例如 42 → '00042'），方便拼出图像路径
        data['item_id'] = data['item_id'].apply(lambda x: f"{int(x):05d}")
        if split == 'test_seen': split = 'test'   # test_seen 作为最终的 test 集
        if split == 'dev': split = 'valid'        # dev 作为验证集 valid
        data.to_pickle(f'{root}/{split}.pkl')
        print(f'hatememes {split}: {data.shape}')


# MM-IMDb 的 23 类标准类别（Arévalo et al. / missing-aware-prompt 通用设定，
# 剔除极少样本的 Adult / News / Reality-TV / Talk-Show）。
MMIMDB_GENRE_CLASSES = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure',
    'Horror', 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography',
    'War', 'History', 'Music', 'Animation', 'Musical', 'Western', 'Sport', 'Short',
    'Film-Noir',
]


def init_data_mmimdb():
    """
    阶段①（MM-IMDb）：把 meta_data/*.json + split.json 转换为 train/valid/test.pkl。

    产物 DataFrame 列：item_id / img / label / text
        - item_id : 与图像/元数据文件名一致的补零字符串（如 '0000005'）
        - img     : '{item_id}.jpeg'
        - label   : 单标签整数（取样本 genres 列表中第一个落在 23 类内的类型）
        - text    : plot 列表中最长的一段剧情简介

    注意：MM-IMDb 原本是多标签任务，而本仓库的训练代码用 cross_entropy 按
    单标签处理（cls_num=23），因此这里取“主类型”。若要复现论文的多标签
    F1 指标，需要改成 multi-hot 标签 + BCE 损失，并替换评测器。

    同时会生成 dataset/mmimdb/class_idx.json 记录类别到索引的映射。
    """
    root = 'dataset/mmimdb'
    class_idx = {g: i for i, g in enumerate(MMIMDB_GENRE_CLASSES)}
    with open(os.path.join(root, 'class_idx.json'), 'w') as f:
        json.dump(class_idx, f, indent=2, ensure_ascii=False)

    split_map = json.load(open(os.path.join(root, 'split.json')))
    for split_name, out_name in [('train', 'train'), ('dev', 'valid'), ('test', 'test')]:
        rows = []
        for item_id in tqdm(split_map[split_name], desc=f'mmimdb {out_name}'):
            item_id = str(item_id)
            meta_path = os.path.join(root, 'meta_data', f'{item_id}.json')
            img_path = os.path.join(root, 'image', f'{item_id}.jpeg')
            if not (os.path.exists(meta_path) and os.path.exists(img_path)):
                continue
            d = json.load(open(meta_path))
            genres = [g for g in d.get('genres', []) if g in class_idx]
            if not genres:
                continue
            plot = d.get('plot') or ['']
            if isinstance(plot, list):
                text = max(plot, key=len) if plot else ''
            else:
                text = str(plot)
            text = ' '.join(str(text).split())
            rows.append({
                'item_id': item_id,
                'img': f'{item_id}.jpeg',
                'label': int(class_idx[genres[0]]),
                'text': text,
            })
        df = pd.DataFrame(rows)
        df.to_pickle(os.path.join(root, f'{out_name}.pkl'))
        print(f'mmimdb {out_name}: {df.shape}')


def init_data_food101(valid_ratio=0.05, seed=2024):
    """
    阶段①（Food101 / UPMC Food-101）：把 *_titles.csv + class_idx.json 转换为
    train/valid/test.pkl。

    产物 DataFrame 列：item_id / img / label / text
        - item_id : 图像文件名去掉 .jpg（如 'apple_pie_851'）
        - img     : '{item_id}.jpg'
        - label   : class_idx.json 给出的 0-100 整数类别
        - text    : csv 中的标题文本

    UPMC Food-101 只提供 train / test 两个划分，这里按类别分层从 train 中
    切出 valid（比例 valid_ratio，随机种子 seed，保证可复现）。
    只保留图像确实存在、且类别名在 class_idx.json 中的样本。
    """
    root = 'dataset/food101'
    class_idx = json.load(open(os.path.join(root, 'class_idx.json')))

    def load_csv(name):
        df = pd.read_csv(
            os.path.join(root, 'meta_data', name),
            header=None, names=['img', 'text', 'class_name'],
        )
        df['item_id'] = df['img'].str.replace(r'\.jpg$', '', regex=True)
        df['label'] = df['class_name'].map(class_idx)
        df = df[df['label'].notna()].copy()
        df['label'] = df['label'].astype(int)
        exists = df['item_id'].apply(
            lambda x: os.path.exists(os.path.join(root, 'image', f'{x}.jpg')))
        df = df[exists].copy()
        df['text'] = df['text'].fillna('').astype(str)
        return df[['item_id', 'img', 'label', 'text']]

    train_df = load_csv('train_titles.csv')
    test_df = load_csv('test_titles.csv')

    # ---- 按类别分层从 train 切出 valid ----
    rng = np.random.RandomState(seed)
    valid_parts = []
    for _, grp in train_df.groupby('label'):
        n_val = max(1, int(round(len(grp) * valid_ratio)))
        valid_parts.append(grp.sample(n=min(n_val, len(grp)), random_state=rng))
    valid_df = pd.concat(valid_parts)
    train_df = train_df.drop(index=valid_df.index).reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    for name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        df.to_pickle(os.path.join(root, f'{name}.pkl'))
        print(f'food101 {name}: {df.shape}')

class MemoryBankGenerator(torch.nn.Module):
    """
    阶段②：记忆库生成器。

    用【冻结参数的预训练 ViLT 模型】对每个样本分别编码文本与图像，
    提取"embedding 层输出"的文本表征 (seq_len, 768) 与图像表征 (n_patch, 768)，
    并保存为 .npy 文件，构成"记忆库（memory bank）"。

    记忆库的作用：
        当某样本缺失某一模态时，训练阶段会从记忆库中检索相似样本的对应
        模态特征，用于重建缺失信息（参见 MMG 模块与 dataloader 的读取逻辑）。
    """

    def __init__(self):
        super(MemoryBankGenerator, self).__init__()
        # 加载预训练的 ViLT 模型（vilt-b32-mlm），只取 embedding 层
        pretrained_vilt = ViltModel.from_pretrained('./src/model/vilt-b32-mlm')
        self.embedding_layer = pretrained_vilt.embeddings
        self._freeze()   # 冻结 embedding 层参数
        # BERT 分词器：把文本转成 token ids
        self.tokenizer = BertTokenizer.from_pretrained('./src/model/vilt-b32-mlm', do_lower_case=True)
        # ViLT 图像处理器：把图像转成 pixel_values / pixel_mask
        self.image_processor = ViltImageProcessor.from_pretrained('./src/model/vilt-b32-mlm')
        self.dataset = 'hatememes'
        self.max_text_len = 128   # 文本 token 数上限
        self.max_image_len = 145  # 图像 patch 数上限（1 CLS + 144 patch）
        # 读取三个划分的 DataFrame
        self.df_train = pd.read_pickle(rf'./dataset/{self.dataset}/train.pkl')
        self.df_test = pd.read_pickle(rf'./dataset/{self.dataset}/test.pkl')
        self.df_valid = pd.read_pickle(rf'./dataset/{self.dataset}/valid.pkl')
        self.batch_size = 64
        # 创建保存记忆库特征（text/image）的目录
        if not os.path.exists(f'./dataset/memory_bank/{self.dataset}/text'):
            os.makedirs(f'./dataset/memory_bank/{self.dataset}/text')
        if not os.path.exists(f'./dataset/memory_bank/{self.dataset}/image'):
            os.makedirs(f'./dataset/memory_bank/{self.dataset}/image')

    def _freeze(self):
        """冻结 ViLT embedding 层，不参与梯度更新。"""
        for param in self.embedding_layer.parameters():
            param.requires_grad = False

    def _encode(self, input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask, image_token_type_idx=1):
        """调用 ViLT embedding 层，得到拼接后的图文表征。"""
        embedding, attention_mask = self.embedding_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            image_embeds=None,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            image_token_type_idx=image_token_type_idx
        )
        return embedding

    def _resize_image(self, img, size=(384, 384)):
        """把图像缩放到指定尺寸（双线性插值）。"""
        return img.resize(size, Image.BILINEAR)

    def _process_batch(self, df, start_idx, end_idx):
        """处理一个 batch：编码文本与图像，并把特征保存为 .npy。"""
        texts = df['text'][start_idx:end_idx]
        ids = df['item_id'][start_idx:end_idx]

        # ---- 文本编码：padding 到 max_length 并截断 ----
        text_encodings = self.tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        input_ids = text_encodings['input_ids']
        attention_mask = text_encodings['attention_mask']
        token_type_ids = text_encodings['token_type_ids']

        # ---- 图像编码：逐张打开、缩放 ----
        images = []
        for id in ids:
            image_path = fr'./dataset/{self.dataset}/image/{id}.png'
            image = Image.open(image_path).convert("RGB")
            image = self._resize_image(image)
            images.append(image)

        encoding_image_processor = self.image_processor(images, return_tensors="pt")
        pixel_values = encoding_image_processor["pixel_values"]
        pixel_mask = encoding_image_processor["pixel_mask"]

        # ---- 前向得到 embedding，再切分出文本/图像两部分 ----
        emb = self._encode(input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask)
        text_emb = emb[:, :self.max_text_len]      # (B, 128, 768)
        image_emb = emb[:, self.max_text_len:]     # (B, 145, 768)

        # ---- 逐样本保存特征为 .npy（文件名 = item_id） ----
        for i, id in enumerate(ids):
            np.save(f'./dataset/memory_bank/{self.dataset}/text/{id}.npy', text_emb[i].detach().numpy())
            np.save(f'./dataset/memory_bank/{self.dataset}/image/{id}.npy', image_emb[i].detach().numpy())

    def run(self):
        """对 train（以及非 food101 时的 valid）集合逐 batch 生成记忆库。"""
        for i in tqdm(range(0, len(self.df_train), self.batch_size)):
            start_idx = i
            end_idx = min(i + self.batch_size, len(self.df_train))
            self._process_batch(self.df_train, start_idx, end_idx)

        # food101 数据集只对 train 生成记忆库；其余数据集额外处理 valid
        if self.dataset != "food101":
            for i in tqdm(range(0, len(self.df_valid), self.batch_size)):
                start_idx = i
                end_idx = min(i + self.batch_size, len(self.df_valid))
                self._process_batch(self.df_valid, start_idx, end_idx)

class MCR():
    """
    阶段③：多通道检索器（Multi-Channel Retriever）。

    使用预训练的 CLIP 模型（clip-vit-large-patch14-336）分别编码图像与文本，
    然后计算样本间的相似度，为每个样本找出 Top-K 个最相似的邻居，并记录
    邻居的 id 与标签，写回 train/valid/test.pkl。

    检索通道有两类：
        - i2i（image-to-image）：图像特征之间的相似度检索
        - t2t（text-to-text）  ：文本特征之间的相似度检索

    检索范围：train + valid 共同构成记忆库（memory bank），test 样本只做
    查询（query），避免信息泄露。
    """

    def __init__(self):
        self.dataset = 'hatememes'
        self.batch_size = 64
        self.top_k = 20                    # 检索返回的邻居数量上限
        self.img_path = os.path.join('./dataset', self.dataset, 'image')
        self.img_name_list = os.listdir(self.img_path)   # 图像文件名列表
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 加载 CLIP 模型与处理器（用于图像/文本编码）
        self.pretrained_model = AutoModel.from_pretrained('./src/model/clip-vit-large-patch14-336')
        self.processor = AutoProcessor.from_pretrained('./src/model/clip-vit-large-patch14-336')
        self.pretrained_model = self.pretrained_model.to(self.device)

        # 读取三个划分
        self.df_train = pd.read_pickle(os.path.join('./dataset', self.dataset, 'train.pkl'))
        self.df_test = pd.read_pickle(os.path.join('./dataset', self.dataset, 'test.pkl'))
        self.df_valid = pd.read_pickle(os.path.join('./dataset', self.dataset, 'valid.pkl'))

    def _compute_similarity_in_batches(self, query_vectors, memory_bank, memory_bank_id, memory_bank_label):
        """
        批量计算查询向量与记忆库向量的余弦相似度，并为每个查询返回
        Top-K 个最相似邻居的 id、相似度与标签。

        参数：
            query_vectors   : 查询特征 (N, D)
            memory_bank     : 记忆库特征 (M, D)
            memory_bank_id  : 记忆库样本 id 列表 (M,)
            memory_bank_label: 记忆库样本标签列表 (M,)
        返回：
            r_id_list   : 每个查询的邻居 id 列表
            sims_list   : 每个查询的邻居相似度列表
            r_label_list: 每个查询的邻居标签列表
        """
        r_id_list = []
        sims_list = []
        r_label_list = []
        for i in tqdm(range(0, len(query_vectors), self.batch_size)):
            batch = query_vectors[i:i+self.batch_size].unsqueeze(1)   # (B, 1, D)
            # 计算查询与所有记忆库向量之间的余弦相似度：(B, M)
            similarity = F.cosine_similarity(batch, memory_bank.unsqueeze(0), dim=-1)
            # 取相似度最高的 top_k 个邻居及索引
            sim_scores, top_k_id = torch.topk(similarity, k=self.top_k, dim=-1)
            for j in range(batch.size(0)):
                id_index = i + j
                id = memory_bank_id[id_index] if id_index < len(memory_bank_id) else None
                # 过滤掉"自己检索到自己"的情况（邻居 id != 自身 id）
                retrieved_ids = [memory_bank_id[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
                retrieved_labels = [memory_bank_label[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
                sim_score = sim_scores[j,1:]   # 去掉第一个（自身）后的相似度
                if len(retrieved_ids) > self.top_k:
                    retrieved_ids = retrieved_ids[:self.top_k]
                    sim_score = sim_score[:self.top_k]
                    retrieved_labels = retrieved_labels[:self.top_k]
                r_id_list.append(retrieved_ids)
                sims_list.append(sim_score.tolist())
                r_label_list.append(retrieved_labels)
        return  r_id_list,sims_list, r_label_list

    def _encode_text(self, text):
        """用 CLIP 文本塔编码文本，返回 [CLS] 位置（末 token）的投影特征。"""
        inputs = self.processor(text=text, return_tensors="pt", padding=True,truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        # CLIP 文本编码流程：embeddings → encoder → final_layer_norm → 投影
        text_features = self.pretrained_model.text_model.embeddings(input_ids, attention_mask)
        text_features = self.pretrained_model.text_model.encoder(text_features).last_hidden_state
        text_features = self.pretrained_model.text_model.final_layer_norm(text_features)
        text_features = (self.pretrained_model.text_projection(text_features))
        # 取 [EOS] 位置（序列最后一个 token）作为整句表征
        return text_features[0, -1, :]

    def _encode_image(self, images):
        """用 CLIP 视觉塔编码图像，返回 [CLS] 位置的投影特征。"""
        with torch.no_grad():
            processed_images = self.processor(images=images, return_tensors="pt").to(self.device)
            # CLIP 视觉编码流程：embeddings → pre_layernorm → encoder → post_layernorm → 投影
            image_features = self.pretrained_model.vision_model.embeddings(processed_images['pixel_values'])
            image_features = self.pretrained_model.vision_model.pre_layrnorm(image_features)
            image_features = self.pretrained_model.vision_model.encoder(image_features).last_hidden_state
            image_features = self.pretrained_model.vision_model.post_layernorm(image_features)
            image_features = self.pretrained_model.visual_projection(image_features)
            # 取 [CLS] 位置（第一个 token）作为整图表征
            return image_features[:, 0, :]

    def _retrieval_vector_generation(self):
        """为所有样本生成图像/文本的检索向量（query），并合并进 DataFrame。"""
        img_name_list = os.listdir(self.img_path)
        print("Loading all images...")
        # 一次性加载所有图像（.copy() 用于规避 PIL 惰性加载的引用问题）
        all_images = [
            Image.open(os.path.join(self.img_path, n)).convert("RGB").copy()  # ← 关键改动
            for n in tqdm(img_name_list, desc="Loading Images")
        ]
        print("==> All images loaded successfully!")

        # 逐 batch 编码图像，得到每个样本的图像查询向量 q_i
        images = []
        batch_size = self.batch_size
        for i in tqdm(range(0, len(all_images), batch_size), desc="Encoding Image Batches"):
            batch_images = all_images[i:i+batch_size]
            batch_outputs = self._encode_image(batch_images)
            images.extend(batch_outputs.cpu().tolist())
        print("==> Image encoding done!")

        # 由图像文件名推导 item_id（去掉扩展名；mmimdb 的扩展名为 5 字符如 .jpeg）
        if self.dataset != "mmimdb":
            id_list = [str(x[:-4]) for x in img_name_list]
        else:
            id_list = [str(x[:-5]) for x in img_name_list]
        # 把图像查询向量合并进 train/test/valid 三个 DataFrame
        df_img_query = pd.DataFrame({'item_id': id_list, 'q_i': images})
        self.df_train = pd.merge(self.df_train, df_img_query, on='item_id', how='inner')
        self.df_test = pd.merge(self.df_test, df_img_query, on='item_id', how='inner')
        self.df_valid = pd.merge(self.df_valid, df_img_query, on='item_id', how='inner')

        # 逐样本编码文本，得到文本查询向量 q_t
        q_t_list = [self._encode_text(text).tolist() for text in tqdm(self.df_train['text'], desc=f"Encoding Text")]
        self.df_train['q_t'] = q_t_list
        q_t_list = [self._encode_text(text).tolist() for text in tqdm(self.df_test['text'], desc=f"Encoding Text")]
        self.df_test['q_t'] = q_t_list
        q_t_list = [self._encode_text(text).tolist() for text in tqdm(self.df_valid['text'], desc=f"Encoding Text")]
        self.df_valid['q_t'] = q_t_list
        print("==> Text encoding done!")

    def _within_retrieval(self):
        """执行检索：对每个查询向量，在记忆库中找 Top-K 相似邻居，并写回 DataFrame。"""
        # train 的查询向量与 id/label
        train_q_i = self.df_train[f'q_i'].tolist()
        train_q_t = self.df_train[f'q_t'].tolist()
        train_item_id = self.df_train['item_id'].tolist()
        train_label = self.df_train['label'].tolist()

        # valid 的查询向量与 id/label
        valid_q_i = self.df_valid[f'q_i'].tolist()
        valid_q_t = self.df_valid[f'q_t'].tolist()
        valid_item_id = self.df_valid['item_id'].tolist()
        valid_label = self.df_valid['label'].tolist()

        # test 只做查询
        test_q_i = self.df_test[f'q_i'].tolist()
        test_q_t = self.df_test[f'q_t'].tolist()

        # ---- 记忆库 = train + valid（test 不参与记忆库，避免泄露）----
        r_v_i = train_q_i + valid_q_i
        r_v_t = train_q_t + valid_q_t
        memory_bank_id = train_item_id + valid_item_id
        memory_bank_label = train_label + valid_label

        # 转为张量并送到设备（squeeze(1) 去除多余的维度）
        r_v_i = torch.tensor(r_v_i).squeeze(1).to(self.device)
        r_v_t = torch.tensor(r_v_t).squeeze(1).to(self.device)
        test_q_i = torch.tensor(test_q_i).squeeze(1).to(self.device)
        test_q_t = torch.tensor(test_q_t).squeeze(1).to(self.device)
        train_q_i = torch.tensor(train_q_i).squeeze(1).to(self.device)
        train_q_t = torch.tensor(train_q_t).squeeze(1).to(self.device)

        # ---- 对 train/test/valid 分别做 t2t 与 i2i 检索 ----
        self.df_train[f't2t_id_list'], self.df_train[f't2t_sims_list'], self.df_train['t2t_label_list'] = self._compute_similarity_in_batches(train_q_t,r_v_t, memory_bank_id, memory_bank_label)
        self.df_train[f'i2i_id_list'], self.df_train[f'i2i_sims_list'], self.df_train['i2i_label_list'] = self._compute_similarity_in_batches(train_q_i,r_v_i, memory_bank_id, memory_bank_label)
        self.df_test[f't2t_id_list'], self.df_test[f't2t_sims_list'], self.df_test['t2t_label_list'] = self._compute_similarity_in_batches(test_q_t,r_v_t,memory_bank_id, memory_bank_label)
        self.df_test[f'i2i_id_list'], self.df_test[f'i2i_sims_list'], self.df_test['i2i_label_list'] = self._compute_similarity_in_batches(test_q_i,r_v_i,memory_bank_id, memory_bank_label)
        valid_q_i = torch.tensor(valid_q_i).squeeze(1).to(self.device)
        valid_q_t = torch.tensor(valid_q_t).squeeze(1).to(self.device)
        self.df_valid[f't2t_id_list'], self.df_valid[f't2t_sims_list'], self.df_valid['t2t_label_list'] = self._compute_similarity_in_batches(valid_q_t, r_v_t, memory_bank_id, memory_bank_label)
        self.df_valid[f'i2i_id_list'], self.df_valid[f'i2i_sims_list'], self.df_valid['i2i_label_list'] = self._compute_similarity_in_batches(valid_q_i, r_v_i, memory_bank_id, memory_bank_label)

        # ---- 保存检索结果 ----
        self.df_train.to_pickle(os.path.join(os.path.join('./dataset', self.dataset, 'train.pkl')))
        self.df_valid.to_pickle(os.path.join(os.path.join('./dataset', self.dataset, 'valid.pkl')))
        self.df_test.to_pickle(os.path.join(os.path.join('./dataset', self.dataset, 'test.pkl')))

        print(f"==> Saved retrieval results for {self.dataset}!")

    def run(self):
        """依次执行：生成检索向量 → 执行检索。"""
        self._retrieval_vector_generation()
        self._within_retrieval()

def generate_missing_table(missing_rate, missing_type, dataset, base_file_path='./dataset/missing_table', **kargs):
    """
    生成"缺失掩码表"（missing_table.pkl）。

    该表为每个样本分配一个缺失掩码值，用于在训练时模拟缺失模态的场景：
        - single（Text/Image 单模态缺失）：0 缺失，1 不缺失
        - both（双模态缺失）：0 缺失文本，1 缺失图像，2 都不缺失

    参数：
        missing_rate : 缺失比例（如 0.7 表示约 70% 样本缺失）
        missing_type : 'Text' / 'Image'（都归为 single）或 'Both'
        dataset      : 数据集名（hatememes / mmimdb / food101）

    产物：保存为 .../missing_table/{single|both}/{dataset}/missing_table.pkl，
          其中列名形如 missing_mask_7（7 = int(0.7 * 10)），表示 70% 缺失率下
          每个样本的缺失状态。
    """
    assert missing_type in ['Text', 'Image', 'Both'], "Invalid missing type"
    assert 0 <= missing_rate <= 1, "Invalid missing rate"
    # Text/Image 都归入 single 目录；Both 归入 both 目录
    if missing_type == 'Text' or missing_type == 'Image':
        missing_type = 'single'
    else:
        missing_type = 'both'

    file_path = f"{base_file_path}/{missing_type}/{dataset}/missing_table.pkl"
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 若表已存在，则读取并在必要时删除旧列后重新生成该缺失率对应的列
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
        if f"missing_mask_{int(missing_rate* 10)}" in df.columns:
            df.drop(f"missing_mask_{int(missing_rate* 10)}", axis=1, inplace=True)
        print("File already exists, regenerating new missing column...")
    else:
        # 表不存在：从 train/valid/test 三个划分取出 item_id 作为骨架
        df = pd.concat([pd.read_pickle(f'./dataset/{dataset}/{split}.pkl') for split in ['train', 'valid', 'test']])
        df = df[['item_id']]
        print("File does not exist, generating new missing table and column...")

    if missing_type == 'single':
        # ---- 单模态缺失：随机挑 num_missing 个样本标记为 0（缺失）----
        num_missing = int(len(df) * missing_rate)
        missing_mask = np.ones(len(df), dtype=int)  # Initialize as all 1s（初始全为 1 = 不缺失）
        missing_indices = np.random.choice(len(df), num_missing, replace=False)  # 随机选缺失样本
        missing_mask[missing_indices] = 0

    elif missing_type == 'both':
        # ---- 双模态缺失：一半样本缺文本（0），一半样本缺图像（1），其余完整（2）----
        num_missing = int(len(df) * missing_rate)
        num_text_missing = num_missing // 2      # 缺文本的样本数
        num_visual_missing = num_missing // 2    # 缺图像的样本数

        missing_mask = np.full(len(df), 2, dtype=int)  # 初始全为 2（完整）

        # 随机选 num_text_missing 个样本标记为 0（缺文本）
        text_missing_indices = np.random.choice(len(df), num_text_missing, replace=False)
        missing_mask[text_missing_indices] = 0

        # 在剩余样本中随机选 num_visual_missing 个标记为 1（缺图像）
        remaining_indices = list(set(range(len(df))) - set(text_missing_indices))
        visual_missing_indices = np.random.choice(remaining_indices, num_visual_missing, replace=False)
        missing_mask[visual_missing_indices] = 1

    # 把该缺失率对应的掩码列写入 DataFrame（列名如 missing_mask_7）
    df[f"missing_mask_{int(missing_rate* 10)}"] = missing_mask

    df.to_pickle(file_path)
    print(f"Missing table has been saved to {file_path}")

def resize_image(img, size=(384, 384)):
    """把 PIL 图像缩放到指定尺寸（双线性插值），ViLT 标准输入为 384x384。"""
    return img.resize(size, Image.BILINEAR)

def load_model(**kargs):
    """工厂函数：加载预训练 ViLT 作为 backbone，并实例化 ANGA 模型。"""
    pretrained_vlit = ViltModel.from_pretrained('./src/model/vilt-b32-mlm')
    model = Model(vilt=pretrained_vlit, **kargs)
    return model

def get_dataset(dataset_name: str, **kargs):
    """
    工厂函数：根据数据集名称动态导入对应的 Dataset 类并实例化。

    等价于：
        import dataloader.hatememes_dataset
        module = dataloader.hatememes_dataset
    通过 importlib 动态导入，避免写死 if-else 的 import 语句。
    """
    module = importlib.import_module(f"dataloader.{dataset_name}_dataset")
    if dataset_name == "hatememes":
        dataset_class = getattr(module, 'HatememesDataset')
    elif dataset_name == "mmimdb":
        dataset_class = getattr(module, 'MMIMDbDataset')
    elif dataset_name == "food101":
        dataset_class = getattr(module, 'Food101Dataset')
    dataset = dataset_class(**kargs)
    return dataset

def get_collator(max_text_len, **kargs):
    """工厂函数：实例化 Collator（负责把一个 batch 编码成张量）。"""
    collator = Collator(max_text_len, **kargs)
    return collator

def get_evaluator(task_id, device):
    """工厂函数：实例化评测器（当前统一使用 HatememesMetric）。"""
    evaluator = HatememesMetric(device)

    return evaluator

class Collator:
    """
    批处理器（Collator）：把一个 batch 的原始样本（来自 Dataset）统一编码、
    对齐为可直接输入模型的张量字典。

    主要工作：
        - 文本：用 BERT 分词器编码成 input_ids / attention_mask / token_type_ids
          （padding 到 max_length，统一长度）
        - 图像：缩放到 384x384 后，用 ViLT 图像处理器得到 pixel_values / pixel_mask
        - 其余字段（label、检索特征、缺失掩码）转为 torch 张量
    """

    def __init__(self, max_text_len, **kargs):
        # 加载 ViLT 图像处理器（负责缩放/归一化/pad 图像）
        self.image_processor = ViltImageProcessor.from_pretrained('./src/model/vilt-b32-mlm')
        # 加载 BERT 分词器（do_lower_case=True 表示转小写）
        self.tokenizer = BertTokenizer.from_pretrained('./src/model/vilt-b32-mlm', do_lower_case=True)
        self.max_text_len = max_text_len

    def __call__(self, batch):

        # ---- 从 dataloader 穿过来的参数：逐字段解包成列表 ----
        text = [item['text'] for item in batch]
        image = [item['image'] for item in batch]
        label = [item['label'] for item in batch]
        r_t_list = [item['r_t_list'] for item in batch]
        r_i_list = [item['r_i_list'] for item in batch]
        missing_mask = [item['missing_mask'] for item in batch]
        r_l_list = [item['r_l_list'] for item in batch]
        id = [item['id'] for item in batch]

        # ---- 文本编码 ----
        text_encoding = self.tokenizer(
            text,
            padding="max_length",        # 全部 padding 到 max_length
            truncation=True,             # 超长截断
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )

        # 文本对应的 token 编号序列，如 I love AI---> [101,132,42,5,0,0,...]
        input_ids = text_encoding['input_ids']
        # 注意力遮罩（1表示有效，0表示padding）,告诉模型哪些 token 是实际文本，哪些是 padding
        attention_mask = text_encoding['attention_mask']
        # 句子类型 ID（也叫 segment ids）
        token_type_ids = text_encoding['token_type_ids']

        # ---- 图像编码 ----
        image = [resize_image(img) for img in image]   # 先缩放到 384x384
        image_encoding = self.image_processor(image, return_tensors="pt")
        # 图像的标准输入张量，形状为 [B, 3, H, W]，可直接输入到模型中
        pixel_values = image_encoding["pixel_values"]
        # 图像掩码，用于处理 padding（ViLT支持变长 patch 时才会用，但通常全为1）
        pixel_mask = image_encoding["pixel_mask"]

        # ---- 各字段转张量 ----
        input_ids = torch.tensor(input_ids,dtype=torch.int64)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask,dtype=torch.int64)

        label = torch.tensor(label,dtype=torch.float)          # 标签（float 以便兼容某些损失）
        r_l_list = torch.tensor(r_l_list,dtype=torch.long)     # 检索邻居标签（用于 label_enhanced 查表）
        r_t_list = torch.tensor(r_t_list,dtype=torch.float)    # 检索文本特征 (B, K, 128, 768)
        r_i_list = torch.tensor(r_i_list,dtype=torch.float)    # 检索图像特征 (B, K, 145, 768)

        return {
            "input_ids": torch.tensor(input_ids,dtype=torch.int64), # 文
            "pixel_values": pixel_values, # 图
            "pixel_mask": pixel_mask, # 图
            "token_type_ids": token_type_ids, # 文
            "attention_mask": attention_mask, # 文
            "label": label, # 标签
            "r_t_list": r_t_list,
            "r_i_list": r_i_list,
            "missing_mask": torch.tensor(missing_mask,dtype=torch.int64),
            "r_l_list": r_l_list,
            "id": id
        }

class EarlyStopping:
    """早停机制：连续 patience 个 epoch 验证指标无提升则提前终止训练。"""

    def __init__(self, patience=5, delta=0.0, path="checkpoints/best_model.pt", trace_func=print):
        """
        Args:
            patience (int): 连续多少个 epoch 验证指标不提升就停止
            delta (float): 认为"提升"的最小幅度
            path (str): 保存最佳模型的路径
            trace_func (func): 打印函数（默认print）
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.counter = 0          # 连续无提升的计数
        self.best_score = None    # 历史最佳分数
        self.early_stop = False   # 是否触发早停

    def __call__(self, val_score, model):
        """
        在每个 epoch 验证后调用，用验证指标更新状态。

        返回：
            early_stop (bool)：是否应该停止训练
        """
        if self.best_score is None:
            # 第一次，直接保存
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.delta:
            # 没提升
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 有提升
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0

        return self.early_stop  # ★ 加这一行

    def save_checkpoint(self, model):
        """保存模型权重"""
        torch.save(model.state_dict(), self.path)


def seed_init(seed):
    """固定所有随机源（random/numpy/torch/cuda），保证实验可复现。"""
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)   # 固定 Python 哈希种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False     # 关闭 cudnn 自动寻优，保证确定性
    torch.backends.cudnn.deterministic = True  # 使用确定性算法

def print_init_msg(logger, args):
    """打印全部训练配置到日志，方便复现与核对。"""
    logger.info('Random Seed: ' + f"{args.seed} ")
    logger.info('Device: ' + f"{args.device} ")
    logger.info('Model: ' + f"{args.model} ")
    logger.info('Backbone: ' + f"{args.backbone}")
    logger.info("Dataset: " + f"{args.dataset}")
    logger.info("Optimizer: " + f"{args.name}(lr = {args.lr})")
    logger.info("Weight Decay: " + f"{args.weight_decay}")
    logger.info("Use Warmup: " + f"{args.use_warmup}")
    logger.info("Warmup Rate: " + f"{int(args.warmup_rate * 100)}%")
    logger.info("Total Epoch: " + f"{args.epochs} Turns")
    logger.info("Early Stop: " + f"{args.patience} Turns")
    logger.info("Batch Size: " + f"{args.batch_size}")
    logger.info("Number of Workers: " + f"{args.num_workers}")
    logger.info("Missing Rate: " + f"{args.missing_rate}")
    logger.info("Missing Type: " + f"{args.missing_type}")
    logger.info("K: " + f"{args.k}")
    logger.info("Prompt Length: " + f"{args.prompt_length}")
    logger.info("Prompt Position: " + f"{args.prompt_position}")

def get_optim(max_steps, model, lr, weight_decay, warmup_rate=0.1, use_warmup=False, **kwargs):
    """
    构建 AdamW 优化器与学习率调度器。

    调度策略（LambdaLR）：
        - 若 use_warmup=True：前 warmup_steps 步学习率线性上升到 lr，
          之后线性衰减到 0（linear decay）。
        - 若 use_warmup=False：从 lr 线性衰减到 0。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Compute warmup steps only if warmup is enabled
    warmup_steps = int(warmup_rate * max_steps) if use_warmup else 0

    def lr_lambda(current_step: int):
        if use_warmup and current_step < warmup_steps:
            # Linear warmup phase（线性预热阶段）
            return float(current_step) / float(max(1, warmup_steps))
        # Linear decay phase（线性衰减阶段）
        return max(
            0.0,
            float(max_steps - current_step) / float(max(1, max_steps - warmup_steps)),
        )

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def compute_loss(output, label, reduction='mean'):
    """
    计算交叉熵损失。
    output: 模型 logits (B, C)；label: 标签 (B,)。
    reduction 参数支持 'mean'（默认）或 'none'（返回每个样本的损失，
    用于 trainer 中对"完整样本/缺失样本"分别求损失）。
    """
    label = label.long()
    loss = F.cross_entropy(output, label, reduction=reduction)
    return loss



class HatememesMetric:
    """
    评测指标类：封装 AUROC 与 ACC（torchmetrics 实现）。

    注：虽名为 HatememesMetric，但实际可用于任意二分类任务；
    由于 get_evaluator 对所有数据集都返回该类，当前评测默认按二分类处理。
    """

    def __init__(self, device):
        self.device = device
        self.auroc = AUROC(task="binary").to(device)   # 二分类 AUROC
        self.acc   = Accuracy(task="binary").to(device)  # 二分类准确率

    def reset(self):
        """重置指标内部状态（每个 epoch 开始时调用）。"""
        self.auroc.reset()
        self.acc.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        累积一个 batch 的预测与标签。

        preds : (B, 2) logits   labels : (B,) int{0,1}
        """
        # 取正类概率（softmax 后第 1 类的概率）
        probs_pos = preds.softmax(dim=1)[:, 1]           # (B,)
        labels = labels.long()

        self.auroc.update(probs_pos, labels)
        self.acc.update(preds.argmax(dim=1), labels)     # logits → 类别

    def compute(self):
        """计算并返回当前累积的指标（计算后自动重置）。"""
        out = {
            "auroc": self.auroc.compute().item(),
            "acc"  : self.acc.compute().item()
        }
        self.reset()
        return out

