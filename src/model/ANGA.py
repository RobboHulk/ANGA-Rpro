"""
================================================================================
ANGA 主模型：Anchor-Guided Gradient Alignment
================================================================================
本文件实现论文《Anchor-Guided Gradient Alignment for Incomplete Multimodal
Learning》的核心模型结构 ANGA（锚点引导的梯度对齐框架）。

设计思路（对应论文三大贡献）：
    1. 实例检索重建（Instance Retrieval）：
       借助检索到的相似样本特征（r_t_list / r_i_list），通过 MMG 模块重建
       缺失模态的表征，缓解信息缺失。

    2. 动态提示（Dynamic Prompt，即 semantic-enhanced adapter）：
       通过 CAP 模块，把检索实例的信息注入当前样本，生成随输入变化的
       文本提示与图像提示，增强模型的鲁棒性。

    3. 优化锚点（Optimization Anchor）：
       训练时（见 utils/trainer.py），以"完整样本 + 可靠补全样本"为锚点，
       对其梯度做对齐，缓解严重缺失条件下的学习不平衡。

结构概览：
    输入文本/图像 → 冻结的 ViLT embedding 层 → 分离文本/图像表征
    → MMG 补全缺失模态 → CAP 生成动态提示 → 在指定层插入提示 token
    → 冻结的 ViLT encoder 逐层编码 → LayerNorm → pooler → 分类头。

注意：ViLT 的主干（embedding + encoder + layernorm）全部冻结（freeze），
只有 MMG、CAP、分类器与 label_enhanced 等新增参数参与训练，属于
parameter-efficient 微调。
================================================================================
"""
import torch
from torch import nn
import torch.nn.functional as F
from .vilt import ViltModel
from .modules import MMG, CAP
# from einops import rearrange

def init_weights(module):
    """对新增模块做权重初始化（参考 BERT/ViT 的初始化方式）。

    规则：
        - Linear / Embedding：均值 0、标准差 0.02 的正态分布
        - LayerNorm：bias 置 0，weight 置 1
        - Linear 的 bias 置 0
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class ANGA(torch.nn.Module):
    def __init__(self,
                 vilt: ViltModel,           # 预训练的 ViLT 模型（backbone）
                 task_id: str,              # 任务名：hatememes / food101 / mmimdb
                 max_text_len: int,         # 文本最大长度（128）
                 max_image_len: int,        # 图像最大 patch 数（145）
                 missing_type: str,         # 缺失类型：Both / Text / Image
                 device: str,               # 运行设备
                 prompt_position: int,      # 动态提示插入到 encoder 的第几层
                 prompt_length: int,        # 每类提示的 token 数
                 dropout_rate: float,       # MMG 中的 dropout 比例
                 hs=768,                    # 隐藏维度（ViLT-B 为 768）
                 **kargs):
        super(ANGA, self).__init__()
        self.device = device
        self.max_text_len = max_text_len
        self.missing_type = missing_type
        self.task_id = task_id
        # ---- 从预训练 ViLT 中取出各子模块 ----
        self.embedding_layer = vilt.embeddings   # 文本/图像嵌入层
        self.encoder_layer = vilt.encoder.layer  # Transformer 编码层的 ModuleList（12 层）
        self.layernorm = vilt.layernorm          # 编码器末尾的 LayerNorm
        self.prompt_length = prompt_length
        self.prompt_position = prompt_position
        self.hs = hs

        # ---- 依据任务确定分类类别数 ----
        if task_id == "hatememes":
            cls_num = 2      # 仇恨梗图二分类
        elif task_id == "food101":
            cls_num = 101    # 食物 101 分类
        elif task_id == "mmimdb":
            cls_num = 23     # 电影 23 分类

        # 冻结预训练的多模态 Transformer 主干（embedding + encoder + layernorm）
        self.freeze()

        # ---- 定义训练组件 ----
        # ViLT 的 pooler：取 [CLS] token 表征做 dense + tanh，得到句级表征
        self.pooler = vilt.pooler

        # ---- 依据缺失类型构建 MMG（用于补全缺失模态）----
        if missing_type == "Text":
            # 只缺文本：一个 MMG，序列长度为文本长度
            self.MMG = MMG(n = max_text_len, d = hs,dropout_rate=dropout_rate)
        elif missing_type == "Image":
            # 只缺图像：一个 MMG，序列长度为图像 patch 数
            self.MMG = MMG(n = max_image_len, d = hs,dropout_rate=dropout_rate)
        elif missing_type == "Both":
            # 文本、图像都可能缺失：分别建两个 MMG
            self.MMG_t = MMG(n = max_text_len, d = hs,dropout_rate=dropout_rate)
            self.MMG_i = MMG(n = max_image_len, d = hs,dropout_rate=dropout_rate)

        # ---- 定义动态提示生成器（CAP）----
        self.dynamic_prompt = CAP(prompt_length=prompt_length)

        # ---- 定义分类头 ----
        self.classifier = nn.Linear(768, cls_num)   # 池化表征 → 类别 logits
        self.classifier.apply(init_weights)         # 对分类头做权重初始化

        # ---- 定义"标签增强嵌入" ----
        # 每个类别学习一个可学习嵌入向量 (cls_num, hs)。
        # 在 forward 中，根据"检索到的邻居标签 r_l_list"取出对应嵌入并平均，
        # 作为额外的提示 token 拼入序列，实现"标签语义增强"。
        self.label_enhanced = nn.Parameter(torch.randn(cls_num, hs))
        # （下面的 Sequential 版分类头为作者早期方案，已注释）
        # self.classifier = nn.Sequential(
        #         nn.Linear(hs * 2, hs * 2),
        #         nn.LayerNorm(hs * 2),
        #         nn.GELU(),
        #         nn.Linear(hs * 2, hs),
        #     )
        # self.classifier.apply(init_weights)

    def freeze(self):
        """冻结 ViLT 主干的所有参数，使其不参与梯度更新。"""
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        for param in self.encoder_layer.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False




    def forward(self,
                input_ids: torch.Tensor,        # 文本 token ids (64,128)
                pixel_values: torch.Tensor,     # 图像像素张量 (64,3,384,384)
                pixel_mask: torch.Tensor,       # 图像掩码 (64,384,384) 好像全是1
                token_type_ids: torch.Tensor,   # 文本 segment ids (64,128) 好像全是0
                attention_mask: torch.Tensor,   # 文本 attention mask (64,128)
                r_t_list: torch.Tensor,         # 检索的文本向量 (64,5,128,768)
                r_i_list: torch.Tensor,         # 检索的图像向量 (64,5,145,768)
                r_l_list: torch.Tensor,         # 检索的标签 (64,5)
                missing_mask = None,            # 缺失掩码 (64,) 若没缺失，则全为1
                image_token_type_idx=1):

        # ---- 第 0 步：得到图文嵌入表征 ----
        # embedding: (64,273,768) = 文本(128) + 图像(145) 拼接
        # attention_mask: (64,273) 对应拼接后的掩码
        embedding, attention_mask = self.embedding_layer(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids,
                                                         inputs_embeds=None,
                                                         image_embeds=None,
                                                         pixel_values=pixel_values,
                                                         pixel_mask=pixel_mask,
                                                         image_token_type_idx=image_token_type_idx)

        # ---- 第 1 步：得到图文表征，mask 后再补全 ----
        # 分离文本嵌入（前 128 个位置）与图像嵌入（后 145 个位置）
        # text_emb: (64,128,768); image_emb: (64,145,768)
        text_emb = embedding[:, :self.max_text_len, :]
        image_emb = embedding[:, self.max_text_len:, :]

        # 依据缺失类型，用 MMG 重建缺失模态
        if self.missing_type == "Text":
            # 仅缺文本：用检索到的文本特征聚合出重建文本表征
            recovered_t = self.MMG(r_t_list)  # (64,128,768) # 仅检索平均
            # 把缺失掩码扩展成 (64,128,768)，作为"保真/替换"的选择开关
            missing_mask_t = missing_mask.view(-1, 1, 1).expand(-1, 128, self.hs)  # (64,128,768)
            # 缺失位置(mask=0)用重建值替换，未缺失位置(mask=1)保留原始值
            text_emb = text_emb * missing_mask_t + recovered_t * (1-missing_mask_t)  # (64,128,768)

        elif self.missing_type == "Image":
            # 仅缺图像：用检索到的图像特征聚合出重建图像表征
            recovered_i = self.MMG(r_i_list)  # (64,145,768) # 仅检索平均
            missing_mask_i = missing_mask.view(-1, 1, 1).expand(-1, 145, self.hs)  # (64,145,768)
            image_emb = image_emb * missing_mask_i + recovered_i * (1-missing_mask_i)  # (64,145,768)

        elif self.missing_type == "Both":
            # 双模态缺失：文本、图像分别重建
            recovered_t = self.MMG_t(r_t_list)
            recovered_i = self.MMG_i(r_i_list)
            # 从 both 掩码（0=缺文本,1=缺图像,2=完整）推导出各自的缺失标志：
            # 文本缺失标志：mask==0 → 缺文本（对应 t_missing_mask=0）
            t_missing_mask = [0 if i == 0 else 1 for i in missing_mask]
            # 图像缺失标志：mask==1 → 缺图像（对应 i_missing_mask=0）
            i_missing_mask = [0 if i == 1 else 1 for i in missing_mask]
            t_missing_mask = torch.tensor(t_missing_mask).to(self.device)
            i_missing_mask = torch.tensor(i_missing_mask).to(self.device)
            missing_mask_t = t_missing_mask.view(-1, 1, 1).expand(-1, 128, self.hs)
            missing_mask_i = i_missing_mask.view(-1, 1, 1).expand(-1, 145, self.hs)
            # 各自在缺失位置用重建值替换
            text_emb = text_emb * missing_mask_t + recovered_t * (1-missing_mask_t)
            image_emb = image_emb * missing_mask_i + recovered_i * (1-missing_mask_i)

        # ---- 第 2 步：构建动态提示 ----
        # 由 CAP 生成文本提示与图像提示（各 prompt_length 个 token）
        # 注意 forward 返回顺序为 (文本提示, 图像提示)
        t_prompt,i_prompt = self.dynamic_prompt(r_i=r_i_list, r_t=r_t_list, T=text_emb, V=image_emb)  # (64,1,768) (64,1,768)
        t_prompt = torch.mean(t_prompt, dim=1)  # (64,768) 文本提示平均
        i_prompt = torch.mean(i_prompt, dim=1)  # (64,768) 图像提示平均

        # ---- 构建标签增强嵌入提示 ----
        # 依据检索邻居的标签 r_l_list (64,5)，查表得到每个邻居的类别嵌入，
        # 再对 K 个邻居求平均，得到 (64,768) 的标签提示，并扩展成 (64,1,768)
        label_emb = self.label_enhanced[r_l_list]  # (64,5,768)
        label_emb = torch.mean(label_emb, dim=1)   # (64,768)
        label_emb = label_emb.view(-1, 1, self.hs) # (64,1,768)

        # ---- 第 3 步：模型训练（逐层前向） ----
        # 先把补全后的文本、图像表征沿序列维拼接
        output = torch.cat([text_emb, image_emb], dim=1)  # (64,273,768)
        for i, layer_module in enumerate(self.encoder_layer):
            if i == self.prompt_position:
                # 在指定层，把 [标签提示, 文本提示, 图像提示] 拼到序列最前面
                # 共 prompt_length*2+1 = 3 个提示 token
                output = torch.cat([label_emb, t_prompt.unsqueeze(1), i_prompt.unsqueeze(1), output], dim=1)  # (64,276,768)
                N = embedding.shape[0]  # int: 64
                # 同步扩展 attention_mask，前 3 个位置为提示 token（全部有效，置 1）
                attention_mask = torch.cat([torch.ones(N, self.prompt_length*2+1).to(self.device), attention_mask], dim=1)  # (64,276)
                layer_outputs = layer_module(output, attention_mask=attention_mask)
                output = layer_outputs[0]  # (64,276,768)
            else:
                # 其余层正常前向
                layer_outputs = layer_module(output, attention_mask=attention_mask)
                output = layer_outputs[0]  # (64,276,768)

        # ---- 第 4 步：池化 + 分类 ----
        output = self.layernorm(output)   # (64,276,768) 最终 LayerNorm
        output = self.pooler(output)      # (64,768) 取 [CLS] 位置做 dense+tanh
        output = self.classifier(output)  # (64,cls_num) 得到类别 logits

        return output