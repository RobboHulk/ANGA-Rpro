"""
================================================================================
ANGA 框架的核心子模块：MMG 与 CAP
================================================================================
本文件定义两个可训练的轻量级模块，是论文提出的 ANGA 框架的组成部分：

    1. MMG（Missing Modality Generator，缺失模态生成器）：
       负责"补全缺失模态"。当某个样本缺失了文本或图像时，MMG 利用
       检索到的 Top-K 个相似样本的对应模态特征，聚合生成该样本的
       "重建模态表征"，用来替换缺失位置。

    2. CAP（Context-Aware Prompter，上下文感知提示器）：
       负责生成"动态提示（dynamic prompt）"。它通过交叉注意力，把
       "检索到的相似实例"的信息注入当前样本的表征，得到随输入动态变化
       的文本提示与图像提示（论文中的 semantic-enhanced adapter）。

两个模块都只需很少的参数即可训练（预训练 ViLT 主干被冻结），符合
parameter-efficient 微调的思路。
================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# abbreviation: MMG: Missing Modality Generator, CAP: Context-Aware Prompter


class MMG(nn.Module):
    """
    缺失模态生成器（Missing Modality Generator）。

    输入：检索到的 Top-K 个相似样本的某一模态特征 F_l，形状 (B, K, seq_len, hidden)
         例如缺文本时输入检索到的文本特征 (B, 5, 128, 768)。
    输出：重建后的模态特征，形状 (B, seq_len, hidden)。

    当前实现采用的是最朴素的做法：对 Top-K 个检索特征在 K 维度上做平均，
    得到 (B, seq_len, hidden) 的重建表征。
    代码中保留了原始设计的频域（FFT）增强实现（已被注释），
    即"在频域用可学习权重 W 对特征做调制后再逆变换回时域"，读者可自行
    对比两种实现的差异。
    """

    def __init__(self, dropout_rate, n, d):
        super(MMG, self).__init__()
        self.n = n   # 序列长度（文本为 max_text_len，图像为 max_image_len）
        self.d = d   # 隐藏维度（ViLT 为 768）
        # 可学习的频域调制权重（复数张量）；当前 forward 中未使用（频域方案被注释）
        self.W = nn.Parameter(torch.randn(n, d, dtype=torch.cfloat))
        self.layer_norm = nn.LayerNorm(d)        # 层归一化，稳定特征分布
        self.dropout = nn.Dropout(dropout_rate)  # 随机失活，防止过拟合
        self.linear = nn.Linear(d, d)            # 线性投影

    # 输入形状示例：(B, 5, 128, 768) = (batch, Top-K, 文本长度, 隐藏维度)
    def forward(self, F_l):
        # 在 Top-K 维度上求平均，得到 (B, 128, 768) 的聚合特征
        F_l = torch.mean(F_l, dim=1)  # (B, seq_len, hidden)

        # ---- 以下为"频域增强"的原始实现，现已被简化版（直接平均）替代 ----
        # 思路：把聚合特征沿序列维度做 FFT 变换到频域 → 用可学习权重 W 调制
        #       → 逆 FFT 变换回时域 → 残差连接 + LayerNorm + 线性投影
        # X_l = torch.fft.fft(F_l, dim=1)               # (B, seq_len, hidden) 复数
        # X_tilde_l = self.W * X_l                       # 频域逐元素调制
        # F_tilde_l = torch.fft.ifft(X_tilde_l, dim=1).real  # 回到时域
        # F_l = self.layer_norm(F_l + self.dropout(F_tilde_l))
        # F_l = self.linear(F_l)
        return F_l


class CAP(nn.Module):
    """
    上下文感知提示器（Context-Aware Prompter）。

    作用：把"检索到的相似实例（Top-K）"作为上下文，通过交叉注意力
    （cross-attention）生成动态提示，增强模型对缺失模态的鲁棒性。

    输入：
        - V：当前样本的图像表征        (B, 145, 768)
        - T：当前样本的文本表征        (B, 128, 768)
        - r_i：检索到的图像特征        (B, 5, 145, 768)
        - r_t：检索到的文本特征        (B, 5, 128, 768)
    输出：
        - T_to_T：由检索文本生成的文本提示  (B, 1, 768)
        - V_to_V：由检索图像生成的图像提示  (B, 1, 768)
    """

    def __init__(self, prompt_length, dim=768):
        super(CAP, self).__init__()
        self.dim = dim
        # 交叉注意力的 Q/K/V 投影层（共享维度 768）
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # 自适应平均池化：把注意力输出从 (B, K, seq, dim) 压缩到 (B, K, prompt_length, dim)
        self.pooling = nn.AdaptiveAvgPool2d((prompt_length, dim))

    def attention(self, query, key_value):
        """
        以 query（当前样本表征）为 Q，以 key_value（检索特征）为 K/V，
        执行一次交叉注意力。
        """
        b, k, s, _ = key_value.shape   # (B, Top-K, seq_len, hidden)

        # Q 由当前样本表征投影得到，并广播到每个检索通道：(B, K, seq, dim)
        q = self.q_proj(query).unsqueeze(1).expand(b, k, -1, -1)   # (B, K, seq, dim)
        k = self.k_proj(key_value)                                 # (B, K, seq, dim)
        v = self.v_proj(key_value)                                 # (B, K, seq, dim)

        # 计算注意力分数并缩放：(B, K, seq, seq)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        # softmax 归一化成注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)                # (B, K, seq, seq)

        # 加权求和得到输出：(B, K, seq, dim)
        output = torch.matmul(attn_probs, v)
        # 池化压缩序列维度 → (B, K, prompt_length, dim)
        output = self.pooling(output)
        # 对 K 个检索通道求平均 → (B, prompt_length, dim)
        output = output.mean(dim=1)
        return output

    # V (B,145,768)
    # T (B,128,768)
    # r_i (B,5,145,768)
    # r_t (B,5,128,768)
    def forward(self, V, T, r_i, r_t):
        # 文本提示：以当前文本 T 为 Q，检索文本 r_t 为 K/V
        V_to_V = self.attention(V, r_i)   # (B, prompt_length, dim)
        # 图像提示：以当前图像 V 为 Q，检索图像 r_i 为 K/V
        T_to_T = self.attention(T, r_t)   # (B, prompt_length, dim)
        # 注意：返回值顺序为 (文本提示, 图像提示)，调用方需对应接收
        return T_to_T, V_to_V
