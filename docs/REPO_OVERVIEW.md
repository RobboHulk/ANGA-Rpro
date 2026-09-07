# ANGA 仓库介绍文档

> 论文：**Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning**（CVPR 2026）
> 作者：Zhi-Hao Guan, Longfei Huang, Yang Yang（南京理工大学）
>
> 本文档面向**复现者**，介绍代码结构、数据流水线、关键模块、超参数，以及
> 当前代码与论文描述之间需要注意的差异。数据集本身的下载与格式见
> [`docs/DATASETS.md`](DATASETS.md)。

---

## 1. 方法一句话概述

在**不完整多模态学习**（部分样本缺失文本或图像）场景下，已有的"模态重建"
方法在高缺失率时会出现**学习不平衡**：重建样本携带语义噪声，主导了优化过程，
削弱了完整样本的表征。ANGA 通过三件事来缓解：

| 组件 | 论文名称 | 代码位置 | 作用 |
|---|---|---|---|
| 多模态实例检索 + 记忆库 | MIR / Memory Bank | `src/utils/core_tools.py` 的 `MemoryBankGenerator` / `MCR`，`src/model/modules.py` 的 `MMG` | 用可用模态检索 Top-K 相似样本，聚合其特征来重建缺失模态 |
| 锚点引导的梯度对齐 + 熵驱动课程 | Anchor-Guided Gradient Alignment | `src/utils/trainer.py` 的 `Trainer._train` / `_ranked_missing_samples` | 用完整样本梯度构造"优化锚点"，把重建样本梯度投影进锚点的锥域，抑制冲突方向 |
| 语义增强适配器 | SEA（Semantic-Enhanced Adapter） | `src/model/modules.py` 的 `CAP` | 用检索到的实例做交叉注意力，生成随输入动态变化的 prompt |

骨干网络是**冻结的预训练 ViLT-B/32**，只训练 MMG、CAP、标签增强嵌入和分类头，
属于参数高效微调。

---

## 2. 目录结构

```
ANGA/
├── README.md / README_CN.md         # 官方简介（环境配置、数据准备、运行命令）
├── requirements.txt                 # 依赖（不完整，见第 7 节）
├── framework.png                    # 论文框架图
├── docs/
│   ├── REPO_OVERVIEW.md             # 本文档
│   └── DATASETS.md                  # 数据集讲解
├── dataset/                         # 数据根目录（需自行放置原始数据）
│   ├── hatememes/{image, meta_data}
│   ├── mmimdb/{image, meta_data, split.json}
│   └── food101/{image, meta_data, class_idx.json}
└── src/
    ├── init_data.py                 # 数据预处理入口（阶段①②③，需逐段取消注释）
    ├── train.py                     # 训练 / 评测入口（argparse 超参）
    ├── config/config.yaml           # 配置的集中记录（仅参考，train.py 实际走命令行）
    ├── utils/
    │   ├── core_tools.py            # 工具箱：数据预处理、工厂函数、Collator、评测器、优化器
    │   └── trainer.py               # 训练器：课程学习 + 梯度对齐 + 验证/早停/测试
    ├── model/
    │   ├── ANGA.py                  # ANGA 主模型（forward 逐层前向 + prompt 注入）
    │   ├── modules.py               # MMG（缺失模态生成器）+ CAP（上下文感知提示器）
    │   ├── backbone.py              # 备用 backbone 代码（当前主流程未使用）
    │   ├── vilt/                    # 从 HuggingFace transformers 复制的 ViLT 实现
    │   └── vilt-b32-mlm/            # ViLT 配置 / 分词器 / 词表（缺 pytorch_model.bin，需自行下载）
    └── dataloader/
        ├── hatememes_dataset.py     # 二分类，图像 .png
        ├── mmimdb_dataset.py        # 23 类，图像 .jpeg
        └── food101_dataset.py       # 101 类，图像 .jpg
```

---

## 3. 数据流水线

复现分为**预处理**和**训练**两步，预处理产物是一次性生成、长期复用的。

### 3.1 阶段① — 生成 `train/valid/test.pkl`

把原始标注（jsonl / json / csv）转换成 `pandas.DataFrame` 并序列化。
产物列：`item_id / img / label / text`。

入口在 `src/init_data.py`（三段代码默认注释，需逐段取消注释运行一次）：

```python
init_data_hatememes()   # HateMemes：读取 train/dev/test_seen.jsonl
init_data_mmimdb()      # MM-IMDb：读取 meta_data/*.json + split.json，并生成 class_idx.json
init_data_food101()     # Food101：读取 *_titles.csv + class_idx.json，并从 train 分层切出 valid
```

> `init_data_mmimdb` / `init_data_food101` 是为了本仓库复现补写的（原代码只提供了
> `init_data_hatememes`）。实现细节与类别定义见 [`docs/DATASETS.md`](DATASETS.md)。

### 3.2 阶段② — 构建记忆库（Memory Bank）

`MemoryBankGenerator` 用**冻结的 ViLT embedding 层**对每个样本的文本
（128 token）和图像（145 patch）分别编码，把 embedding 层输出（尚未过
Transformer encoder）保存成 `.npy`：

```
dataset/memory_bank/{dataset}/text/{item_id}.npy    # (128, 768)
dataset/memory_bank/{dataset}/image/{item_id}.npy   # (145, 768)
```

训练时若某样本缺失某模态，就从记忆库里取"检索到的相似样本"的对应模态特征，
经 `MMG`（K 维平均）聚合后填充到缺失位置。

### 3.3 阶段③ — 构建检索列表（Multi-Channel Retriever, MCR）

`MCR` 用**预训练 CLIP（`clip-vit-large-patch14-336`）**分别编码图像和文本，
计算样本间余弦相似度，为每个样本找 Top-K 邻居，分两条通道：

- **i2i**：图像→图像检索
- **t2t**：文本→文本检索

记忆库 = `train + valid`，`test` 只做查询（避免信息泄露）。检索结果写回
`.pkl`，新增列：`i2i_id_list / i2i_sims_list / i2i_label_list` 和
`t2t_id_list / t2t_sims_list / t2t_label_list`（默认存 Top-20，训练时按 `--k` 截取）。

### 3.4 阶段④ — 生成缺失掩码表（missing_table）

由 `src/train.py --regenerate_missing_table True` 触发，调用
`generate_missing_table`：

```
dataset/missing_table/single/{dataset}/missing_table.pkl   # 单模态缺失：0 缺失，1 不缺失
dataset/missing_table/both/{dataset}/missing_table.pkl     # 双模态缺失：0 缺文本，1 缺图像，2 完整
```

列名形如 `missing_mask_7`（`7 = int(0.7 * 10)`，对应 70% 缺失率）。同一个表可以
容纳多个缺失率的列。

### 3.5 训练 / 评测

`src/train.py` → `Trainer.run()`：

1. 每 5 个 epoch 调用 `_ranked_missing_samples()`，对缺失样本按预测熵升序排序
   （熵低 = 模型更确信 = 更可靠）。
2. 课程比例 `ratio` 按线性函数从 `0.20` 增长到 `0.30`（5 个 epoch 内），取熵最低的
   `ratio` 比例缺失样本"提升"为完整样本，纳入锚点集合。
3. `_train()` 对每个 batch：
   - 逐样本交叉熵损失，按 `missing_mask` 分成完整集 `C` 和缺失集 `M`；
   - `L_C.backward(retain_graph=True)` 得 `g_C`，`L_M.backward()` 得 `g_M`；
   - `a_t = g_C / ||g_C||` 作为锚点主轴；
   - 计算 `cos(g_M, a_t)`：`≤0` 置零；`≥τ` 保留；介于中间则投影到锥边界；
   - 用 `g_total = g_C + g̃_M` 更新参数。
4. `_valid()` 计算 AUROC 并早停；训练结束加载最优权重跑 `_test()`。

---

## 4. 关键模块逐文件说明

### `src/model/ANGA.py` — `ANGA`

`forward(input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask, r_t_list, r_i_list, r_l_list, missing_mask)`：

1. ViLT embedding 层 → `(B, 273, 768)`，拆成 `text_emb (B,128,768)` 与 `image_emb (B,145,768)`。
2. 按 `missing_type` 用 `MMG` 重建缺失模态，在缺失位置用重建值替换（掩码逐元素选择）。
3. `CAP` 生成 `t_prompt` / `i_prompt`（各 `prompt_length` 个 token，随后在 token 维求平均）。
4. 用检索邻居标签 `r_l_list` 查 `label_enhanced` 嵌入表，K 维平均得到 `label_emb`。
5. 逐层过 12 个 ViLT encoder layer；在第 `prompt_position` 层，把
   `[label_emb, t_prompt, i_prompt]` 共 `prompt_length*2+1` 个 token 拼到序列最前，
   同步扩展 `attention_mask`。
6. `layernorm → pooler（取 [CLS]，dense+tanh） → classifier` 输出 `(B, cls_num)` logits。

`cls_num`：hatememes=2，food101=101，mmimdb=23。

### `src/model/modules.py`

- **`MMG`**：输入检索特征 `(B, K, seq, 768)`，当前实现就是**在 K 维求平均**。
  代码里保留了原始设计的频域（FFT）调制实现，但**已被注释**，`self.W`
  （可学习复数权重）当前未参与前向。
- **`CAP`**：以当前样本表征为 Query，检索特征为 Key/Value 做交叉注意力，
  经 `AdaptiveAvgPool2d((prompt_length, 768))` 压缩后在 K 维平均，输出
  `(B, prompt_length, 768)`。注意 `forward` 返回顺序是 `(文本提示, 图像提示)`。

### `src/utils/core_tools.py`

数据预处理（阶段①②③）、`generate_missing_table`、工厂函数
（`load_model` / `get_dataset` / `get_evaluator`）、`Collator`（把 batch 编码成张量）、
`EarlyStopping`、`get_optim`（AdamW + warmup 后线性衰减）、`compute_loss`
（交叉熵）、`HatememesMetric`（AUROC + ACC，二分类实现）。

### `src/utils/trainer.py`

`Trainer` 类，实现第 3.5 节的训练主循环。梯度对齐的锥域投影公式与论文
Eq.(9)–(14) 对应；`t_max = sqrt(1 - τ²) / τ` 与论文 Eq.(11) 的 `κ_max = tan θ` 一致。

### `src/dataloader/*.py`

三个 `Dataset` 类结构一致，区别只在数据路径、图像扩展名、类别数。
`__getitem__` 按 `missing_type` × `missing_mask` 分 6 种情况，决定：
是否把文本替换成占位串（`"I love deep learning" * 1024`）、从记忆库取哪条通道的
邻居特征、`r_l_list` 用哪条通道的邻居标签。

---

## 5. 超参数

`src/train.py` 用 `argparse`，`src/config/config.yaml` 只是**配置的集中记录**，
两者默认值**并不完全一致**。以下是关键参数、`train.py` 默认值，以及**论文
Implementation details 里的取值**：

| 参数 | 含义 | `train.py` 默认 | 论文取值 |
|---|---|---|---|
| `--dataset` | 数据集 | `hatememes` | 三个都做 |
| `--missing_type` | `Both` / `Text` / `Image` | `Text` | 三种都做 |
| `--missing_rate` | 缺失比例 δ | `0.7` | `0.7`（主实验） |
| `--k` | 检索邻居数 Top-K | `5` | 扫 `{1,3,5,7,9}` |
| `--prompt_position` | prompt 插入到第几层 encoder（0 索引） | `0` | 第 2 层 MSA（`b=2`，1 索引） |
| `--prompt_length` | 每类 prompt 的 token 数 | `1` | 小值 |
| `--lr` | 学习率 | `1e-3` | `1e-3` |
| `--weight_decay` | 权重衰减 | `5e-5` | `1e-5` |
| `--batch_size` | 批大小 | `64` | `64` |
| `--epochs` | 训练轮数 | `20` | 未明确 |
| `--name` | 优化器 | `AdamW` | AdamW |
| `--use_warmup` / `--warmup_rate` | 学习率预热 | `True` / `0.1` | — |
| `--patience` | 早停耐心值 | `10` | — |
| `--device` | GPU | `cuda:4` | 单张 RTX 4090 |
| `--save_path` | 权重保存目录 | 绝对路径（需改） | — |

**未暴露为命令行参数、只能改源码的关键量**：

| 量 | 位置 | 当前值 | 论文取值 |
|---|---|---|---|
| 锥域余弦阈值 `τ` | `src/utils/trainer.py`（`_train` 内 `tau = ...`） | `0.7` | **`0.2`** |
| 课程比例下界 `λ_min` | `src/utils/trainer.py`（`run` 内 `ratio_start`） | `0.20` | **`0.1`** |
| 课程比例上界 `λ_max` | `src/utils/trainer.py`（`run` 内 `ratio_end`） | `0.30` | `0.3` |
| 课程阶段长度 `Z_grow` | `src/utils/trainer.py`（`grow_epochs` / `REFRESH_EVERY`） | `5` | `5` |

---

## 6. 复现步骤

```shell
# 0. 环境
conda create -n ANGA python=3.9 && conda activate ANGA
pip install -r requirements.txt      # 注意补全依赖，见第 7 节

# 1. 下载权重（见第 7 节），放到：
#    src/model/vilt-b32-mlm/pytorch_model.bin
#    src/model/clip-vit-large-patch14-336/

# 2. 放置原始数据（见 docs/DATASETS.md），然后逐段运行：
python src/init_data.py              # 依次取消注释：阶段①→②→③

# 3. 生成缺失掩码表
python src/train.py --dataset hatememes --missing_type Text --missing_rate 0.7 \
       --regenerate_missing_table True

# 4. 训练 + 评测（先按第 5 节把 τ 改成 0.2、λ_min 改成 0.1）
mkdir -p checkpoints src/metrics
python src/train.py --dataset hatememes --missing_type Text --missing_rate 0.7 \
       --k 5 --lr 1e-3 --weight_decay 1e-5 --prompt_position 1 \
       --batch_size 64 --epochs 20 --device cuda:0 --save_path checkpoints
```

---

## 7. 已知问题与待办（复现前必看）

### 7.1 缺失的预训练权重

- `src/model/vilt-b32-mlm/` 只有 config / 分词器 / 词表，**缺 `pytorch_model.bin`**，
  需下载 HuggingFace `dandelin/vilt-b32-mlm`。
- `--vilt_weights` 默认 `src/model/vilt/weights/mlm`，该目录**不存在**；
  建议把默认值改成 `src/model/vilt-b32-mlm`，或按此路径放权重。
- 阶段③需要 `src/model/clip-vit-large-patch14-336/`（HuggingFace
  `openai/clip-vit-large-patch14-336`），当前不存在。

### 7.2 硬编码的绝对路径

以下位置仍写死作者机器路径 `/data/gzh/MissingWork/MyWork/...`，运行前需改成
相对路径（`dataset/...`、`src/model/...`）：

- `src/utils/core_tools.py`：`MemoryBankGenerator`、`MCR`、`generate_missing_table`
  的 `base_file_path` 与内部 `pd.concat` 路径、`load_model`、`Collator`
- `src/dataloader/hatememes_dataset.py`（`mmimdb` / `food101` 的 dataloader 已用相对路径）
- `src/utils/trainer.py`：指标 CSV 输出路径 `Path("/data/gzh/.../metrics/...")`
- `src/train.py`：`--save_path` 默认值

### 7.3 预处理脚本里的硬编码常量

- `MemoryBankGenerator` 和 `MCR` 的 `self.dataset` 写死为 `'hatememes'`，跑其他
  数据集要手动改。
- `MemoryBankGenerator._process_batch` 里图像后缀写死 `.png`；MM-IMDb 是
  `.jpeg`、Food101 是 `.jpg`，需按数据集调整。

### 7.4 依赖不完整

`requirements.txt` 缺 `numpy`、`scikit-learn`（`trainer.py` 用到
`sklearn.metrics`）；建议锁版本以兼容 `torchmetrics==0.9.3`：

```
torch==1.13.1
torchvision==0.14.1
torchmetrics==0.9.3
transformers==4.28.1
numpy==1.24.4
scikit-learn==1.3.2
pandas==2.0.3
pillow==10.2.0
tqdm
hydra-core
omegaconf
colorama
```

### 7.5 代码与论文的方法差异

- **`τ = 0.7`（代码） vs `0.2`（论文）**：0.7 甚至不在论文的 τ 扫描范围
  `{0, 0.1, 0.2, 0.4, 0.8}` 内，会显著影响结果，必须改。
- **`prompt_position = 0`（代码） vs 第 2 层（论文）**。
- **`λ_min = 0.20`（代码） vs `0.1`（论文）**。
- **`weight_decay`：`train.py` 默认 5e-5、`config.yaml` 写 2e-2，论文是 1e-5**。
- **MM-IMDb 任务形式**：论文按**多标签**处理，用 **F1-Micro** 评测；本仓库的
  训练代码用 `cross_entropy` 按**单标签**处理（`cls_num=23`），补写的
  `init_data_mmimdb` 取"主类型"作为单标签。若要复现论文 MM-IMDb 的 F1，
  需改成 multi-hot 标签 + BCE 损失，并替换评测器。
- **评测器**：`get_evaluator` 对三个数据集都返回二分类的 `HatememesMetric`
  （`AUROC(task="binary")`）。对 mmimdb / food101，其 `auroc` 实际取
  `softmax(preds)[:, 1]`，指标意义有限，早停/选优也基于它。跑这两个数据集
  前建议实现一个多分类 ACC / F1-Micro 评测器。

### 7.6 运行时目录

`checkpoints/`（`EarlyStopping` 保存 `best_model.pth`）和 `src/metrics/`
（验证指标 CSV）需要预先 `mkdir`。

---

## 8. 硬件与环境要求

- **GPU**：单张 **RTX 4090 / 3090 / A5000（24GB）** 即可跑论文原配置
  （`batch_size=64`，ViLT 主干冻结）。论文原文即在单张 RTX 4090 上完成。
  显存不足时把 `--batch_size` 降到 16–32。代码无 DDP / DataParallel / AMP，多卡无收益。
- **内存**：建议 **≥ 64GB**。阶段③ `MCR._retrieval_vector_generation` 会把整个
  数据集的图片一次性读入内存，Food101（9 万张）解码后可能占 20–35GB，
  内存小的机器需要先把该函数改成分批读取。
- **磁盘**：建议预留 **150–200GB**。记忆库 `.npy` 约 0.84MB/样本
  （HateMemes ~7.5GB、MM-IMDb ~15GB、Food101 ~55GB），再加原始数据。
- **系统 / CUDA**：Linux（Ubuntu 20.04/22.04），NVIDIA 驱动 ≥ 470，CUDA 11.7/11.8。
