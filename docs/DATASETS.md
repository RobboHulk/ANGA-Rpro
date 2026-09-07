# 数据集讲解文档

本文档介绍 ANGA 用到的三个基准数据集：来源与下载方式、原始格式、在本仓库中的
放置位置、预处理产物，以及各自的注意事项。代码层面的预处理逻辑见
[`src/utils/core_tools.py`](../src/utils/core_tools.py) 的
`init_data_hatememes` / `init_data_mmimdb` / `init_data_food101`；
仓库整体结构与复现步骤见 [`docs/REPO_OVERVIEW.md`](REPO_OVERVIEW.md)。

三个数据集处理后统一成同一种"骨架"——`item_id / img / label / text` 四列的
`DataFrame`，再经阶段②（记忆库）、阶段③（检索）扩展出更多列，最终都用同一套
`Trainer` / `Collator` 训练。

---

## 总览

| | HateMemes | MM-IMDb | Food101 |
|---|---|---|---|
| 任务 | 仇恨梗图二分类 | 电影类型分类 | 食物图像分类 |
| 类别数 | 2 | 23 | 101 |
| 论文样本量 | 10,000 | 25,959 | 90,688 |
| 本仓库图像数 | 10,000（`.png`） | 25,959（`.jpeg`） | 90,704（`.jpg`，比 90,688 多出的属正常冗余/重名文件） |
| 官方划分 | train/dev/test_seen | train/dev/test | train/test（无 valid） |
| 标注格式 | jsonl（图文标签同一行） | 每部电影一个 json + 全局 `split.json` | csv（img, 标题文本, 类别名）+ `class_idx.json` |
| 论文任务形式 | 单标签 | **多标签**（本仓库代码按单标签实现，见下文） | 单标签 |
| 论文评测指标 | AUROC | F1-Micro | ACC |

---

## 1. HateMemes（The Hateful Memes Challenge）

- **来源**：Facebook / Meta AI，[Kiela et al., NeurIPS 2020]。
- **下载**：
  - 主数据：<https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset>
  - `test.json` 无标签，需用带标签的 `test_seen.json` 替换：
    <https://www.kaggle.com/datasets/williamberrios/hateful-memes>
    （只替换 `test.json`，其余文件不动）
- **放置**：
  ```
  dataset/hatememes/image/{item_id}.png       # 10,000 张梗图，5 位数字命名
  dataset/hatememes/meta_data/train.jsonl     # 8,500 条
  dataset/hatememes/meta_data/dev.jsonl       # 500 条
  dataset/hatememes/meta_data/test_seen.jsonl # 1,000 条（原 test_seen，即替换后的 test）
  ```
- **原始 jsonl 单行示例**：
  ```json
  {"id": 42953, "img": "img/42953.png", "label": 0, "text": "its their character not their color that matters"}
  ```
- **标签**：`label ∈ {0, 1}`，1 表示仇恨内容。
- **预处理**（`init_data_hatememes`）：
  1. 依次读取 `train.jsonl` / `dev.jsonl` / `test_seen.jsonl`；
  2. 列 `id` 重命名为 `item_id`，并格式化成 5 位字符串（如 `42` → `"00042"`），
     与图像文件名对齐；
  3. `dev` → `valid`，`test_seen` → `test`；
  4. 存为 `dataset/hatememes/{train,valid,test}.pkl`（形状约
     `(8500,4) / (500,4) / (1000,4)`，列 `item_id/img/label/text`）。
- **注意**：README 中写"json 文件放 metadata 文件夹"是笔误，代码统一读取的是
  `meta_data/`（与仓库当前目录一致，不用改）。

---

## 2. MM-IMDb

- **来源**：[Arévalo et al., *Gated Multimodal Units for Information Fusion*, ICLR Workshop 2017]。
- **下载**：<https://archive.org/download/mmimdb/mmimdb.tar.gz>（约 8.2GB）
- **放置**：
  ```
  dataset/mmimdb/image/{item_id}.jpeg       # 25,959 张电影海报
  dataset/mmimdb/meta_data/{item_id}.json   # 25,959 个 IMDb 元数据文件
  dataset/mmimdb/split.json                 # {"train": [...15552个id...], "dev": [...2608...], "test": [...7799...]}
  ```
  `item_id` 是 7 位补零字符串（如 `"0000005"`），元数据文件名、图像文件名、
  `split.json` 里的 id 三者一致。
- **元数据字段**（每个 json 有 27 个字段，训练只用其中两个）：
  - `genres`：字符串列表，如 `["Drama", "Romance"]`——**电影可以同时属于多个类型**，
    因此 MM-IMDb 官方是**多标签**任务；
  - `plot`：字符串列表（IMDb 上多个不同来源撰写的剧情简介，长度不一）。
- **类别定义**：完整数据里出现 27 种 genre，但学术界通用做法（也是本仓库
  `cls_num=23` 所依据的）是剔除样本极少的 4 类（`Adult`=4、`News`=64、
  `Reality-TV`=1、`Talk-Show`=2），保留 23 类：

  ```
  Drama, Comedy, Romance, Thriller, Crime, Action, Adventure, Horror,
  Documentary, Mystery, Sci-Fi, Fantasy, Family, Biography, War, History,
  Music, Animation, Musical, Western, Sport, Short, Film-Noir
  ```

  该列表定义在 `MMIMDB_GENRE_CLASSES`（`src/utils/core_tools.py`）。
- **预处理**（`init_data_mmimdb`，为复现本仓库补写）：
  1. 按 `split.json` 的 `train/dev/test` 遍历 `item_id`；
  2. 读取对应 json，取 `genres` 中**第一个落在 23 类内的类型**作为单标签
     `label`（下文有说明为何是单标签）；
  3. `text` 取 `plot` 列表中**最长的一段**简介（长文本通常信息更完整）；
  4. `img` = `{item_id}.jpeg`；
  5. 同时生成 `dataset/mmimdb/class_idx.json`（genre → 索引，23 项）；
  6. 存为 `dataset/mmimdb/{train,valid,test}.pkl`
     （对应 `dev`→`valid`，划分大小 `15552/2608/7799`，与 `split.json` 完全一致，
     没有样本因缺 genre / 缺文件被丢弃）。
- **⚠️ 多标签 vs 单标签的重要提醒**：论文原文明确说 MM-IMDb
  "formulated as a multi-label task"，用 **F1-Micro** 评测；但本仓库的训练代码
  （`ANGA.classifier` 接 `cross_entropy`，`get_evaluator` 统一用二分类
  `HatememesMetric`）是按**单标签分类**实现的。为了让预处理产物能直接喂给现有
  训练代码，`init_data_mmimdb` 取"主类型"（genres 列表第一项）作为折中方案。
  如果要严格复现论文报告的 MM-IMDb 数值，需要：
  1. 把 `label` 改成 23 维 multi-hot 向量；
  2. `compute_loss` 换成 `BCEWithLogitsLoss`；
  3. 实现一个多标签 F1-Micro 评测器替换 `HatememesMetric`。

---

## 3. Food101（UPMC Food-101，图文版）

- **来源**：[Wang et al., *Recipe Recognition with Large Multimodal Food Dataset*,
  ICME Workshop 2015]，即带网页标题文本的 UPMC Food-101（区别于纯图像的
  ETHZ Food-101）。
- **下载**：<https://www.kaggle.com/datasets/gianmarco96/upmcfood101>
- **放置**：
  ```
  dataset/food101/image/{class_name}_{idx}.jpg   # 90,704 张图，文件名含类别前缀
  dataset/food101/meta_data/train_titles.csv      # 67,972 行
  dataset/food101/meta_data/test_titles.csv       # 22,716 行
  dataset/food101/class_idx.json                  # {"frozen_yogurt": 0, "tacos": 1, ...} 101 类
  ```
  README 里强调的目录约定 `dataset/xxx/image/xxx.jpg|png|jpeg` 在这里体现为
  图片**直接平铺**在 `image/` 下（不是每类一个子文件夹）。
- **csv 格式**（无表头，三列）：
  ```
  apple_pie_851.jpg,Crock-Pot Ladies  Crock-Pot Apple Pie Moonshine,apple_pie
  apple_pie_140.jpg,Mom's Maple-Apple Pie Recipe | Taste of Home,apple_pie
  ```
  依次为：图像文件名、（从网页抓取的）标题文本、类别名。`train_titles.csv` +
  `test_titles.csv` 共 90,688 行，与论文报告的样本量一致（仓库里图片总数
  90,704 略多于此，属正常的少量冗余文件，预处理时会按 csv 行数为准）。
- **预处理**（`init_data_food101`，为复现本仓库补写）：
  1. 分别读取 `train_titles.csv` / `test_titles.csv`，`item_id` = 文件名去掉
     `.jpg`，`label` = `class_idx.json` 查到的类别索引，`text` = 标题；
     过滤掉类别名不在 `class_idx.json` 中或图像文件不存在的行（实测该数据集
     无此类脏数据，过滤后行数不变）；
  2. **UPMC Food-101 官方只有 train / test，没有 valid**：按类别分层
     （`groupby('label')`）从 `train` 里切出 `valid_ratio=5%`，随机种子固定为
     `seed=2024`，保证可复现；剩余部分作为新的 `train`；
  3. 存为 `dataset/food101/{train,valid,test}.pkl`。
- **注意**：`MemoryBankGenerator.run()` 里对 Food101 有特殊分支——**只对
  train 生成记忆库特征、不处理 valid**（因为原作者的 Food101 流水线没有 valid
  概念）；本仓库既然已经切出了 valid，若要让 valid 样本在训练时也能检索到
  自己以外的记忆库特征，需要相应调整该分支（详见
  [`docs/REPO_OVERVIEW.md`](REPO_OVERVIEW.md) 第 7 节）。

---

## 4. 预处理产物一览（三个数据集通用格式）

### 4.1 `dataset/{name}/{train,valid,test}.pkl`

| 阶段 | 新增列 | 说明 |
|---|---|---|
| ① 初始化 | `item_id, img, label, text` | 见上文各数据集 |
| ③ MCR 检索后追加 | `q_i, q_t` | 该样本自身的 CLIP 图像 / 文本查询向量（中间变量） |
| ③ MCR 检索后追加 | `i2i_id_list, i2i_sims_list, i2i_label_list` | 图像→图像检索到的 Top-K（代码里存 Top-20）邻居 id / 相似度 / 标签 |
| ③ MCR 检索后追加 | `t2t_id_list, t2t_sims_list, t2t_label_list` | 文本→文本检索到的邻居 id / 相似度 / 标签 |

### 4.2 `dataset/memory_bank/{name}/{text,image}/{item_id}.npy`

冻结 ViLT embedding 层输出（未过 encoder）：文本 `(128, 768)`，图像
`(145, 768)`，float32。每个样本约 0.84MB（文本+图像合计）。训练时按
`missing_type` 决定用 i2i 还是 t2t 通道的邻居 id，去这里取特征做重建
（`MMG`）和动态 prompt（`CAP`）。

### 4.3 `dataset/missing_table/{single,both}/{name}/missing_table.pkl`

由 `generate_missing_table` 生成，与具体数据集内容无关，只依据
`item_id` 数量随机采样：

- `single/`（`missing_type ∈ {Text, Image}`）：列 `missing_mask_{rate*10}`，
  `0` = 缺失，`1` = 不缺失。
- `both/`（`missing_type = Both`）：列 `missing_mask_{rate*10}`，
  `0` = 缺文本，`1` = 缺图像，`2` = 完整。缺失样本对半分配给"缺文本"和"缺图像"。

---

## 5. 磁盘占用参考

| 数据集 | 原始数据 | memory_bank | pkl（含检索列表） |
|---|---|---|---|
| HateMemes | ~4GB | ~7.5GB（train+valid ≈ 9000 样本） | 数十 MB |
| MM-IMDb | tar 8.2GB（解压后略大） | ~15GB（train+valid ≈ 18160 样本） | 数十 MB |
| Food101 | ~6–10GB | ~55GB（仅 train，≈ 65000+ 样本） | 数十 MB |

合计建议预留 **150–200GB** 可用磁盘空间。更详细的硬件建议（GPU/内存）见
[`docs/REPO_OVERVIEW.md`](REPO_OVERVIEW.md) 第 8 节。
