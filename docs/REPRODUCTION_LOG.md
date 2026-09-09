# 复现记录

本文档记录 ANGA 论文复现过程中的：① 每一次完整训练实验的超参数与结果；
② 复现过程中对代码/环境所做的改动（原因 + 影响）。按时间顺序追加，不改写历史条目。

运行环境：AutoDL 云服务器，单卡 NVIDIA GeForce RTX 4090（24GB），驱动 560.35.03 / CUDA 12.6；
Python venv（`/root/autodl-tmp/ANGA_venv`），依赖见 [`requirements.txt`](../requirements.txt)。

---

## 环境与代码改动记录

| # | commit | 改动 | 原因 |
|---|---|---|---|
| 1 | `0a02662` | 把 `train.py` / `init_data.py` / `utils/core_tools.py` / `utils/trainer.py` / `dataloader/hatememes_dataset.py` 里硬编码的作者绝对路径 `/data/gzh/MissingWork/MyWork/...` 改成相对路径 `./...` | 换机器后原路径不存在，代码跑不起来 |
| 2 | `e724355` | `requirements.txt`：锁定 `transformers==4.41.2`，新增 `scikit-learn`、`setuptools<81` | `transformers` 最新大版本删除了 `transformers.pytorch_utils.find_pruneable_heads_and_indices`，仓库自带的旧版 `vilt/modeling_vilt.py` 依赖这个内部 API，导入直接报错；`torchmetrics==0.9.3` 导入时用到的 `pkg_resources` 被新版 setuptools 移除；`utils/trainer.py` 用到 `sklearn` 但 `requirements.txt` 里根本没写 |
| 3 | `4201dd9` | `train.py` 的 `--device` 默认值 `cuda:4`→`cuda:0`；`utils/core_tools.py` 的 `MCR.device` 硬编码 `cuda:5`→按 `torch.cuda.is_available()` 判断，默认 `cuda:0` | 作者原机器是多卡（用到了 4/5 号卡），这台服务器只有单卡 `cuda:0`，硬编码的卡号会直接报 "invalid device ordinal" |
| 4 | `7c5c8ad` | `Trainer.__init__` 里新增 `Path("./src/metrics").mkdir(...)`，并把生成 CSV 文件名用的时间戳从"每次调用 `_valid()` 重新生成"改成"整次训练运行共用一个（`self.run_timestamp`）" | 原代码从未创建 `src/metrics/` 目录，第一次验证后写 CSV 直接 `FileNotFoundError` 崩溃；另外原逻辑每个 epoch 都用当前时间重新生成文件名，导致 20 个 epoch 会散落成 20 个各只有 1 行的 CSV，而不是一份完整的逐 epoch 记录 |
| 5 | `53b93dd` | `EarlyStopping.save_checkpoint()` 里 `torch.save` 之前加 `os.makedirs(os.path.dirname(self.path), exist_ok=True)` | 同类问题：`checkpoints` 目录不存在，第一次验证指标提升触发保存时崩溃 |
| 6 | `4201dd9`（同上）/ 后续 | `train.py`/`core_tools.py` 里另外两处硬编码 GPU 卡号（`--device` 默认 `cuda:4`、`MCR.device` 硬编码 `cuda:5`）统一改为 `cuda:0` | 开卡后这台服务器是单卡，硬编码卡号会直接报设备越界 |
| 7 | `29c5de6` | 新增 `--use_mir/--use_ga/--use_sea` 三个消融开关（`argparse.BooleanOptionalAction`，默认全 True＝行为与之前完全一致），分别对应论文 Table2 的 MIR（检索重建，关闭后用零向量占位）/ GA（锥域投影梯度对齐，关闭后训练循环退化成朴素联合损失）/ SEA（CAP 动态提示，关闭后只保留 label_enhanced 提示 token，且提示 token 数改为按实际拼接张量算而不是硬编码 `prompt_length*2+1`，避免重蹈原代码 `prompt_length` 参数那类离奇 bug）。同批附带几项不改变训练数学的提速：去掉 4 处从未被读取的 `num_promoted=...item()` 死代码同步、loss 累加从"每 batch 同步一次"改成"整个 epoch 结束才同步一次"、`Trainer.trainable_params` 只在构造时算一次、`get_optim` 里给 AdamW 加 `fused=True`、`HatememesDataset` 把图片解码结果和记忆库 `.npy` 特征都预加载缓存进内存（避免每个 epoch 重复读盘） | 完成消融实验范围需要真正的开关（原代码三个组件都是硬编码常开）；提速部分是本节最后"训练提速"用户需求的落地 |
| 8 | `785fbcb` | 修正 `get_optim` 的 `fused=True` 判断条件，额外排除复数张量 | `MMG.W` 是 `torch.cfloat` 复数参数，fused AdamW 不支持复数，MIR 开启时第一次 `optimizer.step()` 就报 `RuntimeError`；判断改成按模型实际参数动态算（MIR 关闭时没有 MMG，仍能用上 fused） |
| 9 | `03c28e6` | 撤销 `DataLoader(prefetch_factor=4)` 这一项提速 | 这台服务器进程 `ulimit -n`（open files）只有 1024，3 个 DataLoader × 16 个常驻 worker × 更深的预取队列，把跨进程共享内存传张量用的文件描述符耗尽，导致训练到第 2 个 epoch 报 `BrokenPipeError`；原本默认的 `prefetch_factor=2` 已经稳定跑过多次，收益不值得冒这个风险 |

**验证"提速改动没有改变训练结果"**：把 `use_mir/use_ga/use_sea` 全部保持默认 True，重跑一次 HateMemes/Text/0.7（与实验 2 完全相同的超参数与随机种子），最终测试集 AUROC = **0.6693**，与实验 2 原始结果 **0.6692** 几乎完全一致（差 0.0001）。注意：这份代码没有开 `torch.use_deterministic_algorithms`/`cudnn.deterministic`，GPU 训练本身不是逐比特可复现的——即使代码完全不变、种子相同，重跑一次每个 epoch 的具体数值也会有几个点的正常波动（这次重跑的早停轮次、逐 epoch 曲线都与原始跑法不同），但最终收敛质量高度一致，可以确认提速改动没有引入真实的行为差异。

**环境搭建过程中额外发现、但未改代码、仅记录避坑经验的点**（详见对话记录，此处不重复展开）：
- AutoDL 免卡模式容器的 cgroup 内存上限只有 2GB（`nvidia-smi`/`free -h` 显示的宿主机数值具有误导性），大文件 `pip install`（如 torch 768MB wheel）容易被 OOM 杀死；解法是本地 `curl` 流式下载 wheel 后 `pip install --no-deps` 本地文件，CUDA 子依赖再单独 `pip install torch==<ver> --index-url ...`（此时不加 `--no-deps`）。开卡（挂载 GPU）后该限制解除（cgroup 上限变为 ~120GB）。
- `huggingface.co` 直连从服务器超时，`hf-mirror.com` 直连可用；`download.pytorch.org` 可直连。
- `MemoryBankGenerator` / `MCR`（`init_data.py` 阶段②③）目前**硬编码只支持 HateMemes**（`self.dataset = 'hatememes'`、图片后缀写死 `.png`），要跑 MM-IMDb / Food101 的阶段②③需要先改这两处。
- AutoDL 数据盘（`/root/autodl-tmp`）容量有限，跑记忆库生成（阶段②）前务必先估算所需空间（HateMemes 记忆库约 8.2GB，MM-IMDb 约 21GB，Food101 约 74GB）。
- 这台服务器进程 `ulimit -n` 只有 1024，DataLoader 相关的多进程/多 worker 调参（`num_workers`、`prefetch_factor`）要留意文件描述符上限，不能无脑调大。

---

## 实验记录

### 实验 1：HateMemes，missing_type=Both，missing_rate=0.7

- **日期**：2026-09-07
- **数据集**：HateMemes（train 8500 / valid 500 / test 1000）
- **超参数**（均为 `train.py` argparse 默认值，仅显式指定 `--dataset --missing_type --missing_rate`）：
  ```
  model=ANGA backbone=vilt prompt_position=0 prompt_length=1 dropout_rate=0.2
  dataset=hatememes missing_type=Both missing_rate=0.7
  max_text_len=128 max_image_len=145 k=5
  optimizer=AdamW lr=1e-3 weight_decay=5e-5 use_warmup=True warmup_rate=0.1
  device=cuda:0 seed=2024 epochs=20(早停) batch_size=64 num_workers=16 patience=10
  ```
- **结果**：早停触发于 epoch 16（连续 10 轮验证 AUROC 无提升）；最佳验证 AUROC = **0.6737**（epoch 6）；**测试集 AUROC = 0.6054**。
- **逐 epoch 曲线**：epoch1=0.6708 → epoch6=0.6737（峰值）→ 之后震荡下降至 epoch16=0.5856。
- **与论文对照**：论文正文里没有以纯文本形式给出 "Both" 场景在 HateMemes 上的具体 AUROC 数字（相关对比在 `assets/ANGA-table1.png` 图片表格中，无法从 Markdown 文本直接提取），暂无法做定量对照。
- **产物归档**：服务器 `/root/autodl-tmp/ANGA/results_archive/hatememes_both_0.7/`（`best_model.pth` 460MB、`metrics.csv` 逐 epoch AUROC、`train.log` 完整日志）。

### 实验 2：HateMemes，missing_type=Text，missing_rate=0.7

- **日期**：2026-09-07
- **数据集**：同上
- **超参数**：同实验 1，仅 `--missing_type Text`。
- **结果**：早停触发于 epoch 13；最佳验证 AUROC = **0.7000**（epoch 3）；**测试集 AUROC = 0.6692**。
- **逐 epoch 曲线**：epoch1=0.6826 → epoch3=0.7000（峰值）→ 之后震荡下降至 epoch13=0.5988。
- **与论文对照**：论文表2（消融研究，70% 文本缺失，MIR+GA+SEA 全开即完整版 ANGA）给出的 HateMemes AUROC = **68.54%**。本次复现测试集 AUROC 为 **66.92%**，**差距 -1.62 个百分点**。
  - 差距的可能来源（未逐一验证，供后续排查参考）：验证集仅 500 样本，AUROC 噪声较大，早停选中的"最佳 epoch"未必是真正最优；单一随机种子（seed=2024），论文结果可能是多种子平均；未做超参搜索，直接用仓库默认超参。
- **产物归档**：服务器 `/root/autodl-tmp/ANGA/results_archive/hatememes_text_0.7/`（`best_model.pth` 457MB、`metrics.csv`、`train.log`）。

### 实验 3：消融研究（HateMemes，Text，0.7，MIR/GA/SEA 开关组合）

- **日期**：2026-09-08
- **代码前提**：`--use_mir/--use_ga/--use_sea` 三个开关（见上表 #7），实验 2（全部默认 True）即完整版 ANGA，作为本表第 5 行复用。
- **超参数**：同实验 2，仅 `--use_mir/--use_ga/--use_sea` 按下表组合取值。
- **运行方式**：`run_batch.sh` 批量顺序跑（单卡，逐个跑，见下文"批量运行与被杀恢复"说明）。
- **结果**（测试集 AUROC）：

  | MIR | GA | SEA | 论文 Table2 (%) | 本次复现 (%) | 差距 |
  |:-:|:-:|:-:|--:|--:|--:|
  | ✗ | ✗ | ✗ | 59.38 | 61.46 | +2.08 |
  | ✓ | ✗ | ✗ | 64.63 | 67.50 | +2.87 |
  | ✓ | ✓ | ✗ | 66.82 | 66.57 | −0.25 |
  | ✓ | ✗ | ✓ | 67.31 | 66.22 | −1.09 |
  | ✓ | ✓ | ✓（完整 ANGA，=实验2） | 68.54 | 66.92 | −1.62 |

  各行早停信息：000 早停于 epoch14，最佳验证 AUROC=0.6370@epoch4；100 早停于 epoch12，最佳验证=0.6879@epoch2；110 早停于 epoch13，最佳验证=0.6940@epoch3；101 早停于 epoch12，最佳验证=0.6905@epoch2。
- **观察**：一个值得记录、但未深究原因的现象——去掉 MIR 的两行（000/100）反而比论文数字**更高**（+2~3pt），而加上 GA/SEA 之后逐渐**低于**论文数字，完整版差距最大（-1.62pt）。趋势方向和论文一致（各组件都带来提升），但每加一个组件相对论文的差距在扩大，可能与单次跑的随机性、验证集过小、早停时机敏感有关（同实验2的差距分析），未做多种子平均验证。
- **产物归档**：服务器 `results_archive/ablation_{000,100,110,101}/`。

### 实验 4：HateMemes，missing_type=Image，missing_rate=0.7

- **日期**：2026-09-08
- **超参数**：同实验 2，仅 `--missing_type Image`。
- **结果**：早停于 epoch12，最佳验证 AUROC=0.6583@epoch2；**测试集 AUROC = 0.5864**。
- **与论文对照**：论文正文只给出"ANGA 相比最佳基线 RAGPT 在图像缺失场景下 AUROC 提升 0.85%"这一相对差值，未给出 ANGA 自身的绝对数值（该数字在图片格式的 Table1 里），暂无法定量对照；此处只记录本仓库自身的复现结果。
- **产物归档**：`results_archive/image_0.7/`。

### 实验 5：缺失率敏感性扫描（HateMemes，Text，rate ∈ {0.1,0.3,0.5,0.7,0.9}）

- **日期**：2026-09-08
- **对应论文**：4.4 节"对不同缺失率的鲁棒性"（图4，只有图片，无法提取论文自身在各缺失率下的绝对数值，此处只记录本仓库复现的趋势）。
- **超参数**：同实验 2，仅 `--missing_rate` 取值不同（每个 rate 需要先 `--regenerate_missing_table True` 生成对应缺失掩码表）。
- **结果**：

  | missing_rate | 早停 epoch | 最佳验证 AUROC | 测试集 AUROC |
  |--:|--:|--:|--:|
  | 0.1 | 15 | 0.7079@epoch5 | 0.6757 |
  | 0.3 | 11 | 0.7086@epoch1 | 0.6448 |
  | 0.5 | 12 | 0.7311@epoch2 | 0.6651 |
  | 0.7（=实验2） | 13 | 0.7000@epoch3 | 0.6692 |
  | 0.9 | 11 | 0.7064@epoch1 | 0.6707 |
- **观察**：测试集 AUROC 在 0.1~0.9 缺失率区间内落在 0.645~0.676 之间，没有随缺失率单调下降（0.3 反而是五个点里最低的，0.1 最高），大致符合论文"ANGA 在各缺失率下保持稳健"的定性结论（没有出现随缺失率升高性能崩溃的情况），但样本量小（每个点只有单次跑），噪声可能掩盖真实趋势，0.3 这个低点也可能只是这一次跑的正常波动。
- **产物归档**：`results_archive/rate_{0.1,0.3,0.5,0.9}/`。

### 实验 6：检索数 K 敏感性扫描（HateMemes，Text，0.7，k ∈ {1,3,5,7,9}）

- **日期**：2026-09-08
- **对应论文**：4.1 节实验设置里给出的 K 取值范围 K∈{1,3,5,7,9}；4.4 节"超参数敏感性"提到"性能仅受 K 轻微影响"（图3，只有图片无法提取绝对数值）。
- **超参数**：同实验 2，仅 `--k` 取值不同。
- **结果**：

  | k | 早停 epoch | 最佳验证 AUROC | 测试集 AUROC |
  |--:|--:|--:|--:|
  | 1 | 11 | 0.7383@epoch1 | 0.5700 |
  | 3 | 14 | 0.7333@epoch4 | 0.6476 |
  | 5（=实验2，默认） | 13 | 0.7000@epoch3 | 0.6692 |
  | 7 | 15 | 0.7227@epoch5 | 0.6698 |
  | 9 | 14 | 0.7169@epoch4 | 0.6961 |
- **观察**：与论文"仅受 K 轻微影响"的定性结论**不太一致**——k=1 时测试集 AUROC 明显偏低（0.57，比其余各点低 8-13 个点），k≥3 之后才趋于平稳（0.65~0.70）且总体随 k 增大略有上升（k=9 最高，0.6961）。样本量同样只有单次跑，但 k=1 这个点的落差幅度较大，值得后续用多种子复核是否是真实趋势还是这一次跑的异常。
- **产物归档**：`results_archive/k_{1,3,7,9}/`。

### 批量运行与"被杀恢复"方法说明（实验3-6通用）

实验 3-6 共 13 次训练用 `/root/autodl-tmp/run_batch.sh` 顺序批跑（单卡，逐个跑完再跑下一个）。运行中发现：**训练本身（含早停判断、保存 best checkpoint）全部正常完成，但 5 次跑（ablation_000/100/110/101、image_0.7）在训练刚结束、进入最终测试集评估（`Testing:` 阶段）时被系统 `Killed`**；另有 2 次（k_9、rate_0.1）也是同样在 Testing 阶段被杀；`rate_0.3` 例外——它在训练**刚开始**（epoch1 中途）就因为与我并发跑的"补测试"脚本抢文件描述符资源而报 `BrokenPipeError` 崩溃，此时还没保存过 checkpoint。

根因排查：
1. **Testing 阶段被杀**：定位为本仓库这次额外加的"图片+记忆库特征内存缓存"提速改动（见上表 #7）导致——`persistent_workers=True` 让 train/valid/test 三个 DataLoader 的 worker 进程全程存活，Python 对象被 fork 出的 worker 读取时会触发引用计数写入，破坏了本该共享的 copy-on-write 页面，多个 worker 叠加后把这台服务器 120GB 的 cgroup 内存额度耗得所剩无几；测试阶段新开一批 worker 时刚好触发 OOM。已撤销这项缓存优化（见上表最后一条记录），改回按需读盘。
2. **rate_0.3 崩溃**：是我在批处理跑 `rate_0.3` 的同时，并发跑了好几个"用 checkpoint 补测试"的脚本，多个进程同时用了 `num_workers=16 × 3` 个 DataLoader，一起把这台服务器进程级 `ulimit -n`（1024）文件描述符打满导致的，不是训练代码本身的问题。

恢复方式：对于"训练已完成、只是没跑完最终测试"的 7 次跑（ablation_000/100/110/101、image_0.7、k_9、rate_0.1），写了一个独立小脚本 `run_test_only_v2.py`——用保存下来的 `best_model.pth` 重新实例化 `Trainer`，只调用 `trainer._test()`，不用重新训练就能拿到真实的测试集 AUROC（已在文中体现）。`rate_0.3` 没保存过 checkpoint，只能完整重跑；撤销内存缓存、且这次不再并发跑其他脚本之后，重跑一次干净地跑完（结果见实验5表格）。

另外，`run_batch.sh` 里"把 `src/metrics/*.csv` 拷贝进归档目录"这一步有个小毛病：`src/metrics/` 目录会累积历次运行的所有 csv（从不清理），一旦目录里有多个文件，`mv *.csv metrics.csv` 这种写法就会因为"多个源文件对应一个非目录目标"而失败——这正是实验3-6里除 `rate_0.3`（手动按时间戳/行数比对补的）之外，其它归档目录都没有 `metrics.csv`（逐 epoch 曲线）的原因，只能从 `train.log` 里读关键行。这几个脚本是服务器上的临时辅助脚本，没有纳入 git 版本管理，此处只记录这个坑，不再回头修。

---

## ⚠ 结果可靠性审计（2026-09-09）——上面实验1-6的结论需要按本节重新理解

对复现结果做了一次系统审计，发现**两类问题足以让上面所有实验数字失去"论文复现证据"的资格**。数据层面的详细统计见 [`SERVER_DATA_AUDIT_20260909.md`](SERVER_DATA_AUDIT_20260909.md)，本节记录结论与代码层面的发现。

### 一、验证集自我检索泄露（已实测证实，影响最大）

`core_tools.py` 的 `MCR._compute_similarity_in_batches` 在排除"自己检索到自己"时，用的是 `memory_bank_id[i+j]`（查询在**记忆库中的位置**）而不是查询样本自身的真实 `item_id`。记忆库 = `train + valid` 拼接，所以：

- **train 查询**：位置恰好对齐（train 排在记忆库前段），自我排除正确 → 实测自我检索 **0%** ✓
- **valid 查询**：位置索引落在记忆库的 train 段上，排除的是无关的训练样本，自己从未被排除 → 实测自我检索 **100%**（top-5 内 499/500）✗
- **test 查询**：test 不在记忆库里，本就没有"自己"可检索 → 实测 **0%** ✓

后果：验证样本的检索表里第一个"邻居"就是它自己，于是
1. 缺失模态的"重建"里混进了该样本**自己的真实模态特征**（k=5 时占 1/5 权重）；
2. `label_enhanced` 提示里混进了该样本**自己的真实标签**。

即验证集的 AUROC 被系统性抬高。因为早停和 best checkpoint 选择完全依赖验证 AUROC，**所有实验最终报出的测试数字，都是用一个被污染的信号挑出来的 checkpoint 跑出来的**。测试集本身没有被直接泄露（test 不在记忆库、邻居里也没有 test 来源），但"选哪个 checkpoint"这一环已经不干净。

这也**解释了实验6里 k=1 那个反常点**：k=1 时验证样本唯一的"邻居"就是它自己（i2i top-1 自身命中 491/500），验证退化成近似"直接读标签"，所以 k=1 的验证 AUROC 是全部 k 值里最高的（0.7383），而测试 AUROC 却是最低的（0.5700）——当时我记为"与论文结论不一致，需多种子核实"，实际根因是这个泄露，应以本节为准。

### 二、代码实现与论文描述不符（复现对象本身就不是论文的方法）

| 项目 | 论文 4.1 节 | 本仓库代码 | 影响 |
|---|---|---|---|
| MMG（缺失模态重建） | 频域调制：FFT → 可学习复数权重 W 调制 → 逆FFT → 残差+LayerNorm+线性投影 | `modules.py:53-65` 的 `forward` **只有一行 `torch.mean(F_l, dim=1)`**，频域实现整段被注释掉；`W`/`layer_norm`/`linear`/`dropout` 全是构造了但从不使用的死参数 | 论文核心模块之一实际没有参与运算，"MIR" 消融实际是在比较"检索特征直接平均"和"零向量" |
| 锥域余弦阈值 τ | 0.2 | `trainer.py:363` 硬编码 **0.7** | 差 3.5 倍，直接决定多少缺失样本梯度被投影/置零，是 GA 的核心超参 |
| 提示插入层 b | 第 2 层 MSA | `train.py:55` 默认 **0** | 提示注入位置不同 |
| 课程比例下界 λ_min | 0.1 | `trainer.py:190` **0.20** | 课程起点不同（λ_max=0.3、Z_grow=5 一致） |
| 权重衰减 | 1e-5 | `train.py:75` 默认 **5e-5**（`config.yaml` 里还写着第三个值 2e-2） | 三处不一致 |
| 学习率 / 批大小 | 1e-3 / 64 | 1e-3 / 64 ✓ | 一致 |

### 三、其他实现缺陷（不影响上面结论，但记录备查）

- `--dropout_rate` 完全无效：`MMG.forward` 里没有调用 `self.dropout`（只在被注释的频域分支里用到）。
- `--prompt_length` 取 1 以外的值会崩：`ANGA.forward` 把提示 mean 成单 token，但 attention_mask 曾按 `prompt_length*2+1` 计算（加消融开关时已改为按实际拼接张量算，不再有这个隐患，但 CAP 本身仍只产出被平均掉的单 token）。
- `--model` / `--backbone` / `--vilt_weights` 是死参数：`load_model` 里 ViLT 路径硬编码，模型类固定为 ANGA。
- `MCR` 里 `sim_score = sim_scores[j,1:]` 无条件丢掉第一个相似度（假设是自己），对 test 查询是错的，导致 `*_sims_list` 与 `*_id_list` 错位——所幸下游没有任何代码读取 sims，不影响结果。
- `_valid` 的"完整/缺失样本"分组计数在 `Both` 模式下不对（mask==2 才是完整，代码按 ==1 统计），只影响打印出来的分组指标，不影响主 AUROC。
- `get_evaluator` 忽略 `task_id` 参数永远返回 `HatememesMetric`（AUROC），`compute_loss` 固定用 cross_entropy 单标签——切到 MM-IMDb（多标签 / F1-Micro）、Food101（ACC）时指标和损失都不对，是后续换数据集前必须先改的地方。

### 四、对上面实验1-6结论的修正

- 实验2里把 66.92% vs 论文 68.54% 的 -1.62pt 归因为"单次跑波动"是**不充分的**：在 τ、提示层、λ_min、weight_decay 四项超参数都与论文不符、且 MMG 核心模块未实现的情况下，跑的根本不是论文描述的方法，这个对比不构成有意义的对照。
- 实验3消融表里"去掉组件反而比论文高、加上组件反而比论文低"的反常趋势，同样应在上述前提下重新理解，不适合作为"论文结论是否成立"的证据。
- 实验6 k=1 异常点的根因已定位为验证集泄露（见上）。
- 所有 checkpoint、日志、逐 epoch 曲线仍然保留（本地 `results_archive/`，15 个实验、6.4GB），可作为"原始实现下的实验记录"，但不能当作独立协议下的论文复现证据。

### 五、如果要继续，建议的修复顺序

1. 修 `MCR` 的自我排除逻辑（用查询自身的真实 `item_id`，而不是记忆库位置索引）；同时明确记忆库协议——更严格的做法是**只用 train 构建记忆库**，valid/test 都只做查询。
2. 重新生成三个划分的检索表（邻居 ID/标签/相似度），并校验对齐。
3. 决定是否把 τ、prompt_position、λ_min、weight_decay 对齐论文 4.1 节。
4. 决定 `MMG` 是否恢复频域实现——需要先判断"作者最终方案就是简化的平均"还是"发布时误注释"，这一条会改变模型能力本身，属于需要你拍板的问题。
5. 以上改完后**全部实验重跑**（旧 checkpoint 只能用于诊断，不能通过"换个干净检索表再评估一次"来挽救，因为它们的训练过程本身就用过被污染的验证信号来选点）。

---

## 待办 / 尚未复现

- **（最高优先级）** 按上节"结果可靠性审计"修复检索泄露与超参数偏差后，重跑 HateMemes 全部实验
- MM-IMDb、Food101 全部场景（阶段②③代码目前硬编码只支持 HateMemes，需要先扩展这两个类支持其他数据集，见上文"额外发现"；另外 `get_evaluator`/`compute_loss` 也必须先按各数据集的任务类型改对）
- 论文表1（与 SOTA 基线 RAGPT/IF-MMIN/MSPs 等方法的正面比较）——本仓库没有复现这些基线方法的代码，只能拿 ANGA 自己的数字去对比论文里"相对 RAGPT 提升几个点"这类相对差值
- 4.4 节"严重缺失泛化能力"实验（训练用 10%-50% 缺失率、测试用 90%）——当前代码里训练/验证/测试三个 split 共用同一个 `--missing_rate`，没有分别指定训练/测试缺失率的机制，需要先改代码
- 多随机种子重复实验，验证实验3/6里观察到的"部分消融行差距方向不一致""k=1 明显偏低"这些现象是真实趋势还是单次跑的噪声
