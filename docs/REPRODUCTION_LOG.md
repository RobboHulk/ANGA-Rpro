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

---

## 待办 / 尚未复现

- HateMemes，missing_type=Image，missing_rate=0.7（论文表2 应该也有该场景的完整版数字，本仓库暂未跑）
- MM-IMDb、Food101 全部场景（阶段②③代码需要先扩展支持这两个数据集，见上文"额外发现"）
- 论文表1（与 SOTA 基线 RAGPT/IF-MMIN/MSPs 等的比较）、消融变体（关闭 MIR/GA/SEA 中的一个或多个）、不同缺失率（10%–90%）等尚未逐一复现
