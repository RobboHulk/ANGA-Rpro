"""
================================================================================
训练器（Trainer）：课程学习 + 锚点引导的梯度对齐
================================================================================
本文件实现 ANGA 框架的训练流程，是论文"Anchor-Guided Gradient Alignment"
（锚点引导的梯度对齐）思想的核心落地代码。

整体流程（run 方法）：
    每个 epoch 依次执行：
        1. 课程学习（curriculum）：按一定比例挑选"可靠补全样本"（熵最低的
           缺失样本），把它们视作"锚点/完备样本"参与后续梯度对齐；
        2. 训练（_train）：对每个 batch 执行"梯度拆分 + 锥域投影对齐"；
        3. 验证（_valid）：计算 AUROC 等指标，并做早停；
    训练结束后加载最佳权重并在测试集上评测（_test）。

核心算法 —— 梯度对齐（在 _train 中）：
    - 把损失拆成两部分：
        L_C：完备样本（complete，含可靠补全样本）的损失
        L_M：缺失样本（missing）的损失
    - 分别反传得到梯度 g_C 与 g_M；
    - 以 g_C 归一化后的方向 a_t 作为"优化锚点（anchor）"；
    - 把 g_M 投影到"与锚点夹角不超过阈值（tau=0.7 的锥域）"内，得到 g̃_M，
      避免缺失样本的梯度方向与完备样本冲突（缓解学习不平衡）；
    - 最终用 g_total = g_C + g̃_M 更新参数。

课程学习 —— 熵驱动课程（entropy-driven curriculum，在 run / _ranked_missing_samples 中）：
    - 对缺失样本计算预测分布的熵，熵越低表示模型越"确信"，样本越可靠；
    - 每个 epoch 按熵升序（可靠度降序）取一定比例（ratio 从 0.20 线性增长到
      0.30）的缺失样本作为"可靠补全样本"，纳入锚点集合。
================================================================================
"""
import sys, os, copy, time, csv
import numpy as np
from datetime import datetime
import pickle
import torch.nn.functional as F
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from colorama import Back, Fore, Style
from .core_tools import (
    get_optim,
    print_init_msg,
    load_model,
    get_dataset,
    compute_loss,
    get_evaluator,
    Collator,
    EarlyStopping
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.metrics import roc_auc_score, accuracy_score
import random


class Trainer():
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.dataset = args.dataset
        self.device = args.device
        self.missing_type = args.missing_type
        self.task = args.dataset
        self.save_path = args.save_path

        # ---- 加载 ANGA 模型（backbone 为冻结的预训练 ViLT）----
        self.model = load_model(
            missing_type=args.missing_type,
            task_id = self.task,
            device = args.device,
            max_text_len = args.max_text_len,
            max_image_len = args.max_image_len,
            model=args.model,
            backbone=args.backbone,
            vilt_weights=args.vilt_weights,
            prompt_position=args.prompt_position,
            prompt_length=args.prompt_length,
            dropout_rate=args.dropout_rate)
        self.model.to(self.device)


        # ---- 构建 train / valid / test 三个 Dataset ----
        train_dataset = get_dataset(
            dataset_name=self.dataset,
            split='train',
            dataset=args.dataset,
            missing_type=args.missing_type,
            missing_rate=args.missing_rate,
            max_text_len=args.max_text_len,
            max_image_len=args.max_image_len,
            k=args.k
        )

        valid_dataset = get_dataset(
            dataset_name=self.dataset,
            split='valid',
            dataset=args.dataset,
            missing_type=args.missing_type,
            missing_rate=args.missing_rate,
            max_text_len=args.max_text_len,
            max_image_len=args.max_image_len,
            k=args.k
        )

        test_dataset = get_dataset(
            dataset_name=self.dataset,
            split='test',
            dataset=args.dataset,
            missing_type=args.missing_type,
            missing_rate=args.missing_rate,
            max_text_len=args.max_text_len,
            max_image_len=args.max_image_len,
            k=args.k
        )

        # 自定义批处理函数，将原始样本列表统一格式化、编码成张量，供模型直接使用
        collator = Collator(max_text_len=args.max_text_len)

        # ---- 三个 DataLoader：训练集 shuffle，验证/测试集不 shuffle ----
        self.train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            shuffle=True
        )

        self.valid_data_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            shuffle=False
        )

        self.test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            shuffle=False
        )

        # ---- 优化器与学习率调度器 ----
        self.optimizer, self.scheduler = get_optim(
            max_steps=len(self.train_data_loader) * self.epochs,
            model=self.model,
            name=args.name,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_warmup=args.use_warmup,
            warmup_rate=args.warmup_rate
        )

        self.evaluator = get_evaluator(self.dataset, args.device)
        self.early_stopper = EarlyStopping(patience=args.patience, delta=0.0, path=os.path.join(self.save_path, "best_model.pth"))

    def run(self):
        """训练主循环：课程学习 → 训练 → 验证/早停 → 最终测试。"""

        best_score = -1.0  # 当前验证集上最佳 AUROC
        best_epoch = -1
        REFRESH_EVERY = 5  # 每 5 个 epoch 重新评估一次"可靠补全样本"的排序
        pair_ent_id = None  # 保存 (熵, 样本id) 的排序列表

        # 课程表参数（可按需改）
        ratio_start = 0.20  # warmup 之后的起始比例（可靠样本占缺失样本的比例）
        ratio_end = 0.30    # 最终比例
        grow_epochs = 5     # 用多少个 epoch 从 start 线性涨到 end

        for epoch in range(self.epochs):
            print(f"{Fore.RED}Current Epoch: {epoch + 1}{Style.RESET_ALL}")

            # 是否需要刷新：首次或每隔 REFRESH_EVERY 个 epoch 重新排序可靠样本
            need_refresh = (pair_ent_id is None) or (epoch % REFRESH_EVERY == 0)
            if need_refresh:
                pair_ent_id = self._ranked_missing_samples()

            # ---- 线性课程：ratio 从 ratio_start 线性增长到 ratio_end ----
            progress = epoch
            t = min(progress / max(1, grow_epochs), 1.0)
            ratio = ratio_start + (ratio_end - ratio_start) * t
            # 取熵最低（最可靠）的前 ratio 比例的缺失样本 id，作为"锚点集合"
            k = max(1, int(len(pair_ent_id) * ratio)) if pair_ent_id else 0
            reliable_ids = set(id_ for _, id_ in pair_ent_id[:k]) if k > 0 else set()
            print(f"{Fore.RED}Ratio={ratio:.2f}, Reliable_ids={len(reliable_ids)}{Style.RESET_ALL}")
            self._train(reliable_ids)

            # ========== 验证 and 早停 ==========
            val_metrics = self._valid(current_epoch=epoch + 1)
            val_score = float(val_metrics['auroc'])

            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch + 1
            stop = self.early_stopper(val_score, self.model)

            if stop:
                print(f"{Fore.RED}Early stopping at epoch {epoch + 1}. Best AUROC={best_score:.4f} @ epoch {best_epoch}.{Style.RESET_ALL}")
                break

        # ===== 训练结束：加载最佳权重，进行测试=====
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, "best_model.pth"), map_location=self.device))
        self._test()

    def _train(self, reliable_ids=None):
        """
        训练一个 epoch，核心是"锚点引导的梯度对齐"。

        参数：
            reliable_ids : 本 epoch 被课程学习选中的"可靠补全样本"id 集合；
                           这些样本虽然在原始数据里是缺失的，但模型对它们的
                           预测已经很确信（熵低），因此被"提升"为完备样本，
                           参与锚点（g_C）的构建。
        """
        loss_list =  []
        self.model.train()
        pbar = tqdm(self.train_data_loader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}", desc='Training')
        count_zero, count_inside, count_project = 0, 0, 0  # 统计三种梯度投影情况

        for batch in pbar:
            # ---- 数据搬到设备，并分离出标签与样本 id ----
            inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
            labels = inputs.pop('label')
            batch_ids = inputs.pop('id')
            missing_mask = inputs['missing_mask']
            preds = self.model(**inputs)

            # =================================================================
            # 1. 计算每个样本的损失（逐样本），并划分"完备集"与"缺失集"
            # =================================================================
            if self.missing_type == "Text" or self.missing_type == "Image":
                # ---- 单模态缺失：mask==1 为完备，mask==0 为缺失 ----
                per_loss = compute_loss(preds, labels, reduction='none')
                idx_complete = (missing_mask == 1)
                idx_missing = (missing_mask == 0)

                if reliable_ids is None or len(reliable_ids) == 0:
                    # 无课程学习：完备/缺失直接按原始 mask 划分
                    idx_complete_eff = idx_complete
                    idx_missing_eff = idx_missing
                    num_promoted = 0
                else:
                    # 有课程学习：把"缺失但被标记为可靠"的样本提升为完备样本
                    in_set = [i in reliable_ids for i in batch_ids]
                    reliable_mask = torch.tensor(in_set, device=self.device, dtype=torch.bool)
                    # 有效完备集 = 原始完备 ∪（缺失 ∩ 可靠）
                    idx_complete_eff = idx_complete | (idx_missing & reliable_mask)
                    # 有效缺失集 = 缺失 ∩ 非可靠
                    idx_missing_eff = idx_missing & (~reliable_mask)
                    num_promoted = (idx_missing & reliable_mask).sum().item()  # 单个batch中由补全样本转为完备样本的个数

                # 分别求两类样本的平均损失（若某类为空则置 None）
                LC = per_loss[idx_complete_eff].mean() if idx_complete_eff.any() else None
                LM = per_loss[idx_missing_eff].mean() if idx_missing_eff.any() else None
                loss_list.append(((0.0 if LC is None else float(LC)) + (0.0 if LM is None else float(LM))))

            elif self.missing_type == "Both":
                # ---- 双模态缺失：mask==2 完备，mask==1 缺图像，mask==0 缺文本 ----
                per_loss = compute_loss(preds, labels, reduction='none')
                idx_complete = (missing_mask == 2)
                idx_missing_image = (missing_mask == 1)
                idx_missing_text = (missing_mask == 0)
                # 缺失样本并集（缺文本 ∪ 缺图像）
                idx_missing_union = idx_missing_image | idx_missing_text

                if reliable_ids is None or len(reliable_ids) == 0:
                    idx_complete_eff = idx_complete
                    idx_missing_eff = idx_missing_union
                    num_promoted = 0
                else:
                    in_set = [i in reliable_ids for i in batch_ids]
                    reliable_mask = torch.tensor(in_set, device=self.device, dtype=torch.bool)
                    # 有效完备集 = 原始完备 ∪（缺失并集 ∩ 可靠）
                    idx_complete_eff = idx_complete | (idx_missing_union & reliable_mask)
                    # 有效缺失集 = 缺失并集 ∩ 非可靠
                    idx_missing_eff = idx_missing_union & (~reliable_mask)
                    num_promoted = (idx_missing_union & reliable_mask).sum().item()  # 单个batch中由补全样本转为完备样本的个数

                LC = per_loss[idx_complete_eff].mean() if idx_complete_eff.any() else None
                LM = per_loss[idx_missing_eff].mean() if idx_missing_eff.any() else None
                loss_list.append(((0.0 if LC is None else float(LC)) + (0.0 if LM is None else float(LM))))

            # =================================================================
            # 2. 梯度拆分：分别求完备样本梯度 g_C 与缺失样本梯度 g_M
            # =================================================================
            trainable = [p for p in self.model.parameters() if p.requires_grad]

            # 先清零，反传 L_C 得到 g_C（retain_graph 保留计算图供后面 L_M 使用）
            self.optimizer.zero_grad(set_to_none=True)
            if LC is not None:
                LC.backward(retain_graph=True)
            gC = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in trainable]

            # 再次清零，反传 L_M 得到 g_M（最后一次反传无需保留图）
            self.optimizer.zero_grad(set_to_none=True)
            if LM is not None:
                LM.backward()
            gM = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in trainable]

            # =================================================================
            # 3. 锚点构建：以 g_C 的方向作为优化锚点 a_t
            # =================================================================
            with torch.no_grad():
                # 3.1 计算 gC 的整体范数（把各层梯度看成一个拼接大向量）
                gC_sqsum = torch.tensor(0.0, device=self.device)
                for gc in gC:
                    gC_sqsum += (gc.float() ** 2).sum()
                a_norm = gC_sqsum.sqrt().clamp_min(1e-12)   # 防止除零

                # 3.2 归一化得到主轴各参数分量（与 trainable 一一对应）
                a_t = [gc / a_norm for gc in gC]

            # =================================================================
            # 4. 锥域投影：把 g_M 投影到以 a_t 为轴、阈值 cos>=tau 的锥域内
            #    目的：保证缺失样本的梯度与完备样本梯度方向"足够一致"，
            #    避免两者冲突造成优化震荡（即缓解学习不平衡）。
            # =================================================================
            tau = 0.7    # 夹角余弦阈值（锥域边界）
            eps = 1e-12  # 数值稳定项

            # 4.1 计算 cos(gM, a_t) = (gM·a_t) / (||gM|| * ||a_t||)，a_t 是单位向量
            dot = torch.tensor(0.0, device=self.device)
            gm_sq = torch.tensor(0.0, device=self.device)
            for gm, ah in zip(gM, a_t):
                dot += (gm * ah).float().sum()   # 分子：内积
                gm_sq += (gm.float() ** 2).sum() # 分母：||gM||^2
            gm_norm = gm_sq.sqrt().clamp_min(eps)
            cos_gm_a = (dot / gm_norm).clamp(-1.0, 1.0)  # 一个标量值

            # 4.2 分三种情况处理 g_M
            if cos_gm_a <= 0:  # 情况①：反向（与锚点夹角 > 90°）→ 直接置零
                gM_proj = [torch.zeros_like(gm) for gm in gM]
                count_zero += 1

            elif cos_gm_a >= tau:  # 情况②：已在锥内（夹角足够小）→ 不动
                gM_proj = gM
                count_inside += 1

            else:
                # 情况③：介于 0 与 tau 之间 → 投影到锥边界
                # 平行分量 g_parallel = (gM·a)a；注意 a_t 是单位向量
                g_par = [dot * ah for ah in a_t]

                # 垂直分量 g_perp = gM - g_parallel
                g_perp = [gm - gp for gm, gp in zip(gM, g_par)]

                # 允许上限 t_max = sqrt(1 - tau^2) / tau
                # （由三角关系推导：把向量投影到锥边界时垂直分量的缩放系数）
                t_max = torch.sqrt(torch.tensor(1.0, device=self.device) - tau ** 2) / (tau + eps)

                # ||g_parallel|| = |gM·a| （因为 a 是单位向量）
                gpar_norm = torch.abs(dot)

                # ||g_perp||
                gperp_sq = torch.tensor(0.0, device=self.device)
                for gp in g_perp:
                    gperp_sq += (gp.float() ** 2).sum()
                gperp_norm = gperp_sq.sqrt().clamp_min(eps)

                # s = t_max * ||g_parallel|| / (||g_perp|| + eps)：缩放系数
                s = t_max * gpar_norm / (gperp_norm + eps)

                # g̃ = g_parallel + s * g_perp（把垂直分量缩放到锥边界）
                gM_proj = [gp_par + s * gp_perp for gp_par, gp_perp in zip(g_par, g_perp)]
                count_project += 1

            # =================================================================
            # 5. 写回总梯度并更新：g_total = gC + g̃M
            # =================================================================
            self.optimizer.zero_grad(set_to_none=True)
            for p, gc, gm in zip(trainable, gC, gM_proj):
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(gc + gm)   # 对齐后的缺失梯度 + 完备梯度
            self.optimizer.step()
            self.scheduler.step()

        print(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list):.4f}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Stats: Zero={count_zero}, Inside={count_inside}, Project={count_project}{Style.RESET_ALL}")

    def _valid(self, current_epoch=None):
        """验证阶段：在 valid 集上计算 AUROC/ACC（区分完整样本与缺失样本）。"""
        self.model.eval()
        loss_list = []

        missing_cnt, complete_cnt = 0, 0
        # 三个评测器副本：全部样本 / 只完整样本 / 只缺失样本
        eva_all = copy.deepcopy(self.evaluator); eva_all.reset()
        eva_complete = copy.deepcopy(self.evaluator); eva_complete.reset()
        eva_missing = copy.deepcopy(self.evaluator); eva_missing.reset()

        self.test_data_loader

        pbar = tqdm(self.valid_data_loader, bar_format=f"{Fore.YELLOW}{{l_bar}}{{bar}}{{r_bar}}", desc='Validating')

        with torch.no_grad():
            for batch in pbar:
                missing_mask = batch["missing_mask"].to(self.device)
                # 统计缺失样本与完整样本数量（注意：这里按 mask==0 计缺失、mask==1 计完整，
                # 对 Both 情况只是近似统计，用于打印展示）
                missing_cnt += int((missing_mask == 0).sum())
                complete_cnt += int((missing_mask == 1).sum())

                # ---- 前向 ----
                inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
                labels = inputs.pop('label')
                batch_ids = inputs.pop('id')
                preds = self.model(**inputs)
                loss = compute_loss(preds, labels)
                loss_list.append(loss.item())

                # ---- 更新 evaluator ----
                eva_all.update(preds=preds, labels=labels)

                # 分别统计完整样本与缺失样本的指标
                idx_complete = (missing_mask == 1).nonzero(as_tuple=True)[0]
                idx_missing = (missing_mask == 0).nonzero(as_tuple=True)[0]
                if idx_complete.numel():
                    eva_complete.update(preds=preds[idx_complete], labels=labels[idx_complete])
                if idx_missing.numel():
                    eva_missing.update(preds=preds[idx_missing], labels=labels[idx_missing])

        # ---- 计算指标 ----
        metrics_all = eva_all.compute()
        metrics_complete = eva_complete.compute() if complete_cnt > 0 else {"auroc": float("nan"), "acc": float("nan")}
        metrics_missing = eva_missing.compute() if missing_cnt > 0 else {"auroc": float("nan"), "acc": float("nan")}

        print(f"{Fore.YELLOW}Valid: Loss: {np.mean(loss_list):.4f}{Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}AUROC (all)      {metrics_all['auroc']:.4f} | ACC (all)      {metrics_all['acc']:.4f}{Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}AUROC (complete) {metrics_complete['auroc']:.4f} | ACC (complete) {metrics_complete['acc']:.4f}{Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}AUROC (missing)  {metrics_missing['auroc']:.4f} | ACC (missing)  {metrics_missing['acc']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}AUROC (all)      {metrics_all['auroc']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}AUROC (complete) {metrics_complete['auroc']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}AUROC (missing)  {metrics_missing['auroc']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Valid: 完整样本 {complete_cnt}，缺失样本 {missing_cnt}{Style.RESET_ALL}")

        # ---- 保存指标到 CSV 日志（按时间戳命名） ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"./src/metrics/metrics_{timestamp}.csv")

        if not log_path.exists():
            with open(log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "auroc_all", "auroc_complete", "auroc_missing"])

        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                current_epoch if current_epoch is not None else -1,
                metrics_all["auroc"],
                metrics_complete["auroc"],
                metrics_missing["auroc"],
            ])

        return metrics_all

    def _test(self):
        """测试阶段：在 test 集上计算 AUROC（加载最佳权重后调用）。"""
        self.model.eval()

        eva = copy.deepcopy(self.evaluator); eva.reset()

        pbar = tqdm(self.test_data_loader, bar_format=f"{Fore.RED}{{l_bar}}{{bar}}{{r_bar}}", desc='Testing')

        with torch.no_grad():
            for batch in pbar:
                # ---- 前向 ----
                inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
                labels = inputs.pop('label')
                batch_ids = inputs.pop('id')
                preds = self.model(**inputs)

                # ---- 更新 evaluator ----
                eva.update(preds=preds, labels=labels)

        # ---- 计算指标 ----
        metrics_all = eva.compute()
        print(f"{Fore.RED}AUROC (all)      {metrics_all['auroc']:.4f}{Style.RESET_ALL}")

    def _ranked_missing_samples(self):
        """
        熵驱动课程学习：返回"缺失样本按熵升序"排序的 (熵, id) 列表。

        思路（entropy-driven curriculum）：
            - 对训练集中的【缺失样本】，计算其预测分布的熵
              ent = -Σ p log p；
            - 熵越低，说明模型对该样本的预测越"确信"，即该补全样本越可靠；
            - 排序后，run 方法会按比例选取熵最低的若干缺失样本作为锚点集合。
        """
        self.model.eval()
        pbar = tqdm(self.train_data_loader, bar_format=f"{Fore.RED}{{l_bar}}{{bar}}{{r_bar}}", desc='Selecting Reliable Samples')

        pair_ent_id = []
        with torch.no_grad():
            for batch in pbar:
                ids = batch['id']
                missing_mask = batch['missing_mask'].to(self.device)

                # 去掉 label 与 id，只保留模型输入
                inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                          for k, v in batch.items()
                          if k not in ['label', 'id']}

                preds = self.model(**inputs)
                # 计算预测分布的熵（越小越可靠），并 clamp 防止 log(0)
                probs = torch.softmax(preds, dim=-1).clamp_min(1e-12)
                ent = -(probs * probs.log()).sum(dim=-1)  # [B] 熵，越小越可靠

                # 只保留缺失样本（mask==0）
                miss_mask = (missing_mask == 0)
                if miss_mask.any():
                    sel_idx = torch.nonzero(miss_mask, as_tuple=False).squeeze(1).cpu().tolist()
                    ids_sel = [ids[i] for i in sel_idx]
                    ents_sel = ent[miss_mask].detach().cpu().tolist()
                    pair_ent_id.extend(zip(ents_sel, ids_sel))

        # 按熵升序排序（熵小的排前面 = 更可靠）
        pair_ent_id.sort(key=lambda x: x[0])

        return pair_ent_id

















