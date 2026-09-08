"""
从已保存的 checkpoint 单独重跑测试集评估，不用重新训练。

用途：训练本身（含早停、保存 best_model.pth）已经正常跑完，但进入最终
Testing 阶段时进程被杀（本仓库这次是因为一处已撤销的内存缓存改动导致
OOM，见 docs/REPRODUCTION_LOG.md），此时用这个脚本对着保存下来的
checkpoint 补跑一次 trainer._test()，就能拿到真实的测试集 AUROC，不用
重新训练一遍。

用法（在仓库根目录下）：
    python scripts/run_test_only_v2.py \
        --missing_type Text --missing_rate 0.7 --no-use_ga \
        --checkpoint results_archive/ablation_101/best_model.pth
参数需要跟当初训练这个 checkpoint 时的 --missing_type/--missing_rate/--k/
--use_mir/--use_ga/--use_sea 完全一致（否则模型结构或数据集不匹配）。
"""
import sys, os
sys.path.insert(0, 'src')
import argparse
import torch
from utils import Trainer

p = argparse.ArgumentParser()
p.add_argument('--missing_type', default='Text')
p.add_argument('--missing_rate', type=float, default=0.7)
p.add_argument('--k', type=int, default=5)
p.add_argument('--use_mir', action=argparse.BooleanOptionalAction, default=True)
p.add_argument('--use_ga', action=argparse.BooleanOptionalAction, default=True)
p.add_argument('--use_sea', action=argparse.BooleanOptionalAction, default=True)
p.add_argument('--checkpoint', required=True)  # 指向 results_archive/<name>/best_model.pth
cli = p.parse_args()

class Args:
    model='ANGA'; backbone='vilt'; vilt_weights='src/model/vilt/weights/mlm'
    prompt_position=0; prompt_length=1; dropout_rate=0.2
    dataset='hatememes'
    missing_type=cli.missing_type; missing_rate=cli.missing_rate
    max_text_len=128; max_image_len=145; k=cli.k
    use_mir=cli.use_mir; use_ga=cli.use_ga; use_sea=cli.use_sea
    name='AdamW'; lr=1e-3; weight_decay=5e-5; use_warmup=True; warmup_rate=0.1
    device='cuda:0'; seed=2024; epochs=20; batch_size=64; num_workers=16
    patience=10; save_path='./src/checkpoints'; regenerate_missing_table=False

args = Args()
trainer = Trainer(args)
trainer.model.load_state_dict(torch.load(cli.checkpoint, map_location=args.device))
trainer._test()
