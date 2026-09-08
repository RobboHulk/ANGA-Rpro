"""
================================================================================
训练入口脚本（Training Entry）
================================================================================
本文件是 ANGA 模型的【训练与评测】主入口，职责如下：

    1. 解析全部命令行超参数（模型 / 数据 / 优化器 / 全局配置）；
    2. 若指定 --regenerate_missing_table=True，则先生成"缺失掩码表"
       （missing_table.pkl），然后退出（该表是训练前的必要准备）；
    3. 固定随机种子以保证实验可复现；
    4. 实例化 Trainer 并调用 run()，执行"课程学习 + 梯度对齐"的完整训练流程。

关于"缺失"（missing）的核心概念：
    - 多模态学习中，某个样本可能天然缺失文本或图像模态。
      本框架通过一个 missing_mask（缺失掩码）来标记每个样本的缺失状态。
    - missing_mask 的取值语义取决于 missing_type（缺失类型）：

        * missing_type = "Text" 或 "Image"（单模态缺失）：
            0 表示【该模态缺失】；1 表示【该模态不缺失】

        * missing_type = "Both"（双模态都可能缺失）：
            0 表示【缺失文本】；1 表示【缺失图像】；2 表示【两模态都完整】

运行方式：
    python src/train.py [--dataset hatememes] [--missing_type Both] ...
================================================================================
"""
import argparse
import sys
from utils import (
    seed_init,
    generate_missing_table,
    Trainer)
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def main():

    if True:
        # 创建命令行参数解析器，用于灵活控制训练配置
        parser = argparse.ArgumentParser(description="Model and Training Configuration")

        # ---------------------------------------------------------------------
        # 缺失掩码的取值语义（便于理解下面的 missing_type 参数）：
        #   one modality missing (单模态缺失): 0 缺失, 1 不缺失
        #   two modality missing (双模态缺失): 0 缺失文本, 1 缺失图像, 2 都不缺失
        # ---------------------------------------------------------------------

        # ---------- 模型参数（Model parameters） ----------
        parser.add_argument('--model', type=str, default="ANGA")                          # 使用的模型名（论文提出的 ANGA 框架）
        parser.add_argument('--backbone', type=str, default="vilt")                       # 视觉-语言骨干网络（ViLT）
        parser.add_argument('--vilt_weights', type=str, default="src/model/vilt/weights/mlm")  # 预训练 ViLT 权重路径
        parser.add_argument('--prompt_position', type=int, default=0)                     # 动态提示（prompt）插入到 Transformer 的第几层
        parser.add_argument('--prompt_length', type=int, default=1)                       # 每类提示的 token 长度（文本提示/图像提示各 1 个）
        parser.add_argument('--dropout_rate', type=float, default=0.2)                    # MMG 模块中的 dropout 比例
        # ---------- 消融开关（对应论文 Table2 的 MIR / GA / SEA 三大组件）----------
        # 用 --no-use_xxx 关闭对应组件；默认全部为 True，与未加开关前的模型行为完全一致
        parser.add_argument('--use_mir', action=argparse.BooleanOptionalAction, default=True)  # 是否用 MIR（检索重建缺失模态）
        parser.add_argument('--use_ga', action=argparse.BooleanOptionalAction, default=True)   # 是否用 GA（锥域投影梯度对齐）
        parser.add_argument('--use_sea', action=argparse.BooleanOptionalAction, default=True)  # 是否用 SEA（CAP 动态提示）

        # ---------- 数据参数（Data parameters） ----------
        parser.add_argument('--dataset', type=str, default="hatememes", choices=["hatememes", "mmimdb", "food101"])  # 数据集名称
        parser.add_argument('--missing_type', type=str, default="Text", choices=["Both", "Text", "Image"])           # 缺失类型：Both 双模态 / Text 缺文本 / Image 缺图像
        parser.add_argument('--missing_rate', type=float, default=0.7)                     # 缺失比例（0.7 表示 70% 样本缺失）
        parser.add_argument('--max_text_len', type=int, default=128)                       # 文本序列最大长度（token 数）
        parser.add_argument('--max_image_len', type=int, default=145)                      # 图像 patch 序列最大长度（含 1 个 CLS + 144 个 patch）
        parser.add_argument('--k', type=int, default=5)                                    # 检索返回的相似样本数（Top-K）

        # ---------- 优化器参数（Optimizer parameters） ----------
        parser.add_argument('--name', type=str, default="AdamW")                           # 优化器名称
        parser.add_argument('--lr', type=float, default=1e-3)                              # 学习率
        parser.add_argument('--weight_decay', type=float, default=5e-5)                    # 权重衰减（L2 正则化系数）
        parser.add_argument('--use_warmup', type=bool, default=True)                       # 是否使用学习率预热（warmup）
        parser.add_argument('--warmup_rate', type=float, default=0.1)                      # 预热步数占总步数的比例

        # ---------- 全局参数（Global parameters） ----------
        parser.add_argument('--device', type=str, default="cuda:0")                        # 使用的 GPU 设备
        parser.add_argument('--seed', type=int, default=2024)                              # 随机种子
        parser.add_argument('--epochs', type=int, default=20)                              # 训练总轮数
        parser.add_argument('--batch_size', type=int, default=64)                          # 批大小
        parser.add_argument('--num_workers', type=int, default=16)                         # DataLoader 的并行加载进程数
        parser.add_argument('--patience', type=int, default=10)                            # 早停耐心值（连续多少轮无提升则停止）
        parser.add_argument('--save_path', type=str, default="./src/checkpoints")  # 模型保存目录
        parser.add_argument('--regenerate_missing_table', type=bool, default=False)        # 是否重新生成缺失掩码表

        args = parser.parse_args()

    pd.set_option('future.no_silent_downcasting', True) # 控制 Pandas 中的数据类型自动降级行为的选项
    seed_init(args.seed)  # 固定随机种子，保证实验可复现

    # -------------------------------------------------------------------------
    # 若指定了 --regenerate_missing_table=True，则只生成缺失掩码表后退出。
    # 缺失掩码表记录了"每个样本缺失哪个模态"，是训练时模拟缺失场景的依据。
    #
    # single（单模态缺失）掩码语义：0 缺失；1 不缺失
    # both（双模态缺失）掩码语义：0 缺失文本；1 缺失图像；2 都不缺失
    # -------------------------------------------------------------------------
    if args.regenerate_missing_table:
        data_para = {
            'dataset': args.dataset,
            'missing_type': args.missing_type,
            'missing_rate': args.missing_rate,}
        generate_missing_table(**data_para)
        sys.exit(0)

    # # ----------（调试用）加载并查看 missing_table.pkl 的内容 ----------
    # file_path = './dataset/missing_table/both/hatememes/missing_table.pkl'  # 修改为你的文件路径
    # df = pd.read_pickle(file_path)
    # total_items = df['item_id'].nunique()
    # missing_counts = df['missing_mask_7'].value_counts()
    # # 输出结果
    # print(f"Total number of unique item_ids: {total_items}")
    # print(f"Missing data (0): {missing_counts.get(0, 0)}")
    # print(f"Missing data (1): {missing_counts.get(1, 0)}")
    # print(f"Missing data (2): {missing_counts.get(2, 0)}")
    # print(df[:10])
    # print(df.shape)
    # sys.exit(0)

    # 实例化 Trainer 并开始训练+评测
    trainer = Trainer(args)
    trainer.run()

if  __name__ == '__main__':
    main()