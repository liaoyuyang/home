"""
训练入口

支持单品种训练或按分组批量训练

用法:
    # 单品种
    python pipeline/05_train.py --group 油脂油粕 --symbol A

    # 分组批量
    python pipeline/05_train.py --group 油脂油粕 --all-symbols
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.training_adapter import TrainingPipeline, build_model_folder_name


def train_symbol(group_name: str, symbol: str, train_end_date: str | None = None, train_label: int | None = None):
    """训练单个品种"""
    cm = ConfigManager()
    cfg = cm.get_pipeline()["training"]
    train_end_date = train_end_date or cfg["train_end_date"]
    train_label = train_label or cfg["train_label"]

    pipeline = TrainingPipeline(group_name, symbol)

    # 1. 加载数据
    print(f"\n{'='*60}")
    print(f"[{group_name}] {symbol} | 加载数据...")
    df = pipeline.load_data(train_end_date=train_end_date, train_label=train_label)
    print(f"数据形状: {df.shape}, 时间范围: {df.index.min()} ~ {df.index.max()}")

    # 2. 预训练（如果结果不存在）
    print(f"[{symbol}] 预训练...")
    pretrainer = pipeline.run_pretraining(df, train_end_date, train_label)

    # 3. 因子筛选（简化版：直接取重要性 top 300）
    # TODO: 接入 FactorFilter 完整筛选逻辑
    importance = pretrainer.importance_df
    factor_col = importance.head(300).index.tolist()
    factor_col = [c for c in factor_col if c in df.columns]
    print(f"[{symbol}] 筛选后因子数: {len(factor_col)}")

    # 4. KFold 训练
    folder_name = build_model_folder_name(symbol, train_label, train_end_date)
    print(f"[{symbol}] KFold 训练 -> {folder_name}")
    results = pipeline.run_training(
        df=df,
        factor_col=factor_col,
        train_end_date=train_end_date,
        model_folder_name=folder_name,
        n_splits=cfg["n_splits"],
    )

    print(f"[{symbol}] 训练完成")
    return results


def main():
    parser = argparse.ArgumentParser(description="训练入口")
    parser.add_argument("--group", type=str, required=True, help="分组名称")
    parser.add_argument("--symbol", type=str, help="指定单个品种")
    parser.add_argument("--all-symbols", action="store_true", help="训练该分组下所有品种")
    parser.add_argument("--train-end-date", type=str, help="训练截止日期")
    parser.add_argument("--train-label", type=int, help="预测标签周期")
    args = parser.parse_args()

    cm = ConfigManager()
    symbols = cm.get_group_symbols(args.group)

    if args.symbol:
        if args.symbol not in symbols:
            raise ValueError(f"{args.symbol} 不在分组 {args.group} 中")
        targets = [args.symbol]
    elif args.all_symbols:
        targets = symbols
    else:
        print("请指定 --symbol 或 --all-symbols")
        return

    for symbol in targets:
        train_symbol(
            args.group,
            symbol,
            train_end_date=args.train_end_date,
            train_label=args.train_label,
        )

    print("\n全部完成")


if __name__ == "__main__":
    main()
