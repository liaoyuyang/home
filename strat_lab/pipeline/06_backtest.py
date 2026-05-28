"""
回测入口

用法:
    python pipeline/06_backtest.py --group 油脂油粕 --symbol A --model-folder A_pred5_2025-01-01_v0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest_adapter import BacktestPipeline
from src.config_manager import ConfigManager


def main():
    parser = argparse.ArgumentParser(description="回测入口")
    parser.add_argument("--group", type=str, required=True, help="分组名称")
    parser.add_argument("--symbol", type=str, required=True, help="品种代码")
    parser.add_argument("--model-folder", type=str, required=True, help="模型文件夹名")
    parser.add_argument("--test-start", type=str, help="回测开始日期")
    parser.add_argument("--test-end", type=str, help="回测结束日期")
    parser.add_argument("--th1", type=float, help="开多阈值")
    parser.add_argument("--th2", type=float, help="平多阈值")
    args = parser.parse_args()

    cm = ConfigManager()
    symbols = cm.get_group_symbols(args.group)
    if args.symbol not in symbols:
        raise ValueError(f"{args.symbol} 不在分组 {args.group} 中")

    pipeline = BacktestPipeline(args.group, args.symbol)
    merged = pipeline.run(
        model_folder_name=args.model_folder,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        th1=args.th1,
        th2=args.th2,
    )

    # 快速统计
    stats = pipeline.analyze(merged)
    print("\n回测统计:")
    for k, v in list(stats.items())[:10]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
