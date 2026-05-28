"""
拼因子表

将单品种因子 + 跨品种因子拼接为 all_factor.feather，按分组隔离存储

用法:
    python pipeline/04_make_factor_df.py --group 油脂油粕
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.group_builder import GroupBuilder


def main():
    parser = argparse.ArgumentParser(description="拼接因子表")
    parser.add_argument("--group", type=str, required=True, help="分组名称")
    parser.add_argument(
        "--single-source",
        type=str,
        default="mnt",
        choices=["mnt", "local"],
        help="单品种因子来源",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="拼单品种因子时的并行度（默认1，和同事共用机器请谨慎提高）",
    )
    args = parser.parse_args()

    builder = GroupBuilder(args.group)
    saved = builder.build_all_factor(
        single_factor_source=args.single_source,
        n_jobs=args.n_jobs,
    )

    print(f"\nAll factor tables built for group [{args.group}]:")
    for symbol, path in saved.items():
        print(f"  {symbol}: {path}")


if __name__ == "__main__":
    main()
