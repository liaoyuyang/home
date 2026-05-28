"""
按分组计算跨品种因子

用法:
    python pipeline/03_calc_cross_factors.py --group 油脂油粕
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.group_builder import GroupBuilder


def main():
    parser = argparse.ArgumentParser(description="计算跨品种因子")
    parser.add_argument("--group", type=str, required=True, help="分组名称")
    parser.add_argument("--method", type=str, default="default", help="计算方法")
    args = parser.parse_args()

    builder = GroupBuilder(args.group)
    df = builder.compute_cross_factors(method=args.method)
    print(f"Cross factors shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}...")


if __name__ == "__main__":
    main()
