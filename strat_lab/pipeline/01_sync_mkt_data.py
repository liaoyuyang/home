"""
同步主力合约市场数据到本地

从 mnt 复制主力合约 1min 数据和合约切换表到项目本地 data/raw/
支持增量同步（按文件修改时间判断）
"""

import argparse
import shutil
from pathlib import Path

from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.path_resolver import PathResolver


def sync_symbol(symbol: str, force: bool = False) -> None:
    """同步单个品种的主力合约数据"""
    pr = PathResolver()
    cm = ConfigManager()

    # 目标路径
    local_dir = pr.ensure_dir(pr.resolve("local", "raw", symbol))

    # 1. 同步主力合约 1min 数据
    src_1min = pr.resolve("mnt", "data_1min_active") / f"main_{symbol}.csv"
    dst_1min = local_dir / "main_1min.csv"

    if src_1min.exists():
        if force or not dst_1min.exists() or src_1min.stat().st_mtime > dst_1min.stat().st_mtime:
            shutil.copy2(src_1min, dst_1min)
            print(f"[{symbol}] Synced main_1min.csv")
        else:
            print(f"[{symbol}] main_1min.csv up-to-date")
    else:
        print(f"[{symbol}] WARNING: mnt main_1min.csv not found: {src_1min}")

    # 2. 同步合约切换表
    src_info = pr.resolve("mnt", "future_info", symbol) / "main_instrument.csv"
    dst_info = local_dir / "contract_info.csv"

    if src_info.exists():
        if force or not dst_info.exists() or src_info.stat().st_mtime > dst_info.stat().st_mtime:
            shutil.copy2(src_info, dst_info)
            print(f"[{symbol}] Synced contract_info.csv")
        else:
            print(f"[{symbol}] contract_info.csv up-to-date")
    else:
        print(f"[{symbol}] WARNING: mnt contract_info.csv not found: {src_info}")


def main():
    parser = argparse.ArgumentParser(description="同步主力合约数据到本地")
    parser.add_argument("--symbols", nargs="+", help="指定品种列表，如 A M Y P")
    parser.add_argument("--group", type=str, help="指定分组名，同步该分组下所有品种")
    parser.add_argument("--force", action="store_true", help="强制覆盖，忽略修改时间")
    parser.add_argument("--all", action="store_true", help="同步所有已配置品种")
    args = parser.parse_args()

    cm = ConfigManager()

    if args.all:
        symbols = cm.list_symbols()
    elif args.group:
        symbols = cm.get_group_symbols(args.group)
    elif args.symbols:
        symbols = args.symbols
    else:
        print("请指定 --symbols、--group 或 --all")
        return

    print(f"Syncing {len(symbols)} symbols: {symbols}")
    for symbol in tqdm(symbols, desc="Sync"):
        sync_symbol(symbol, force=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
