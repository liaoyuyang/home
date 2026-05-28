"""
同步单品种因子到本地

从 mnt 复制某品种的所有单品种因子（m2m/t2m/t2t）到 data/processed/
支持按模式筛选（只同步 m2m、或只同步 t2t 等）
"""

import argparse
import shutil
from pathlib import Path

from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.path_resolver import PathResolver


def sync_symbol_factors(symbol: str, modes: list[str] | None = None, force: bool = False) -> None:
    """
    同步单个品种的单品种因子

    Parameters
    ----------
    symbol : str
    modes : list[str], optional
        要同步的模式列表，如 ['m2m', 't2m', 't2t']，None 则同步全部
    force : bool
    """
    pr = PathResolver()
    if modes is None:
        modes = ["m2m", "t2m", "t2t"]

    local_base = pr.ensure_dir(pr.resolve("local", "processed", symbol))

    for mode in modes:
        src_dir = pr.resolve("mnt", "factor", symbol, mode)
        dst_dir = local_base / mode

        if not src_dir.exists():
            print(f"[{symbol}] {mode} not found on mnt, skip")
            continue

        pr.ensure_dir(dst_dir)

        # 遍历源目录下所有 feather 文件
        src_files = list(src_dir.rglob("*.feather"))
        synced = 0
        skipped = 0

        for src_file in tqdm(src_files, desc=f"{symbol}-{mode}", leave=False):
            rel_path = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel_path
            pr.ensure_dir(dst_file.parent)

            if force or not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                shutil.copy2(src_file, dst_file)
                synced += 1
            else:
                skipped += 1

        print(f"[{symbol}] {mode}: synced={synced}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(description="同步单品种因子到本地")
    parser.add_argument("--symbols", nargs="+", help="指定品种列表")
    parser.add_argument("--group", type=str, help="指定分组名")
    parser.add_argument("--modes", nargs="+", default=["m2m", "t2m", "t2t"], help="同步模式")
    parser.add_argument("--force", action="store_true", help="强制覆盖")
    parser.add_argument("--all", action="store_true", help="同步所有品种")
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

    print(f"Syncing single factors for {len(symbols)} symbols, modes={args.modes}")
    for symbol in symbols:
        sync_symbol_factors(symbol, modes=args.modes, force=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
