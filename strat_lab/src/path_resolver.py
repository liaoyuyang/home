"""
统一路径解析器
支持 mnt 侧和本地侧路径的集中管理，换机器时只需改 config/paths.yaml
"""

import os
from pathlib import Path
from typing import Literal

import yaml


class PathResolver:
    """
    路径解析器

    用法:
        pr = PathResolver()
        pr.resolve('mnt', 'factor', 'A', 'all_fac')
        pr.resolve('local', 'group', '油脂油粕', 'models')
    """

    def __init__(self, config_path: str = "config/paths.yaml"):
        self.project_root = Path(__file__).parent.parent.resolve()
        config_path = self.project_root / config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # 允许环境变量覆盖 mnt_root
        self.mnt_root = Path(
            os.environ.get("STRAT_LAB_MNT_ROOT", self.cfg["mnt"]["root"])
        )
        self.local_root = self.project_root / self.cfg["local"]["root"].lstrip("./")

    def resolve(
        self,
        side: Literal["mnt", "local"],
        *parts: str,
    ) -> Path:
        """
        解析路径

        Parameters
        ----------
        side : 'mnt' | 'local'
            选择 mnt 侧还是本地侧
        *parts : str
            路径层级，支持关键字映射

        支持的关键字映射:
            mnt 侧:
                'factor', '{symbol}', 'all_fac' → mnt/factor/{symbol}/all_fac/all_factor.feather
                'factor', '{symbol}', 'm2m'     → mnt/factor/{symbol}/m2m
                'data_1min_active'               → mnt/data/1min/active
                'future_info', '{symbol}'        → mnt/data/future_info/{symbol}

            local 侧:
                'raw', '{symbol}'                → local/data/raw/{symbol}
                'processed', '{symbol}'          → local/data/processed/{symbol}
                'group', '{group_name}', 'all_factor' → local/data/groups/{group_name}/all_factor
                'group', '{group_name}', 'models'     → local/data/groups/{group_name}/models
        """
        if side == "mnt":
            return self._resolve_mnt(parts)
        return self._resolve_local(parts)

    def _resolve_mnt(self, parts: tuple[str, ...]) -> Path:
        key = parts[0] if parts else ""

        # 关键字快捷映射
        if key == "factor" and len(parts) >= 2:
            symbol = parts[1]
            sub = parts[2] if len(parts) > 2 else ""
            base = self.mnt_root / "factor" / symbol
            if sub == "all_fac":
                return base / "all_fac" / "all_factor.feather"
            if sub:
                return base / sub
            return base

        if key == "data_1min_active":
            return self.mnt_root / "data" / "1min" / "active"

        if key == "data_1min" and len(parts) >= 2:
            symbol = parts[1]
            return self.mnt_root / "data" / "1min" / symbol

        if key == "future_info" and len(parts) >= 2:
            symbol = parts[1]
            return self.mnt_root / "data" / "future_info" / symbol

        if key == "model" and len(parts) >= 2:
            # parts[1] 为模型文件夹名
            return self.mnt_root / "model" / "lightgbm" / "KFoldModel" / "models" / parts[1]

        # 兜底：按 cfg 中定义的键直接查找
        if key in self.cfg["mnt"]:
            base = Path(self.cfg["mnt"][key])
            if not base.is_absolute():
                base = self.mnt_root / base
            return base / "/".join(parts[1:])

        raise KeyError(f"未识别的 mnt 路径关键字: {parts}")

    def _resolve_local(self, parts: tuple[str, ...]) -> Path:
        key = parts[0] if parts else ""

        if key == "raw" and len(parts) >= 2:
            symbol = parts[1]
            return self.local_root / "raw" / symbol

        if key == "processed" and len(parts) >= 2:
            symbol = parts[1]
            return self.local_root / "processed" / symbol

        if key == "group" and len(parts) >= 2:
            group_name = parts[1]
            group_root = self.local_root / "groups" / group_name
            if len(parts) >= 3:
                sub = parts[2]
                return group_root / sub
            return group_root

        if key == "logs":
            return self.project_root / "logs"

        # 兜底
        if key in self.cfg["local"]:
            base = Path(self.cfg["local"][key])
            if not base.is_absolute():
                base = self.local_root / base
            return base / "/".join(parts[1:])

        raise KeyError(f"未识别的 local 路径关键字: {parts}")

    def ensure_dir(self, path: Path) -> Path:
        """确保目录存在，返回 Path"""
        path.mkdir(parents=True, exist_ok=True)
        return path
