"""
配置管理器
统一加载和校验 config/ 下的所有 yaml 配置
"""

from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    """
    配置管理器

    用法:
        cm = ConfigManager()
        paths = cm.get("paths")
        groups = cm.get("groups")
        symbols = cm.get_groups()["油脂油粕"]["symbols"]
    """

    def __init__(self, config_dir: str = "config"):
        self.project_root = Path(__file__).parent.parent.resolve()
        self.config_dir = self.project_root / config_dir
        self._cache: dict[str, Any] = {}

    def _load(self, name: str) -> Any:
        if name not in self._cache:
            path = self.config_dir / f"{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"配置文件不存在: {path}")
            with open(path, "r", encoding="utf-8") as f:
                self._cache[name] = yaml.safe_load(f)
        return self._cache[name]

    def get(self, name: str) -> Any:
        """加载指定配置文件"""
        return self._load(name)

    def get_paths(self) -> dict:
        """获取路径配置"""
        return self._load("paths")

    def get_groups(self) -> dict:
        """获取分组配置"""
        return self._load("groups").get("groups", {})

    def get_instruments(self) -> dict:
        """获取品种配置"""
        return self._load("instruments").get("instruments", {})

    def get_pipeline(self) -> dict:
        """获取流程默认参数"""
        return self._load("pipeline")

    def get_group_symbols(self, group_name: str) -> list[str]:
        """获取指定分组包含的品种列表"""
        groups = self.get_groups()
        if group_name not in groups:
            raise KeyError(f"分组不存在: {group_name}，可用分组: {list(groups.keys())}")
        return groups[group_name]["symbols"]

    def get_instrument_config(self, symbol: str) -> dict:
        """获取单个品种的配置"""
        instruments = self.get_instruments()
        if symbol not in instruments:
            raise KeyError(f"品种未配置: {symbol}")
        return instruments[symbol]

    def list_groups(self) -> list[str]:
        """列出所有分组名称"""
        return list(self.get_groups().keys())

    def list_symbols(self) -> list[str]:
        """列出所有已配置品种"""
        return list(self.get_instruments().keys())
