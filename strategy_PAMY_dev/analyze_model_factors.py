#!/usr/bin/env python3
"""
分析模型因子并自动替换 strategies.py 中的 generate_factor_dataframe 函数体

用法：
    python analyze_model_factors.py           # 更新所有品种
    python analyze_model_factors.py A P       # 只更新 A 和 P
    python analyze_model_factors.py --dry-run # 预览，不修改文件
"""
import os
import re
import shutil
import argparse
from datetime import datetime, timezone, timedelta

BEIJING_TZ = timezone(timedelta(hours=8))

import lightgbm as lgb

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STRATEGIES_FILE = os.path.join(BASE_DIR, "strategies.py")


# ============================================================
# 因子名称解析
# ============================================================
PARAM_ALIASES = {
    "Volraiseap": "Volraise",
    "TrendRevmean": "trend_rev",
    "corrAskwap": "corrAsk",
    "corrBidwap": "corrBid",
}


def _get_all_symbols():
    """获取所有已知品种代码（models 目录 + config.json）"""
    symbols = set()
    # 从 models 目录
    if os.path.isdir(MODELS_DIR):
        for entry in sorted(os.listdir(MODELS_DIR)):
            p = os.path.join(MODELS_DIR, entry)
            if os.path.isdir(p) and any(f.endswith(".lgb") for f in os.listdir(p)):
                symbols.add(entry)
    # 从 config.json
    config_path = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        symbols.update(cfg.get("symbols", []))
        symbols.update(cfg.get("instruments", {}).keys())
    return sorted(symbols, key=len, reverse=True)


def _try_parse_cross_prefix(name: str, all_symbols: list):
    """
    尝试从因子名称中解析跨品种前缀 symbol1_symbol2_
    返回 (symbol1, symbol2, rest) 或 (None, None, None)
    优先使用已知品种列表匹配，匹配失败则从已知后缀反推。
    """
    sorted_symbols = sorted(set(all_symbols), key=len, reverse=True)

    # 方法1: 用已知品种列表匹配（如 CS 优先于 C）
    for s1 in sorted_symbols:
        if name.startswith(f"{s1}_"):
            rest_after_s1 = name[len(s1) + 1 :]
            for s2 in sorted_symbols:
                if rest_after_s1.startswith(f"{s2}_"):
                    rest = rest_after_s1[len(s2) + 1 :]
                    return s1, s2, rest

    # 方法2: fallback，从已知跨品种因子后缀反推
    known_suffix_patterns = [
        r"closepctchg\d+_sub",
        r"cvcorr10_diff",
        r"oi5_diff",
        r"vcorr10",
        r"volumediv\d+_diff\d+",
    ]
    for pat in known_suffix_patterns:
        m = re.match(rf"^(.+)_{pat}$", name)
        if m:
            prefix = m.group(1)
            parts = prefix.split("_")
            if len(parts) >= 2:
                for i in range(1, len(parts)):
                    s1 = "_".join(parts[:i])
                    s2 = "_".join(parts[i:])
                    if re.match(r"^[A-Z]+$", s1) and re.match(r"^[A-Z]+$", s2):
                        rest = name[len(prefix) + 1 :]
                        return s1, s2, rest
            break

    return None, None, None


def parse_fac_factor(name: str):
    """
    解析 FAC_ 因子名称，返回 (函数名, 参数)
    简单规则：将最后一个下划线后的部分作为参数
    """
    if not name.startswith("FAC_"):
        return None, None

    rest = name[4:]
    last_underscore = rest.rfind("_")
    if last_underscore == -1:
        return f"FAC_{rest}", None

    func_name = f"FAC_{rest[:last_underscore]}"
    param = rest[last_underscore + 1 :]
    param = PARAM_ALIASES.get(param, param)
    return func_name, param


def parse_cross_factor(name: str, main_symbol: str, all_symbols: list):
    """
    解析跨品种因子名称，返回函数调用字符串（不含 fac_generator. 前缀）
    """
    symbol1, symbol2, rest = _try_parse_cross_prefix(name, all_symbols)
    if symbol1 is None or symbol2 is None:
        return None

    # closepctchg
    if "closepctchg" in rest:
        m = re.search(r"(\d+)", rest)
        window = int(m.group(1)) if m else None
        parts = [
            f"closepctchg_sub(main_symbol='{main_symbol}'",
            f"symbol1='{symbol1}'",
            f"symbol2='{symbol2}'",
        ]
        if window is not None:
            parts.append(f"window={window}")
        return ", ".join(parts) + ")"

    # cvcorr
    if "cvcorr" in rest:
        return f"cvcorr10_diff(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}')"

    # oi5_diff
    if "oi" in rest and "oi5" in rest:
        return f"oi5_diff(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}')"

    # vcorr
    if "vcorr" in rest:
        return f"vcorr10(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}')"

    # volumediv
    if "volumediv" in rest:
        nums = re.findall(r"(\d+)", rest)
        if len(nums) >= 2:
            return (
                f"volumediv_diff(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}', "
                f"window1={int(nums[0])}, window2={int(nums[1])})"
            )
        return f"volumediv_diff(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}')"

    return None


# ============================================================
# 核心分析逻辑
# ============================================================
def analyze_one_model(model_path: str):
    """
    分析单个模型目录，返回 (main_symbol, 生成的因子赋值代码行列表, 总因子数)
    """
    main_symbol = os.path.basename(model_path)

    model_files = [f for f in os.listdir(model_path) if f.endswith(".lgb")]
    if not model_files:
        raise FileNotFoundError(f"未找到 .lgb 模型文件: {model_path}")

    model_file = os.path.join(model_path, model_files[0])
    model = lgb.Booster(model_file=model_file)
    feature_names = model.feature_name()

    # 获取所有已知品种代码，用于跨品种因子匹配
    all_symbols = _get_all_symbols()

    # 分类
    fac_factors = []
    cross_factors = []
    normal_factors = []

    for name in feature_names:
        if name.startswith("FAC_"):
            fac_factors.append(name)
        else:
            s1, s2, rest = _try_parse_cross_prefix(name, all_symbols)
            if s1 and s2:
                cross_factors.append(name)
            else:
                normal_factors.append(name)

    lines = []

    # FAC 因子
    for name in sorted(fac_factors):
        func_name, param = parse_fac_factor(name)
        if func_name and param:
            lines.append(f"    factor_dict['{name}'] = fac_generator.{func_name}('{param}')")
        elif func_name:
            lines.append(f"    factor_dict['{name}'] = fac_generator.{func_name}()")
        else:
            lines.append(f"    # 无法解析: {name}")

    # 跨品种因子
    if cross_factors:
        lines.append(f"    # ========== 跨品种因子 ({len(cross_factors)}个) ==========")
        for name in sorted(cross_factors):
            call = parse_cross_factor(name, main_symbol, all_symbols)
            if call:
                lines.append(f"    factor_dict['{name}'] = fac_generator.{call}")
            else:
                lines.append(f"    # 未知跨品种因子: {name}")

    # 普通因子
    if normal_factors:
        lines.append(f"    # ========== 普通因子 ({len(normal_factors)}个) ==========")
        for name in sorted(normal_factors):
            lines.append(f"    factor_dict['{name}'] = fac_generator.{name}()")

    return main_symbol, lines, len(feature_names)


# ============================================================
# 文件替换逻辑
# ============================================================
def _make_function_template(symbol: str, body_lines: list) -> str:
    """为新品种生成完整的 generate_factor_dataframe_{symbol} 函数代码"""
    body = "\n".join(body_lines)
    return f'''

@parallel_factor_compute
def generate_factor_dataframe_{symbol}(fac_generator,
                                valid_index: pd.Index,
                                factor_col) -> pd.DataFrame:
    from typing import Dict

    print(f'{{valid_index[0]}} - {{valid_index[-1]}}')
    factor_dict: Dict[str, np.ndarray] = {{}}
{body}
    # 创建DataFrame
    fac_df = pd.DataFrame(factor_dict, index=valid_index)
    fac_df = fac_df.replace([np.inf, -np.inf], np.nan)

    print(fac_df.shape)

    return fac_df[factor_col].round(8)
'''


def update_strategies_py(symbol: str, new_body_lines: list):
    """
    在 strategies.py 中找到 generate_factor_dataframe_{symbol} 函数，
    替换 factor_dict 声明之后到 # 创建DataFrame 之前的代码。
    如果函数不存在，则在文件末尾追加新函数。
    """
    with open(STRATEGIES_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    # 1) 定位函数定义
    func_idx = None
    for i, line in enumerate(lines):
        if f"def generate_factor_dataframe_{symbol}(" in line:
            func_idx = i
            break

    # 如果函数不存在，追加新函数
    if func_idx is None:
        new_func = _make_function_template(symbol, new_body_lines)
        # 在文件末尾（动态生成策略类之前）插入
        insert_pos = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("# 动态生成策略子类") or lines[i].strip().startswith("_IMPORTED_FACTOR_FUNCS"):
                insert_pos = i
                break
        new_lines = lines[:insert_pos] + [new_func] + lines[insert_pos:]
        new_content = "\n".join(new_lines)
    else:
        # 2) 定位 factor_dict: Dict[str, np.ndarray] = {}
        dict_start = None
        for i in range(func_idx, len(lines)):
            if "factor_dict: Dict[str, np.ndarray] = {}" in lines[i]:
                dict_start = i
                break
        if dict_start is None:
            raise ValueError(
                f"在 generate_factor_dataframe_{symbol} 中找不到 factor_dict 声明"
            )

        # 3) 定位 # 创建DataFrame 或 fac_df = pd.DataFrame
        df_create = None
        for i in range(dict_start + 1, len(lines)):
            if "# 创建DataFrame" in lines[i] or "fac_df = pd.DataFrame(factor_dict" in lines[i]:
                df_create = i
                break
        if df_create is None:
            raise ValueError(
                f"在 generate_factor_dataframe_{symbol} 中找不到 DataFrame 创建代码"
            )

        # 4) 组装新内容
        new_lines = lines[: dict_start + 1] + new_body_lines + lines[df_create:]
        new_content = "\n".join(new_lines)

    # 5) 备份 & 写入
    backup_path = STRATEGIES_FILE + f".backup_{datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(STRATEGIES_FILE, backup_path)

    with open(STRATEGIES_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)

    return backup_path


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="分析模型因子并自动替换 strategies.py 中的因子代码"
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="指定要更新的品种（如 A P M Y），默认更新 models/ 下所有品种",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印生成的代码，不修改 strategies.py",
    )
    args = parser.parse_args()

    # 确定要处理的品种
    if args.symbols:
        targets = []
        for sym in args.symbols:
            p = os.path.join(MODELS_DIR, sym)
            if os.path.isdir(p) and any(f.endswith(".lgb") for f in os.listdir(p)):
                targets.append(sym)
            else:
                print(f"⚠️ 跳过无效品种: {sym}（目录不存在或无 .lgb 模型）")
    else:
        targets = []
        for entry in sorted(os.listdir(MODELS_DIR)):
            p = os.path.join(MODELS_DIR, entry)
            if os.path.isdir(p) and any(f.endswith(".lgb") for f in os.listdir(p)):
                targets.append(entry)

    if not targets:
        print("未找到任何有效模型，退出。")
        return

    print(f"将要处理的品种: {targets}")
    if args.dry_run:
        print("【dry-run 模式】不会修改 strategies.py\n")

    for sym in targets:
        model_path = os.path.join(MODELS_DIR, sym)
        main_symbol, body_lines, total = analyze_one_model(model_path)
        print(f"\n{'='*80}")
        print(f"品种: {main_symbol} | 总因子数: {total}")
        print(f"{'='*80}")

        if args.dry_run:
            print("\n".join(body_lines))
        else:
            backup = update_strategies_py(main_symbol, body_lines)
            print(f"✅ 已更新 strategies.py -> generate_factor_dataframe_{main_symbol}")
            print(f"📦 备份文件: {backup}")

    if not args.dry_run:
        print(f"\n{'='*80}")
        print("🎉 全部完成！strategies.py 已更新。")
        print("   如需回滚，请使用 .backup_xxx 文件恢复。")


if __name__ == "__main__":
    main()
