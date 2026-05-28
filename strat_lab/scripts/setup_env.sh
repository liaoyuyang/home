#!/bin/bash
# 环境初始化脚本
# 校验 mnt 挂载、创建必要目录、检查依赖

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$PROJECT_ROOT/config/paths.yaml"

echo "============================================"
echo "Strat Lab 环境初始化"
echo "============================================"
echo "项目路径: $PROJECT_ROOT"

# 1. 检查 Python 依赖
echo ""
echo "[1/4] 检查 Python 依赖..."
python3 -c "import pandas, numpy, lightgbm, yaml, tqdm" 2>/dev/null || {
    echo "WARNING: 部分依赖缺失，建议安装:"
    echo "  pip install pandas numpy lightgbm pyyaml tqdm joblib matplotlib plotly ipywidgets"
}

# 2. 检查 mnt 挂载
echo ""
echo "[2/4] 检查 mnt 挂载..."
MNT_ROOT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['mnt']['root'])" 2>/dev/null || echo "/mnt/Data/writable/liaoyuyang")

if [ -d "$MNT_ROOT" ]; then
    echo "  ✓ mnt 已挂载: $MNT_ROOT"
else
    echo "  ✗ mnt 未挂载: $MNT_ROOT"
    echo "    请检查挂载状态，或修改 config/paths.yaml 中的 mnt.root"
    exit 1
fi

# 3. 创建目录结构
echo ""
echo "[3/4] 创建目录结构..."
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/data/groups"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/research/factor_analysis"
mkdir -p "$PROJECT_ROOT/research/resample_study"
mkdir -p "$PROJECT_ROOT/research/cross_group_compare"
echo "  ✓ 目录结构已创建"

# 4. 检查关键配置文件
echo ""
echo "[4/4] 检查配置文件..."
for cfg in paths.yaml groups.yaml instruments.yaml pipeline.yaml; do
    if [ -f "$PROJECT_ROOT/config/$cfg" ]; then
        echo "  ✓ $cfg"
    else
        echo "  ✗ $cfg 缺失"
    fi
done

echo ""
echo "============================================"
echo "初始化完成"
echo "============================================"
echo ""
echo "建议下一步:"
echo "  1. 同步市场数据: python pipeline/01_sync_mkt_data.py --group 油脂油粕"
echo "  2. 同步单品种因子: python pipeline/02_sync_single_factors.py --group 油脂油粕"
echo "  3. 计算跨品种因子: python pipeline/03_calc_cross_factors.py --group 油脂油粕"
echo "  4. 拼接因子表: python pipeline/04_make_factor_df.py --group 油脂油粕"
