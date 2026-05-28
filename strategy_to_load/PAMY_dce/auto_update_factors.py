#!/usr/bin/env python3
"""
自动更新策略主程序中的因子代码
每次更换新模型后，运行此脚本自动更新四个品种的主程序
"""
import subprocess
import re
import os

# 品种列表
SYMBOLS = ['A', 'M', 'P', 'Y']

# 基础路径
BASE_PATH = "/home/strategy_to_load/PAMY_dce"


def run_analyze_model_factors(symbol):
    """
    运行 analyze_model_factors.py 获取生成的因子代码
    
    Args:
        symbol: 品种代码 (A, M, P, Y)
    
    Returns:
        生成的因子代码字符串
    """
    model_path = f"{BASE_PATH}/models/{symbol}"
    script_path = f"{BASE_PATH}/analyze_model_factors.py"
    
    print(f"  正在分析 {symbol} 品种的模型...")
    result = subprocess.run(
        ['python3', script_path, model_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  错误: 运行 analyze_model_factors.py 失败")
        print(f"  stderr: {result.stderr}")
        return None
    
    return result.stdout


def parse_factor_code(output):
    """
    解析 analyze_model_factors.py 的输出，提取因子定义代码
    
    Args:
        output: analyze_model_factors.py 的输出字符串
    
    Returns:
        提取的因子定义代码字符串
    """
    lines = output.split('\n')
    factor_lines = []
    
    # 找到因子定义的开始（以 factor_dict[ 开头的行）
    in_factor_section = False
    
    for line in lines:
        stripped = line.strip()
        
        # 跳过空行和注释行（除了分类注释）
        if not stripped:
            continue
        
        # 检测因子定义开始
        if stripped.startswith("factor_dict["):
            in_factor_section = True
        
        # 如果在因子定义区域内
        if in_factor_section:
            # 保留分类注释和因子定义
            if stripped.startswith('#') or stripped.startswith("factor_dict["):
                factor_lines.append(line)
        
        # 检测结束（遇到创建DataFrame的注释或代码）
        if '创建DataFrame' in stripped or stripped.startswith('fac_df = pd.DataFrame'):
            break
    
    return '\n'.join(factor_lines)


def update_main_strategy(symbol, factor_code):
    """
    更新主程序文件中的 generate_factor_dataframe 函数
    
    Args:
        symbol: 品种代码 (A, M, P, Y)
        factor_code: 新的因子定义代码
    
    Returns:
        是否成功更新
    """
    file_path = f"{BASE_PATH}/main_strategy_{symbol}.py"
    
    print(f"  正在更新 {file_path}...")
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到 generate_factor_dataframe 函数的位置
    # 函数开始：factor_dict: Dict[str, np.ndarray] = {}
    func_start_pattern = r'(factor_dict: Dict\[str, np\.ndarray\] = \{\}\s*\n)'
    
    # 函数结束：# 创建DataFrame
    func_end_pattern = r'(\n\s*# 创建DataFrame)'
    
    # 查找函数开始位置
    start_match = re.search(func_start_pattern, content)
    if not start_match:
        print(f"  错误: 无法找到函数开始位置 (factor_dict: Dict[str, np.ndarray] = {{}})")
        return False
    
    # 查找函数结束位置
    end_match = re.search(func_end_pattern, content)
    if not end_match:
        print(f"  错误: 无法找到函数结束位置 (# 创建DataFrame)")
        return False
    
    start_pos = start_match.end()
    end_pos = end_match.start()
    
    # 构建新的函数体 - 保持4空格缩进
    indented_factor_code = '\n'.join('    ' + line if line.strip() else '' for line in factor_code.split('\n'))
    new_content = content[:start_pos] + indented_factor_code + content[end_pos:]
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  ✓ {symbol} 品种更新完成")
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("开始自动更新因子代码")
    print("=" * 60)
    
    success_count = 0
    
    for symbol in SYMBOLS:
        print(f"\n[{symbol}] 处理中...")
        
        # 1. 运行 analyze_model_factors.py 获取因子代码
        output = run_analyze_model_factors(symbol)
        if output is None:
            print(f"  ✗ {symbol} 品种处理失败")
            continue
        
        # 2. 解析输出，提取因子定义
        factor_code = parse_factor_code(output)
        if not factor_code:
            print(f"  错误: 无法解析因子代码")
            continue
        
        print(f"  提取到 {len(factor_code.split(chr(10)))} 行因子代码")
        
        # 3. 更新主程序文件
        if update_main_strategy(symbol, factor_code):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"更新完成: {success_count}/{len(SYMBOLS)} 个品种成功")
    print("=" * 60)


if __name__ == "__main__":
    main()
