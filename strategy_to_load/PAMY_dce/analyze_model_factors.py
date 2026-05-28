#!/usr/bin/env python3
"""
分析模型因子并生成generate_factor_dataframe函数代码
"""
import os
import re
import lightgbm as lgb

def parse_fac_factor(name):
    """
    解析FAC因子名称，返回函数名和参数
    简单规则：将最后一个下划线后的部分作为参数
    特殊处理以下两个因子：
    - FAC_PVcorrsub_a2b2_Volraiseap -> FAC_PVcorrsub_a2b2, Volraise
    - FAC_VWAP_Deviation_TrendRevmean -> FAC_VWAP_Deviation, trend_rev
    """
    if not name.startswith('FAC_'):
        return None, None
    
    # 移除FAC_前缀
    rest = name[4:]
    
    # 找到最后一个下划线的位置
    last_underscore = rest.rfind('_')
    
    func_name = f'FAC_{rest[:last_underscore]}'
    param = rest[last_underscore + 1:]
    if param == "Volraiseap":
        param = "Volraise"
    if param == "TrendRevmean":
        param = "trend_rev"
    if param == "corrAskwap":
        param = "corrAsk"
    if param == "corrBidwap":
        param = "corrBid"
    return func_name, param

def analyze_model_factors(model_path):
    """分析模型中的因子并按类别分类"""
    
    # 从模型路径提取主品种代码
    main_symbol = os.path.basename(model_path)
    
    # 加载第一个模型获取特征名称
    model_files = [f for f in os.listdir(model_path) if f.endswith('.lgb')]
    if not model_files:
        print(f"未找到模型文件在: {model_path}")
        return
    
    model_file = os.path.join(model_path, model_files[0])
    model = lgb.Booster(model_file=model_file)
    feature_names = model.feature_name()
    
    print(f"模型路径: {model_path}")
    print(f"模型文件: {model_files[0]}")
    print(f"总因子数: {len(feature_names)}")
    print("="*80)
    
    # 分类因子
    fac_factors = []  # FAC_开头的因子
    cross_factors = []  # 跨品种因子 (包含_和品种代码)
    normal_factors = []  # 普通因子
    
    # 定义品种代码模式 (A_M_, A_Y_, P_A_等)
    symbol_pattern = re.compile(r'^([A-Z])_([A-Z])_')
    
    for name in feature_names:
        if name.startswith('FAC_'):
            fac_factors.append(name)
        elif symbol_pattern.match(name):
            cross_factors.append(name)
        else:
            normal_factors.append(name)
    
    # 打印FAC因子（排序后）
    print(f"\n# ===================== FAC_开头的因子 ({len(fac_factors)}个) =====================")
    for name in sorted(fac_factors):
        func_name, param = parse_fac_factor(name)
        if func_name and param:
            print(f"factor_dict['{name}'] = fac_generator.{func_name}('{param}')")
        elif func_name:
            print(f"factor_dict['{name}'] = fac_generator.{func_name}()")
        else:
            print(f"# 无法解析: {name}")
    
    # 打印跨品种因子（排序后）
    print(f"\n# ========== 跨品种因子 ({len(cross_factors)}个) ==========")
    for name in sorted(cross_factors):
        # 解析跨品种因子
        # 格式: 品种1_品种2_因子类型参数
        # 例如: A_M_closepctchg20_sub -> closepctchg_sub(main_symbol='A', symbol1='A', symbol2='M', window=20)
        match = symbol_pattern.match(name)
        if match:
            symbol1 = match.group(1)  # 第一个品种代码
            symbol2 = match.group(2)  # 第二个品种代码
            
            # 提取因子类型和参数部分
            rest = name[match.end():]  # 去掉品种前缀后的部分
            
            # 提取窗口参数
            window = None
            window1 = None
            window2 = None
            
            if 'closepctchg' in rest:
                # closepctchg20_sub 或 closepctchg5_sub
                factor_type = 'closepctchg_sub'
                match_num = re.search(r'(\d+)', rest)
                if match_num:
                    window = int(match_num.group(1))
                func_call = f"closepctchg_sub(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}'"
                if window:
                    func_call += f", window={window}"
                func_call += ")"
                
            elif 'cvcorr' in rest:
                # cvcorr10_diff
                factor_type = 'cvcorr10_diff'
                func_call = f"cvcorr10_diff(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}')"
                
            elif 'oi' in rest and 'oi5' in rest:
                # oi5_diff
                factor_type = 'oi5_diff'
                func_call = f"oi5_diff(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}')"
                
            elif 'vcorr' in rest:
                # vcorr10
                factor_type = 'vcorr10'
                func_call = f"vcorr10(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}')"
                
            elif 'volumediv' in rest:
                # volumediv20_diff5
                factor_type = 'volumediv_diff'
                match_nums = re.findall(r'(\d+)', rest)
                if len(match_nums) >= 2:
                    window1 = int(match_nums[0])
                    window2 = int(match_nums[1])
                func_call = f"volumediv_diff(main_symbol='{main_symbol}', symbol1='{symbol1}', symbol2='{symbol2}'"
                if window1 and window2:
                    func_call += f", window1={window1}, window2={window2}"
                func_call += ")"
            else:
                factor_type = 'UNKNOWN'
                func_call = f"UNKNOWN_FACTOR_TYPE('{rest}')"
            
            print(f"factor_dict['{name}'] = fac_generator.{func_call}")
    
    # 打印普通因子（排序后）
    print(f"\n# ========== 普通因子 ({len(normal_factors)}个) ==========")
    for name in sorted(normal_factors):
        print(f"factor_dict['{name}'] = fac_generator.{name}()")
    
    return {
        'fac_factors': sorted(fac_factors),
        'cross_factors': sorted(cross_factors),
        'normal_factors': sorted(normal_factors)
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # 默认分析A模型
        main_symbol = "A"
        model_path = f"/home/strategy_to_load/PAMY_dce/models/{main_symbol}"
    
    analyze_model_factors(model_path)