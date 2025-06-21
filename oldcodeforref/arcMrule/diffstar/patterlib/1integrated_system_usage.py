"""
整合系统使用示例
"""

def analyze_arc_task_with_integrated_system(task_data):
    """使用整合系统分析ARC任务"""
    # 初始化整合系统
    integrated_system = IntegratedARCPatternSystem(debug=True)
    
    # 1. 从训练数据中提取对象和映射规则
    mapping_rules = []
    all_objects = {'input': [], 'output': []}
    
    for pair_id, (input_grid, output_grid) in enumerate(task_data['train']):
        # 提取输入对象
        input_objects = extract_objects_from_grid(input_grid)
        all_objects['input'].append((pair_id, input_objects))
        
        # 提取输出对象
        output_objects = extract_objects_from_grid(output_grid)
        all_objects['output'].append((pair_id, output_objects))
        
        # 创建映射规则
        rule = create_mapping_rule(pair_id, input_objects, output_objects)
        mapping_rules.append(rule)
    
    # 2. 构建关系库
    integrated_system.build_libraries_from_data(mapping_rules, all_objects)
    
    # 3. 提取模式和规则
    patterns_and_rules = integrated_system.extract_patterns_and_rules()
    
    # 4. 分析结果
    print("\n==== 分析结果 ====")
    
    print("\n提取的模式:")
    for category, patterns in patterns_and_rules["extracted_patterns"].items():
        print(f"  {category} 类别: {len(patterns)} 个模式")
        for i, pattern in enumerate(patterns[:3]):  # 只显示前3个
            print(f"    {i+1}. {pattern.get('pattern_name')}: {pattern.get('description', '')}")
    
    print("\n生成的规则:")
    for i, rule in enumerate(patterns_and_rules["composite_rules"][:5]):  # 只显示前5个
        print(f"  {i+1}. {rule.get('description')}")
        print(f"     类型: {rule.get('rule_type')}")
        print(f"     置信度: {rule.get('confidence')}")
    
    # 5. 应用规则到测试输入
    if 'test' in task_data:
        test_input = task_data['test'][0]['input']
        
        # 提取测试特征
        test_features = {
            'shapes': extract_shapes_from_grid(test_input),
            'colors': extract_colors_from_grid(test_input)
        }
        
        # 应用规则
        predicted_output = integrated_system.apply_rules_to_grid(test_input, test_features)
        
        print("\n预测输出:")
        print_grid(predicted_output)
        
        # 保存结果
        integrated_system.export_to_json("integrated_analysis_results.json")
        
        return predicted_output
    
    return patterns_and_rules

# 辅助函数
def extract_objects_from_grid(grid):
    """从网格中提取对象"""
    # ... 实现对象提取逻辑 ...
    return []

def create_mapping_rule(pair_id, input_objects, output_objects):
    """创建映射规则"""
    # ... 实现规则创建逻辑 ...
    return {}

def extract_shapes_from_grid(grid):
    """从网格中提取形状"""
    # ... 实现形状提取逻辑 ...
    return []

def extract_colors_from_grid(grid):
    """从网格中提取颜色"""
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:  # 假设0是背景色
                colors.add(cell)
    return list(colors)

def print_grid(grid):
    """打印网格"""
    for row in grid:
        print(" ".join(str(cell) for cell in row))