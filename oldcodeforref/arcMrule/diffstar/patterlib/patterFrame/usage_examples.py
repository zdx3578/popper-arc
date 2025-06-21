def analyze_arc_task_with_pattern_framework(task_data):
    """使用模式框架分析ARC任务"""
    # 初始化系统
    system = IntegratedARCPatternSystem(debug=True)
    
    # 处理训练数据
    for pair_id, (input_grid, output_grid) in enumerate(task_data['train']):
        # 存储原始网格
        system.store_grid(pair_id, 'input', input_grid)
        system.store_grid(pair_id, 'output', output_grid)
        
        # 提取对象
        input_objects = extract_objects(input_grid, pair_id, 'input')
        output_objects = extract_objects(output_grid, pair_id, 'output')
        
        # 添加对象到系统
        for obj in input_objects:
            system._process_single_object(pair_id, 'input', obj)
        
        for obj in output_objects:
            system._process_single_object(pair_id, 'output', obj)
    
    # 提取并分析模式
    result = system.analyze_with_pattern_framework()
    
    print("\n=== 检测到的模式 ===")
    for category, patterns in groupby(result['detected_patterns'], key=lambda x: x.get('category')):
        pattern_list = list(patterns)
        print(f"{category}: {len(pattern_list)} 个模式实例")
    
    print("\n=== 生成的规则 ===")
    for rule in result['generated_rules']:
        print(f"{rule.get('pattern_name')}: {rule.get('description')}")
        print(f"  支持数据对: {rule.get('supporting_pairs')}")
        print(f"  置信度: {rule.get('confidence'):.2f}")
    
    # 应用规则到测试输入
    if 'test' in task_data:
        test_input = task_data['test'][0]['input']
        predicted_output = system.apply_generated_rules(test_input)
        
        print("\n=== 预测结果 ===")
        print_grid(predicted_output)
        
        return predicted_output
    
    return result

def detect_specific_pattern(grid, pattern_type='four_box_pattern'):
    """检测特定类型的模式"""
    framework = ARCPatternFramework()
    
    # 准备输入数据
    input_data = {'grid': grid}
    
    # 检测模式
    detected_patterns = framework.detect_pattern(pattern_type, input_data)
    
    print(f"检测到 {len(detected_patterns)} 个 {pattern_type} 模式实例:")
    for pattern in detected_patterns:
        if pattern_type == 'four_box_pattern':
            print(f"中心位置 {pattern.get('center_position')}: "
                  f"中心颜色 {pattern.get('center_color')} "
                  f"被颜色 {pattern.get('surrounding_color')} 包围")
    
    return detected_patterns

def apply_pattern_transformation(grid, pattern_instance, transformation):
    """应用模式变换"""
    framework = ARCPatternFramework()
    
    # 添加变换信息
    pattern_instance['transformation'] = transformation
    
    # 应用模式
    result = framework.apply_pattern(
        pattern_instance.get('pattern_id', 'four_box_pattern'),
        pattern_instance,
        {'grid': grid}
    )
    
    return result.get('grid')