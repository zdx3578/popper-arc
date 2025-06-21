def test_4box_pattern_detection():
    """测试4-Box模式检测"""
    # 创建简单的测试网格
    test_grid = [
        [0, 2, 0, 0, 0],
        [2, 1, 2, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 2, 3, 2],
        [0, 0, 0, 2, 0]
    ]
    
    # 初始化系统
    system = IntegratedARCPatternSystem(debug=True)
    
    # 存储测试网格
    if not hasattr(system, 'original_grids'):
        system.original_grids = {}
    system.original_grids[0] = {'input': test_grid}
    
    # 设置pair_id为0
    system.objects_by_pair[0] = {'input': []}
    
    # 手动添加一个 4-Box 模式实例
    pattern = {
        'pattern_type': 'four_box_pattern',
        'center_color': 1,
        'surrounding_color': 2,
        'surrounding_ratio': 1.0,
        'supporting_pairs': [0],
        'confidence': 1.0,
        'description': "颜色1的对象被颜色2包围"
    }
    
    # 创建规则
    rule = system._create_four_box_pattern_rule(pattern)
    
    # 应用规则（假设我们想将中心颜色改为5）
    rule['transformation'] = {
        'type': 'color_change',
        'parameters': {'new_color': 5},
        'confidence': 1.0
    }
    
    # 复制网格并应用规则
    output_grid = [row[:] for row in test_grid]
    output_grid = system._apply_4box_pattern_rule(rule, output_grid)
    
    print("输入网格:")
    for row in test_grid:
        print(row)
    
    print("\n输出网格:")
    for row in output_grid:
        print(row)
    
    # 检测网格中的4-Box模式
    detected_patterns = system._detect_four_box_pattern()
    print(f"\n检测到 {len(detected_patterns)} 个4-Box模式")
    for p in detected_patterns:
        print(f"中心颜色: {p.get('center_color')}, 围绕颜色: {p.get('surrounding_color')}")
        print(f"描述: {p.get('description')}")

# 执行测试
test_4box_pattern_detection()