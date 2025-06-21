# 初始化模式库和提取器
pattern_library = PatternLibrary(library_path="pattern_library.json")
pattern_extractor = EnhancedPatternExtractor(pattern_library=pattern_library, debug=True)

# 添加自定义模式到库中
pattern_library.add_user_pattern("transformation", {
    "id": "custom_color_spread_pattern",
    "name": "颜色扩散模式",
    "description": "一种颜色向周围扩散",
    "detector": pattern_library._detect_color_spread_pattern,
    "parameters": {"spread_threshold": 2}
})

# 在分析ARC任务时调用
def analyze_arc_task(task_data):
    # ... 前置处理代码 ...
    
    # 使用增强版规则集成
    analyzer._integrate_rules()
    
    # 打印发现的模式
    print("发现以下模式:")
    for category, patterns in analyzer.extracted_patterns.items():
        print(f"  {category} 类别: {len(patterns)} 个模式")
        for pattern in patterns[:3]:  # 只显示前3个
            print(f"    - {pattern.get('pattern_name')}: {pattern.get('description')}")
    
    # 打印生成的规则
    print("\n生成以下规则:")
    for rule in analyzer.composite_rules[:5]:  # 只显示前5个
        print(f"  - {rule.get('description')}")
        print(f"    支持数据对: {len(rule.get('supporting_pairs', []))}")
        print(f"    置信度: {rule.get('confidence')}")
    
    # ... 后续处理代码 ...