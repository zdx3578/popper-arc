"""
完整ARC模式分析系统集成
"""

class EnhancedARCSystem:
    """
    增强版ARC系统 - 集成关系库、模式框架和规则引擎
    """
    
    def __init__(self, debug=False):
        """初始化增强版ARC系统"""
        self.debug = debug
        self.debug_print = lambda x: print(x) if debug else None
        
        # 初始化各组件
        self.relationship_system = IntegratedARCPatternSystem(debug=debug)
        self.pattern_framework = ARCPatternFramework()
        
        # 存储训练和测试数据
        self.training_data = []
        self.test_data = []
        
        # 规则存储
        self.extracted_rules = []
        
        # 最终结果
        self.predicted_outputs = {}
    
    def load_task(self, task_data):
        """加载ARC任务数据"""
        # 存储训练数据
        self.training_data = task_data.get('train', [])
        
        # 存储测试数据
        self.test_data = task_data.get('test', [])
        
        # 处理训练数据
        for pair_id, (input_grid, output_grid) in enumerate(self.training_data):
            # 存储网格
            self.relationship_system.store_grid(pair_id, 'input', input_grid)
            self.relationship_system.store_grid(pair_id, 'output', output_grid)
            
            # 提取并处理对象
            input_objects = extract_objects(input_grid, pair_id, 'input')
            output_objects = extract_objects(output_grid, pair_id, 'output')
            
            for obj in input_objects:
                self.relationship_system._process_single_object(pair_id, 'input', obj)
            
            for obj in output_objects:
                self.relationship_system._process_single_object(pair_id, 'output', obj)
        
        return self
    
    def analyze(self):
        """分析任务数据，提取模式和规则"""
        # 1. 使用关系系统提取基础特征和关系
        if hasattr(self.relationship_system, 'build_relationship_libraries'):
            self.relationship_system.build_relationship_libraries()
        
        # 2. 使用模式框架分析模式
        analysis_result = self.relationship_system.analyze_with_pattern_framework()
        
        # 3. 获取生成的规则
        self.extracted_rules = analysis_result.get('generated_rules', [])
        
        # 4. 打印分析结果
        if self.debug:
            print("\n=== 模式分析结果 ===")
            pattern_count = len(analysis_result.get('detected_patterns', []))
            rule_count = len(self.extracted_rules)
            
            print(f"检测到 {pattern_count} 个模式实例")
            print(f"生成了 {rule_count} 条规则")
            
            # 显示最高优先级的规则
            if self.extracted_rules:
                top_rules = sorted(self.extracted_rules, key=lambda x: x.get('priority', 0), reverse=True)[:3]
                print("\n优先级最高的规则:")
                for i, rule in enumerate(top_rules):
                    print(f"{i+1}. {rule.get('description')}")
                    print(f"   类型: {rule.get('pattern_name')}")
                    print(f"   置信度: {rule.get('confidence', 0):.2f}")
                    print(f"   支持数据对: {rule.get('supporting_pairs')}")
        
        return self
    
    def predict(self):
        """对测试数据进行预测"""
        # 对每个测试输入应用规则
        for i, test_case in enumerate(self.test_data):
            test_input = test_case['input']
            
            # 应用提取的规则
            predicted_output = self.relationship_system.apply_generated_rules(
                test_input, self.extracted_rules
            )
            
            # 存储预测结果
            self.predicted_outputs[i] = predicted_output
            
            if self.debug:
                print(f"\n=== 测试用例 {i} 的预测结果 ===")
                print_grid(predicted_output)
        
        return self.predicted_outputs

def run_arc_task(task_data):
    """使用增强系统运行ARC任务"""
    system = EnhancedARCSystem(debug=True)
    
    # 加载任务并执行分析
    result = (
        system
        .load_task(task_data)
        .analyze()
        .predict()
    )
    
    return result