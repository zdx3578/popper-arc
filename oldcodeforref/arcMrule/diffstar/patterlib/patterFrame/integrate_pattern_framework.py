"""
将模式分析框架整合到IntegratedARCPatternSystem
"""

class IntegratedARCPatternSystem:
    """增强版ARC关系与模式系统"""
    
    def __init__(self, debug=False, debug_print=None):
        """初始化整合系统"""
        self.debug = debug
        self.debug_print = debug_print or (lambda x: print(x) if debug else None)
        
        # ... 原有的初始化代码 ...
        
        # 初始化模式分析框架
        self.pattern_framework = ARCPatternFramework()
        
        # 原始网格存储
        self.original_grids = {}
        
        # ... 其他原有的初始化代码 ...
    
    def store_grid(self, pair_id, io_type, grid):
        """存储原始网格"""
        if pair_id not in self.original_grids:
            self.original_grids[pair_id] = {}
        self.original_grids[pair_id][io_type] = grid
    
    def analyze_with_pattern_framework(self, test_input=None, categories=None):
        """
        使用模式框架分析数据
        
        Args:
            test_input: 可选的测试输入网格
            categories: 可选的模式类别列表
            
        Returns:
            分析结果，包括检测到的模式和生成的规则
        """
        # 如果提供了测试输入，存储它
        if test_input is not None:
            self.store_grid(-1, 'test_input', test_input)
        
        detected_patterns = []
        
        # 1. 对每个训练对分析模式
        for pair_id in self.objects_by_pair:
            # 准备输入数据
            pair_data = {
                'pair_id': pair_id,
                'objects_by_attribute': self.objects_by_attribute,
                'object_attributes': self.object_attributes
            }
            
            # 添加输入网格
            if pair_id in self.original_grids and 'input' in self.original_grids[pair_id]:
                pair_data['grid'] = self.original_grids[pair_id]['input']
            
            # 检测所有模式
            pair_patterns = self.pattern_framework.detect_all_patterns(
                pair_data, categories=categories
            )
            
            # 添加源信息
            for pattern in pair_patterns:
                pattern['source_pair_id'] = pair_id
                pattern['source_type'] = 'input'
            
            detected_patterns.extend(pair_patterns)
            
            # 同样处理输出网格
            if pair_id in self.original_grids and 'output' in self.original_grids[pair_id]:
                pair_data['grid'] = self.original_grids[pair_id]['output']
                
                output_patterns = self.pattern_framework.detect_all_patterns(
                    pair_data, categories=categories
                )
                
                for pattern in output_patterns:
                    pattern['source_pair_id'] = pair_id
                    pattern['source_type'] = 'output'
                
                detected_patterns.extend(output_patterns)
        
        # 2. 如果提供了测试输入，也检测模式
        if test_input is not None:
            test_data = {
                'pair_id': -1,
                'grid': test_input,
                'objects_by_attribute': self.objects_by_attribute,
                'object_attributes': self.object_attributes
            }
            
            test_patterns = self.pattern_framework.detect_all_patterns(
                test_data, categories=categories
            )
            
            for pattern in test_patterns:
                pattern['source_pair_id'] = -1
                pattern['source_type'] = 'test_input'
            
            detected_patterns.extend(test_patterns)
        
        # 3. 聚合和分析模式
        pattern_groups = self._group_similar_patterns(detected_patterns)
        
        # 4. 生成规则
        generated_rules = self._generate_rules_from_patterns(pattern_groups)
        
        # 5. 按优先级排序规则
        generated_rules.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return {
            'detected_patterns': detected_patterns,
            'pattern_groups': pattern_groups,
            'generated_rules': generated_rules
        }
    
    def _group_similar_patterns(self, patterns):
        """
        将相似的模式分组
        
        Args:
            patterns: 检测到的模式列表
            
        Returns:
            模式组字典
        """
        pattern_groups = {}
        
        # 按模式类型分组
        for pattern in patterns:
            pattern_id = pattern.get('pattern_id')
            
            if pattern_id not in pattern_groups:
                pattern_groups[pattern_id] = []
            
            # 查找相似的现有组
            found_group = False
            for group in pattern_groups[pattern_id]:
                # 如果组内有至少一个相似模式，添加到该组
                for existing_pattern in group:
                    similarity = self.pattern_framework.calculate_pattern_similarity(
                        pattern_id, pattern, existing_pattern
                    )
                    
                    if similarity > 0.7:  # 相似度阈值
                        group.append(pattern)
                        found_group = True
                        break
                
                if found_group:
                    break
            
            # 如果没找到相似组，创建新组
            if not found_group:
                pattern_groups[pattern_id].append([pattern])
        
        return pattern_groups
    
    def _generate_rules_from_patterns(self, pattern_groups):
        """
        从模式组生成规则
        
        Args:
            pattern_groups: 模式组字典
            
        Returns:
            生成的规则列表
        """
        rules = []
        
        # 对每种模式类型
        for pattern_id, groups in pattern_groups.items():
            for group in groups:
                # 根据出现频次筛选有意义的模式组
                if len(group) >= 2:
                    # 分析模式变换
                    rule = self._analyze_pattern_transformations(pattern_id, group)
                    if rule:
                        rules.append(rule)
        
        return rules
    
    def _analyze_pattern_transformations(self, pattern_id, pattern_group):
        """
        分析模式组中的变换规律
        
        Args:
            pattern_id: 模式ID
            pattern_group: 模式组
            
        Returns:
            生成的规则或None
        """
        # 区分输入和输出模式
        input_patterns = [p for p in pattern_group if p.get('source_type') == 'input']
        output_patterns = [p for p in pattern_group if p.get('source_type') == 'output']
        
        # 如果没有足够的数据，返回None
        if not input_patterns or not output_patterns:
            return None
        
        # 按数据对组织模式
        patterns_by_pair = {}
        for pattern in pattern_group:
            pair_id = pattern.get('source_pair_id')
            source_type = pattern.get('source_type')
            
            if pair_id not in patterns_by_pair:
                patterns_by_pair[pair_id] = {'input': [], 'output': []}
            
            if source_type in patterns_by_pair[pair_id]:
                patterns_by_pair[pair_id][source_type].append(pattern)
        
        # 分析变换类型
        transformations = []
        
        for pair_id, pair_patterns in patterns_by_pair.items():
            if pair_id == -1:  # 跳过测试数据
                continue
                
            # 只考虑同时有输入和输出模式的数据对
            if not pair_patterns['input'] or not pair_patterns['output']:
                continue
            
            # 对每个输入模式
            for input_pattern in pair_patterns['input']:
                # 查找最相似的输出模式
                best_match = None
                best_similarity = 0
                
                for output_pattern in pair_patterns['output']:
                    similarity = self.pattern_framework.calculate_pattern_similarity(
                        pattern_id, input_pattern, output_pattern
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = output_pattern
                
                # 如果找到匹配，分析变换
                if best_match and best_similarity > 0.5:
                    transformation = self._extract_transformation(input_pattern, best_match)
                    if transformation:
                        transformation['pair_id'] = pair_id
                        transformation['similarity'] = best_similarity
                        transformations.append(transformation)
        
        # 如果没有发现变换，返回None
        if not transformations:
            return None
        
        # 寻找最常见的变换类型
        transformation_types = {}
        for t in transformations:
            t_type = t['type']
            if t_type not in transformation_types:
                transformation_types[t_type] = []
            transformation_types[t_type].append(t)
        
        # 选择支持最多的变换类型
        best_type = max(transformation_types.items(), key=lambda x: len(x[1]))
        type_name, type_transformations = best_type
        
        # 创建规则
        rule = {
            'rule_type': 'pattern_based_rule',
            'pattern_id': pattern_id,
            'pattern_name': self.pattern_framework.pattern_definitions[pattern_id]['name'],
            'transformation': {
                'type': type_name,
                'parameters': self._merge_transformation_parameters(type_transformations)
            },
            'supporting_pairs': list(set(t['pair_id'] for t in type_transformations)),
            'confidence': len(type_transformations) / len(transformations),
            'priority': self.pattern_framework.pattern_priorities.get(pattern_id, 50) * (len(type_transformations) / len(transformations)),
            'description': f"基于{self.pattern_framework.pattern_definitions[pattern_id]['name']}的规则: {type_name}变换"
        }
        
        return rule
    
    def _extract_transformation(self, input_pattern, output_pattern):
        """
        提取两个模式实例之间的变换
        
        Args:
            input_pattern: 输入模式
            output_pattern: 输出模式
            
        Returns:
            变换描述或None
        """
        # 针对不同模式类型的变换提取逻辑
        pattern_id = input_pattern.get('pattern_id')
        
        # 示例：处理4-Box模式
        if pattern_id == 'four_box_pattern':
            # 检查中心颜色变化
            in_center_color = input_pattern.get('center_color')
            out_center_color = output_pattern.get('center_color')
            
            if in_center_color != out_center_color:
                return {
                    'type': 'color_change',
                    'from_color': in_center_color,
                    'to_color': out_center_color
                }
            
            # 检查周围颜色变化
            in_surr_color = input_pattern.get('surrounding_color')
            out_surr_color = output_pattern.get('surrounding_color')
            
            if in_surr_color != out_surr_color:
                return {
                    'type': 'surrounding_color_change',
                    'from_color': in_surr_color,
                    'to_color': out_surr_color
                }
        
        # 其他模式类型的变换提取...
        
        return None
    
    def _merge_transformation_parameters(self, transformations):
        """
        合并多个变换实例的参数
        
        Args:
            transformations: 变换列表
            
        Returns:
            合并后的参数
        """
        if not transformations:
            return {}
            
        # 以第一个变换为基础
        base_params = transformations[0].copy()
        del base_params['pair_id']
        del base_params['similarity']
        
        # 对于某些变换类型，需要特殊处理
        if base_params['type'] == 'color_change':
            # 统计颜色变换频率
            color_changes = {}
            for t in transformations:
                key = (t['from_color'], t['to_color'])
                if key not in color_changes:
                    color_changes[key] = 0
                color_changes[key] += 1
            
            # 选择最常见的颜色变换
            most_common = max(color_changes.items(), key=lambda x: x[1])
            from_color, to_color = most_common[0]
            
            return {
                'from_color': from_color,
                'to_color': to_color,
                'frequency': most_common[1] / len(transformations)
            }
        
        # 其他类型的参数合并逻辑...
        
        return base_params
    
    def apply_generated_rules(self, input_grid, rules=None):
        """
        应用生成的规则到输入网格
        
        Args:
            input_grid: 输入网格
            rules: 可选的规则列表，如果不提供则使用最近分析生成的规则
            
        Returns:
            应用规则后的输出网格
        """
        if rules is None:
            # 重新分析并获取规则
            result = self.analyze_with_pattern_framework(input_grid)
            rules = result.get('generated_rules', [])
        
        # 初始化输出网格为输入网格的副本
        output_grid = [row[:] for row in input_grid]
        
        # 检测输入网格中的模式实例
        input_data = {
            'grid': input_grid,
            'pair_id': -1,
            'objects_by_attribute': self.objects_by_attribute,
            'object_attributes': self.object_attributes
        }
        
        # 按优先级应用规则
        for rule in sorted(rules, key=lambda x: x.get('priority', 0), reverse=True):
            pattern_id = rule.get('pattern_id')
            transformation = rule.get('transformation', {})
            
            # 检测模式实例
            try:
                pattern_instances = self.pattern_framework.detect_pattern(pattern_id, input_data)
                
                # 对每个检测到的实例应用变换
                for instance in pattern_instances:
                    instance['transformation'] = transformation
                    
                    # 应用模式
                    result = self.pattern_framework.apply_pattern(
                        pattern_id, instance, {'grid': output_grid}
                    )
                    
                    # 更新输出网格
                    if 'grid' in result:
                        output_grid = result['grid']
                
            except Exception as e:
                if self.debug:
                    self.debug_print(f"应用规则时出错: {e}")
        
        return output_grid