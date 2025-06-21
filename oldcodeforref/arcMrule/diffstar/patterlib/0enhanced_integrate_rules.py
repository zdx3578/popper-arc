def _integrate_rules(self):
    """
    增强版规则集成方法 - 不仅整合全局和条件规则，还提取通用模式
    """
    # 1. 基础规则整合（保留原有功能）
    self._integrate_basic_rules()
    
    # 2. 使用增强型模式提取器
    extractor = EnhancedPatternExtractor(debug=self.debug, debug_print=self.debug_print)
    
    # 准备对象数据 - 根据您的数据结构进行适配
    objects_data = self._prepare_objects_data_for_extractor()
    
    # 提取模式并获取结果
    extraction_results = extractor.process_objects_data(objects_data)
    
    # 3. 合并提取的模式和规则
    self.extracted_patterns = extraction_results["extracted_patterns"]
    
    # 将提取器的复合规则添加到现有规则中
    for rule in extraction_results["composite_rules"]:
        if rule not in self.composite_rules:  # 避免重复
            self.composite_rules.append(rule)
    
    # 4. 按覆盖率和置信度排序
    self._sort_composite_rules()

def _prepare_objects_data_for_extractor(self):
    """准备用于模式提取器的对象数据结构"""
    objects_data = {
        'input': {},
        'output': {}
    }
    
    # 为输入和输出对象创建数据结构
    for io_type in ['input', 'output']:
        for pair_id in self.objects_by_pair:
            if io_type in self.objects_by_pair[pair_id]:
                objects_data[io_type][pair_id] = {}
                
                for obj_id in self.objects_by_pair[pair_id][io_type]:
                    # 收集该对象的所有相关信息
                    obj_info = self._collect_object_info(pair_id, io_type, obj_id)
                    objects_data[io_type][pair_id][obj_id] = obj_info
    
    return objects_data

def _collect_object_info(self, pair_id, io_type, obj_id):
    """收集单个对象的所有相关信息"""
    obj_info = {
        'obj_id': obj_id,
        'pair_id': pair_id,
        'io_type': io_type
    }
    
    # 添加形状信息
    for shape_hash, objects in self.objects_by_shape.items():
        if (pair_id, io_type, obj_id) in objects:
            obj_info['shape_hash'] = shape_hash
            break
    
    # 添加颜色信息
    for color, objects in self.objects_by_color.items():
        if (pair_id, io_type, obj_id) in objects:
            obj_info['color'] = color
            break
    
    # 添加位置信息（如果有）
    # ...
    
    return obj_info

def _integrate_basic_rules(self):
    """整合基本的全局操作规则和条件规则"""
    # 按颜色分组条件规则
    color_to_conditional_rules = defaultdict(list)
    for rule in self.conditional_rules:
        from_color = rule['effect']['from_color']
        color_to_conditional_rules[from_color].append(rule)

    # 查找可以组合的规则
    for global_rule in self.global_operation_rules:
        color = global_rule.get('color')
        operation = global_rule.get('operation')

        # 如果是移除操作，查找相关的条件规则
        if operation == 'removed' and color in color_to_conditional_rules:
            related_conditional_rules = color_to_conditional_rules[color]

            # 创建组合规则
            composite_rule = {
                'rule_type': 'composite_rule',
                'base_rule': {
                    'type': 'global_color_operation',
                    'color': color,
                    'operation': operation
                },
                'conditional_rules': [],
                'all_supporting_pairs': set(global_rule.get('supporting_pairs', [])),
                'confidence': global_rule.get('confidence', 0),
                'description': f"处理颜色为{color}的对象的综合规则"
            }

            # 添加相关的条件规则
            for cond_rule in related_conditional_rules:
                composite_rule['conditional_rules'].append(cond_rule)
                composite_rule['all_supporting_pairs'].update(cond_rule.get('supporting_pairs', []))

            # 计算综合置信度
            if composite_rule['conditional_rules']:
                avg_cond_confidence = sum(r.get('confidence', 0) for r in composite_rule['conditional_rules']) / len(composite_rule['conditional_rules'])
                composite_rule['confidence'] = (global_rule.get('confidence', 0) + avg_cond_confidence) / 2

            # 转换supporting_pairs为列表，以便序列化
            composite_rule['all_supporting_pairs'] = list(composite_rule['all_supporting_pairs'])

            # 计算覆盖率分数
            composite_rule['coverage_score'] = len(composite_rule['all_supporting_pairs'])

            # 添加到复合规则列表
            self.composite_rules.append(composite_rule)

def _sort_composite_rules(self):
    """按覆盖率和置信度排序复合规则"""
    self.composite_rules.sort(
        key=lambda x: (x.get('coverage_score', 0), x.get('confidence', 0)),
        reverse=True
    )