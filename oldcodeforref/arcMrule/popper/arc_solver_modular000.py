from arcMrule.diffstar.weighted_analyzer.analyzer_core import WeightedARCDiffAnalyzer
from arcMrule.diffstar.weighted_analyzer.object_matching import ObjectMatcher

class ARCSolverModular:
    # 保留之前的代码...

    def analyze_transformation(self, input_grid, output_grid):
        """分析输入输出转换，使用WeightedARCDiffAnalyzer"""
        # 创建分析器实例
        analyzer = WeightedARCDiffAnalyzer(debug=self.debug)

        # 设置背景色
        background_color = 0  # 默认黑色背景，可根据需要调整
        if hasattr(self, 'background_colors') and self.background_colors:
            background_color = list(self.background_colors)[0]

        # 提取参数设置
        param = (True, True, False)  # 默认参数，可根据需要调整

        # 分析训练对
        pair_id = 0  # 由于只分析一个对，使用固定ID
        analyzer.add_train_pair(pair_id, input_grid, output_grid, param, background_color)

        # 获取分析结果
        if analyzer.oneInOut_mapping_rules:
            mapping_rule = analyzer.oneInOut_mapping_rules[0]
            input_objects = [obj for pid, objs in analyzer.all_objects['input'] for obj in objs if pid == pair_id]
            output_objects = [obj for pid, objs in analyzer.all_objects['output'] for obj in objs if pid == pair_id]

            # 获取转换规则
            transformation = mapping_rule.get('input_to_output_transformation', {})

            return transformation, input_objects, output_objects
        else:
            # 如果分析失败，返回空结果
            return {}, [], []

    def prepare_popper_data(self):
        """准备Popper训练数据 - 使用插件架构和WeightedARCDiffAnalyzer"""
        all_facts = []
        positive_examples = []
        negative_examples = []

        # 处理每个训练对
        for pair_id, (input_grid, output_grid) in enumerate(self.train_pairs):
            # 使用适合的分析器分析转换
            transformation, input_objects, output_objects = self.analyze_transformation(input_grid, output_grid)
            self.oneInOut_mapping_rules[pair_id] = transformation
            self.all_objects[pair_id] = {"input": input_objects, "output": output_objects}

            # 生成Popper事实
            pair_facts = self._convert_weighted_objects_to_popper_facts(
                pair_id, input_grid, output_grid, input_objects, output_objects, transformation
            )
            all_facts.extend(pair_facts)

            # 应用所有适用的插件生成特定事实和例子
            for plugin in self.applicable_plugins:
                plugin_facts = plugin.generate_facts(pair_id, input_objects, output_objects)
                plugin_positives = plugin.generate_positive_examples(pair_id)
                plugin_negatives = plugin.generate_negative_examples(pair_id)

                all_facts.extend(plugin_facts)
                positive_examples.extend(plugin_positives)
                negative_examples.extend(plugin_negatives)

        return all_facts, positive_examples, negative_examples

    def _convert_weighted_objects_to_popper_facts(self, pair_id, input_grid, output_grid,
                                                input_objects, output_objects, transformation):
        """将加权对象转换为Popper可用的事实"""
        facts = []

        # 添加网格尺寸信息
        height_in, width_in = len(input_grid), len(input_grid[0])
        facts.append(f"grid_size({pair_id}, {width_in}, {height_in}).")

        # 添加通用对象信息
        for i, obj in enumerate(input_objects):
            obj_id = f"in_{pair_id}_{i}"
            facts.append(f"object({obj_id}).")
            facts.append(f"input_object({obj_id}).")
            facts.append(f"color({obj_id}, {obj.main_color}).")

            # 添加位置和尺寸信息
            facts.append(f"x_min({obj_id}, {obj.left}).")
            facts.append(f"y_min({obj_id}, {obj.top}).")
            facts.append(f"x_max({obj_id}, {obj.left + obj.width}).")
            facts.append(f"y_max({obj_id}, {obj.top + obj.height}).")
            facts.append(f"width({obj_id}, {obj.width}).")
            facts.append(f"height({obj_id}, {obj.height}).")
            facts.append(f"size({obj_id}, {obj.size}).")
            facts.append(f"weight({obj_id}, {obj.obj_weight}).")

            # 添加形状信息
            facts.append(f"shape_hash({obj_id}, \"{obj.obj_000}\").")

            # 如果有特殊形状特征
            if hasattr(obj, 'is_rectangle') and obj.is_rectangle:
                facts.append(f"is_rectangle({obj_id}).")
            if hasattr(obj, 'touches_edge') and obj.touches_edge:
                facts.append(f"touches_edge({obj_id}).")

        for i, obj in enumerate(output_objects):
            obj_id = f"out_{pair_id}_{i}"
            facts.append(f"object({obj_id}).")
            facts.append(f"output_object({obj_id}).")
            facts.append(f"color({obj_id}, {obj.main_color}).")

            # 添加位置和尺寸信息
            facts.append(f"x_min({obj_id}, {obj.left}).")
            facts.append(f"y_min({obj_id}, {obj.top}).")
            facts.append(f"x_max({obj_id}, {obj.left + obj.width}).")
            facts.append(f"y_max({obj_id}, {obj.top + obj.height}).")
            facts.append(f"width({obj_id}, {obj.width}).")
            facts.append(f"height({obj_id}, {obj.height}).")
            facts.append(f"size({obj_id}, {obj.size}).")
            facts.append(f"weight({obj_id}, {obj.obj_weight}).")

            # 添加形状信息
            facts.append(f"shape_hash({obj_id}, \"{obj.obj_000}\").")

            # 如果有特殊形状特征
            if hasattr(obj, 'is_rectangle') and obj.is_rectangle:
                facts.append(f"is_rectangle({obj_id}).")
            if hasattr(obj, 'touches_edge') and obj.touches_edge:
                facts.append(f"touches_edge({obj_id}).")

        # 添加转换关系事实
        if transformation:
            # 保留的对象
            for preserved in transformation.get('preserved_objects', []):
                in_id = preserved.get('input_obj_id')
                out_id = preserved.get('output_obj_id')
                if in_id is not None and out_id is not None:
                    in_idx = next((i for i, obj in enumerate(input_objects) if obj.obj_id == in_id), None)
                    out_idx = next((i for i, obj in enumerate(output_objects) if obj.obj_id == out_id), None)
                    if in_idx is not None and out_idx is not None:
                        facts.append(f"preserved({pair_id}, in_{pair_id}_{in_idx}, out_{pair_id}_{out_idx}).")

            # 修改的对象
            for modified in transformation.get('modified_objects', []):
                in_id = modified.get('input_obj_id')
                out_id = modified.get('output_obj_id')
                if in_id is not None and out_id is not None:
                    in_idx = next((i for i, obj in enumerate(input_objects) if obj.obj_id == in_id), None)
                    out_idx = next((i for i, obj in enumerate(output_objects) if obj.obj_id == out_id), None)
                    if in_idx is not None and out_idx is not None:
                        facts.append(f"modified({pair_id}, in_{pair_id}_{in_idx}, out_{pair_id}_{out_idx}).")

                        # 添加变换细节
                        transform = modified.get('transformation', {})
                        if transform:
                            transform_type = transform.get('type', '')
                            facts.append(f"transform_type({pair_id}, in_{pair_id}_{in_idx}, out_{pair_id}_{out_idx}, {transform_type}).")

                            # 位置变化
                            pos_change = transform.get('position_change', {})
                            if pos_change:
                                dr = pos_change.get('delta_row', 0)
                                dc = pos_change.get('delta_col', 0)
                                facts.append(f"position_change({pair_id}, in_{pair_id}_{in_idx}, out_{pair_id}_{out_idx}, {dr}, {dc}).")

                            # 颜色变换
                            color_trans = transform.get('color_transform', {})
                            if color_trans and 'color_mapping' in color_trans:
                                for from_color, to_color in color_trans['color_mapping'].items():
                                    facts.append(f"color_change({pair_id}, {from_color}, {to_color}).")

            # 移除的对象
            for removed in transformation.get('removed_objects', []):
                in_id = removed.get('input_obj_id')
                if in_id is not None:
                    in_idx = next((i for i, obj in enumerate(input_objects) if obj.obj_id == in_id), None)
                    if in_idx is not None:
                        facts.append(f"removed({pair_id}, in_{pair_id}_{in_idx}).")

            # 添加的对象
            for added in transformation.get('added_objects', []):
                out_id = added.get('output_obj_id')
                if out_id is not None:
                    out_idx = next((i for i, obj in enumerate(output_objects) if obj.obj_id == out_id), None)
                    if out_idx is not None:
                        facts.append(f"added({pair_id}, out_{pair_id}_{out_idx}).")

                        # 如果有生成信息
                        sources = added.get('generated_from', [])
                        for source in sources:
                            source_type = source.get('type', '')
                            facts.append(f"generated_by({pair_id}, out_{pair_id}_{out_idx}, {source_type}).")

        return facts




