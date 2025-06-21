"""
规则应用模块

负责将识别出的模式和规则应用到新数据上。
"""

from typing import List, Dict, Any, Callable, Optional


class RuleApplier:
    """处理规则应用的类"""

    def __init__(self, debug_print=None):
        """
        初始化规则应用器

        Args:
            debug_print: 调试打印函数（可选）
        """
        self.debug_print = debug_print

    def apply_patterns(self, input_grid, common_patterns, input_obj_infos, debug=False):
        """
        将识别的共有模式应用到输入网格上

        Args:
            input_grid: 输入网格
            common_patterns: 共有模式字典
            input_obj_infos: 输入对象信息列表
            debug: 是否启用调试模式

        Returns:
            预测的输出网格
        """
        # 创建输出网格（初始为输入的副本）
        output_grid = [list(row) for row in input_grid]
        height, width = len(input_grid), len(input_grid[0])

        # 创建2D的转换记录，跟踪每个位置是否已被转换
        transformed = [[False for _ in range(width)] for _ in range(height)]

        # 按权重对对象排序（权重高的优先处理）
        sorted_obj_infos = sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True)

        # 1. 应用颜色映射，优先考虑高权重映射
        if "color_mappings" in common_patterns:
            # 获取颜色映射并按加权置信度排序
            color_mappings = common_patterns["color_mappings"].get("mappings", {})
            sorted_mappings = sorted(
                [(from_color, mapping) for from_color, mapping in color_mappings.items()],
                key=lambda x: x[1].get("weighted_confidence", 0),
                reverse=True
            )

            for from_color, mapping in sorted_mappings:
                to_color = mapping["to_color"]
                cells_changed = 0

                # 根据对象权重应用颜色映射
                for obj_info in sorted_obj_infos:
                    for val, (i, j) in obj_info.original_obj:
                        if val == from_color:
                            output_grid[i][j] = to_color
                            transformed[i][j] = True
                            cells_changed += 1

                if debug and cells_changed > 0 and self.debug_print:
                    weighted_conf = mapping.get("weighted_confidence", 0)
                    self.debug_print(f"应用颜色映射: {from_color} -> {to_color}, 加权置信度: {weighted_conf:.2f}, 改变了 {cells_changed} 个单元格")

        # 2. 应用颜色模式（如颜色偏移）
        if "color_mappings" in common_patterns:
            for pattern in common_patterns["color_mappings"].get("patterns", []):
                if pattern["type"] == "color_offset" and pattern.get("weighted_score", 0) > 1:
                    offset = pattern["offset"]
                    cells_changed = 0

                    # 优先处理高权重对象
                    for obj_info in sorted_obj_infos:
                        for val, (i, j) in obj_info.original_obj:
                            if val != 0 and not transformed[i][j]:  # 不处理背景和已转换的单元格
                                try:
                                    new_val = (int(val) + offset) % 10  # 假设颜色范围是0-9
                                    output_grid[i][j] = new_val
                                    transformed[i][j] = True
                                    cells_changed += 1
                                except (ValueError, TypeError):
                                    # 跳过无法进行数值运算的颜色
                                    pass

                    if debug and cells_changed > 0 and self.debug_print:
                        weighted_score = pattern.get("weighted_score", 0)
                        self.debug_print(f"应用颜色偏移: +{offset}, 加权得分: {weighted_score:.2f}, 改变了 {cells_changed} 个单元格")

        # 3. 应用位置变化，优先考虑高权重变化
        position_changes = common_patterns.get("position_changes", [])
        if position_changes:
            # 找到最高加权得分的位置变化
            best_position_change = max(position_changes, key=lambda x: x.get("weight_score", 0))

            if best_position_change["type"] == "absolute_position" and best_position_change.get("weight_score", 0) > 1:
                # 应用绝对位置变化
                dr, dc = best_position_change["delta_row"], best_position_change["delta_col"]

                # 创建临时网格保存结果
                temp_grid = [[0 for _ in range(width)] for _ in range(height)]
                cells_moved = 0

                # 对每个对象应用位置变化，优先处理高权重对象
                for obj_info in sorted_obj_infos:
                    # 对象中的每个像素
                    for val, (r, c) in obj_info.original_obj:
                        nr, nc = int(r + dr), int(c + dc)
                        if 0 <= nr < height and 0 <= nc < width:
                            temp_grid[nr][nc] = val  # 保留原始颜色
                            cells_moved += 1

                # 合并结果到最终输出网格
                for i in range(height):
                    for j in range(width):
                        if temp_grid[i][j] != 0:  # 只覆盖非零值
                            output_grid[i][j] = temp_grid[i][j]
                            transformed[i][j] = True

                if debug and cells_moved > 0 and self.debug_print:
                    weight_score = best_position_change.get("weight_score", 0)
                    self.debug_print(f"应用位置变化: ({dr}, {dc}), 加权得分: {weight_score:.2f}, 移动了 {cells_moved} 个单元格")

        # 4. 应用形状变换（如果有）
        shape_transformations = common_patterns.get("shape_transformations", [])
        if shape_transformations:
            best_shape_transform = max(shape_transformations, key=lambda x: x.get("weight_score", 0))

            # 目前形状变换与位置变化的应用方式类似，可以在此基础上扩展特殊的形状变换处理
            if debug and self.debug_print:
                self.debug_print(f"发现形状变换: {best_shape_transform['transform_type']}-{best_shape_transform['transform_name']}")

        return output_grid

    def apply_transformation_rules(self, input_grid, common_patterns, input_obj_infos, transformation_rules=None, debug=False):
        """
        应用提取的转换规则，将输入网格转换为预测的输出网格

        Args:
            input_grid: 输入网格
            common_patterns: 识别的共有模式
            input_obj_infos: 输入对象信息列表
            transformation_rules: 可选，特定的转换规则列表，如果不提供则使用当前累积的规则
            debug: 是否启用调试模式

        Returns:
            预测的输出网格
        """
        if debug and self.debug_print:
            self.debug_print(f"\n\n开始应用转换规则生成预测输出")

        # 如果没有提供转换规则或规则为空，使用常规模式匹配方法
        if not transformation_rules:
            if debug and self.debug_print:
                self.debug_print("未提供转换规则，使用共有模式方法")
            return self.apply_patterns(input_grid, common_patterns, input_obj_infos, debug)

        # 深拷贝输入网格作为初始输出
        output_grid = [list(row) for row in input_grid]
        height, width = len(input_grid), len(input_grid[0])

        # 创建2D的转换记录，跟踪每个位置是否已被转换
        transformed = [[False for _ in range(width)] for _ in range(height)]

        # 按权重对对象排序（权重高的优先处理）
        sorted_obj_infos = sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True)

        # 4. 尝试应用每个转换规则
        for rule in transformation_rules:
            # 根据规则中的"保留对象"列表，保留位置相同的对象
            if 'preserved_objects' in rule and rule['preserved_objects']:
                for preserved in rule['preserved_objects']:
                    # 这些对象在输入和输出中位置相同，无需更改
                    pass

            # 应用"修改对象"列表，将输入对象按规则转换
            if 'modified_objects' in rule and rule['modified_objects']:
                for modification in rule['modified_objects']:
                    # 找到输入中与修改对象特征匹配的对象
                    for input_obj in sorted_obj_infos:
                        # 检查是否可以应用此转换
                        if self._can_apply_transformation(input_obj, modification):
                            # 应用转换得到新对象
                            transformed_obj = self._apply_object_transformation(
                                input_obj, modification['transformation'])

                            # 更新输出网格和转换记录
                            for color, (i, j) in transformed_obj:
                                if 0 <= i < height and 0 <= j < width:
                                    output_grid[i][j] = color
                                    transformed[i][j] = True

            # 应用"移除对象"列表，从输出中移除指定对象
            if 'removed_objects' in rule and rule['removed_objects']:
                for removed in rule['removed_objects']:
                    # 找到输入中与移除对象特征匹配的对象
                    for input_obj in sorted_obj_infos:
                        if self._match_object_pattern(input_obj, removed):
                            # 从输出中移除此对象
                            for _, (i, j) in input_obj.original_obj:
                                if 0 <= i < height and 0 <= j < width:
                                    # 设置为背景色（通常为0）
                                    output_grid[i][j] = 0
                                    transformed[i][j] = True

            # 应用"添加对象"列表，在输出中添加新对象
            if 'added_objects' in rule and rule['added_objects']:
                for added in rule['added_objects']:
                    # 检查添加对象的生成来源
                    if 'generated_from' in added and added['generated_from']:
                        for source in added['generated_from']:
                            if source['type'] == 'transformed_input':
                                # 从输入对象生成新对象
                                input_id = source.get('input_obj_id')
                                # 找到对应的输入对象
                                for input_obj in sorted_obj_infos:
                                    if hasattr(input_obj, 'obj_id') and str(input_obj.obj_id) == str(input_id):
                                        # 应用变换
                                        new_obj = self._generate_object_from_transformation(
                                            input_obj, source['transformation'])

                                        # 更新输出网格
                                        for color, (i, j) in new_obj:
                                            if 0 <= i < height and 0 <= j < width:
                                                output_grid[i][j] = color
                                                transformed[i][j] = True

                            elif source['type'] == 'combined_inputs':
                                # 组合多个输入对象
                                # 实现对象组合逻辑
                                pass

        # 5. 应用共有模式（如果转换规则没有处理的区域）
        patterns_applied = self._apply_remaining_patterns(
            input_grid, output_grid, transformed, common_patterns, sorted_obj_infos, debug)

        if debug and self.debug_print:
            self.debug_print(f"完成规则应用，应用了 {len(transformation_rules)} 条转换规则")
            if patterns_applied:
                self.debug_print("并应用了额外的共有模式到未处理区域")

        return output_grid

    def _apply_remaining_patterns(self, input_grid, output_grid, transformed, common_patterns, input_obj_infos, debug=False):
        """应用共有模式到未被转换规则处理的区域"""
        patterns_applied = False
        height, width = len(input_grid), len(input_grid[0])

        # 应用位置变化模式
        if "position_changes" in common_patterns and common_patterns["position_changes"]:
            best_pattern = max(common_patterns["position_changes"], key=lambda x: x.get("weight_score", 0))

            if best_pattern["type"] == "absolute_position" and best_pattern.get("confidence", 0) > 0.5:
                dr, dc = best_pattern["delta_row"], best_pattern["delta_col"]

                # 创建临时网格保存结果
                temp_grid = [[0 for _ in range(width)] for _ in range(height)]
                cells_moved = 0

                # 对未处理的区域应用位置变化
                for i in range(height):
                    for j in range(width):
                        if not transformed[i][j] and input_grid[i][j] != 0:
                            ni, nj = i + dr, j + dc
                            if 0 <= ni < height and 0 <= nj < width:
                                temp_grid[ni][nj] = input_grid[i][j]
                                cells_moved += 1

                # 合并结果
                for i in range(height):
                    for j in range(width):
                        if temp_grid[i][j] != 0:
                            output_grid[i][j] = temp_grid[i][j]
                            transformed[i][j] = True

                if cells_moved > 0:
                    patterns_applied = True
                    if debug and self.debug_print:
                        self.debug_print(f"应用位置变化模式到未处理区域: ({dr}, {dc}), 移动了 {cells_moved} 个单元格")

        # 应用颜色映射模式
        if "color_mappings" in common_patterns and common_patterns["color_mappings"].get("mappings"):
            mappings = common_patterns["color_mappings"]["mappings"]
            sorted_mappings = sorted(
                [(from_color, mapping) for from_color, mapping in mappings.items()],
                key=lambda x: x[1].get("weighted_confidence", 0),
                reverse=True
            )

            cells_changed = 0
            for from_color, mapping in sorted_mappings:
                to_color = mapping["to_color"]

                # 应用到未处理区域
                for i in range(height):
                    for j in range(width):
                        if not transformed[i][j] and input_grid[i][j] == from_color:
                            output_grid[i][j] = to_color
                            transformed[i][j] = True
                            cells_changed += 1

            if cells_changed > 0:
                patterns_applied = True
                if debug and self.debug_print:
                    self.debug_print(f"应用颜色映射模式到未处理区域, 改变了 {cells_changed} 个单元格")

        return patterns_applied

    def _can_apply_transformation(self, input_obj, modification):
        """检查是否可以将变换规则应用到输入对象"""
        # 如果有对象ID，可以直接比较
        if 'input_obj_id' in modification and hasattr(input_obj, 'obj_id'):
            return str(input_obj.obj_id) == str(modification['input_obj_id'])

        # 否则基于对象特征进行匹配
        # 例如：大小、颜色分布、位置等
        return True  # 简化版，实际应用中需要更复杂的匹配逻辑

    def _apply_object_transformation(self, input_obj, transformation):
        """应用变换到输入对象，返回变换后的对象"""
        transform_type = transformation.get('type', '')
        result_obj = set()

        if transform_type == 'same_shape_different_position':
            # 应用位置变化
            position_change = transformation.get('position_change', {})
            delta_row = position_change.get('delta_row', 0)
            delta_col = position_change.get('delta_col', 0)

            for color, (i, j) in input_obj.original_obj:
                new_i = i + delta_row
                new_j = j + delta_col
                result_obj.add((color, (new_i, new_j)))

        elif transform_type in ['rotate', 'flip', 'scale']:
            # 应用形状变换（简化版）
            # 在实际应用中，需要实现完整的旋转、翻转和缩放逻辑
            result_obj = input_obj.original_obj

        elif 'color_transform' in transformation:
            # 应用颜色变换
            color_map = transformation.get('color_transform', {}).get('color_mapping', {})

            for color, pos in input_obj.original_obj:
                new_color = color_map.get(str(color), color)
                result_obj.add((new_color, pos))
        else:
            # 默认不变
            result_obj = input_obj.original_obj

        return result_obj

    def _generate_object_from_transformation(self, input_obj, transformation):
        """根据变换生成新对象"""
        # 类似于_apply_object_transformation，但可能有不同的生成逻辑
        return self._apply_object_transformation(input_obj, transformation)

    def _match_object_pattern(self, obj, pattern):
        """检查对象是否匹配特定模式"""
        # 简化版匹配逻辑
        if 'input_obj_id' in pattern and hasattr(obj, 'obj_id'):
            return str(obj.obj_id) == str(pattern['input_obj_id'])

        # 可以添加基于形状、大小、颜色等的匹配
        return False