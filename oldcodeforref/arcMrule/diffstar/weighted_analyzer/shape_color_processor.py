"""
形状-颜色规则处理器

处理基于形状特征的颜色变化规则，以及跨对象的属性依赖关系。
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict


class ShapeColorProcessor:
    """处理形状-颜色规则的组件"""

    def __init__(self, debug_print_fn=None):
        """
        初始化形状-颜色处理器

        Args:
            debug_print_fn: 调试打印函数，为None则不打印调试信息
        """
        self.debug_print = debug_print_fn

    def apply_shape_color_rules(self, input_grid, input_objects, shape_color_rules):
        """
        应用形状-颜色规则到输入网格

        Args:
            input_grid: 输入网格
            input_objects: 输入对象列表 (WeightedObjInfo实例)
            shape_color_rules: 形状-颜色规则列表

        Returns:
            修改后的网格
        """
        if not shape_color_rules or not input_objects:
            return input_grid

        # 创建网格副本
        output_grid = tuple(tuple(row) for row in input_grid)
        output_grid = [list(row) for row in output_grid]

        for rule in shape_color_rules:
            if rule['pattern_type'] == 'shape_to_color':
                # 应用基于对象形状的颜色变化规则
                if self.debug_print:
                    self.debug_print(f"应用形状-颜色规则: {rule['color_change']}")

                output_grid = self._apply_shape_to_color_rule(
                    output_grid, input_objects, rule
                )

            elif rule['pattern_type'] == 'cross_shape_to_color':
                # 应用跨对象的形状-颜色规则
                if self.debug_print:
                    self.debug_print(f"应用跨对象形状-颜色规则，目标颜色: {rule['resulting_color']}")

                output_grid = self._apply_cross_shape_color_rule(
                    output_grid, input_objects, rule
                )

        # 将列表转换回元组格式
        return tuple(tuple(row) for row in output_grid)

    def apply_attribute_dependency_rules(self, input_grid, input_objects, attr_dependency_rules):
        """
        应用属性依赖规则到输入网格

        Args:
            input_grid: 输入网格
            input_objects: 输入对象列表
            attr_dependency_rules: 属性依赖规则列表

        Returns:
            修改后的网格
        """
        if not attr_dependency_rules or not input_objects:
            return input_grid

        # 创建网格副本
        output_grid = tuple(tuple(row) for row in input_grid)
        output_grid = [list(row) for row in output_grid]

        for rule in attr_dependency_rules:
            if rule['pattern_type'] == 'self_attr_color_change':
                # 应用基于自身属性的颜色变化规则
                if self.debug_print:
                    self.debug_print(f"应用自我属性颜色变化规则: {rule['color_change']}")

                output_grid = self._apply_self_attr_color_rule(
                    output_grid, input_objects, rule
                )

            elif rule['pattern_type'] == 'cross_obj_color_dependency':
                # 应用跨对象的属性依赖规则
                if self.debug_print:
                    self.debug_print(f"应用跨对象颜色依赖规则")

                output_grid = self._apply_cross_obj_attr_rule(
                    output_grid, input_objects, rule
                )

        # 将列表转换回元组格式
        return tuple(tuple(row) for row in output_grid)

    def _apply_shape_to_color_rule(self, grid, objects, rule):
        """应用形状-颜色规则到网格"""
        color_from = rule['color_change']['from']
        color_to = rule['color_change']['to']
        shape_conditions = rule.get('shape_conditions', {})

        for obj in objects:
            # 检查对象颜色是否匹配起始颜色
            if obj.main_color == color_from:
                # 检查对象形状是否满足条件
                if self._check_shape_conditions(obj, shape_conditions):
                    # 修改对象区域的颜色
                    for i in range(obj.top, obj.top + obj.height):
                        for j in range(obj.left, obj.left + obj.width):
                            if (i < len(grid) and j < len(grid[0]) and
                                i >= 0 and j >= 0 and
                                grid[i][j] == color_from):
                                grid[i][j] = color_to

        return grid

    def _apply_cross_shape_color_rule(self, grid, objects, rule):
        """应用跨对象的形状-颜色规则"""
        resulting_color = rule['resulting_color']

        # 按照对象权重排序
        sorted_objects = sorted(objects, key=lambda obj: obj.obj_weight, reverse=True)

        if len(sorted_objects) >= 2:
            # 获取最高权重的对象（可能是形状决定因素）
            reference_obj = sorted_objects[0]

            # 应用到其他对象
            for target_obj in sorted_objects[1:]:
                # 根据对象的属性获取其位置信息
                # 使用obj.obj.top和obj.obj.left替代直接访问top和left
                try:
                    top = getattr(target_obj.obj, 'top', 0)
                    left = getattr(target_obj.obj, 'left', 0)
                    height = getattr(target_obj.obj, 'height', 1)
                    width = getattr(target_obj.obj, 'width', 1)

                    # 修改目标对象区域的颜色
                    for i in range(top, top + height):
                        for j in range(left, left + width):
                            if (i < len(grid) and j < len(grid[0]) and
                                i >= 0 and j >= 0):
                                grid[i][j] = resulting_color
                except AttributeError:
                    if self.debug_print:
                        self.debug_print(f"警告: 无法获取对象位置信息")

        return grid

    def _apply_self_attr_color_rule(self, grid, objects, rule):
        """应用基于自身属性的颜色变化规则"""
        color_from = rule['color_change']['from']
        color_to = rule['color_change']['to']
        dependent_on = rule.get('dependent_on')

        # 根据依赖属性类型应用规则
        if dependent_on == 'shape':
            # 这里复用形状-颜色规则的逻辑
            return self._apply_shape_to_color_rule(
                grid, objects,
                {'color_change': {'from': color_from, 'to': color_to},
                 'shape_conditions': {}}
            )

        return grid

    def _apply_cross_obj_attr_rule(self, grid, objects, rule):
        """应用跨对象的属性依赖规则"""
        # 跨对象规则的实现与跨对象形状-颜色规则类似
        if len(objects) >= 2:
            # 按权重排序对象
            sorted_objects = sorted(objects, key=lambda obj: obj.obj_weight, reverse=True)

            # 假设第一个对象影响其他对象
            influencer = sorted_objects[0]

            # 推断结果颜色（简化实现）
            result_color = (influencer.main_color + 1) % 10  # 简单变换

            # 应用到其他对象
            for target in sorted_objects[1:]:
                try:
                    top = getattr(target.obj, 'top', 0)
                    left = getattr(target.obj, 'left', 0)
                    height = getattr(target.obj, 'height', 1)
                    width = getattr(target.obj, 'width', 1)

                    for i in range(top, top + height):
                        for j in range(left, left + width):
                            if (i < len(grid) and j < len(grid[0]) and
                                i >= 0 and j >= 0):
                                grid[i][j] = result_color
                except AttributeError:
                    if self.debug_print:
                        self.debug_print(f"警告: 无法获取对象位置信息")

        return grid

    def _check_shape_conditions(self, obj, conditions):
        """检查对象是否满足形状条件"""
        if not conditions:
            return True  # 没有条件时默认满足

        # 检查每个条件，确保安全访问属性
        for feature_name, expected_value in conditions.items():
            if feature_name == 'height':
                height = getattr(obj, 'height', None) or getattr(obj.obj, 'height', 0)
                if abs(height - expected_value) > 1:
                    return False
            elif feature_name == 'width':
                width = getattr(obj, 'width', None) or getattr(obj.obj, 'width', 0)
                if abs(width - expected_value) > 1:
                    return False
            elif feature_name == 'size':
                size = getattr(obj, 'size', None) or getattr(obj.obj, 'size', 0)
                if abs(size - expected_value) > size * 0.2:  # 允许20%误差
                    return False
            elif feature_name == 'aspect_ratio':
                width = getattr(obj, 'width', None) or getattr(obj.obj, 'width', 1)
                height = getattr(obj, 'height', None) or getattr(obj.obj, 'height', 1)
                actual_ratio = width / max(1, height)
                if abs(actual_ratio - expected_value) > 0.2:  # 允许0.2误差
                    return False
            # 其他特征条件检查...

        return True  # 所有条件都满足