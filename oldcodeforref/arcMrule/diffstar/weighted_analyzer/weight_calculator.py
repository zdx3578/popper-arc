"""
对象权重计算模块

负责计算和调整对象的权重。
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Callable


class WeightCalculator:
    """处理对象权重计算的类"""

    def __init__(self, pixel_threshold_pct=40, weight_increment=1,
                 diff_weight_increment=2, debug_print=None):
        """
        初始化权重计算器

        Args:
            pixel_threshold_pct: 颜色占比阈值（百分比），超过此阈值的颜色视为背景
            weight_increment: 对象权重增量
            diff_weight_increment: 差异区域权重增量
            debug_print: 调试打印函数（可选）
        """
        self.pixel_threshold_pct = pixel_threshold_pct
        self.weight_increment = weight_increment
        self.diff_weight_increment = diff_weight_increment
        self.debug_print = debug_print
        self.background_colors = set()  # 全局背景色集合

    def set_background_colors(self, background_colors):
        """
        设置全局背景色

        Args:
            background_colors: 背景色集合
        """
        self.background_colors = background_colors
        if self.debug_print:
            self.debug_print(f"设置全局背景色: {background_colors}")

    def calculate_object_weights(self, pair_id, input_grid, output_grid,
                                input_obj_infos, output_obj_infos,
                                diff_in_obj_infos, diff_out_obj_infos,
                                diff_in=None, diff_out=None):
        """
        为所有对象计算权重

        Args:
            pair_id: 训练对ID
            input_grid, output_grid: 输入输出网格
            input_obj_infos, output_obj_infos: 输入输出对象信息
            diff_in_obj_infos, diff_out_obj_infos: 差异对象信息
            diff_in, diff_out: 差异网格
        """
        if self.debug_print:
            self.debug_print(f"计算训练对 {pair_id} 的对象权重")

        # 1. 初始权重 - 基于对象大小
        for obj_list in [input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos]:
            for obj_info in obj_list:
                # 基础权重为对象大小
                obj_info.obj_weight = 0

        # 2. 增加差异区域对象的权重
        for obj_info in diff_in_obj_infos + diff_out_obj_infos:
            obj_info.increase_weight(self.diff_weight_increment)

        # 3. 处理位于差异位置的原始对象
        # 创建坐标到对象的映射
        input_pos_to_obj = self._create_position_object_map(input_obj_infos)
        output_pos_to_obj = self._create_position_object_map(output_obj_infos)

        # 跟踪已增加权重的对象，避免重复增加
        increased_input_objs = set()
        increased_output_objs = set()

        # 如果有差异区域，为涉及差异的原始对象增加权重
        if diff_in is not None and diff_out is not None:
            for i in range(len(diff_in)):
                for j in range(len(diff_in[0])):
                    if diff_in[i][j] is not None:  # 发现差异
                        # 检查该位置是否有输入对象
                        pos = (i, j)
                        if pos in input_pos_to_obj:
                            for obj_info in input_pos_to_obj[pos]:
                                # 确保每个对象只增加一次权重
                                if obj_info.obj_id not in increased_input_objs:
                                    obj_info.increase_weight(self.diff_weight_increment)
                                    increased_input_objs.add(obj_info.obj_id)
                                    if self.debug_print:
                                        self.debug_print(f"增加位置涉及差异的输入对象 {obj_info.obj_id} 权重，现在为 {obj_info.obj_weight}")

                        # 检查该位置是否有输出对象
                        if pos in output_pos_to_obj:
                            for obj_info in output_pos_to_obj[pos]:
                                # 确保每个对象只增加一次权重
                                if obj_info.obj_id not in increased_output_objs:
                                    obj_info.increase_weight(self.diff_weight_increment)
                                    increased_output_objs.add(obj_info.obj_id)
                                    if self.debug_print:
                                        self.debug_print(f"增加位置涉及差异的输出对象 {obj_info.obj_id} 权重，现在为 {obj_info.obj_weight}")

        # 4. 基于形状匹配增加权重
        self._add_shape_matching_weights(input_obj_infos, output_obj_infos)

        # 5. 考虑颜色占比，调整背景对象权重 #!（计算背景色，去掉这里权重设置）
        # self._adjust_background_object_weights(input_grid, input_obj_infos)
        # self._adjust_background_object_weights(output_grid, output_obj_infos)

    def calculate_test_object_weights(self, input_grid, input_obj_infos, shape_library):
        """
        计算测试输入对象的权重

        Args:
            input_grid: 输入网格
            input_obj_infos: 输入对象信息列表
            shape_library: 形状库，用于匹配形状
        """
        # 1. 初始权重 - 基于对象大小
        for obj_info in input_obj_infos:
            obj_info.obj_weight = obj_info.size

        # 2. 基于形状库匹配增加权重
        for obj_info in input_obj_infos:
            normalized_obj = obj_info.obj_000
            hashable_obj = self._get_hashable_representation(normalized_obj)

            # 检查是否在形状库中
            for shape_key, shape_info in shape_library.items():
                lib_shape = shape_info["normalized_shape"]
                lib_hashable = self._get_hashable_representation(lib_shape)

                if hashable_obj == lib_hashable:
                    # 如果形状匹配，增加权重
                    match_bonus = shape_info["count"] * 2  # 根据出现次数增加权重
                    obj_info.increase_weight(match_bonus)
                    if self.debug_print:
                        self.debug_print(f"对象 {obj_info.obj_id} 匹配形状库中的形状，增加权重 +{match_bonus}")
                    break

        # 3. 考虑颜色占比，调整背景对象权重
        # self._adjust_background_object_weights(input_grid, input_obj_infos)

    def _create_position_object_map(self, obj_infos):
        """
        创建坐标到对象的映射，用于快速查找特定位置的对象

        Args:
            obj_infos: 对象信息列表

        Returns:
            字典 {(row, col): [obj_info, ...]}
        """
        pos_to_obj = defaultdict(list)

        for obj_info in obj_infos:
            for _, (i, j) in obj_info.original_obj:
                pos_to_obj[(i, j)].append(obj_info)

        return pos_to_obj

    def _add_shape_matching_weights(self, input_obj_infos, output_obj_infos):
        """
        基于形状匹配增加对象权重

        Args:
            input_obj_infos: 输入对象信息列表
            output_obj_infos: 输出对象信息列表
        """
        # 创建形状匹配字典
        normalized_shapes = {}

        # 处理输入对象
        for obj_info in input_obj_infos:
            # 获取规范化形状
            normalized_obj = obj_info.obj_000

            # 使用可哈希的表示
            hashable_obj = self._get_hashable_representation(normalized_obj)

            if hashable_obj not in normalized_shapes:
                normalized_shapes[hashable_obj] = []
            normalized_shapes[hashable_obj].append(('input', obj_info))

        # 处理输出对象
        for obj_info in output_obj_infos:
            # 获取规范化形状
            normalized_obj = obj_info.obj_000

            # 使用可哈希的表示
            hashable_obj = self._get_hashable_representation(normalized_obj)

            if hashable_obj not in normalized_shapes:
                normalized_shapes[hashable_obj] = []
            normalized_shapes[hashable_obj].append(('output', obj_info))

        # 为相同形状的对象增加权重
        for shape, obj_list in normalized_shapes.items():
            if len(obj_list) <= 1:
                continue  # 跳过没有匹配的形状

            # 相同形状的对象数量作为额外权重
            shape_bonus = len(obj_list)

            for _, obj_info in obj_list:
                obj_info.increase_weight(shape_bonus)
                if self.debug_print:
                    self.debug_print(f"增加对象 {obj_info.obj_id} 的形状匹配权重 +{shape_bonus}，现在为 {obj_info.obj_weight}")

    def _adjust_background_object_weights(self, grid, obj_infos):
        """
        基于颜色占比调整背景对象权重

        Args:
            grid: 网格
            obj_infos: 对象信息列表
        """
        background_colors = self.background_colors

        # 如果没有预设的背景色，则使用原始方法计算
        if not background_colors:
            # 计算每种颜色的像素数
            color_counts = defaultdict(int)
            total_pixels = len(grid) * len(grid[0])

            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    color_counts[grid[i][j]] += 1

            # 找出背景颜色（占比超过阈值的颜色）
            for color, count in color_counts.items():
                percentage = (count / total_pixels) * 100
                if percentage > self.pixel_threshold_pct:
                    background_colors.add(color)
                    if self.debug_print:
                        self.debug_print(f"单独网格中识别到背景颜色: {color}, 占比: {percentage:.2f}%")

        # 调整背景对象的权重
        for obj_info in obj_infos:
            # 检查对象主色是否为背景色
            if obj_info.main_color in background_colors:
                # 计算对象中背景色的占比
                bg_pixels = sum(1 for val, _ in obj_info.original_obj if val in background_colors)
                bg_percentage = (bg_pixels / obj_info.size) * 100

                # 如果对象主要由背景色组成，降低其权重
                if bg_percentage > 80:  # 80%以上为背景色
                    # 将权重设为0
                    new_weight = 0
                    obj_info.set_weight(new_weight)
                    if self.debug_print:
                        self.debug_print(f"降低背景对象 {obj_info.obj_id} 权重至 {new_weight}，背景色占比 {bg_percentage:.1f}%")

    def _get_hashable_representation(self, obj_set):
        """
        将对象集合转换为可哈希的表示

        Args:
            obj_set: 对象集合

        Returns:
            可哈希的表示（元组）
        """
        sorted_elements = []
        for value, loc in obj_set:
            i, j = loc
            sorted_elements.append((value, i, j))

        return tuple(sorted(sorted_elements))