import numpy as np
from typing import List, Dict, Tuple, Any, Set, FrozenSet, Optional, Union
from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
import copy

# 导入现有函数和类
from objutil import pureobjects_from_grid, objects_fromone_params, shift_pure_obj_to_0_0_0
from objutil import uppermost, leftmost, lowermost, rightmost, palette, extend_obj
from weightgird import grid2grid_fromgriddiff, apply_color_matching_weights, display_weight_grid, display_matrices
from arc_diff_analyzer import ARCDiffAnalyzer, ObjInfo, JSONSerializer


# 修改ObjInfo类，添加权重属性
class WeightedObjInfo(ObjInfo):
    """增强型对象信息类，存储对象的所有相关信息、变换和权重"""

    def __init__(self, pair_id, in_or_out, obj, obj_params=None, grid_hw=None, background=0):
        """
        初始化对象信息

        Args:
            pair_id: 训练对ID
            in_or_out: 'in', 'out', 'diff_in', 'diff_out'等
            obj: 原始对象 (frozenset形式)
            obj_params: 对象参数 (univalued, diagonal, without_bg)
            grid_hw: 网格尺寸 [height, width]
            background: 背景值
        """
        # 调用父类初始化
        super().__init__(pair_id, in_or_out, obj, obj_params, grid_hw, background)

        # 新增：对象权重属性
        self.obj_weight = 0  # 初始权重为0，后续根据各种规则增加权重

    def to_dict(self):
        """转换为可序列化的字典表示"""
        result = super().to_dict()
        result["obj_weight"] = self.obj_weight  # 添加权重到字典表示
        return result

    def increase_weight(self, amount):
        """增加对象权重"""
        self.obj_weight += amount
        return self.obj_weight

    def set_weight(self, value):
        """设置对象权重"""
        self.obj_weight = value
        return self.obj_weight


class WeightedARCDiffAnalyzer(ARCDiffAnalyzer):
    """
    扩展ARCDiffAnalyzer，整合对象权重系统，优化对象分析
    """

    def __init__(self, debug=True, debug_dir="debug_output", pixel_threshold_pct=60,
                 weight_increment=1, diff_weight_increment=2):
        """
        初始化加权分析器

        Args:
            debug: 是否启用调试模式
            debug_dir: 调试信息输出目录
            pixel_threshold_pct: 颜色占比阈值（百分比），超过此阈值的颜色视为背景
            weight_increment: 对象权重增量
            diff_weight_increment: 差异区域权重增量
        """
        # 调用父类初始化
        super().__init__(debug, debug_dir)

        # 权重相关参数
        self.pixel_threshold_pct = pixel_threshold_pct

        self.weight1 = 1
        self.weight2 = 2
        self.weight3 = 3
        self.weight4 = 4
        self.weight5 = 5

        self.weight_increment = weight_increment
        self.diff_weight_increment = diff_weight_increment

        # 保存颜色映射统计
        self.color_statistics = {}
        self.transformation_rules = []

        # 重写对象存储结构，使用WeightedObjInfo替代ObjInfo
        self.all_objects = {
            'input': [],  # [(pair_id, [WeightedObjInfo]), ...]
            'output': [], # [(pair_id, [WeightedObjInfo]), ...]
            'diff_in': [], # [(pair_id, [WeightedObjInfo]), ...]
            'diff_out': []  # [(pair_id, [WeightedObjInfo]), ...]
        }

    def add_train_pair(self, pair_id, input_grid, output_grid, param):
        """
        添加一对训练数据，提取对象并计算权重

        Args:
            pair_id: 训练对ID
            input_grid: 输入网格
            output_grid: 输出网格
        """
        if self.debug:
            self._debug_print(f"处理训练对 {pair_id}")
            self._debug_save_grid(input_grid, f"input_{pair_id}")
            self._debug_save_grid(output_grid, f"output_{pair_id}")

        # 确保网格是元组的元组格式
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)
        if isinstance(output_grid, list):
            output_grid = tuple(tuple(row) for row in output_grid)

        # 保存原始网格对
        self.train_pairs.append((input_grid, output_grid))

        # 计算差异网格
        diff_in, diff_out = grid2grid_fromgriddiff(input_grid, output_grid)
        self.diff_pairs.append((diff_in, diff_out))

        if self.debug:
            self._debug_save_grid(diff_in, f"diff_in_{pair_id}")
            self._debug_save_grid(diff_out, f"diff_out_{pair_id}")

        # 获取网格尺寸
        height_in, width_in = len(input_grid), len(input_grid[0])
        height_out, width_out = len(output_grid), len(output_grid[0])

        # 提取对象
        input_objects = pureobjects_from_grid(
            param, pair_id, 'in', input_grid, [height_in, width_in]
        )
        output_objects = pureobjects_from_grid(
            param, pair_id, 'out', output_grid, [height_out, width_out]
        )

        # 转换为加权对象信息
        input_obj_infos = [
            WeightedObjInfo(pair_id, 'in', obj, obj_params=None, grid_hw=[height_in, width_in])
            for obj in input_objects
        ]

        output_obj_infos = [
            WeightedObjInfo(pair_id, 'out', obj, obj_params=None, grid_hw=[height_out, width_out])
            for obj in output_objects
        ]

        if self.debug:
            self._debug_print(f"从输入网格提取了 {len(input_obj_infos)} 个对象")
            self._debug_print(f"从输出网格提取了 {len(output_obj_infos)} 个对象")

        # 为diff网格也提取对象
        if diff_in is not None and diff_out is not None:
            height_diff, width_diff = len(diff_in), len(diff_in[0])
            diff_in_objects = pureobjects_from_grid(
                param, pair_id, 'diff_in', diff_in, [height_diff, width_diff]
            )
            diff_out_objects = pureobjects_from_grid(
                param, pair_id, 'diff_out', diff_out, [height_diff, width_diff]
            )

            # 转换为加权对象信息
            diff_in_obj_infos = [
                WeightedObjInfo(pair_id, 'diff_in', obj, obj_params=None, grid_hw=[height_diff, width_diff])
                for obj in diff_in_objects
            ]

            diff_out_obj_infos = [
                WeightedObjInfo(pair_id, 'diff_out', obj, obj_params=None, grid_hw=[height_diff, width_diff])
                for obj in diff_out_objects
            ]

            if self.debug:
                self._debug_print(f"从差异输入网格提取了 {len(diff_in_obj_infos)} 个对象")
                self._debug_print(f"从差异输出网格提取了 {len(diff_out_obj_infos)} 个对象")
        else:
            diff_in_obj_infos = []
            diff_out_obj_infos = []
            if self.debug:
                self._debug_print("差异网格为空")

        # 存储提取的对象
        self.all_objects['input'].append((pair_id, input_obj_infos))
        self.all_objects['output'].append((pair_id, output_obj_infos))
        self.all_objects['diff_in'].append((pair_id, diff_in_obj_infos))
        self.all_objects['diff_out'].append((pair_id, diff_out_obj_infos))

        # 更新形状库
        self._update_shape_library(input_obj_infos + output_obj_infos)

        # 分析对象间的部分-整体关系
        self._analyze_part_whole_relationships(input_obj_infos)
        self._analyze_part_whole_relationships(output_obj_infos)

        # 应用权重计算 - 为每个对象设置权重
        self._calculate_object_weights(pair_id, input_grid, output_grid,
                                      input_obj_infos, output_obj_infos,
                                      diff_in_obj_infos, diff_out_obj_infos)

        # 分析diff映射关系
        mapping_rule = self._analyze_diff_mapping_with_weights(
            pair_id, input_grid, output_grid, diff_in, diff_out,
            input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos
        )

        self.mapping_rules.append(mapping_rule)

        if self.debug:
            self._debug_save_json(mapping_rule, f"mapping_rule_{pair_id}")
            self._debug_print(f"完成训练对 {pair_id} 的分析和权重计算")
            self._debug_print_object_weights(input_obj_infos, f"input_obj_weights_{pair_id}")
            self._debug_print_object_weights(output_obj_infos, f"output_obj_weights_{pair_id}")
            self._debug_print_object_weights(diff_in_obj_infos, f"diff_in_obj_weights_{pair_id}")
            self._debug_print_object_weights(diff_out_obj_infos, f"diff_out_obj_weights_{pair_id}")

    def _calculate_object_weights(self, pair_id, input_grid, output_grid,
                                 input_obj_infos, output_obj_infos,
                                 diff_in_obj_infos, diff_out_obj_infos):
        """
        为所有对象计算权重

        Args:
            pair_id: 训练对ID
            input_grid, output_grid: 输入输出网格
            input_obj_infos, output_obj_infos: 输入输出对象信息
            diff_in_obj_infos, diff_out_obj_infos: 差异对象信息
        """
        if self.debug:
            self._debug_print(f"计算训练对 {pair_id} 的对象权重")

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
        diff_in, diff_out = self.diff_pairs[-1]  # 最新添加的diff对

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
                                    if self.debug:
                                        self._debug_print(f"增加位置涉及差异的输入对象 {obj_info.obj_id} 权重，现在为 {obj_info.obj_weight}")

                        # 检查该位置是否有输出对象
                        if pos in output_pos_to_obj:
                            for obj_info in output_pos_to_obj[pos]:
                                # 确保每个对象只增加一次权重
                                if obj_info.obj_id not in increased_output_objs:
                                    obj_info.increase_weight(self.diff_weight_increment)
                                    increased_output_objs.add(obj_info.obj_id)
                                    if self.debug:
                                        self._debug_print(f"增加位置涉及差异的输出对象 {obj_info.obj_id} 权重，现在为 {obj_info.obj_weight}")



        # 4. 基于形状匹配增加权重
        self._add_shape_matching_weights(input_obj_infos, output_obj_infos)

        # 5. 考虑颜色占比，调整背景对象权重
        self._adjust_background_object_weights(input_grid, input_obj_infos)
        self._adjust_background_object_weights(output_grid, output_obj_infos)

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
                if self.debug:
                    self._debug_print(f"增加对象 {obj_info.obj_id} 的形状匹配权重 +{shape_bonus}，现在为 {obj_info.obj_weight}")

    def _adjust_background_object_weights(self, grid, obj_infos):
        """
        基于颜色占比调整背景对象权重

        Args:
            grid: 网格
            obj_infos: 对象信息列表
        """
        # 计算每种颜色的像素数
        color_counts = defaultdict(int)
        total_pixels = len(grid) * len(grid[0])

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                color_counts[grid[i][j]] += 1

        # 找出背景颜色（占比超过阈值的颜色）
        background_colors = set()
        for color, count in color_counts.items():
            percentage = (count / total_pixels) * 100
            if percentage > self.pixel_threshold_pct:
                background_colors.add(color)
                if self.debug:
                    self._debug_print(f"识别到背景颜色: {color}, 占比: {percentage:.2f}%")

        # 调整背景对象的权重
        for obj_info in obj_infos:
            # 检查对象主色是否为背景色
            if obj_info.main_color in background_colors:
                # 计算对象中背景色的占比
                bg_pixels = sum(1 for val, _ in obj_info.original_obj if val in background_colors)
                bg_percentage = (bg_pixels / obj_info.size) * 100

                # 如果对象主要由背景色组成，降低其权重
                if bg_percentage > 80:  # 80%以上为背景色
                    # 将权重设为初始权重的一半
                    # new_weight = max(1, obj_info.obj_weight // 2)
                    new_weight = 0

                    obj_info.set_weight(new_weight)
                    if self.debug:
                        self._debug_print(f"降低背景对象 {obj_info.obj_id} 权重至 {new_weight}，背景色占比 {bg_percentage:.1f}%")
                        # display_matrices(obj_info.original_obj,obj_info.grid_hw )

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

    def _analyze_diff_mapping_with_weights(self, pair_id, input_grid, output_grid, diff_in, diff_out,
                                          input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos):
        """
        基于对象权重分析差异网格映射关系

        Args:
            pair_id: 训练对ID
            input_grid, output_grid: 输入输出网格
            diff_in, diff_out: 差异网格
            input_obj_infos, output_obj_infos: 输入输出对象信息
            diff_in_obj_infos, diff_out_obj_infos: 差异对象信息

        Returns:
            映射规则字典
        """
        if self.debug:
            self._debug_print(f"基于权重分析差异映射关系，pair_id={pair_id}")

        mapping_rule = {
            "pair_id": pair_id,
            "object_mappings": [],
            "shape_transformations": [],
            "color_mappings": {},
            "position_changes": [],
            "part_whole_relationships": [],
            "weighted_objects": [],  # 添加权重信息
            "combined_transformations": []  # 新增：捕获多维度变换的组合规则
        }

        # 准备分析diff对象间的映射
        if not diff_in_obj_infos or not diff_out_obj_infos:
            if self.debug:
                self._debug_print("差异对象为空，无法分析映射")
            return mapping_rule

        # 添加权重信息
        for obj_info in diff_in_obj_infos + diff_out_obj_infos:
            mapping_rule["weighted_objects"].append({
                "obj_id": obj_info.obj_id,
                "weight": obj_info.obj_weight,
                "size": obj_info.size,
                "type": obj_info.in_or_out
            })

        # 按权重降序排序对象，优先考虑高权重对象
        sorted_diff_in = sorted(diff_in_obj_infos, key=lambda x: x.obj_weight, reverse=True)
        sorted_diff_out = sorted(diff_out_obj_infos, key=lambda x: x.obj_weight, reverse=True)

        # 基于形状匹配寻找对象映射，但考虑权重
        object_mappings = self._find_object_mappings_by_shape_and_weight(sorted_diff_in, sorted_diff_out)

        if self.debug:
            self._debug_print(f"找到 {len(object_mappings)} 个基于形状和权重的对象匹配")

        # 分析每个映射的变换
        for in_obj, out_obj, match_info in object_mappings:
            # 分析颜色变换
            color_transformation = in_obj.get_color_transformation(out_obj)

            # 分析位置变换
            position_change = in_obj.get_positional_change(out_obj)

            # 添加到映射规则
            mapping_rule["object_mappings"].append({
                "diff_in_object": in_obj.to_dict(),
                "diff_out_object": out_obj.to_dict(),
                "match_info": match_info,
                "weight_product": in_obj.obj_weight * out_obj.obj_weight  # 添加权重乘积作为匹配强度
            })

            # 记录形状变换
            mapping_rule["shape_transformations"].append({
                "in_obj_id": in_obj.obj_id,
                "out_obj_id": out_obj.obj_id,
                "transform_type": match_info["transform_type"],
                "transform_name": match_info["transform_name"],
                "confidence": match_info["confidence"],
                "weight_in": in_obj.obj_weight,
                "weight_out": out_obj.obj_weight
            })

            # 记录颜色映射
            if color_transformation and color_transformation.get("color_mapping"):
                for from_color, to_color in color_transformation["color_mapping"].items():
                    if from_color not in mapping_rule["color_mappings"]:
                        mapping_rule["color_mappings"][from_color] = {
                            "to_color": to_color,
                            "weight": in_obj.obj_weight  # 使用输入对象权重作为颜色映射权重
                        }
                    elif in_obj.obj_weight > mapping_rule["color_mappings"][from_color]["weight"]:
                        # 如果当前对象权重更高，更新颜色映射
                        mapping_rule["color_mappings"][from_color] = {
                            "to_color": to_color,
                            "weight": in_obj.obj_weight
                        }

            # 记录位置变化
            mapping_rule["position_changes"].append({
                "in_obj_id": in_obj.obj_id,
                "out_obj_id": out_obj.obj_id,
                "delta_row": position_change["delta_row"],
                "delta_col": position_change["delta_col"],
                "direction": position_change.get("direction"),
                "orientation": position_change.get("orientation"),
                "weight_in": in_obj.obj_weight,
                "weight_out": out_obj.obj_weight
            })

            # 创建组合变换记录
            combined_transform = {
                "in_obj_id": in_obj.obj_id,
                "out_obj_id": out_obj.obj_id,
                "shape_transform": {
                    "transform_type": match_info["transform_type"],
                    "transform_name": match_info["transform_name"],
                },
                "position_change": position_change,
                "color_change": color_transformation.get("color_mapping", {}),
                "weight_score": in_obj.obj_weight * out_obj.obj_weight,
                "confidence": match_info["confidence"],
                # 记录对象的原始上下文信息
                "original_context": self._capture_object_context(in_obj, input_grid),
                "transformed_context": self._capture_object_context(out_obj, output_grid)
            }

            mapping_rule["combined_transformations"].append(combined_transform)

        transformation_rule = self._analyze_input_to_output_transformation(
            pair_id, input_grid, output_grid,
            input_obj_infos, output_obj_infos,
            diff_in_obj_infos, diff_out_obj_infos,
            mapping_rule
        )

        # 将转换规则合并到映射规则中
        mapping_rule["input_to_output_transformation"] = transformation_rule
        self.transformation_rules.append(transformation_rule)

        return mapping_rule

    def _find_object_mappings_by_shape_and_weight(self, diff_in_obj_infos, diff_out_obj_infos):
        """
        基于形状匹配和权重寻找对象映射

        Args:
            diff_in_obj_infos: 差异输入对象列表，已按权重降序排序
            diff_out_obj_infos: 差异输出对象列表，已按权重降序排序

        Returns:
            列表 [(in_obj, out_obj, match_info), ...]
        """
        mappings = []
        matched_out_objs = set()  # 跟踪已匹配的输出对象

        # 为每个输入对象找到最匹配的输出对象，优先考虑高权重对象
        for in_obj in diff_in_obj_infos:
            best_match = None
            best_match_info = None
            best_confidence = -1
            best_weight_score = -1

            for out_obj in diff_out_obj_infos:
                # 跳过已匹配的输出对象
                if out_obj.obj_id in matched_out_objs:
                    continue

                # 检查是否匹配任何变换形式
                matches, transform_type, transform_name = in_obj.matches_with_transformation(out_obj)

                if matches:
                    # 计算匹配置信度（基本置信度）
                    confidence = 0.7  # 初始置信度

                    # 对于完全相同形状，增加置信度
                    if transform_type == "same_shape":
                        confidence += 0.3

                    # 计算权重得分（输入权重 * 输出权重）
                    weight_score = in_obj.obj_weight * out_obj.obj_weight

                    # 如果权重得分更高，或权重得分相同但置信度更高，则更新最佳匹配
                    if weight_score > best_weight_score or (weight_score == best_weight_score and confidence > best_confidence):
                        best_weight_score = weight_score
                        best_confidence = confidence
                        best_match = out_obj
                        best_match_info = {
                            "transform_type": transform_type,
                            "transform_name": transform_name,
                            "confidence": confidence,
                            "weight_score": weight_score
                        }

            if best_match and best_confidence > 0:
                mappings.append((in_obj, best_match, best_match_info))
                matched_out_objs.add(best_match.obj_id)  # 标记该输出对象已匹配

        return mappings

    def analyze_common_patterns_with_weights(self):
        """
        分析多对训练数据的共有模式，考虑权重因素

        Returns:
            共有模式字典
        """
        if self.debug:
            self._debug_print("基于权重分析共有模式")

        if not self.mapping_rules:
            return {}

        # 分析共有的形状变换模式
        common_shape_transformations = self._find_common_shape_transformations_with_weights()

        # 分析共有的颜色映射模式
        common_color_mappings = self._find_common_color_mappings_with_weights()

        # 分析共有的位置变化模式
        common_position_changes = self._find_common_position_changes_with_weights()

        self.common_patterns = {
            "shape_transformations": common_shape_transformations,
            "color_mappings": common_color_mappings,
            "position_changes": common_position_changes
        }

        if self.debug:
            self._debug_save_json(self.common_patterns, "weighted_common_patterns")
            self._debug_print(f"找到 {len(common_shape_transformations)} 个加权共有形状变换模式")
            self._debug_print(f"找到 {len(common_color_mappings.get('mappings', {}))} 个加权共有颜色映射")
            self._debug_print(f"找到 {len(common_position_changes)} 个加权共有位置变化模式")

        return self.common_patterns

    def _find_common_shape_transformations_with_weights(self):
        """寻找共有的形状变换模式，考虑权重"""
        # 收集所有形状变换
        all_transformations = []
        for rule in self.mapping_rules:
            for transform in rule.get("shape_transformations", []):
                # 添加权重信息，如果有的话
                if "weight_in" in transform and "weight_out" in transform:
                    transform["weight_score"] = transform["weight_in"] * transform["weight_out"]
                else:
                    transform["weight_score"] = 1  # 默认权重
                all_transformations.append(transform)

        if not all_transformations:
            return []

        # 按变换类型分组
        transform_types = defaultdict(list)
        for transform in all_transformations:
            key = (transform["transform_type"], transform["transform_name"])
            transform_types[key].append(transform)

        common_transforms = []

        # 分析各种变换类型
        for (t_type, t_name), transforms in transform_types.items():
            # 如果变换至少出现两次，认为是共有模式
            if len(transforms) >= 2:
                # 计算加权平均置信度
                total_weight = sum(t.get("weight_score", 1) for t in transforms)
                avg_confidence = sum(t["confidence"] * t.get("weight_score", 1) for t in transforms) / total_weight

                common_transforms.append({
                    "transform_type": t_type,
                    "transform_name": t_name,
                    "count": len(transforms),
                    "confidence": avg_confidence,
                    "weight_score": total_weight,
                    "examples": [t["in_obj_id"] + "->" + t["out_obj_id"] for t in transforms]
                })

        # 按加权得分和出现次数排序
        return sorted(common_transforms, key=lambda x: (x["weight_score"], x["count"]), reverse=True)

    def _find_common_color_mappings_with_weights(self):
        """寻找共有的颜色映射模式，考虑权重"""
        # 收集所有颜色映射
        all_mappings = defaultdict(list)

        for rule in self.mapping_rules:
            for from_color, mapping in rule.get("color_mappings", {}).items():
                if isinstance(mapping, dict) and "to_color" in mapping:
                    # 新格式：包含权重
                    to_color = mapping["to_color"]
                    weight = mapping.get("weight", 1)
                    all_mappings[(from_color, to_color)].append(weight)
                else:
                    # 旧格式：直接是目标颜色
                    to_color = mapping
                    all_mappings[(from_color, to_color)].append(1)  # 默认权重为1

        # 找出共有的映射
        common_mappings = {}
        total_examples = len(self.mapping_rules)

        for (from_color, to_color), weights in all_mappings.items():
            if len(weights) > 1:  # 至少在两个示例中出现
                avg_weight = sum(weights) / len(weights)
                confidence = len(weights) / total_examples
                # 计算加权置信度
                weighted_confidence = confidence * avg_weight

                common_mappings[from_color] = {
                    "to_color": to_color,
                    "count": len(weights),
                    "confidence": confidence,
                    "avg_weight": avg_weight,
                    "weighted_confidence": weighted_confidence
                }

        # 分析颜色变化模式
        color_patterns = []

        # 检查是否有统一的颜色偏移
        offsets = []
        for from_color, mapping in common_mappings.items():
            to_color = mapping["to_color"]
            try:
                # 尝试计算颜色偏移
                offset = to_color - from_color
                offsets.append((offset, mapping["weighted_confidence"]))
            except (TypeError, ValueError):
                pass  # 跳过无法计算偏移的颜色对

        if offsets:
            # 对偏移值进行加权统计
            offset_counts = defaultdict(float)
            for offset, weight in offsets:
                offset_counts[offset] += weight

            # 找出权重最高的偏移
            if offset_counts:
                best_offset, best_score = max(offset_counts.items(), key=lambda x: x[1])
                color_patterns.append({
                    "type": "color_offset",
                    "offset": best_offset,
                    "weighted_score": best_score
                })

        return {
            "mappings": common_mappings,
            "patterns": color_patterns
        }

    def _find_common_position_changes_with_weights(self):
        """寻找共有的位置变化模式，考虑权重"""
        # 收集所有位置变化
        all_changes = []
        for rule in self.mapping_rules:
            for change in rule.get("position_changes", []):
                # 添加权重得分
                if "weight_in" in change and "weight_out" in change:
                    change["weight_score"] = change["weight_in"] * change["weight_out"]
                else:
                    change["weight_score"] = 1  # 默认权重
                all_changes.append(change)

        if not all_changes:
            return []

        # 按位移大小分组
        delta_groups = defaultdict(list)
        for change in all_changes:
            # 取整以处理浮点误差
            delta_row = round(change["delta_row"])
            delta_col = round(change["delta_col"])
            key = (delta_row, delta_col)
            delta_groups[key].append(change)

        # 按方向分组
        direction_groups = defaultdict(list)
        for change in all_changes:
            if "direction" in change and "orientation" in change:
                key = (change["direction"], change["orientation"])
                direction_groups[key].append(change)

        common_changes = []

        # 分析位移组，考虑权重
        for (dr, dc), changes in delta_groups.items():
            if len(changes) >= 2:  # 至少出现两次
                # 计算加权得分
                total_weight = sum(change.get("weight_score", 1) for change in changes)

                common_changes.append({
                    "type": "absolute_position",
                    "delta_row": dr,
                    "delta_col": dc,
                    "count": len(changes),
                    "confidence": len(changes) / len(all_changes),
                    "weight_score": total_weight
                })

        # 分析方向组，考虑权重
        for (direction, orientation), changes in direction_groups.items():
            if len(changes) >= 2:  # 至少出现两次
                # 计算加权得分
                total_weight = sum(change.get("weight_score", 1) for change in changes)

                common_changes.append({
                    "type": "directional",
                    "direction": direction,
                    "orientation": orientation,
                    "count": len(changes),
                    "confidence": len(changes) / len(all_changes),
                    "weight_score": total_weight
                })

        # 按加权得分和出现次数排序
        return sorted(common_changes, key=lambda x: (x["weight_score"], x["count"]), reverse=True)

    def analyze_common_patterns(self):
        """覆盖父类方法，使用加权版本"""
        return self.analyze_common_patterns_with_weights()

    def apply_common_patterns(self, input_grid, param):
        """
        将共有模式应用到新的输入网格，考虑权重

        Args:
            input_grid: 输入网格

        Returns:
            预测的输出网格
        """
        if self.debug:
            self._debug_print("开始应用加权共有模式到测试输入")
            self._debug_save_grid(input_grid, "test_input")

        # 分析共有模式，确保考虑权重
        if not self.common_patterns:
            self.analyze_common_patterns_with_weights()

        # 确保输入网格是元组格式
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)

        # 获取网格尺寸
        height, width = len(input_grid), len(input_grid[0])

        # 提取输入网格中的对象
        input_objects = pureobjects_from_grid(
            param, -1, 'test_in', input_grid, [height, width]
        )

        # 转换为加权对象信息
        input_obj_infos = [
            WeightedObjInfo(-1, 'test_in', obj, obj_params=None, grid_hw=[height, width])
            for obj in input_objects
        ]

        # 计算测试输入对象的权重
        self._calculate_test_object_weights(input_grid, input_obj_infos)

        if self.debug:
            self._debug_print(f"从测试输入提取了 {len(input_obj_infos)} 个对象")
            self._debug_print_object_weights(input_obj_infos, "test_input_objects")

        # 创建输出网格（初始为输入的副本）
        output_grid = [list(row) for row in input_grid]

        # 应用颜色映射，优先考虑高权重映射
        if "color_mappings" in self.common_patterns:
            # 获取颜色映射并按加权置信度排序
            color_mappings = self.common_patterns["color_mappings"].get("mappings", {})
            sorted_mappings = sorted(
                [(from_color, mapping) for from_color, mapping in color_mappings.items()],
                key=lambda x: x[1].get("weighted_confidence", 0),
                reverse=True
            )

            for from_color, mapping in sorted_mappings:
                to_color = mapping["to_color"]
                cells_changed = 0

                # 根据对象权重应用颜色映射
                for obj_info in sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True):
                    for val, (i, j) in obj_info.original_obj:
                        if val == from_color:
                            output_grid[i][j] = to_color
                            cells_changed += 1

                if self.debug and cells_changed > 0:
                    weighted_conf = mapping.get("weighted_confidence", 0)
                    self._debug_print(f"应用颜色映射: {from_color} -> {to_color}, 加权置信度: {weighted_conf:.2f}, 改变了 {cells_changed} 个单元格")

        # 应用颜色模式
        if "color_mappings" in self.common_patterns:
            for pattern in self.common_patterns["color_mappings"].get("patterns", []):
                if pattern["type"] == "color_offset" and pattern.get("weighted_score", 0) > 1:
                    offset = pattern["offset"]
                    cells_changed = 0

                    # 优先处理高权重对象
                    for obj_info in sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True):
                        for val, (i, j) in obj_info.original_obj:
                            if val != 0:  # 不处理背景
                                output_grid[i][j] = (val + offset) % 10  # 假设颜色范围是0-9
                                cells_changed += 1

                    if self.debug and cells_changed > 0:
                        weighted_score = pattern.get("weighted_score", 0)
                        self._debug_print(f"应用颜色偏移: +{offset}, 加权得分: {weighted_score:.2f}, 改变了 {cells_changed} 个单元格")

        # 应用位置变化，优先考虑高权重变化
        position_changes = self.common_patterns.get("position_changes", [])
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
                for obj_info in sorted(input_obj_infos, key=lambda x: x.obj_weight, reverse=True):
                    obj_color = obj_info.main_color
                    # 对象中的每个像素
                    for val, (r, c) in obj_info.original_obj:
                        nr, nc = int(r + dr), int(c + dc)
                        if 0 <= nr < height and 0 <= nc < width:
                            temp_grid[nr][nc] = val  # 保留原始颜色
                            cells_moved += 1

                # 合并结果
                for i in range(height):
                    for j in range(width):
                        if temp_grid[i][j] != 0:  # 只覆盖非零值
                            output_grid[i][j] = temp_grid[i][j]

                if self.debug and cells_moved > 0:
                    weight_score = best_position_change.get("weight_score", 0)
                    self._debug_print(f"应用位置变化: ({dr}, {dc}), 加权得分: {weight_score:.2f}, 移动了 {cells_moved} 个单元格")

        if self.debug:
            self._debug_save_grid(output_grid, "test_output_predicted")
            self._debug_print("完成测试预测")

        return output_grid

    def _calculate_test_object_weights(self, input_grid, input_obj_infos):
        """
        计算测试输入对象的权重

        Args:
            input_grid: 输入网格
            input_obj_infos: 输入对象信息列表
        """
        # 1. 初始权重 - 基于对象大小
        for obj_info in input_obj_infos:
            obj_info.obj_weight = obj_info.size

        # 2. 基于形状库匹配增加权重
        for obj_info in input_obj_infos:
            normalized_obj = obj_info.obj_000
            hashable_obj = self._get_hashable_representation(normalized_obj)

            # 检查是否在形状库中
            for shape_key, shape_info in self.shape_library.items():
                lib_shape = shape_info["normalized_shape"]
                lib_hashable = self._get_hashable_representation(lib_shape)

                if hashable_obj == lib_hashable:
                    # 如果形状匹配，增加权重
                    match_bonus = shape_info["count"] * 2  # 根据出现次数增加权重
                    obj_info.increase_weight(match_bonus)
                    if self.debug:
                        self._debug_print(f"对象 {obj_info.obj_id} 匹配形状库中的形状，增加权重 +{match_bonus}")
                    break

        # 3. 考虑颜色占比，调整背景对象权重
        self._adjust_background_object_weights(input_grid, input_obj_infos)

    def _debug_print_object_weights(self, obj_infos, name):
        """
        输出对象权重信息到调试文件

        Args:
            obj_infos: 对象信息列表
            name: 输出文件名
        """
        if not self.debug:
            return

        weight_info = []
        for obj_info in sorted(obj_infos, key=lambda x: x.obj_weight, reverse=True):
            weight_info.append({
                "obj_id": obj_info.obj_id,
                "weight": obj_info.obj_weight,
                "size": obj_info.size,
                "main_color": obj_info.main_color,
                "height": obj_info.height,
                "width": obj_info.width
            })

        self._debug_save_json(weight_info, name)

        # 打印权重信息
        self._debug_print(f"对象权重信息 ({name}):")
        for info in weight_info[:5]:  # 只打印前5个
            self._debug_print(f"  对象 {info['obj_id']}: 权重={info['weight']}, 大小={info['size']}, 颜色={info['main_color']}")
        if len(weight_info) > 5:
            self._debug_print(f"  ... 还有 {len(weight_info) - 5} 个对象")

    def _analyze_input_to_output_transformation(self, pair_id, input_grid, output_grid,
                                            input_obj_infos, output_obj_infos,
                                            diff_in_obj_infos, diff_out_obj_infos,
                                            diff_mapping):
        """
        分析从输入到输出的转换规则，找出如何生成输出网格

        Args:
            pair_id: 训练对ID
            input_grid, output_grid: 输入输出网格
            input_obj_infos, output_obj_infos: 输入输出对象信息
            diff_in_obj_infos, diff_out_obj_infos: 差异对象信息
            diff_mapping: 已识别的差异映射规则

        Returns:
            转换规则字典
        """
        if self.debug:
            self._debug_print(f"分析输入到输出的转换规则，pair_id={pair_id}")

        # 初始化转换规则
        transformation_rule = {
            "pair_id": pair_id,
            "preserved_objects": [],  # 保留的对象
            "modified_objects": [],   # 修改的对象
            "removed_objects": [],    # 移除的对象
            "added_objects": [],      # 新增的对象
            "transformation_patterns": [], # 抽象转换模式
            "object_operations": []   # 对象操作序列
        }

        # 为快速查找创建对象映射
        input_obj_by_id = {obj.obj_id: obj for obj in input_obj_infos}
        output_obj_by_id = {obj.obj_id: obj for obj in output_obj_infos}

        # 创建标准化形状到对象的映射，用于形状匹配
        input_by_shape = {}
        for obj in input_obj_infos:
            shape_key = self._get_hashable_representation(obj.obj_000)
            if shape_key not in input_by_shape:
                input_by_shape[shape_key] = []
            input_by_shape[shape_key].append(obj)

        output_by_shape = {}
        for obj in output_obj_infos:
            shape_key = self._get_hashable_representation(obj.obj_000)
            if shape_key not in output_by_shape:
                output_by_shape[shape_key] = []
            output_by_shape[shape_key].append(obj)

        # 1. 分析保留的对象 - 形状和位置都相同
        for in_obj in input_obj_infos:
            for out_obj in output_obj_infos:
                if (self._get_hashable_representation(in_obj.original_obj) ==
                    self._get_hashable_representation(out_obj.original_obj)):
                    transformation_rule["preserved_objects"].append({
                        "input_obj_id": in_obj.obj_id,
                        "output_obj_id": out_obj.obj_id,
                        "weight": in_obj.obj_weight,
                        "object": in_obj.to_dict()
                    })
                    # 标记此对象已处理
                    input_obj_by_id.pop(in_obj.obj_id, None)
                    output_obj_by_id.pop(out_obj.obj_id, None)

        # 2. 分析修改的对象 - 基于形状匹配或位置关系
        for in_obj in list(input_obj_by_id.values()):
            best_match = None
            best_transformation = None
            best_score = -1

            in_shape = self._get_hashable_representation(in_obj.obj_000)

            for out_obj in list(output_obj_by_id.values()):
                out_shape = self._get_hashable_representation(out_obj.obj_000)

                # 检查形状匹配
                if in_shape == out_shape:
                    # 相同形状，可能是位置变化
                    position_change = in_obj.get_positional_change(out_obj)
                    color_transform = in_obj.get_color_transformation(out_obj)

                    # 计算匹配得分
                    score = (in_obj.obj_weight * out_obj.obj_weight) * 0.8
                    if position_change["delta_row"] == 0 and position_change["delta_col"] == 0:
                        score += 0.5  # 位置相同加分

                    if score > best_score:
                        best_score = score
                        best_match = out_obj
                        best_transformation = {
                            "type": "same_shape_different_position",
                            "position_change": position_change,
                            "color_transform": color_transform
                        }
                else:
                    # 形状不同，检查是否有变换关系
                    matches, transform_type, transform_name = in_obj.matches_with_transformation(out_obj)
                    if matches:
                        # 计算得分
                        score = (in_obj.obj_weight * out_obj.obj_weight) * 0.6

                        if score > best_score:
                            best_score = score
                            best_match = out_obj
                            best_transformation = {
                                "type": transform_type,
                                "name": transform_name,
                                "position_change": in_obj.get_positional_change(out_obj),
                                "color_transform": in_obj.get_color_transformation(out_obj)
                            }

            if best_match and best_score > 0:
                transformation_rule["modified_objects"].append({
                    "input_obj_id": in_obj.obj_id,
                    "output_obj_id": best_match.obj_id,
                    "transformation": best_transformation,
                    "confidence": best_score / (in_obj.obj_weight * best_match.obj_weight + 0.01),
                    "weight_product": in_obj.obj_weight * best_match.obj_weight
                })

                # 标记已处理
                input_obj_by_id.pop(in_obj.obj_id, None)
                output_obj_by_id.pop(best_match.obj_id, None)

                # 记录转换操作
                transformation_rule["object_operations"].append({
                    "operation": "transform",
                    "input_obj_id": in_obj.obj_id,
                    "output_obj_id": best_match.obj_id,
                    "details": best_transformation
                })

        # 3. 分析移除的对象 - 输入中有但输出中没有的
        for in_obj in input_obj_by_id.values():
            transformation_rule["removed_objects"].append({
                "input_obj_id": in_obj.obj_id,
                "weight": in_obj.obj_weight,
                "object": in_obj.to_dict()
            })

            # 记录删除操作
            transformation_rule["object_operations"].append({
                "operation": "remove",
                "input_obj_id": in_obj.obj_id,
                "details": {"reason": "no_match_in_output"}
            })

        # 4. 分析新增的对象 - 输出中有但输入中没有的
        for out_obj in output_obj_by_id.values():
            # 检查是否可以从某些输入对象组合生成
            generated_from = self._check_if_generated_from(out_obj, input_obj_infos, diff_in_obj_infos)

            transformation_rule["added_objects"].append({
                "output_obj_id": out_obj.obj_id,
                "weight": out_obj.obj_weight,
                "object": out_obj.to_dict(),
                "generated_from": generated_from
            })

            # 记录添加操作
            transformation_rule["object_operations"].append({
                "operation": "add",
                "output_obj_id": out_obj.obj_id,
                "details": {
                    "generated_from": generated_from
                }
            })

        # 5. 提取抽象转换模式
        transformation_patterns = self._extract_transformation_patterns(
            transformation_rule["preserved_objects"],
            transformation_rule["modified_objects"],
            transformation_rule["removed_objects"],
            transformation_rule["added_objects"]
        )

        transformation_rule["transformation_patterns"] = transformation_patterns

        if self.debug:
            self._debug_print(f"找到 {len(transformation_rule['preserved_objects'])} 个保留的对象")
            self._debug_print(f"找到 {len(transformation_rule['modified_objects'])} 个修改的对象")
            self._debug_print(f"找到 {len(transformation_rule['removed_objects'])} 个移除的对象")
            self._debug_print(f"找到 {len(transformation_rule['added_objects'])} 个新增的对象")
            self._debug_print(f"提取了 {len(transformation_patterns)} 个抽象转换模式")
            self._debug_save_json(transformation_rule, f"transformation_rule_{pair_id}")

        return transformation_rule

    def _check_if_generated_from(self, output_obj, input_objs, diff_input_objs):
        """检查输出对象是否可能由输入对象生成"""
        possible_sources = []

        # 分析可能的来源：
        # 1. 可能是输入对象的复制、移动或变换
        for input_obj in input_objs:
            matches, transform_type, transform_name = input_obj.matches_with_transformation(output_obj)
            if matches:
                possible_sources.append({
                    "type": "transformed_input",
                    "input_obj_id": input_obj.obj_id,
                    "transformation": {
                        "type": transform_type,
                        "name": transform_name
                    },
                    "confidence": 0.7,
                    "weight_product": input_obj.obj_weight * output_obj.obj_weight
                })

        # 2. 可能是多个输入对象的组合
        if len(input_objs) >= 2:
            for i, obj1 in enumerate(input_objs):
                for obj2 in input_objs[i+1:]:
                    # 检查是否是两个对象的合并
                    combined = self._check_if_combined(output_obj, obj1, obj2)
                    if combined:
                        possible_sources.append({
                            "type": "combined_inputs",
                            "input_obj_ids": [obj1.obj_id, obj2.obj_id],
                            "combination_type": combined["type"],
                            "confidence": combined["confidence"],
                            "weight_product": obj1.obj_weight * obj2.obj_weight * output_obj.obj_weight
                        })

        # 3. 可能是输入对象的填充、描边等操作
        for input_obj in input_objs:
            operation = self._check_common_operations(output_obj, input_obj)
            if operation:
                possible_sources.append({
                    "type": "operation_on_input",
                    "input_obj_id": input_obj.obj_id,
                    "operation": operation["type"],
                    "confidence": operation["confidence"],
                    "weight_product": input_obj.obj_weight * output_obj.obj_weight
                })

        # 按置信度排序
        possible_sources.sort(key=lambda x: (x.get("weight_product", 0), x.get("confidence", 0)), reverse=True)

        return possible_sources

    def _check_if_combined(self, output_obj, input_obj1, input_obj2):
        """检查输出对象是否是两个输入对象的组合"""

        out_pixels = {loc for _, loc in output_obj.original_obj}
        in1_pixels = {loc for _, loc in input_obj1.original_obj}
        in2_pixels = {loc for _, loc in input_obj2.original_obj}

        # 计算像素重叠
        union_pixels = in1_pixels.union(in2_pixels)
        overlap_with_output = out_pixels.intersection(union_pixels)

        # 如果合并的输入像素大部分存在于输出中，可能是组合
        if len(overlap_with_output) > 0.8 * len(union_pixels):
            return {
                "type": "spatial_union",
                "confidence": len(overlap_with_output) / len(union_pixels)
            }

        return None

    def _check_common_operations(self, output_obj, input_obj):
        """检查常见操作如填充、描边等"""
        # 提取像素集
        out_pixels = {loc for _, loc in output_obj.original_obj}
        in_pixels = {loc for _, loc in input_obj.original_obj}

        # 检查填充
        if out_pixels.issuperset(in_pixels) and len(out_pixels) > len(in_pixels):
            # 如果输出包含所有输入像素且更大，可能是填充
            return {
                "type": "fill",
                "confidence": len(in_pixels) / len(out_pixels)
            }

        # 检查描边 - 简化版，只检查输出是否围绕输入
        if not out_pixels.intersection(in_pixels) and self._is_surrounding(out_pixels, in_pixels):
            return {
                "type": "outline",
                "confidence": 0.8
            }

        return None

    def _is_surrounding(self, outline_pixels, inner_pixels):
        """检查一组像素是否围绕另一组像素"""
        # 简化实现，检查是否存在相邻关系
        for i, j in inner_pixels:
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for ni, nj in neighbors:
                if (ni, nj) in outline_pixels:
                    return True
        return False

    def _extract_transformation_patterns(self, preserved, modified, removed, added):
        """提取抽象转换模式"""
        patterns = []

        # 分析是否有一致的移动模式
        position_changes = []
        for obj in modified:
            if "transformation" in obj and "position_change" in obj["transformation"]:
                change = obj["transformation"]["position_change"]
                position_changes.append((change["delta_row"], change["delta_col"], obj["weight_product"]))

        if position_changes:
            # 按位置变化分组
            changes_by_delta = {}
            for dr, dc, weight in position_changes:
                key = (dr, dc)
                if key not in changes_by_delta:
                    changes_by_delta[key] = []
                changes_by_delta[key].append(weight)

            # 找出最常见的位置变化
            if changes_by_delta:
                best_delta, weights = max(changes_by_delta.items(), key=lambda x: sum(x[1]))
                dr, dc = best_delta
                avg_weight = sum(weights) / len(weights)

                patterns.append({
                    "type": "position_pattern",
                    "delta_row": dr,
                    "delta_col": dc,
                    "count": len(weights),
                    "avg_weight": avg_weight,
                    "confidence": len(weights) / max(1, len(modified))
                })

        # 分析是否有一致的颜色变换模式
        color_transformations = []
        for obj in modified:
            if "transformation" in obj and "color_transform" in obj["transformation"]:
                color_trans = obj["transformation"]["color_transform"]
                if "color_mapping" in color_trans:
                    for from_color, to_color in color_trans["color_mapping"].items():
                        color_transformations.append((from_color, to_color, obj["weight_product"]))

        if color_transformations:
            # 按颜色变化分组
            changes_by_color = {}
            for from_color, to_color, weight in color_transformations:
                key = (from_color, to_color)
                if key not in changes_by_color:
                    changes_by_color[key] = []
                changes_by_color[key].append(weight)

            # 找出最常见的颜色变化
            if changes_by_color:
                best_color_change, weights = max(changes_by_color.items(), key=lambda x: sum(x[1]))
                from_color, to_color = best_color_change
                avg_weight = sum(weights) / len(weights)

                patterns.append({
                    "type": "color_pattern",
                    "from_color": from_color,
                    "to_color": to_color,
                    "count": len(weights),
                    "avg_weight": avg_weight,
                    "confidence": len(weights) / max(1, len(modified))
                })

        # 分析是否有添加/删除对象的模式
        if added:
            # 分析新增对象的模式
            patterns.append({
                "type": "addition_pattern",
                "count": len(added),
                "avg_weight": sum(obj["weight"] for obj in added) / len(added),
                "description": "可能添加了新对象"
            })

        if removed:
            # 分析移除对象的模式
            patterns.append({
                "type": "removal_pattern",
                "count": len(removed),
                "avg_weight": sum(obj["weight"] for obj in removed) / len(removed),
                "description": "可能移除了某些对象"
            })

        return patterns

    def apply_transformation_rules(self, input_grid, common_patterns, transformation_rules=None):
        """
        应用提取的转换规则，将输入网格转换为预测的输出网格

        Args:
            input_grid: 输入网格
            common_patterns: 识别的共有模式
            transformation_rules: 可选，特定的转换规则列表，如果不提供则使用当前累积的规则

        Returns:
            预测的输出网格
        """
        if self.debug:
            self._debug_print(f"\n\n开始应用转换规则生成预测输出")

        # 如果没有提供转换规则，使用当前的规则
        if transformation_rules is None:
            transformation_rules = self.transformation_rules

        # 如果没有规则，无法生成预测
        if not transformation_rules:
            if self.debug:
                self._debug_print("未提供转换规则，无法生成预测")
            return input_grid  # 返回输入网格作为默认预测

        # 深拷贝输入网格作为初始输出
        output_grid = [list(row) for row in input_grid]

        # 创建2D的转换记录，跟踪每个位置是否已被转换
        transformed = [[False for _ in range(len(input_grid[0]))] for _ in range(len(input_grid))]

        # 1. 提取输入网格的对象
        input_obj_infos = []
        for param in [(True, True, False), (True, False, False), (False, False, False), (False, True, False)]:
            height, width = len(input_grid), len(input_grid[0])
            objects = self._extract_objects_with_param(input_grid, param)

            for obj in objects:
                obj_info = self._create_obj_info(0, 'test_in', obj, param, height, width)
                input_obj_infos.append(obj_info)

        # 2. 计算输入对象的权重 - 使用正确的测试对象权重计算方法
        self._calculate_test_object_weights(input_grid, input_obj_infos)

        # 3. 按权重对对象排序（权重高的优先处理）
        input_obj_infos.sort(key=lambda x: x.obj_weight if hasattr(x, 'obj_weight') else 0, reverse=True)

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
                    for input_obj in input_obj_infos:
                        # 检查是否可以应用此转换
                        if self._can_apply_transformation(input_obj, modification):
                            # 应用转换得到新对象
                            transformed_obj = self._apply_object_transformation(
                                input_obj, modification['transformation'])

                            # 更新输出网格和转换记录
                            for color, (i, j) in transformed_obj:
                                if 0 <= i < len(output_grid) and 0 <= j < len(output_grid[0]):
                                    output_grid[i][j] = color
                                    transformed[i][j] = True

            # 应用"移除对象"列表，从输出中移除指定对象
            if 'removed_objects' in rule and rule['removed_objects']:
                for removed in rule['removed_objects']:
                    # 找到输入中与移除对象特征匹配的对象
                    for input_obj in input_obj_infos:
                        if self._match_object_pattern(input_obj, removed):
                            # 从输出中移除此对象
                            for _, (i, j) in input_obj.original_obj:
                                if 0 <= i < len(output_grid) and 0 <= j < len(output_grid[0]):
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
                                for input_obj in input_obj_infos:
                                    if hasattr(input_obj, 'obj_id') and str(input_obj.obj_id) == str(input_id):
                                        # 应用变换
                                        new_obj = self._generate_object_from_transformation(
                                            input_obj, source['transformation'])

                                        # 更新输出网格
                                        for color, (i, j) in new_obj:
                                            if 0 <= i < len(output_grid) and 0 <= j < len(output_grid[0]):
                                                output_grid[i][j] = color
                                                transformed[i][j] = True

                            elif source['type'] == 'combined_inputs':
                                # 组合多个输入对象
                                # 实现对象组合逻辑
                                pass

        # 5. 应用抽象转换模式
        for pattern in common_patterns:
            if isinstance(pattern, dict) and pattern.get('type') == 'position_pattern' and pattern.get('confidence', 0) > 0.5:
                # 应用位置变化模式
                delta_row = pattern.get('delta_row', 0)
                delta_col = pattern.get('delta_col', 0)

                # 对尚未转换的位置应用位置变化
                height, width = len(input_grid), len(input_grid[0])
                new_grid = [list(row) for row in output_grid]

                for i in range(height):
                    for j in range(width):
                        if not transformed[i][j] and input_grid[i][j] != 0:
                            new_i = i + delta_row
                            new_j = j + delta_col

                            # 确保新位置在网格范围内
                            if 0 <= new_i < height and 0 <= new_j < width:
                                new_grid[new_i][new_j] = input_grid[i][j]
                                new_grid[i][j] = 0  # 清除原位置

                # 更新输出网格
                output_grid = new_grid

            elif isinstance(pattern, dict) and pattern.get('type') == 'color_pattern' and pattern.get('confidence', 0) > 0.5:
                # 应用颜色变化模式
                from_color = pattern.get('from_color')
                to_color = pattern.get('to_color')

                # 对尚未转换的位置应用颜色变化
                for i in range(len(output_grid)):
                    for j in range(len(output_grid[0])):
                        if not transformed[i][j] and input_grid[i][j] == from_color:
                            output_grid[i][j] = to_color

        if self.debug:
            self._debug_save_grid(output_grid, "predicted_output_from_rules")
            self._debug_print("完成规则应用，生成预测输出")

        return output_grid

    def _can_apply_transformation(self, input_obj, modification):
        """检查是否可以将变换规则应用到输入对象"""
        # 实现规则匹配逻辑
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

    def get_prediction_confidence(self, predicted_output, actual_output):
        """
        计算预测与实际输出的匹配程度，返回置信度得分

        Args:
            predicted_output: 预测的输出网格
            actual_output: 实际的输出网格

        Returns:
            匹配置信度 (0-1)
        """
        if predicted_output == actual_output:
            return 1.0  # 完全匹配

        # 计算网格大小
        if not predicted_output or not actual_output:
            return 0.0

        height_pred, width_pred = len(predicted_output), len(predicted_output[0])
        height_act, width_act = len(actual_output), len(actual_output[0])

        # 如果尺寸不同，返回较低的置信度
        if height_pred != height_act or width_pred != width_act:
            return 0.1  # 尺寸不匹配，几乎没有信心

        # 计算像素匹配率
        total_pixels = height_pred * width_pred
        matching_pixels = 0

        for i in range(height_pred):
            for j in range(width_pred):
                if predicted_output[i][j] == actual_output[i][j]:
                    matching_pixels += 1

        # 基本置信度：匹配像素比例
        base_confidence = matching_pixels / total_pixels

        # 优化：考虑重要区域的匹配程度
        # 例如：非背景像素（非0像素）的匹配更重要
        non_zero_pred = sum(1 for row in predicted_output for pixel in row if pixel != 0)
        non_zero_act = sum(1 for row in actual_output for pixel in row if pixel != 0)

        # 计算非零像素的匹配
        non_zero_matching = 0
        for i in range(height_pred):
            for j in range(width_pred):
                if predicted_output[i][j] != 0 and predicted_output[i][j] == actual_output[i][j]:
                    non_zero_matching += 1

        # 非零像素匹配率（避免除零）
        if max(non_zero_pred, non_zero_act) > 0:
            non_zero_confidence = non_zero_matching / max(non_zero_pred, non_zero_act)
        else:
            non_zero_confidence = 1.0  # 如果两者都没有非零像素，则认为匹配

        # 加权组合两种置信度，非零区域匹配更重要
        combined_confidence = 0.3 * base_confidence + 0.7 * non_zero_confidence

        if self.debug:
            self._debug_print(f"预测置信度: 基本={base_confidence:.4f}, 非零区域={non_zero_confidence:.4f}, 组合={combined_confidence:.4f}")

        return combined_confidence

    def calculate_rule_confidence(self, input_grid, predicted_output):
        """
        计算基于规则生成的预测输出的置信度

        Args:
            input_grid: 输入网格
            predicted_output: 预测的输出网格

        Returns:
            规则预测置信度 (0-1)
        """
        # 如果没有规则，置信度低
        if not self.transformation_rules:
            return 0.2

        # 获取应用规则的数量
        num_rules_applied = 0
        total_rule_confidence = 0.0

        # 计算各种规则的应用情况
        for rule in self.transformation_rules:
            # 检查规则是否适用于当前输入/输出
            if self._is_rule_applicable(rule, input_grid, predicted_output):
                num_rules_applied += 1

                # 计算规则的置信度
                rule_conf = 0.0

                # 1. 如果规则在训练数据中频繁出现，提高置信度
                if 'pair_id' in rule:
                    rule_conf += 0.3  # 基础置信度

                # 2. 考虑对象权重
                if 'weighted_objects' in rule and rule['weighted_objects']:
                    avg_weight = sum(obj['weight'] for obj in rule['weighted_objects']) / len(rule['weighted_objects'])
                    weight_factor = min(1.0, avg_weight / 5.0)  # 规范化到0-1范围
                    rule_conf += weight_factor * 0.3

                # 3. 考虑模式匹配
                if 'transformation_patterns' in rule and rule['transformation_patterns']:
                    patterns = rule['transformation_patterns']
                    for pattern in patterns:
                        if pattern.get('confidence', 0) > 0.7:
                            rule_conf += 0.2
                            break

                # 累加总置信度
                total_rule_confidence += rule_conf

        # 如果没有应用规则，返回低置信度
        if num_rules_applied == 0:
            return 0.3

        # 计算平均规则置信度，并确保不超过1.0
        avg_rule_confidence = min(1.0, total_rule_confidence / num_rules_applied)

        if self.debug:
            self._debug_print(f"规则预测置信度: {avg_rule_confidence:.4f} (应用了 {num_rules_applied} 条规则)")

        return avg_rule_confidence

    def _is_rule_applicable(self, rule, input_grid, predicted_output):
        """检查规则是否适用于给定的输入/输出对"""
        # 简化版规则适用性检查
        # 在实际应用中，可以根据规则的具体内容进行更复杂的检查
        # 例如检查对象匹配、位置变化、颜色变换等是否符合规则

        # 检查输入网格中是否存在与规则相关的特征
        if 'object_mappings' in rule and rule['object_mappings']:
            # 抽取输入网格中的对象
            height, width = len(input_grid), len(input_grid[0])
            input_objects = []
            for param in [(True, True, False), (False, False, False)]:
                objects = self._extract_objects_with_param(input_grid, param)
                for obj in objects:
                    input_objects.append(self._create_obj_info(0, 'test_in', obj, param, height, width))

            # 检查是否有对象匹配规则中的对象
            for mapping in rule['object_mappings']:
                if 'diff_in_object' in mapping:
                    in_obj_info = mapping['diff_in_object']
                    # 简化检查：只检查是否有类似大小和颜色的对象
                    for obj in input_objects:
                        if (hasattr(obj, 'size') and hasattr(obj, 'main_color') and
                            abs(obj.size - in_obj_info.get('size', 0)) < 3 and
                            obj.main_color == in_obj_info.get('main_color')):
                            return True

        # 默认返回True，表示规则适用
        return True  # 简化版，实际应用中需要更复杂的匹配逻辑

    def _learn_dynamic_rules(self, patterns):
        """从模式中学习动态规则"""
        rules = []

        for pattern in patterns:
            # 分析模式中的一致性和变化性
            consistencies = self._find_consistencies(pattern)
            variations = self._find_variations(pattern)

            # 创建动态规则
            rule = {
                "conditions": self._extract_conditions(consistencies),
                "transformations": self._extract_transformations(consistencies, variations),
                "confidence": pattern.get("confidence", 0),
                "priority": self._calculate_rule_priority(pattern)
            }
            rules.append(rule)

        return rules

    def _find_consistencies(self, pattern):
        """找出模式中的一致性特征"""
        # 假设pattern是一组变换的集合
        examples = pattern.get("examples", [])
        if not examples or len(examples) < 2:
            return {}

        # 分析例子中的共同属性
        consistencies = {
            "position": {},
            "color": {},
            "shape": {},
            "context": {}
        }

        # 位置一致性
        positions = [ex.get("position_change", {}) for ex in examples if "position_change" in ex]
        if positions:
            delta_rows = [p.get("delta_row") for p in positions if "delta_row" in p]
            delta_cols = [p.get("delta_col") for p in positions if "delta_col" in p]

            # 检查是否所有例子都有相同的位置变化
            if delta_rows and all(dr == delta_rows[0] for dr in delta_rows):
                consistencies["position"]["delta_row"] = delta_rows[0]
            if delta_cols and all(dc == delta_cols[0] for dc in delta_cols):
                consistencies["position"]["delta_col"] = delta_cols[0]

        # 颜色一致性
        color_changes = [ex.get("color_change", {}) for ex in examples if "color_change" in ex]
        if color_changes:
            # 合并所有颜色映射
            all_mappings = {}
            for change in color_changes:
                for from_color, to_color in change.items():
                    if from_color not in all_mappings:
                        all_mappings[from_color] = []
                    all_mappings[from_color].append(to_color)

            # 找出一致的颜色变换
            for from_color, to_colors in all_mappings.items():
                if all(tc == to_colors[0] for tc in to_colors):
                    consistencies["color"][from_color] = to_colors[0]

        return consistencies

    def _find_variations(self, pattern):
        """找出模式中的变化性特征"""
        examples = pattern.get("examples", [])
        if not examples or len(examples) < 2:
            return {}

        variations = {
            "position": {},
            "color": {},
            "shape": {},
            "context_dependent": []
        }

        # 分析位置变化的变化性
        positions = [ex.get("position_change", {}) for ex in examples if "position_change" in ex]
        if positions:
            delta_rows = [p.get("delta_row") for p in positions if "delta_row" in p]
            delta_cols = [p.get("delta_col") for p in positions if "delta_col" in p]

            # 检查位置变化是否不一致
            if delta_rows and not all(dr == delta_rows[0] for dr in delta_rows):
                # 可能是基于位置的条件变化
                variations["position"]["delta_row_varies"] = True
            if delta_cols and not all(dc == delta_cols[0] for dc in delta_cols):
                variations["position"]["delta_col_varies"] = True

        # 分析颜色变化的变化性
        color_changes = [ex.get("color_change", {}) for ex in examples if "color_change" in ex]
        if color_changes:
            # 合并所有颜色映射
            all_mappings = {}
            for change in color_changes:
                for from_color, to_color in change.items():
                    if from_color not in all_mappings:
                        all_mappings[from_color] = []
                    all_mappings[from_color].append(to_color)

            # 检查颜色变化是否不一致
            for from_color, to_colors in all_mappings.items():
                if not all(tc == to_colors[0] for tc in to_colors):
                    variations["color"][from_color] = to_colors

        # 分析上下文相关变化
        for example in examples:
            if "original_context" in example and "transformed_context" in example:
                # 上下文相关变化
                context = example["original_context"]
                position = context.get("position_category")

                if position:
                    # 添加位置上下文依赖
                    ctx_dep = {
                        "type": "position_context",
                        "position": position,
                        "transformation": {
                            "position_change": example.get("position_change", {}),
                            "color_change": example.get("color_change", {})
                        }
                    }
                    variations["context_dependent"].append(ctx_dep)

        return variations

    def _extract_conditions(self, consistencies):
        """从一致性中提取条件"""
        conditions = []

        # 位置条件
        pos_consistencies = consistencies.get("position", {})
        if pos_consistencies:
            conditions.append({
                "type": "position",
                "details": pos_consistencies
            })

        # 颜色条件
        color_consistencies = consistencies.get("color", {})
        if color_consistencies:
            conditions.append({
                "type": "color",
                "details": color_consistencies
            })

        # 形状条件
        shape_consistencies = consistencies.get("shape", {})
        if shape_consistencies:
            conditions.append({
                "type": "shape",
                "details": shape_consistencies
            })

        return conditions

    def _extract_transformations(self, consistencies, variations):
        """提取变换规则"""
        transformations = []

        # 从一致性和变化性中提取变换

        # 1. 位置变换
        if "position" in consistencies and consistencies["position"]:
            # 静态位置变换
            transformations.append({
                "type": "position_change",
                "static": consistencies["position"],
                "dynamic": variations.get("position", {})
            })

        # 2. 颜色变换
        if "color" in consistencies and consistencies["color"]:
            # 静态颜色变换
            transformations.append({
                "type": "color_change",
                "static": consistencies["color"],
                "dynamic": variations.get("color", {})
            })

        # 3. 上下文相关变换
        context_deps = variations.get("context_dependent", [])
        if context_deps:
            for ctx_dep in context_deps:
                transformations.append({
                    "type": "context_dependent",
                    "context": ctx_dep.get("type"),
                    "conditions": ctx_dep,
                    "transformations": ctx_dep.get("transformation", {})
                })

        return transformations

    def _calculate_rule_priority(self, pattern):
        """计算规则优先级"""
        # 基础优先级
        priority = 1.0

        # 基于出现频率调整
        count = pattern.get("instance_count", 0)
        if count > 0:
            priority += min(2.0, count * 0.2)  # 最多+2.0

        # 基于权重调整
        weight = pattern.get("avg_weight", 0)
        if weight > 0:
            priority += min(3.0, weight * 0.3)  # 最多+3.0

        # 基于信心调整
        confidence = pattern.get("confidence", 0)
        if confidence > 0:
            priority += confidence * 4.0  # 最多+4.0

        return priority