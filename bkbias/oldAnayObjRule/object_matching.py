"""
对象匹配和映射模块

负责识别和分析对象之间的映射关系。
"""

from typing import List, Dict, Tuple, Any, Callable, Optional
from collections import defaultdict

from .utils import get_hashable_representation


class ObjectMatcher:
    """处理对象匹配和映射的类"""

    def __init__(self, debug_print=False):
        """
        初始化对象匹配器

        Args:
            debug_print: 调试打印函数（可选）
        """
        self.debug_print = print
        self.debug = debug_print

    def analyze_diff_mapping_with_weights(self, pair_id, input_grid, output_grid, diff_in, diff_out,
                                        input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos,debug):
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

        #! !
        diff_in_obj_infos = input_obj_infos
        diff_out_obj_infos = output_obj_infos

        self.debug_print = print
        self.debug = debug

        if self.debug_print:
            self.debug_print(f"基于权重分析差异映射关系，pair_id={pair_id}")

        mapping_rule = {
            "pair_id": pair_id,
            "Rule_object_mappings": [],
            "Rule_shape_transformations": [],
            "Rule_color_mappings": {},
            "position_changes": [],
            "part_whole_relationships": [],
            "weighted_objects": [],  # 添加权重信息
            "combined_transformations": []  # 新增：捕获多维度变换的组合规则
        }

        # 准备分析diff对象间的映射
        if not diff_in_obj_infos or not diff_out_obj_infos:
            if self.debug_print:
                self.debug_print("差异对象为空，无法分析映射")
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

        if self.debug_print:
            self.debug_print(f"找到 {len(object_mappings)} 个基于形状和权重的对象匹配")

        # # 分析每个映射的变换
        # for in_obj, out_obj, match_info in object_mappings:
        #     # 分析颜色变换
        #     color_transformation = in_obj.get_color_transformation(out_obj)

        #     # 分析位置变换
        #     position_change = in_obj.get_positional_change(out_obj)

        #     # 添加到映射规则
        #     mapping_rule["Rule_object_mappings"].append({
        #         "diff_in_object": in_obj.to_dict(),
        #         "diff_out_object": out_obj.to_dict(),
        #         "match_info": match_info,
        #         "weight_product": in_obj.obj_weight * out_obj.obj_weight  # 添加权重乘积作为匹配强度
        #     })

        #     # 记录形状变换
        #     mapping_rule["Rule_shape_transformations"].append({
        #         "in_obj_id": in_obj.obj_id,
        #         "out_obj_id": out_obj.obj_id,
        #         "transform_type": match_info["transform_type"],
        #         "transform_name": match_info["transform_name"],
        #         "confidence": match_info["confidence"],
        #         "weight_in": in_obj.obj_weight,
        #         "weight_out": out_obj.obj_weight
        #     })

        #     # 记录颜色映射
        #     if color_transformation and color_transformation.get("color_mapping"):
        #         for from_color, to_color in color_transformation["color_mapping"].items():
        #             if from_color not in mapping_rule["Rule_color_mappings"]:
        #                 mapping_rule["Rule_color_mappings"][from_color] = {
        #                     "to_color": to_color,
        #                     "weight": in_obj.obj_weight,  # 使用输入对象权重作为颜色映射权重
        #                     "in_obj_id": in_obj.obj_id,   # 添加输入对象ID
        #                     "out_obj_id": out_obj.obj_id  # 添加输出对象ID
        #                 }
        #             elif in_obj.obj_weight > mapping_rule["Rule_color_mappings"][from_color]["weight"]:
        #                 # 如果当前对象权重更高，更新颜色映射
        #                 mapping_rule["Rule_color_mappings"][from_color] = {
        #                     "to_color": to_color,
        #                     "weight": in_obj.obj_weight,
        #                     "in_obj_id": in_obj.obj_id,   # 添加输入对象ID
        #                     "out_obj_id": out_obj.obj_id  # 添加输出对象ID
        #                 }

        #     # 记录位置变化
        #     mapping_rule["position_changes"].append({
        #         "in_obj_id": in_obj.obj_id,
        #         "out_obj_id": out_obj.obj_id,
        #         "delta_row": position_change["delta_row"],
        #         "delta_col": position_change["delta_col"],
        #         "direction": position_change.get("direction"),
        #         "orientation": position_change.get("orientation"),
        #         "weight_in": in_obj.obj_weight,
        #         "weight_out": out_obj.obj_weight
        #     })

        #     # 创建组合变换记录
        #     combined_transform = {
        #         "in_obj_id": in_obj.obj_id,
        #         "out_obj_id": out_obj.obj_id,
        #         "shape_transform": {
        #             "transform_type": match_info["transform_type"],
        #             "transform_name": match_info["transform_name"],
        #         },
        #         "position_change": position_change,
        #         "color_change": color_transformation.get("color_mapping", {}),
        #         "weight_score": in_obj.obj_weight * out_obj.obj_weight,
        #         "confidence": match_info["confidence"],
        #         # 记录对象的原始上下文信息
        #         "original_context": self._capture_object_context(in_obj, input_grid),
        #         "transformed_context": self._capture_object_context(out_obj, output_grid)
        #     }

        #     mapping_rule["combined_transformations"].append(combined_transform)
        #!

        transformation_rule = self._analyze_input_to_output_transformation(
            pair_id, input_grid, output_grid,
            input_obj_infos, output_obj_infos,
            diff_in_obj_infos, diff_out_obj_infos,
            mapping_rule
        )

        # 将转换规则合并到映射规则中
        mapping_rule["input_to_output_transformation"] = transformation_rule
        # if self.debug:
        if self.debug:
            print('\n\n\n one pair in out mapping_rule:\n\n', mapping_rule)

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

    def _capture_object_context(self, obj_info, grid):
        """
        捕获对象的上下文环境（周围的颜色和对象）

        Args:
            obj_info: 对象信息
            grid: 所在网格

        Returns:
            上下文信息字典
        """
        context = {
            "surrounding_colors": {},
            "object_position": "unknown"  # 例如：中心、边缘、角落等
        }

        # 如果对象或网格为空，则返回空上下文
        if not obj_info or not grid:
            return context

        # 获取对象的边界框
        min_row = min(i for _, (i, _) in obj_info.original_obj)
        max_row = max(i for _, (i, _) in obj_info.original_obj)
        min_col = min(j for _, (_, j) in obj_info.original_obj)
        max_col = max(j for _, (_, j) in obj_info.original_obj)

        # 确定对象在网格中的位置
        grid_height, grid_width = len(grid), len(grid[0])

        # 计算对象的中心点在网格中的相对位置
        center_row = (min_row + max_row) / 2
        center_col = (min_col + max_col) / 2
        rel_row = center_row / grid_height
        rel_col = center_col / grid_width

        # 确定对象位置描述
        if rel_row < 0.25:
            v_pos = "top"
        elif rel_row > 0.75:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        if rel_col < 0.25:
            h_pos = "left"
        elif rel_col > 0.75:
            h_pos = "right"
        else:
            h_pos = "center"

        context["object_position"] = f"{v_pos}-{h_pos}"

        # 创建对象像素位置集合，方便快速查找
        obj_pixels = {(i, j) for _, (i, j) in obj_info.original_obj}

        # 扫描周围像素，收集上下文信息
        for i in range(max(0, min_row-1), min(grid_height, max_row+2)):
            for j in range(max(0, min_col-1), min(grid_width, max_col+2)):
                # 如果是对象外部的像素
                if (i, j) not in obj_pixels:
                    color = grid[i][j]
                    if color not in context["surrounding_colors"]:
                        context["surrounding_colors"][color] = 0
                    context["surrounding_colors"][color] += 1

        return context

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
        if self.debug_print:
            self.debug_print(f"\n\nInToOut:分析输入到输出的转换规则，pair_id={pair_id}")

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
            shape_key = get_hashable_representation(obj.obj_000)
            if shape_key not in input_by_shape:
                input_by_shape[shape_key] = []
            input_by_shape[shape_key].append(obj)

        output_by_shape = {}
        for obj in output_obj_infos:
            shape_key = get_hashable_representation(obj.obj_000)
            if shape_key not in output_by_shape:
                output_by_shape[shape_key] = []
            output_by_shape[shape_key].append(obj)

        # 1. 分析保留的对象 - 形状和位置都相同
        for in_obj in input_obj_infos:
            for out_obj in output_obj_infos:
                if (get_hashable_representation(in_obj.original_obj) ==
                    get_hashable_representation(out_obj.original_obj)):
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

            in_shape = get_hashable_representation(in_obj.obj_000)

            for out_obj in list(output_obj_by_id.values()):
                out_shape = get_hashable_representation(out_obj.obj_000)

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

        if self.debug_print:
            self.debug_print(f"OneInOut找到 {len(transformation_rule['preserved_objects'])} 个保留的对象")
            self.debug_print(f"OneInOut找到 {len(transformation_rule['modified_objects'])} 个修改的对象")
            self.debug_print(f"OneInOut找到 {len(transformation_rule['removed_objects'])} 个移除的对象")
            self.debug_print(f"OneInOut找到 {len(transformation_rule['added_objects'])} 个新增的对象")
            self.debug_print(f"OneInOut找到 {len(transformation_patterns)} 个抽象转换模式")

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