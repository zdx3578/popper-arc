"""
模式分析器模块

用于分析和提取数据中的共有模式。
"""

from collections import defaultdict
from typing import List, Dict, Any, Callable, Optional


class PatternAnalyzer:
    """处理共有模式分析的类"""

    def __init__(self, debug_print=None):
        """
        初始化模式分析器

        Args:
            debug_print: 调试打印函数（可选）
        """
        self.debug_print = debug_print

    def analyze_common_patterns(self, oneInOut_mapping_rules):
        """
        分析多对训练数据的共有模式，考虑权重因素

        Args:
            oneInOut_mapping_rules: 映射规则列表

        Returns:
            共有模式字典
        """
        if not oneInOut_mapping_rules:
            return {}

        # 分析共有的形状变换模式
        common_shape_transformations = self._find_common_shape_transformations(oneInOut_mapping_rules)

        # 分析共有的颜色映射模式
        common_color_mappings = self._find_common_color_mappings(oneInOut_mapping_rules)

        # 分析共有的位置变化模式
        common_position_changes = self._find_common_position_changes(oneInOut_mapping_rules)

        common_patterns = {
            "shape_transformations": common_shape_transformations,
            "color_mappings": common_color_mappings,
            "position_changes": common_position_changes
        }
        # print("共有模式分析结果：", common_patterns)
        return common_patterns

    def _find_common_shape_transformations(self, oneInOut_mapping_rules):
        """
        寻找共有的形状变换模式，考虑权重

        Args:
            oneInOut_mapping_rules: 映射规则列表

        Returns:
            共有形状变换模式列表
        """
        # 收集所有形状变换
        all_transformations = []
        for rule in oneInOut_mapping_rules:
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

    def _find_common_color_mappings(self, oneInOut_mapping_rules):
        """
        寻找共有的颜色映射模式，考虑权重

        Args:
            oneInOut_mapping_rules: 映射规则列表

        Returns:
            共有颜色映射模式字典
        """
        # 收集所有颜色映射
        all_mappings = defaultdict(list)

        for rule in oneInOut_mapping_rules:
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
        total_examples = len(oneInOut_mapping_rules)

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
                offset = int(to_color) - int(from_color)
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

    def _find_common_position_changes(self, oneInOut_mapping_rules):
        """
        寻找共有的位置变化模式，考虑权重

        Args:
            oneInOut_mapping_rules: 映射规则列表

        Returns:
            共有位置变化模式列表
        """
        # 收集所有位置变化
        all_changes = []
        for rule in oneInOut_mapping_rules:
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