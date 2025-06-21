"""
加权ARC差异分析器核心类

实现加权版的ARC差异分析器，扩展基础分析器功能。
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Set, FrozenSet, Optional, Union
from collections import defaultdict

from arcMrule.diffstar.arc_diff_analyzer import ARCDiffAnalyzer
from objutil import pureobjects_from_grid
from weightgird import grid2grid_fromgriddiff

from .weighted_obj_info import WeightedObjInfo
from .weight_calculator import WeightCalculator
from .object_matching import ObjectMatcher
from .pattern_analyzer import PatternAnalyzer
from .rule_applier import RuleApplier
from .utils import get_hashable_representation,get_obj_shape_hash
from .optimized_rule_applier import OptimizedRuleApplier



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

        # 新增: 形状-颜色关系规则
        self.shape_color_rules = []

        # 新增: 属性依赖规则
        self.attribute_dependency_rules = []

        # 重写对象存储结构，使用WeightedObjInfo替代ObjInfo
        self.all_objects = {
            'input': [],  # [(pair_id, [WeightedObjInfo]), ...]
            'output': [], # [(pair_id, [WeightedObjInfo]), ...]
            'diff_in': [], # [(pair_id, [WeightedObjInfo]), ...]
            'diff_out': [],  # [(pair_id, [WeightedObjInfo]), ...]
            'test_input': [],  # [(pair_id, [WeightedObjInfo]), ...]
        }

        # 初始化辅助组件
        self.weight_calculator = WeightCalculator(
            self.pixel_threshold_pct,
            self.weight_increment,
            self.diff_weight_increment,
            self._debug_print if debug else None
        )

        self.object_matcher = ObjectMatcher(self._debug_print if debug else None)
        self.pattern_analyzer = PatternAnalyzer(self._debug_print if debug else None)
        self.rule_applier = RuleApplier(self._debug_print if debug else None)
        self.optimized_rule_applier = OptimizedRuleApplier(debug=debug, debug_print=self._debug_print)

    def set_background_colors(self, background_colors):
        """
        设置全局背景色

        Args:
            background_colors: 背景色集合
        """
        self.background_colors = background_colors
        # 如果使用了权重计算器，将背景色传递给它
        if hasattr(self, 'weight_calculator'):
            self.weight_calculator.set_background_colors(background_colors)

    def add_train_pair(self, pair_id, input_grid, output_grid, param, background_color,test = False):
        """
        添加一对训练数据，提取对象并计算权重

        Args:
            pair_id: 训练对ID
            input_grid: 输入网格
            output_grid: 输出网格
            param: 对象提取参数
        """
        self.background_color = background_color

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
            param, pair_id, 'in', input_grid, [height_in, width_in], background_color=background_color
        )
        output_objects = pureobjects_from_grid(
            param, pair_id, 'out', output_grid, [height_out, width_out], background_color=background_color
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
                param, pair_id, 'diff_in', diff_in, [height_diff, width_diff], background_color=background_color
            )
            diff_out_objects = pureobjects_from_grid(
                param, pair_id, 'diff_out', diff_out, [height_diff, width_diff], background_color=background_color
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

        # self.all_objects['test_input'].append((pair_id, input_obj_infos))

        # 更新形状库
        self._update_shape_library(input_obj_infos + output_obj_infos)

        # 分析对象间的部分-整体关系
        self._analyze_part_whole_relationships(input_obj_infos)
        self._analyze_part_whole_relationships(output_obj_infos)

        # 应用权重计算 - 为每个对象设置权重
        self.weight_calculator.calculate_object_weights(
            pair_id, input_grid, output_grid,
            input_obj_infos, output_obj_infos,
            diff_in_obj_infos, diff_out_obj_infos,
            diff_in, diff_out
        )



        # 分析diff映射关系
        mapping_rule = self.object_matcher.analyze_diff_mapping_with_weights(
            pair_id, input_grid, output_grid, diff_in, diff_out,
            input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos,self.debug
        )

        self.oneInOut_mapping_rules.append(mapping_rule)

        # # 新增: 提取基于形状的颜色变化规则
        # shape_color_rule = self._extract_shape_color_rules(
        #     pair_id, input_obj_infos, output_obj_infos,
        #     diff_in_obj_infos, diff_out_obj_infos
        # )
        # if shape_color_rule:
        #     self.oneInOut_mapping_rules.append(shape_color_rule)

        # # 新增: 提取更通用的属性依赖规则
        # attr_rules = self._extract_attribute_dependency_rules(
        #     pair_id, input_obj_infos, output_obj_infos
        # )
        # self.oneInOut_mapping_rules.extend(attr_rules)


        if self.debug:
            self._debug_save_json(mapping_rule, f"mapping_rule_{pair_id}")
            self._debug_print(f"\n\nadd_train_pair:完成训练对 {pair_id} 的分析和权重计算")
            self._debug_print_object_weights(input_obj_infos, f"input_obj_weights_{pair_id}")
            self._debug_print_object_weights(output_obj_infos, f"output_obj_weights_{pair_id}")
            self._debug_print_object_weights(diff_in_obj_infos, f"diff_in_obj_weights_{pair_id}")
            self._debug_print_object_weights(diff_out_obj_infos, f"diff_out_obj_weights_{pair_id}")

            # if shape_color_rule:
            #     self._debug_save_json(shape_color_rule, f"shape_color_rule_{pair_id}")
            #     self._debug_print(f"提取了 {len(shape_color_rule.get('rules', []))} 个形状-颜色规则")

            # if attr_rules:
            #     self._debug_save_json(attr_rules, f"attr_dependency_rules_{pair_id}")
            #     self._debug_print(f"提取了 {len(attr_rules)} 个属性依赖规则")



    def add_test_pair(self, pair_id, input_grid, output_grid , param, background_color,test = True):

        if self.debug:
            self._debug_print(f"处理训练对test  {pair_id}")
            self._debug_save_grid(input_grid, f"test_input_{pair_id}")


        # 确保网格是元组的元组格式
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)

        # 保存原始网格对
        self.test_pairs.append((input_grid, output_grid))

        # 获取网格尺寸
        height_in, width_in = len(input_grid), len(input_grid[0])

        # 提取对象
        test_input_objects = pureobjects_from_grid(
            param, pair_id, 'in', input_grid, [height_in, width_in], background_color=background_color
        )

        # 转换为加权对象信息
        test_input_obj_infos = [
            WeightedObjInfo(pair_id, 'in', obj, obj_params=None, grid_hw=[height_in, width_in])
            for obj in test_input_objects
        ]

        if self.debug:
            self._debug_print(f"从输入网格提取了 {len(test_input_obj_infos)} 个对象")

        # 存储提取的对象
        self.all_objects['test_input'].append((pair_id, test_input_obj_infos))


        # 更新形状库
        self._update_shape_library(test_input_obj_infos )

        # 分析对象间的部分-整体关系
        self._analyze_part_whole_relationships(test_input_obj_infos)


        # # 应用权重计算 - 为每个对象设置权重
        # self.weight_calculator.calculate_object_weights(
        #     pair_id, input_grid,             test_input_obj_infos        )


        # if self.debug:
        #     # self._debug_save_json(mapping_rule, f"mapping_rule_{pair_id}")
        #     self._debug_print(f"\n\nadd_test_pair:完成训练对 {pair_id} 的分析和权重计算")
        #     self._debug_print_object_weights(test_input_obj_infos, f"test_input_obj_weights_{pair_id}")





    def _extract_shape_color_rules(self, pair_id, input_obj_infos, output_obj_infos,
                                diff_in_obj_infos, diff_out_obj_infos):
        """
        提取基于形状的颜色变化规则

        Args:
            pair_id: 训练对ID
            input_obj_infos: 输入对象信息列表
            output_obj_infos: 输出对象信息列表
            diff_in_obj_infos: 差异输入对象信息列表
            diff_out_obj_infos: 差异输出对象信息列表

        Returns:
            形状-颜色规则字典
        """
        rules = []

        # 首先匹配输入和输出对象
        matched_objects = self._match_input_output_objects(input_obj_infos, output_obj_infos)

        # 分析每个匹配对，寻找形状与颜色变化的关系
        for in_obj, out_obj in matched_objects:
            # 只关注颜色发生变化的对象
            if in_obj.main_color != out_obj.main_color:
                # 检查是否有形状特征可以解释颜色变化
                shape_features = self._extract_shape_features(in_obj)

                # 形成规则: 基于形状特征的颜色变化
                rule = {
                    "rule_type": "shape_to_color",
                    "pair_id": pair_id,
                    "object_id": in_obj.obj_id,
                    "shape_features": shape_features,
                    "original_color": in_obj.main_color,
                    "new_color": out_obj.main_color,
                    "weight": in_obj.obj_weight,  # 使用对象权重表示规则重要性
                    "confidence": 0.7  # 初始置信度
                }

                # 增强规则: 检查其他对象是否有相同形状特征且进行了相同颜色变化
                similar_changes = 0
                for other_in, other_out in matched_objects:
                    if (other_in != in_obj and
                        other_in.main_color == in_obj.main_color and
                        other_out.main_color == out_obj.main_color and
                        self._shape_similarity(other_in, in_obj) > 0.7):
                        similar_changes += 1
                        rule["confidence"] = min(1.0, rule["confidence"] + 0.1)

                if similar_changes > 0:
                    rule["similar_changes"] = similar_changes

                rules.append(rule)

        # 分析差异区域对象的颜色变化
        if diff_in_obj_infos and diff_out_obj_infos:
            diff_matched_objects = self._match_input_output_objects(diff_in_obj_infos, diff_out_obj_infos)

            for in_obj, out_obj in diff_matched_objects:
                if in_obj.main_color != out_obj.main_color:
                    shape_features = self._extract_shape_features(in_obj)

                    rule = {
                        "rule_type": "diff_shape_to_color",
                        "pair_id": pair_id,
                        "object_id": in_obj.obj_id,
                        "shape_features": shape_features,
                        "original_color": in_obj.main_color,
                        "new_color": out_obj.main_color,
                        "weight": in_obj.obj_weight * 1.5,  # 差异区域权重更高
                        "confidence": 0.8  # 差异区域置信度更高
                    }
                    rules.append(rule)

        # 寻找跨对象的形状-颜色关联
        # 例如: 对象A的形状决定对象B的颜色
        self._extract_cross_object_shape_color_rules(
            pair_id, input_obj_infos, output_obj_infos, rules
        )

        if rules:
            return {
                "pair_id": pair_id,
                "rules": rules
            }
        return None

    def _extract_attribute_dependency_rules(self, pair_id, input_obj_infos, output_obj_infos):
        """
        提取更通用的属性依赖规则

        Args:
            pair_id: 训练对ID
            input_obj_infos: 输入对象信息列表
            output_obj_infos: 输出对象信息列表

        Returns:
            属性依赖规则列表
        """
        rules = []

        # 匹配对象
        matched_objects = self._match_input_output_objects(input_obj_infos, output_obj_infos)

        # 属性变化分析
        for in_obj, out_obj in matched_objects:
            # 分析各种属性变化
            changes = {
                "color": in_obj.main_color != out_obj.main_color,
                "position": (in_obj.top != out_obj.top or in_obj.left != out_obj.left),
                "size": in_obj.size != out_obj.size,
                "shape": self._shape_similarity(in_obj, out_obj) < 0.9
            }

            # 如果有属性发生变化
            if any(changes.values()):
                # 尝试找出变化的依据
                for attr_name, changed in changes.items():
                    if changed:
                        # 尝试根据自身其他属性解释变化
                        self_rule = self._find_attribute_dependency(
                            in_obj, out_obj, attr_name, input_obj_infos, output_obj_infos
                        )

                        if self_rule:
                            self_rule["pair_id"] = pair_id
                            rules.append(self_rule)

                        # 尝试根据其他对象属性解释变化
                        for other_in in input_obj_infos:
                            if other_in != in_obj:
                                cross_rule = self._find_cross_object_dependency(
                                    in_obj, out_obj, other_in, attr_name
                                )

                                if cross_rule:
                                    cross_rule["pair_id"] = pair_id
                                    rules.append(cross_rule)

        return rules

    def _extract_cross_object_shape_color_rules(self, pair_id, input_obj_infos, output_obj_infos, rules_list):
        """提取跨对象的形状-颜色规则"""
        # 对于每个输出对象
        for out_obj in output_obj_infos:
            # 检查其颜色是否可能受到其他输入对象形状的影响
            for in_obj in input_obj_infos:
                # 跳过可能是同一对象变化前后的情况
                if self._shape_similarity(in_obj, out_obj) > 0.8:
                    continue

                shape_features = self._extract_shape_features(in_obj)

                # 尝试寻找规律: 输入对象in_obj的形状影响输出对象out_obj的颜色
                rule = {
                    "rule_type": "cross_shape_to_color",
                    "pair_id": pair_id,
                    "in_object_id": in_obj.obj_id,
                    "out_object_id": out_obj.obj_id,
                    "shape_features": shape_features,
                    "determining_color": in_obj.main_color,
                    "resulting_color": out_obj.main_color,
                    "weight": (in_obj.obj_weight + out_obj.obj_weight) / 2,
                    "confidence": 0.6
                }

                # 如果输入对象权重高，提高规则置信度
                if in_obj.obj_weight > 3:
                    rule["confidence"] = min(1.0, rule["confidence"] + 0.2)

                rules_list.append(rule)

    def _match_input_output_objects(self, input_objects, output_objects):
        """匹配输入和输出对象，返回(输入对象,输出对象)对列表"""
        matches = []

        # 简单启发式匹配: 优先考虑位置和形状相似性
        for in_obj in input_objects:
            best_match = None
            best_score = -1

            for out_obj in output_objects:
                # 计算相似度分数
                shape_sim = self._shape_similarity(in_obj, out_obj)
                # pos_sim = self._position_similarity(in_obj, out_obj)

                # 综合分数
                # score = 0.6 * shape_sim + 0.4 * pos_sim
                score = shape_sim

                if score > best_score:
                    best_score = score
                    best_match = out_obj

            # 只有当相似度足够高时才认为匹配有效
            if best_score > 0.8 and best_match:
                matches.append((in_obj, best_match))

        return matches

    def _extract_shape_features(self, obj_info):
        """提取对象的形状特征"""
        #! if same color
        return {
            "height": obj_info.height,
            "width": obj_info.width,
            "size": obj_info.size,
            "aspect_ratio": obj_info.width / max(1, obj_info.height),
            "is_symmetric_h": self._check_horizontal_symmetry(obj_info),
            "is_symmetric_v": self._check_vertical_symmetry(obj_info),
            "compactness": obj_info.size / (obj_info.height * obj_info.width),
            "num_corners": self._estimate_corners(obj_info)
        }

    def _check_horizontal_symmetry(self, obj_info):
        """检查水平对称性"""
        # 简化实现，根据实际情况可以完善
        return True  # 占位实现

    def _check_vertical_symmetry(self, obj_info):
        """检查垂直对称性"""
        # 简化实现，根据实际情况可以完善
        return True  # 占位实现

    def _estimate_corners(self, obj_info):
        """估计对象的角点数量"""
        # 简化实现，根据实际情况可以完善
        return 4  # 占位实现

    def _shape_similarity(self, obj1, obj2):
        """计算两个对象的形状相似度"""
        # 简单实现，可以根据需要增强
        # size_sim = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
        # aspect_sim = min(obj1.width/max(1,obj1.height), obj2.width/max(1,obj2.height)) / \
        #             max(obj1.width/max(1,obj1.height), obj2.width/max(1,obj2.height))

        # return 0.7 * size_sim + 0.3 * aspect_sim
        #! now if same; todo if part subparts
        if obj1.obj_000 == obj2.obj_000:
            return 1.0
        elif obj1.obj_000 != obj2.obj_000:
            return 0.0

    def _position_similarity(self, obj1, obj2):
        """计算两个对象的位置相似度，简化为只比较左上角位置"""
        try:
            # 尝试从对象或对象的obj属性获取位置信息
            x1 = getattr(obj1, 'left', None) or getattr(obj1.obj, 'left', 0)
            y1 = getattr(obj1, 'top', None) or getattr(obj1.obj, 'top', 0)

            x2 = getattr(obj2, 'left', None) or getattr(obj2.obj, 'left', 0)
            y2 = getattr(obj2, 'top', None) or getattr(obj2.obj, 'top', 0)

            # 计算左上角位置的距离
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            # 获取网格尺寸用于归一化距离
            grid_hw1 = getattr(obj1, 'grid_hw', [10, 10])
            grid_hw2 = getattr(obj2, 'grid_hw', [10, 10])

            # 使用最大的网格尺寸进行归一化
            max_grid_size = max(max(grid_hw1), max(grid_hw2))

            # 将距离归一化为相似度
            return max(0, 1 - dist / max_grid_size)
        except (AttributeError, TypeError) as e:
            # 如果无法获取位置信息，则返回中性相似度值
            if self.debug:
                self._debug_print(f"位置相似度计算错误: {e}")
            return 0.5  # 返回中性值，既不完全相似也不完全不相似

    def _find_attribute_dependency(self, in_obj, out_obj, changed_attr, all_in_objs, all_out_objs):
        """寻找对象属性变化的依据"""
        # 这里实现寻找属性依赖的逻辑
        # 例如：如果颜色变化，尝试找出依赖于形状的规则

        if changed_attr == "color":
            # 检查是否基于形状的颜色变化规则
            shape_features = self._extract_shape_features(in_obj)

            return {
                "rule_type": "self_attr_dependency",
                "changed_attr": "color",
                "from_value": in_obj.main_color,
                "to_value": out_obj.main_color,
                "dependent_on": "shape",
                "shape_features": shape_features,
                "object_id": in_obj.obj_id,
                "confidence": 0.7
            }

        # 可以添加其他属性变化的依赖分析
        return None

    def _find_cross_object_dependency(self, in_obj, out_obj, other_in_obj, changed_attr):
        """寻找跨对象的属性依赖"""
        if changed_attr == "color":
            # 检查颜色变化是否依赖于其他对象的形状
            cross_rule = {
                "rule_type": "cross_obj_attr_dependency",
                "target_object_id": in_obj.obj_id,
                "reference_object_id": other_in_obj.obj_id,
                "changed_attr": "color",
                "from_value": in_obj.main_color,
                "to_value": out_obj.main_color,
                "dependent_on": "shape",
                "reference_shape": self._extract_shape_features(other_in_obj),
                "confidence": 0.65
            }
            return cross_rule

        return None

    """
    将多维度关系库系统集成到ARC分析流程中的示例
    """

    # from .improved_analyze_common_patterns_with_weights import analyze_common_patterns_with_weights as  analyze_common_patterns_with_weights2

    # # def analyze_common_patterns_with_weights(self):
    # #     return analyze_common_patterns_with_weights(self)
    # analyze_common_patterns_with_weights3 = analyze_common_patterns_with_weights2


    """
    增强的ARC分析流程：集成测试形状匹配和组合规则提取
    ##  analyze_with_test_data_matching
    """

    def enhanced_analyze_common_patterns_with_test_data_matching(self, task=None):
        """
        增强的模式分析函数：将测试数据与模式库进行匹配

        Args:
            task: 可选的测试输入数据

        Returns:
            带权重的通用模式和针对测试数据的优化规则
        """
        print("\n\n\n\n\n\nCOMMON ommon_patterns_with_weights开始分析共同模式\n\n")
        if not self.oneInOut_mapping_rules:
            return {}

        # 1. 初始化并构建关系库
        from arc_relationship_libraries import ARCRelationshipLibraries
        relationship_libs = ARCRelationshipLibraries(debug=self.debug, debug_print=self._debug_print)
        relationship_libs.build_libraries_from_data(self.oneInOut_mapping_rules, self.all_objects)

        # 2. 查找跨数据对模式
        cross_pair_patterns = relationship_libs.find_patterns_across_pairs()

        if self.debug:
            print(f"\n\n\n\n\n ! ! pairs : 找到  cross_pair_patterns  {len(cross_pair_patterns)} 个跨数据对模式:\n\n\n\n",cross_pair_patterns)

        # test_input = task
        for i, example in enumerate(task['test']):
                if self.debug:
                    print(f"\n----- 测试样例 {i+1} -----")
                test_input = example['input']

        # 3. 提取测试输入的形状和颜色特征
        test_features = None
        if test_input:
            test_features = self._extract_test_features(test_input)
            if self.debug:
                self._debug_print("从测试输入提取特征:")
                if test_features.get('shapes'):
                    self._debug_print(f"  - 发现 {len(test_features['shapes'])} 个形状")
                if test_features.get('colors'):
                    self._debug_print(f"  - 发现颜色: {test_features['colors']}")

        # 4. 使用增强版模式学习器提取规则并匹配测试数据
        from enhanced_pattern_meta_analyzer import EnhancedPatternMetaAnalyzer
        meta_analyzer = EnhancedPatternMetaAnalyzer(debug=self.debug, debug_print=self._debug_print, task=task)
        advanced_rules = meta_analyzer.process_patterns(cross_pair_patterns, test_features)

        # 确保底层模式分析结果被包含在结果中
        if 'global_rules' in advanced_rules:
            for rule in advanced_rules['global_rules']:
                if rule.get('operation') == 'added' and 'underlying_pattern' in rule:
                    if self.debug:
                        self._debug_print(f"发现带底层模式的添加规则: {rule['description']}")

        if self.debug:
            print(f"\n\n\n\n\n ! ! pairs : enhanced_pattern_meta_analyzer : advanced_rules : 找到 {len(advanced_rules)} 个增强规则:\n\n\n", advanced_rules)

        # 5. 调用原始模式分析器获取基本模式
        basic_patterns = self.pattern_analyzer.analyze_common_patterns(self.oneInOut_mapping_rules)

        # 6. 组合所有结果
        combined_results = {
            "basic_patterns": basic_patterns,
            "cross_pair_patterns": cross_pair_patterns,
            "advanced_rules": advanced_rules
        }
        #! plan for test
        if test_features:
            combined_results["test_features"] = test_features

            # 7. 生成针对测试数据的优化应用计划
            optimized_plan = self._generate_optimized_plan(
                basic_patterns,
                cross_pair_patterns, advanced_rules, test_features)

            combined_results["optimized_plan"] = optimized_plan

        # 8. 保存结果（可选）
        if self.debug:
            self._debug_save_json(advanced_rules, "advanced_rules_with_test_matching")
            if test_features:
                self._debug_save_json(test_features, "test_features")
                self._debug_save_json(combined_results.get("optimized_plan", {}), "optimized_application_plan")

        return combined_results

    def _extract_test_features(self, test_input):
        """
        从测试输入中提取形状和颜色特征

        Args:
            test_input: 测试输入数据

        Returns:
            包含形状和颜色信息的特征字典
        """
        features = {
            'shapes': [],
            'colors': set(),
            'objects': []
        }

        # 提取形状和颜色的逻辑，需要根据实际数据结构调整
        # 例如，对于网格数据:

        # 1. 提取颜色 (非零值)
        if isinstance(test_input, list) or isinstance(test_input, tuple):
            for row in test_input:
                for cell in row:
                    if cell != 0:  # 假设0是背景色
                        features['colors'].add(cell)

        # 2. 提取对象，这需要使用现有的对象提取函数
        # objects = self._extract_objects_from_grid(test_input)
        objects = self.all_objects['test_input'][-1][1] if self.all_objects['test_input'] else []
        features['objects'] = objects

        # 3. 从对象计算形状哈希
        for obj in objects:
            shape_hash = get_obj_shape_hash(obj)
            if shape_hash:
                features['shapes'].append(shape_hash)

        # # 确保颜色是列表而非集合，以便序列化
        features['colors'] = list(features['colors'])

        return features

    def _generate_optimized_plan(self, basic_patterns, cross_patterns, advanced_rules, test_features):
        """
        生成针对测试数据的优化应用计划

        Args:
            basic_patterns: 基本模式
            cross_patterns: 跨对模式
            advanced_rules: 高级规则
            test_features: 测试特征

        Returns:
            优化的应用计划
        """
        # 获取推荐执行计划
        execution_plan = advanced_rules.get('recommended_execution_plan', [])

        if not execution_plan:
            return {"message": "无法生成优化计划，未找到匹配的规则"}

        # 创建应用计划
        application_plan = {
            "steps": [],
            "matched_test_features": {
                "shapes": [],
                "colors": []
            }
        }

        # 添加匹配的形状和颜色
        for step in execution_plan:
            match_info = step.get('match_info', {})
            elements = match_info.get('matched_elements', {})

            if isinstance(elements, dict) and 'shape' in elements:
                application_plan['matched_test_features']['shapes'].append(elements['shape'])
            if isinstance(elements, dict) and 'color' in elements:
                application_plan['matched_test_features']['colors'].append(elements['color'])
            if isinstance(elements, list):
                for elem in elements:
                    if isinstance(elem, (int, str)):  # 如果是简单类型，假设是颜色
                        application_plan['matched_test_features']['colors'].append(elem)

        # 去重
        application_plan['matched_test_features']['shapes'] = list(set(application_plan['matched_test_features']['shapes']))
        application_plan['matched_test_features']['colors'] = list(set(application_plan['matched_test_features']['colors']))

        # 构建应用步骤
        for idx, step in enumerate(execution_plan):
            step_info = {
                "step_id": idx + 1,
                "step_type": step['step_type'],
                "priority": step['priority']
            }

            # 根据规则类型添加具体操作
            rule = step.get('rule', {})
            if rule.get('rule_type') == 'composite_rule':
                step_info["action"] = {
                    "base_operation": {
                        "type": rule['base_rule'].get('type'),
                        "color": rule['base_rule'].get('color'),
                        "operation": rule['base_rule'].get('operation')
                    },
                    "conditional_changes": []
                }

                # 添加条件变化
                for cond in rule.get('conditional_rules', []):
                    step_info["action"]["conditional_changes"].append({
                        "when_removed_shape_change_color": cond['condition'].get('shape_hash'),
                        "color_change": {
                            "from": cond['effect']['from_color'],
                            "to": cond['effect']['to_color']
                        }
                    })

            elif rule.get('rule_type') == 'conditional_rule':
                step_info["action"] = {
                    "when_removed_shape_change_color": rule['condition'].get('shape_hash'),
                    "color_change": {
                        "from": rule['effect']['from_color'],
                        "to": rule['effect']['to_color']
                    }
                }

            elif rule.get('rule_type') == 'global_color_operation':
                step_info["action"] = {
                    "apply_to_color": rule.get('color'),
                    "operation": rule.get('operation')
                }

            application_plan["steps"].append(step_info)

        return application_plan

    # def _extract_objects_from_grid(self, grid):
    #     """从网格中提取对象"""
    #     # 实现对象提取逻辑，可以使用现有的对象提取函数
    #     # 这是一个简化版本
    #     objects = []
    #     # ... 对象提取代码 ...
    #     return objects

    # def _calculate_shape_hash(self, obj):
    #     """计算对象的形状哈希"""
    #     # 实现形状哈希计算逻辑
    #     # 这是一个简化版本
    #     shape_hash = None
    #     # ... 形状哈希计算代码 ...
    #     return shape_hash






    """
    集成高级模式学习器到ARC分析流程
    """

    def enhanced_analyze_common_patterns_with_weights(self):
        """
        增强版分析函数：整合多维度关系库和高级模式学习器

        Returns:
            带权重的通用模式与高级规则
        """
        if not self.oneInOut_mapping_rules:
            return {}

        # 1. 初始化多维度关系库系统
        from arc_relationship_libraries import ARCRelationshipLibraries
        relationship_libs = ARCRelationshipLibraries(debug=self.debug, debug_print=self._debug_print)

        # 2. 构建关系库
        relationship_libs.build_libraries_from_data(self.oneInOut_mapping_rules, self.all_objects)

        # 3. 查找跨数据对的模式
        cross_pair_patterns = relationship_libs.find_patterns_across_pairs()

        # 4. 使用高级模式学习器提取通用规则
        from pattern_meta_analyzer import PatternMetaAnalyzer
        meta_analyzer = PatternMetaAnalyzer(debug=self.debug, debug_print=self._debug_print)
        high_level_rules = meta_analyzer.process_patterns(cross_pair_patterns)

        # 5. 调用原始的模式分析器获取基本模式
        # basic_patterns = self.pattern_analyzer.analyze_common_patterns(self.oneInOut_mapping_rules)

        # 6. 结合基本模式、跨数据对模式和高级规则
        combined_patterns = {
            # "basic": basic_patterns,
            "cross_instance": cross_pair_patterns,
            "high_level_rules": high_level_rules
        }

        # 7. 对所有模式和规则进行权重计算和排序
        weighted_patterns = self._compute_enhanced_pattern_weights(combined_patterns)

        # 8. 保存关系库状态和高级规则到文件（可选）
        if self.debug:
            relationship_libs.export_libraries_to_json(f"{self.debug_dir}/relationship_libraries.json")
            self._debug_save_json(high_level_rules, "high_level_rules")
            self._debug_save_json(weighted_patterns, "weighted_patterns_enhanced")

        # 9. 返回带权重的模式和规则
        return weighted_patterns

    def _compute_enhanced_pattern_weights(self, combined_patterns):
        """
        计算增强版的模式权重，包含高级规则

        Args:
            combined_patterns: 组合的模式字典

        Returns:
            带权重的排序模式和规则
        """
        # 提取所有模式和规则到一个列表
        all_patterns_and_rules = []

        # 处理基本模式
        basic = combined_patterns.get("basic", {})
        # [基本模式处理逻辑...]

        # 处理跨实例模式
        for pattern in combined_patterns.get("cross_instance", []):
            pattern_type = pattern.get("type", "unknown")
            subtype = pattern.get("subtype", "")

            all_patterns_and_rules.append({
                "type": f"{pattern_type}_{subtype}" if subtype else pattern_type,
                "source": "cross_instance",
                "data": pattern,
                "confidence": pattern.get("confidence", 0.5),
                "raw_weight": pattern.get("weight", 1.0)
            })

        # 处理高级规则，给予更高的权重
        for rule in combined_patterns.get("high_level_rules", []):
            rule_type = rule.get("rule_type", "unknown")

            all_patterns_and_rules.append({
                "type": rule_type,
                "source": "high_level_rule",
                "data": rule,
                "confidence": rule.get("confidence", 0.5),
                "raw_weight": rule.get("score", 1.0) * 1.2  # 高级规则权重增加20%
            })

        # 计算最终权重分数 (0.7 * confidence + 0.3 * raw_weight)
        for item in all_patterns_and_rules:
            item["weight"] = 0.7 * item["confidence"] + 0.3 * item["raw_weight"]

        # 按权重排序
        all_patterns_and_rules.sort(key=lambda x: x["weight"], reverse=True)

        # 构建最终结果
        result = {
            "patterns_and_rules": all_patterns_and_rules,
            "top_items": all_patterns_and_rules[:min(10, len(all_patterns_and_rules))],
            "total_items": len(all_patterns_and_rules),
            "original": combined_patterns
        }

        return result





    def analyze_common_patterns_with_weights(self):
        """
        改进的版本：分析多对训练数据的共同模式，考虑权重因素，
        通过多维度关系库发现复杂的跨实例模式

        Returns:
            带权重的共同模式字典
        """
        print("\n\n\n\nCOMMON ommon_patterns_with_weights开始分析共同模式\n\n")
        if not self.oneInOut_mapping_rules:
            return {}

        # 1. 初始化多维度关系库系统
        from arc_relationship_libraries import ARCRelationshipLibraries
        relationship_libs = ARCRelationshipLibraries(debug=self.debug, debug_print=self._debug_print)

        # 2. 构建关系库
        relationship_libs.build_libraries_from_data(self.oneInOut_mapping_rules, self.all_objects)

        # 3. 查找跨数据对的模式
        cross_pair_patterns = relationship_libs.find_patterns_across_pairs()

        # 4. 调用原始的模式分析器获取基本模式
        # basic_patterns = self.pattern_analyzer.analyze_common_patterns(self.oneInOut_mapping_rules)

        # 5. 结合基本模式和跨数据对模式
        combined_patterns = {
            # "basic": basic_patterns,
            "cross_instance": cross_pair_patterns
        }

        # 6. 对所有模式进行权重计算和排序
        weighted_patterns = self._compute_pattern_weights(combined_patterns)
        patterup2 = uppatter2(combined_patterns)

        # 7. 保存关系库状态到文件（可选）
        if self.debug:
            relationship_libs.export_libraries_to_json(f"{self.debug_dir}/relationship_libraries.json")
            self._debug_save_json(weighted_patterns, "weighted_patterns_from_libs")
            print(f"weighted_patterns")
            print(weighted_patterns)

        # 8. 返回带权重的模式
        return weighted_patterns

    def _compute_pattern_weights(self, combined_patterns):
        """
        计算所有模式的权重并排序

        Args:
            combined_patterns: 组合的模式字典

        Returns:
            带权重的排序模式
        """
        # 提取所有模式到一个列表
        all_patterns = []

        # 处理基本模式
        basic = combined_patterns.get("basic", {})

        # 形状变换模式
        for pattern in basic.get("shape_transformations", []):
            all_patterns.append({
                "type": "shape_transformation",
                "source": "basic",
                "data": pattern,
                "confidence": pattern.get("confidence", 0.5),
                "raw_weight": 1.0
            })

        # 颜色映射模式
        for from_color, mapping in basic.get("color_mappings", {}).get("mappings", {}).items():
            all_patterns.append({
                "type": "color_mapping",
                "source": "basic",
                "data": {"from": from_color, "to": mapping.get("to_color")},
                "confidence": mapping.get("confidence", 0.5),
                "raw_weight": 1.0
            })

        # 位置变化模式
        for pattern in basic.get("position_changes", []):
            all_patterns.append({
                "type": "position_change",
                "source": "basic",
                "data": pattern,
                "confidence": pattern.get("confidence", 0.5),
                "raw_weight": 1.0
            })

        # 处理跨实例模式
        for pattern in combined_patterns.get("cross_instance", []):
            pattern_type = pattern.get("type", "unknown")
            subtype = pattern.get("subtype", "")

            all_patterns.append({
                "type": f"{pattern_type}_{subtype}" if subtype else pattern_type,
                "source": "cross_instance",
                "data": pattern,
                "confidence": pattern.get("confidence", 0.5),
                "raw_weight": pattern.get("weight", 1.0)
            })

        # 计算最终权重分数 (0.7 * confidence + 0.3 * raw_weight)
        for pattern in all_patterns:
            pattern["weight"] = 0.7 * pattern["confidence"] + 0.3 * pattern["raw_weight"]

        # 按权重排序
        all_patterns.sort(key=lambda x: x["weight"], reverse=True)

        # 构建最终结果
        result = {
            "patterns": all_patterns,
            "top_patterns": all_patterns[:min(10, len(all_patterns))],
            "total_patterns": len(all_patterns),
            "original": combined_patterns
        }

        return result



    def _induce_shape_color_patterns(self):
        """归纳形状-颜色规则模式"""
        if not self.shape_color_rules:
            return []

        patterns = []
        rule_groups = self._group_similar_shape_color_rules()

        for group in rule_groups:
            if len(group) >= 2:  # 至少需要2个相似规则才能形成模式
                pattern = self._create_shape_color_pattern(group)
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _group_similar_shape_color_rules(self):
        """将相似的形状-颜色规则分组"""
        if not self.shape_color_rules:
            return []

        # 将所有规则平铺
        all_rules = []
        for rule_set in self.shape_color_rules:
            all_rules.extend(rule_set.get('rules', []))

        # 按规则类型分组
        rule_type_groups = defaultdict(list)
        for rule in all_rules:
            rule_type_groups[rule.get('rule_type', '')].append(rule)

        # 对每种规则类型内的规则进行相似性分组
        all_groups = []

        for rule_type, rules in rule_type_groups.items():
            if rule_type == 'shape_to_color':
                # 按颜色变化分组
                color_change_groups = defaultdict(list)
                for rule in rules:
                    key = (rule.get('original_color'), rule.get('new_color'))
                    color_change_groups[key].append(rule)

                # 将各组添加到结果中
                for group in color_change_groups.values():
                    if group:
                        all_groups.append(group)

            elif rule_type == 'cross_shape_to_color':
                # 跨对象规则按结果颜色分组
                result_color_groups = defaultdict(list)
                for rule in rules:
                    key = rule.get('resulting_color')
                    result_color_groups[key].append(rule)

                for group in result_color_groups.values():
                    if group:
                        all_groups.append(group)

        return all_groups

    def _create_shape_color_pattern(self, rules_group):
        """从规则组创建形状-颜色模式"""
        if not rules_group:
            return None

        rule_type = rules_group[0].get('rule_type')

        if rule_type == 'shape_to_color':
            # 提取共有颜色变化
            color_change = (rules_group[0].get('original_color'), rules_group[0].get('new_color'))

            # 提取共有形状特征
            common_shape_features = {}
            for feature_name in rules_group[0].get('shape_features', {}):
                # 检查该特征是否在所有规则中都相似
                values = [rule['shape_features'].get(feature_name) for rule in rules_group
                        if feature_name in rule.get('shape_features', {})]

                if values and all(abs(v - values[0]) < 0.2 for v in values):
                    common_shape_features[feature_name] = sum(values) / len(values)

            # 创建模式
            pattern = {
                "pattern_type": "shape_to_color",
                "color_change": {"from": color_change[0], "to": color_change[1]},
                "shape_conditions": common_shape_features,
                "supporting_rules": len(rules_group),
                "confidence": sum(rule.get('confidence', 0) for rule in rules_group) / len(rules_group),
                "weight": sum(rule.get('weight', 1) for rule in rules_group) / len(rules_group)
            }

            return pattern

        elif rule_type == 'cross_shape_to_color':
            # 处理跨对象规则
            resulting_color = rules_group[0].get('resulting_color')

            pattern = {
                "pattern_type": "cross_shape_to_color",
                "resulting_color": resulting_color,
                "confidence": sum(rule.get('confidence', 0) for rule in rules_group) / len(rules_group),
                "supporting_rules": len(rules_group),
                "weight": sum(rule.get('weight', 1) for rule in rules_group) / len(rules_group)
            }

            return pattern

        return None

    def _induce_attribute_dependency_patterns(self):
        """归纳属性依赖规则模式"""
        if not self.attribute_dependency_rules:
            return []

        # 按规则类型和变化属性分组
        rule_groups = defaultdict(list)
        for rule in self.attribute_dependency_rules:
            key = (rule.get('rule_type', ''), rule.get('changed_attr', ''))
            rule_groups[key].append(rule)

        patterns = []

        # 处理每种规则类型
        for (rule_type, changed_attr), rules in rule_groups.items():
            if len(rules) < 2:
                continue

            if rule_type == 'self_attr_dependency' and changed_attr == 'color':
                # 处理基于自身属性的颜色变化规则
                color_change_groups = defaultdict(list)
                for rule in rules:
                    key = (rule.get('from_value'), rule.get('to_value'))
                    color_change_groups[key].append(rule)

                # 为每组颜色变化创建一个模式
                for color_change, group_rules in color_change_groups.items():
                    if len(group_rules) >= 2:
                        pattern = {
                            "pattern_type": "self_attr_color_change",
                            "color_change": {"from": color_change[0], "to": color_change[1]},
                            "dependent_on": group_rules[0].get('dependent_on'),
                            "supporting_rules": len(group_rules),
                            "confidence": sum(rule.get('confidence', 0) for rule in group_rules) / len(group_rules)
                        }
                        patterns.append(pattern)

            elif rule_type == 'cross_obj_attr_dependency' and changed_attr == 'color':
                # 处理跨对象的颜色变化规则
                cross_pattern = {
                    "pattern_type": "cross_obj_color_dependency",
                    "changed_attr": changed_attr,
                    "dependent_on": rules[0].get('dependent_on'),
                    "supporting_rules": len(rules),
                    "confidence": sum(rule.get('confidence', 0) for rule in rules) / len(rules)
                }
                patterns.append(cross_pattern)

        return patterns

    def analyze_common_patterns(self):
        """覆盖父类方法，使用加权版本"""
        return self.analyze_common_patterns_with_weights()

    def apply_common_patterns(self, input_grid, param):
        """
        将共有模式应用到新的输入网格，考虑权重

        Args:
            input_grid: 输入网格
            param: 对象提取参数

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
        self.weight_calculator.calculate_test_object_weights(input_grid, input_obj_infos, self.shape_library)

        if self.debug:
            self._debug_print(f"从测试输入提取了 {len(input_obj_infos)} 个对象")
            self._debug_print_object_weights(input_obj_infos, "test_input_objects")

        # 由RuleApplier应用规则，生成输出网格
        output_grid = self.rule_applier.apply_patterns(
            input_grid, self.common_patterns, input_obj_infos, self.debug
        )

        if self.debug:
            self._debug_save_grid(output_grid, "test_output_predicted")
            self._debug_print("完成测试预测")

        return output_grid


    def apply_transformation_rules(self, trainortest, pair_id, input_grid, common_patterns=None, transformation_rules=None):
        """应用转换规则到输入网格"""
        if self.debug:
            self._debug_print("调用转换规则应用功能")

        # 使用当前的共有模式（如果未提供）
        if common_patterns is None:
            if not self.common_patterns:
                self.common_patterns = self.analyze_with_test_data_matching(input_grid)
            common_patterns = self.common_patterns

        # 获取网格尺寸和复制网格
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)

        height, width = len(input_grid), len(input_grid[0])

        # 首先应用基于模式的规则（如4Box模式）
        # 检查是否有全局规则
        pattern_based_output = None

        # 检查是否有包含underlying_pattern的全局规则
        has_underlying_pattern = False
        if 'advanced_rules' in common_patterns and 'global_rules' in common_patterns['advanced_rules']:
            for rule in common_patterns['advanced_rules']['global_rules']:
                if 'underlying_pattern' in rule:
                    has_underlying_pattern = True
                    break

        if has_underlying_pattern:
            # 尝试应用模式规则
            pattern_based_output = self.execute_pattern_based_rules(
                input_grid,
                common_patterns['advanced_rules']['global_rules']
            )

            # 如果模式规则产生了变化，使用它作为输出
            if pattern_based_output and pattern_based_output != [row[:] for row in input_grid]:
                if self.debug:
                    self._debug_print("应用了基于模式的规则，跳过传统规则应用")
                return pattern_based_output

        # 准备输入对象（如果模式规则没有生效）
        if trainortest == 'train':
            input_objects = self.all_objects['input'][pair_id][1]
        else:
            input_objects = self.all_objects['test_input'][pair_id][1]

        # 委托给优化规则应用器
        return self.optimized_rule_applier.apply_transformation_rules(
            input_grid,
            common_patterns,
            input_objects,
            traditional_rule_applier=self.rule_applier,
            background_color=self.background_color
        )

    def apply_transformation_rules11(self, trainortest, pair_id, input_grid, common_patterns=None, transformation_rules=None):
        """
        应用提取的转换规则，将输入网格转换为预测的输出网格

        Args:
            input_grid: 输入网格
            common_patterns: 识别的共有模式，如果不提供则使用当前的共有模式
            transformation_rules: 可选，特定的转换规则列表，如果不提供则使用当前累积的规则

        Returns:
            预测的输出网格
        """
        if self.debug:
            self._debug_print("调用转换规则应用功能")

        # 使用当前的共有模式（如果未提供）
        if common_patterns is None:
            if not self.common_patterns:
                # 使用增强版分析，包含测试匹配
                self.common_patterns = self.analyze_with_test_data_matching(input_grid)
            common_patterns = self.common_patterns

        # 使用当前的转换规则（如果未提供）
        # if transformation_rules is None:
        #     transformation_rules = self.transformation_rules

        # 获取网格尺寸
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)

        height, width = len(input_grid), len(input_grid[0])

        # 提取输入网格中的对象
        input_objects = []
        # for param in [(True, True, False), (True, False, False), (False, False, False), (False, True, False)]:
        #     objects = pureobjects_from_grid(param, -1, 'test_in', input_grid, [height, width])
        #     for obj in objects:
        #         input_objects.append(WeightedObjInfo(-1, 'test_in', obj, obj_params=None, grid_hw=[height, width]))

        # # 计算测试输入对象的权重
        # self.weight_calculator.calculate_test_object_weights(input_grid, input_objects, self.shape_library)

        if trainortest == 'train':
            input_objects = self.all_objects['input'][pair_id][1]
        else:
            input_objects = self.all_objects['test_input'][pair_id][1]

        # 委托给优化规则应用器
        return self.optimized_rule_applier.apply_transformation_rules(
            input_grid,
            common_patterns,
            input_objects,
            # transformation_rules,
            traditional_rule_applier=self.rule_applier,  # 传递传统规则应用器作为回退
            background_color=self.background_color
        )

    def apply_transformation_rules00(self, input_grid, common_patterns=None, transformation_rules=None):
        """
        应用提取的转换规则，将输入网格转换为预测的输出网格（委托给 rule_applier）

        Args:
            input_grid: 输入网格
            common_patterns: 识别的共有模式，如果不提供则使用当前的共有模式
            transformation_rules: 可选，特定的转换规则列表，如果不提供则使用当前累积的规则

        Returns:
            预测的输出网格
        """
        if self.debug:
            self._debug_print("调用转换规则应用功能")

        # 使用当前的共有模式（如果未提供）
        if common_patterns is None:
            if not self.common_patterns:
                self.analyze_common_patterns_with_weights()
            common_patterns = self.common_patterns

        # 使用当前的转换规则（如果未提供）
        if transformation_rules is None:
            transformation_rules = self.transformation_rules

        # 获取网格尺寸
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)

        height, width = len(input_grid), len(input_grid[0])

        # 提取输入网格中的对象
        input_objects = []
        for param in [(True, True, False), (True, False, False), (False, False, False), (False, True, False)]:
            objects = pureobjects_from_grid(param, -1, 'test_in', input_grid, [height, width])
            for obj in objects:
                input_objects.append(WeightedObjInfo(-1, 'test_in', obj, obj_params=None, grid_hw=[height, width]))

        # 计算测试输入对象的权重
        self.weight_calculator.calculate_test_object_weights(input_grid, input_objects, self.shape_library)

        # 委托给 rule_applier 处理转换规则的应用
        return self.rule_applier.apply_transformation_rules(
            input_grid, common_patterns, input_objects, transformation_rules, self.debug
        )

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
                objects = pureobjects_from_grid(param, -1, 'test_in', input_grid, [height, width])
                for obj in objects:
                    input_objects.append(WeightedObjInfo(-1, 'test_in', obj, obj_params=None, grid_hw=[height, width]))

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

    def execute_pattern_based_rules(self, input_grid, rules):
        """
        执行基于模式的规则

        Args:
            input_grid: 输入网格
            rules: 规则列表

        Returns:
            应用规则后的输出网格
        """
        # 复制输入网格作为输出
        output_grid = [row[:] for row in input_grid]

        # 检查规则是否包含underlying_pattern
        pattern_rules = [r for r in rules if 'underlying_pattern' in r]

        if not pattern_rules:
            return output_grid  # 没有基于模式的规则，返回原网格

        # 处理每个基于模式的规则
        for rule in pattern_rules:
            pattern = rule['underlying_pattern']

            # 如果有检测和执行函数字段
            if 'detect_fun' in pattern and 'execute_function' in pattern:
                # 获取函数名
                detect_func_name = pattern['detect_fun']
                execute_func_name = pattern['execute_function']

                # 需要确保这些函数存在
                from arcMrule.diffstar.patterlib.pattern_analysis_mixin import PatternAnalysisMixin
                mixin = PatternAnalysisMixin()

                if hasattr(mixin, detect_func_name) and hasattr(mixin, execute_func_name):
                    # 创建执行规则
                    exec_rule = {
                        'center_color': pattern.get('center_color'),
                        'surrounding_color': pattern.get('surrounding_color'),
                        'target_color': rule.get('color'),
                        'min_directions': 4 if pattern.get('complete_ratio', 0) < 0.8 else 4
                    }

                    # 执行模式检测和应用
                    output_grid_list = [list(row) for row in output_grid]
                    output_grid = getattr(mixin, execute_func_name)(output_grid_list, exec_rule)

                    if self.debug:
                        self._debug_print(f"应用了{pattern['pattern_type']}模式规则")

        return output_grid


