import abc
from typing import Dict, List, Tuple, Any
import numpy as np
import json
import os
import traceback
from datetime import datetime
import shutil

from arcMrule.diffstar.weighted_analyzer.analyzer_core import WeightedARCDiffAnalyzer
from arcMrule.diffstar.weighted_analyzer.object_matching import ObjectMatcher

from arcMrule.popper.grid_extension_plugin_integration import GridExtensionPriorKnowledge

from base_classes import PriorKnowledgePlugin

# class PriorKnowledgePlugin(abc.ABC):
#     """先验知识插件接口"""

#     @abc.abstractmethod
#     def get_plugin_name(self) -> str:
#         """获取插件名称"""
#         pass

#     @abc.abstractmethod
#     def is_applicable(self, task_data: Dict) -> bool:
#         """判断此先验知识是否适用于当前任务"""
#         pass

#     @abc.abstractmethod
#     def generate_facts(self, pair_id: int, input_objects: List, output_objects: List) -> List[str]:
#         """生成特定于任务的Popper事实"""
#         pass

#     @abc.abstractmethod
#     def generate_positive_examples(self, pair_id: int) -> List[str]:
#         """生成特定于任务的正例"""
#         pass

#     @abc.abstractmethod
#     def generate_negative_examples(self, pair_id: int) -> List[str]:
#         """生成特定于任务的负例"""
#         pass

#     def generate_bias(self) -> str:
#         """生成特定于任务的Popper偏置"""
#         return ""

#     def apply_solution(self, input_grid, learned_rules=None):
#         """应用插件特定的解决方案"""
#         # 默认实现返回输入网格的副本
#         return [row[:] for row in input_grid]


class ARCSolverModular:
    """模块化的ARC求解器"""

    def __init__(self, debug=True):
        self.debug = debug
        self.popper_facts = []
        self.oneInOut_mapping_rules = {}
        self.all_objects = {}
        self.task_data = None
        self.train_pairs = []
        self.test_pairs = []

        # 插件注册表
        self.prior_knowledge_plugins = []
        self.applicable_plugins = []
        self.register_default_plugins()

    def register_default_plugins(self):
        """注册默认插件"""
        grid_plugin = GridExtensionPriorKnowledge()
        self.register_plugin(grid_plugin)

        if self.debug:
            print(f"已注册默认插件: {grid_plugin.get_plugin_name()}")

    def register_plugin(self, plugin: PriorKnowledgePlugin):
        """注册新的先验知识插件"""
        if self.debug:
            print(f"注册插件: {plugin.get_plugin_name()}")
        self.prior_knowledge_plugins.append(plugin)

    def load_task(self, task_path):
        """加载ARC任务"""
        with open(task_path, 'r') as f:
            self.task_data = json.load(f)

        self.train_pairs = [(pair['input'], pair['output']) for pair in self.task_data['train']]
        self.test_pairs = [(pair['input'], pair['output']) for pair in self.task_data['test']]

        # 找到适用于此任务的插件
        self.applicable_plugins = self._find_applicable_plugins()

        if self.debug:
            print(f"加载了 {len(self.train_pairs)} 个训练对和 {len(self.test_pairs)} 个测试对")
            print(f"找到 {len(self.applicable_plugins)} 个适用的先验知识插件:")
            for plugin in self.applicable_plugins:
                print(f"  - {plugin.get_plugin_name()}")

        return self.train_pairs, self.test_pairs

    def _find_applicable_plugins00(self) -> List[PriorKnowledgePlugin]:
        """找出适用于当前任务的所有插件"""
        applicable = []
        for plugin in self.prior_knowledge_plugins:
            if plugin.is_applicable(self.task_data):
                applicable.append(plugin)
        return applicable
    def _find_applicable_plugins(self) -> List[PriorKnowledgePlugin]:
        """找出适用于当前任务的所有插件"""
        applicable = []

        if self.debug:
            print(f"检查 {len(self.prior_knowledge_plugins)} 个插件的适用性...")

        for plugin in self.prior_knowledge_plugins:
            is_app = plugin.is_applicable(self.task_data)
            if is_app:
                applicable.append(plugin)
                if self.debug:
                    print(f"  - 插件 '{plugin.get_plugin_name()}' 适用")
            else:
                if self.debug:
                    print(f"  - 插件 '{plugin.get_plugin_name()}' 不适用")

        if not applicable and self.debug:
            print("警告: 没有适用的插件! 将使用默认策略生成正反例。")

        return applicable

    def extract_objects(self, grid):
        """从网格中提取对象"""
        objects = []
        visited = set()
        height, width = len(grid), len(grid[0])

        # 特殊处理: 提取水平和垂直线条
        h_lines, v_lines = self._extract_lines(grid)
        for line in h_lines + v_lines:
            for x, y in line['pixels']:
                visited.add((x, y))
            objects.append(line)

        # 提取其他连通对象
        for y in range(height):
            for x in range(width):
                if (x, y) not in visited and grid[y][x] != 0:  # 0是背景
                    obj_pixels = self._extract_connected_component(grid, x, y, visited)

                    # 计算对象属性
                    color = grid[y][x]
                    x_values = [p[0] for p in obj_pixels]
                    y_values = [p[1] for p in obj_pixels]

                    obj = {
                        "type": "blob",
                        "color": color,
                        "pixels": obj_pixels,
                        "x_min": min(x_values),
                        "y_min": min(y_values),
                        "x_max": max(x_values),
                        "y_max": max(y_values),
                        "width": 1 + max(x_values) - min(x_values),
                        "height": 1 + max(y_values) - min(y_values),
                        "size": len(obj_pixels)
                    }

                    # 添加形状特征
                    self._add_shape_features(obj, grid)
                    objects.append(obj)

        return objects

    def _extract_lines(self, grid):
        """特别提取水平和垂直线条"""
        height, width = len(grid), len(grid[0])
        h_lines = []
        v_lines = []
        visited = set()

        # 查找水平线条 (主要是颜色6-蓝色)
        for y in range(height):
            for color in range(1, 10):  # 检查各种颜色
                line_pixels = []
                for x in range(width):
                    if grid[y][x] == color and (x, y) not in visited:
                        line_pixels.append((x, y))

                if len(line_pixels) > 1:
                    if max(p[0] for p in line_pixels) - min(p[0] for p in line_pixels) + 1 == len(line_pixels):
                        h_lines.append({
                            "type": "h_line",
                            "color": color,
                            "pixels": line_pixels,
                            "y": y,
                            "x_min": min(p[0] for p in line_pixels),
                            "x_max": max(p[0] for p in line_pixels),
                            "length": len(line_pixels)
                        })
                        for px in line_pixels:
                            visited.add(px)

        # 查找垂直线条
        for x in range(width):
            for color in range(1, 10):
                line_pixels = []
                for y in range(height):
                    if grid[y][x] == color and (x, y) not in visited:
                        line_pixels.append((x, y))

                if len(line_pixels) > 1:
                    if max(p[1] for p in line_pixels) - min(p[1] for p in line_pixels) + 1 == len(line_pixels):
                        v_lines.append({
                            "type": "v_line",
                            "color": color,
                            "pixels": line_pixels,
                            "x": x,
                            "y_min": min(p[1] for p in line_pixels),
                            "y_max": max(p[1] for p in line_pixels),
                            "length": len(line_pixels)
                        })
                        for px in line_pixels:
                            visited.add(px)

        return h_lines, v_lines

    def _extract_connected_component(self, grid, start_x, start_y, visited):
        """提取连通分量"""
        height, width = len(grid), len(grid[0])
        color = grid[start_y][start_x]
        queue = [(start_x, start_y)]
        pixels = []

        while queue:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue

            visited.add((x, y))
            pixels.append((x, y))

            # 检查四个相邻像素
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if (0 <= nx < width and 0 <= ny < height and
                    grid[ny][nx] == color and (nx, ny) not in visited):
                    queue.append((nx, ny))

        return pixels

    def _add_shape_features(self, obj, grid):
        """添加对象的形状特征"""
        # 检测是否是矩形
        if obj["size"] == obj["width"] * obj["height"]:
            obj["is_rectangle"] = True
        else:
            obj["is_rectangle"] = False

        # 检测是否沿边
        if (obj["x_min"] == 0 or obj["x_max"] == len(grid[0])-1 or
            obj["y_min"] == 0 or obj["y_max"] == len(grid)-1):
            obj["touches_edge"] = True
        else:
            obj["touches_edge"] = False


    def analyze_transformation(self, input_grid, output_grid,pair_id):
        """分析输入输出转换，使用WeightedARCDiffAnalyzer"""
        # 创建分析器实例
        analyzer = WeightedARCDiffAnalyzer(debug=self.debug)

        # 设置背景色 #!
        background_color = 0  # 默认黑色背景，可根据需要调整
        if hasattr(self, 'background_colors') and self.background_colors:
            background_color = list(self.background_colors)[0]

        # 提取参数设置
        param = (True, True, False)  # 默认参数，可根据需要调整

        # 分析训练对
        # pair_id = 0  # 由于只分析一个对，使用固定ID
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


    def analyze_transformation000(self, input_grid, output_grid):
        """分析输入输出转换"""
        input_objects = self.extract_objects(input_grid)
        output_objects = self.extract_objects(output_grid)

        transformation = {
            'preserved_objects': [],
            'modified_objects': [],
            'removed_objects': [],
            'added_objects': [],
            'patterns': {}
        }

        # 分析对象映射关系
        for i, in_obj in enumerate(input_objects):
            matched = False
            for j, out_obj in enumerate(output_objects):
                if self._objects_match(in_obj, out_obj):
                    transformation['preserved_objects'].append({
                        'input_id': i,
                        'output_id': j
                    })
                    matched = True
                    break
                elif self._objects_related(in_obj, out_obj):
                    transformation['modified_objects'].append({
                        'input_id': i,
                        'output_id': j,
                        'changes': self._get_changes(in_obj, out_obj)
                    })
                    matched = True
                    break

            if not matched:
                transformation['removed_objects'].append({
                    'input_id': i
                })

        # 查找新增对象
        matched_out_ids = set()
        for mapping in transformation['preserved_objects'] + transformation['modified_objects']:
            if 'output_id' in mapping:
                matched_out_ids.add(mapping['output_id'])

        for j, out_obj in enumerate(output_objects):
            if j not in matched_out_ids:
                transformation['added_objects'].append({
                    'output_id': j
                })

        return transformation, input_objects, output_objects

    def _objects_match(self, obj1, obj2):
        """检查两个对象是否完全匹配"""
        if obj1.get('color') != obj2.get('color'):
            return False

        # 对于线条，检查位置
        if obj1.get('type') in ['h_line', 'v_line'] and obj1.get('type') == obj2.get('type'):
            if obj1.get('type') == 'h_line' and obj1.get('y') == obj2.get('y'):
                return True
            if obj1.get('type') == 'v_line' and obj1.get('x') == obj2.get('x'):
                return True

        # 对于其他对象，检查中心点和大小
        obj1_center = ((obj1.get('x_min', 0) + obj1.get('x_max', 0)) / 2,
                       (obj1.get('y_min', 0) + obj1.get('y_max', 0)) / 2)
        obj2_center = ((obj2.get('x_min', 0) + obj2.get('x_max', 0)) / 2,
                       (obj2.get('y_min', 0) + obj2.get('y_max', 0)) / 2)

        if (abs(obj1_center[0] - obj2_center[0]) < 2 and
            abs(obj1_center[1] - obj2_center[1]) < 2 and
            obj1.get('size') == obj2.get('size')):
            return True

        return False

    def _objects_related(self, obj1, obj2):
        """检查两个对象是否相关（可能经过变换）"""
        # 对于颜色相同的对象，检查位置重叠
        if obj1.get('color') == obj2.get('color'):
            # 检查是否有重叠
            x_overlap = (obj1.get('x_min', 0) <= obj2.get('x_max', 0) and
                        obj2.get('x_min', 0) <= obj1.get('x_max', 0))
            y_overlap = (obj1.get('y_min', 0) <= obj2.get('y_max', 0) and
                        obj2.get('y_min', 0) <= obj1.get('y_max', 0))

            if x_overlap and y_overlap:
                return True

            # 检查中心点是否接近
            obj1_center = ((obj1.get('x_min', 0) + obj1.get('x_max', 0)) / 2,
                           (obj1.get('y_min', 0) + obj1.get('y_max', 0)) / 2)
            obj2_center = ((obj2.get('x_min', 0) + obj2.get('x_max', 0)) / 2,
                           (obj2.get('y_min', 0) + obj2.get('y_max', 0)) / 2)

            if (abs(obj1_center[0] - obj2_center[0]) < 5 and
                abs(obj1_center[1] - obj2_center[1]) < 5):
                return True

        return False

    def _get_changes(self, obj1, obj2):
        """获取从obj1到obj2的变化"""
        changes = {}

        # 检查颜色变化
        if obj1.get('color') != obj2.get('color'):
            changes['color_change'] = {
                'from': obj1.get('color'),
                'to': obj2.get('color')
            }

        # 检查大小变化
        if obj1.get('size') != obj2.get('size'):
            changes['size_change'] = {
                'from': obj1.get('size'),
                'to': obj2.get('size'),
                'ratio': obj2.get('size') / obj1.get('size') if obj1.get('size') > 0 else float('inf')
            }

        # 检查位置变化
        obj1_center = ((obj1.get('x_min', 0) + obj1.get('x_max', 0)) / 2,
                       (obj1.get('y_min', 0) + obj1.get('y_max', 0)) / 2)
        obj2_center = ((obj2.get('x_min', 0) + obj2.get('x_max', 0)) / 2,
                       (obj2.get('y_min', 0) + obj2.get('y_max', 0)) / 2)

        dx = obj2_center[0] - obj1_center[0]
        dy = obj2_center[1] - obj1_center[1]

        if abs(dx) > 0 or abs(dy) > 0:
            changes['position_change'] = {
                'dx': dx,
                'dy': dy
            }

        return changes

    # def _convert_to_popper_facts(self, pair_id, input_grid, output_grid, input_objects, output_objects):
    #     """转换为通用的Popper事实 - 不依赖于特定任务"""
    #     facts = []

    #     # 添加网格尺寸信息
    #     height, width = len(input_grid), len(input_grid[0])
    #     facts.append(f"grid_size({pair_id}, {width}, {height}).")

    #     # 添加通用对象信息
    #     for i, obj in enumerate(input_objects):
    #         obj_id = f"in_{pair_id}_{i}"
    #         facts.append(f"object({obj_id}).")
    #         facts.append(f"input_object({obj_id}).")
    #         facts.append(f"color({obj_id}, {obj['color']}).")

    #         if 'type' in obj:
    #             facts.append(f"type({obj_id}, {obj['type']}).")

    #         # 添加通用位置和尺寸信息
    #         if 'x_min' in obj:
    #             facts.append(f"x_min({obj_id}, {obj['x_min']}).")
    #             facts.append(f"y_min({obj_id}, {obj['y_min']}).")
    #             facts.append(f"x_max({obj_id}, {obj['x_max']}).")
    #             facts.append(f"y_max({obj_id}, {obj['y_max']}).")
    #             facts.append(f"width({obj_id}, {obj['width']}).")
    #             facts.append(f"height({obj_id}, {obj['height']}).")
    #             facts.append(f"size({obj_id}, {obj['size']}).")

    #             # 特定形状特征
    #             if obj.get('is_rectangle', False):
    #                 facts.append(f"is_rectangle({obj_id}).")
    #             if obj.get('touches_edge', False):
    #                 facts.append(f"touches_edge({obj_id}).")

    #     for i, obj in enumerate(output_objects):
    #         obj_id = f"out_{pair_id}_{i}"
    #         facts.append(f"object({obj_id}).")
    #         facts.append(f"output_object({obj_id}).")
    #         facts.append(f"color({obj_id}, {obj['color']}).")

    #         if 'type' in obj:
    #             facts.append(f"type({obj_id}, {obj['type']}).")

    #         # 添加位置和尺寸信息
    #         if 'x_min' in obj:
    #             facts.append(f"x_min({obj_id}, {obj['x_min']}).")
    #             facts.append(f"y_min({obj_id}, {obj['y_min']}).")
    #             facts.append(f"x_max({obj_id}, {obj['x_max']}).")
    #             facts.append(f"y_max({obj_id}, {obj['y_max']}).")
    #             facts.append(f"width({obj_id}, {obj['width']}).")
    #             facts.append(f"height({obj_id}, {obj['height']}).")
    #             facts.append(f"size({obj_id}, {obj['size']}).")

    #             # 特定形状特征
    #             if obj.get('is_rectangle', False):
    #                 facts.append(f"is_rectangle({obj_id}).")
    #             if obj.get('touches_edge', False):
    #                 facts.append(f"touches_edge({obj_id}).")

    #     # 对象间关系
    #     for i, obj1 in enumerate(input_objects):
    #         for j, obj2 in enumerate(input_objects):
    #             if i != j:
    #                 obj1_id = f"in_{pair_id}_{i}"
    #                 obj2_id = f"in_{pair_id}_{j}"

    #                 # 相对位置关系
    #                 if obj1.get('x_max', 0) < obj2.get('x_min', 0):
    #                     facts.append(f"left_of({obj1_id}, {obj2_id}).")
    #                 if obj1.get('y_max', 0) < obj2.get('y_min', 0):
    #                     facts.append(f"above({obj1_id}, {obj2_id}).")

    #                 # 相同颜色关系
    #                 if obj1.get('color') == obj2.get('color'):
    #                     facts.append(f"same_color({obj1_id}, {obj2_id}).")

    #     return facts

    def prepare_popper_data(self):
        """准备Popper训练数据 - 使用插件架构"""
        all_facts = []
        positive_examples = []
        negative_examples = []

        # 处理每个训练对
        for pair_id, (input_grid, output_grid) in enumerate(self.train_pairs):
            # 分析转换
            transformation, input_objects, output_objects = self.analyze_transformation(input_grid, output_grid,pair_id)
            self.oneInOut_mapping_rules[pair_id] = transformation
            self.all_objects[pair_id] = {"input": input_objects, "output": output_objects}

            # 生成通用Popper事实
            pair_facts = self._convert_weighted_objects_to_popper_facts(pair_id, input_grid, output_grid,
                                                    input_objects, output_objects, transformation)
            all_facts.extend(pair_facts)

            # 应用所有适用的插件生成特定事实和例子
            plugin_generated = False
            for plugin in self.applicable_plugins:
                plugin_facts = plugin.generate_facts(pair_id, input_objects, output_objects)
                plugin_positives = plugin.generate_positive_examples(pair_id)
                plugin_negatives = plugin.generate_negative_examples(pair_id)

                all_facts.extend(plugin_facts)
                positive_examples.extend(plugin_positives)
                negative_examples.extend(plugin_negatives)
                plugin_generated = True

            # 如果没有任何插件生成正例和反例，使用默认生成策略
            if not plugin_generated:
                if self.debug:
                    print(f"警告: 没有适用的插件为pair_id={pair_id}生成正反例。使用默认生成策略。")

                # 默认正例 - 根据任务特征生成
                if self._has_blue_lines(input_grid) and self._has_blue_lines(output_grid):
                    positive_examples.append(f"task_involves_grid({pair_id}).")

                if self._has_color(input_grid, 4) and self._has_color(output_grid, 4):  # 黄色
                    positive_examples.append(f"preserves_yellow({pair_id}).")

                if self._has_color(output_grid, 2) and not self._has_color(input_grid, 2):  # 绿色在输出中出现
                    positive_examples.append(f"introduces_green({pair_id}).")

                # 默认反例 - 与任务明显不符的行为
                negative_examples.append(f"removes_all_objects({pair_id}).")
                negative_examples.append(f"inverts_colors({pair_id}).")

        # 如果依然没有正例，添加通用正例
        if not positive_examples:
            for pair_id in range(len(self.train_pairs)):
                positive_examples.append(f"transforms_grid({pair_id}).")

        # 如果依然没有反例，添加通用反例
        if not negative_examples:
            for pair_id in range(len(self.train_pairs)):
                negative_examples.append(f"no_transformation({pair_id}).")

        if self.debug:
            print(f"生成了 {len(all_facts)} 条事实")
            print(f"生成了 {len(positive_examples)} 条正例")
            print(f"生成了 {len(negative_examples)} 条反例")

        return all_facts, positive_examples, negative_examples

    def _has_blue_lines(self, grid):
        """检查网格中是否有蓝色线条(颜色6)"""
        return any(6 in row for row in grid)

    def _has_color(self, grid, color):
        """检查网格中是否有特定颜色"""
        return any(color in row for row in grid)


    def prepare_popper_data00(self):
        """准备Popper训练数据 - 使用插件架构和WeightedARCDiffAnalyzer"""
        all_facts = []
        positive_examples = []
        negative_examples = []

        # 处理每个训练对
        for pair_id, (input_grid, output_grid) in enumerate(self.train_pairs):
            # 使用适合的分析器分析转换
            transformation, input_objects, output_objects = self.analyze_transformation(input_grid, output_grid,pair_id)
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





    def prepare_popper_data000(self):
        """准备Popper训练数据 - 使用插件架构"""
        all_facts = []
        positive_examples = []
        negative_examples = []

        # 处理每个训练对
        for pair_id, (input_grid, output_grid) in enumerate(self.train_pairs):
            # 分析转换

            transformation, input_objects, output_objects = self.analyze_transformation(input_grid, output_grid)
            self.oneInOut_mapping_rules[pair_id] = transformation
            self.all_objects[pair_id] = {"input": input_objects, "output": output_objects}

            # 生成通用Popper事实
            pair_facts = self._convert_to_popper_facts(pair_id, input_grid, output_grid,
                                                      input_objects, output_objects)
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


    def generate_popper_bias(self) -> str:
        """生成偏置文件"""
        bias = []

        # 目标谓词
        bias.append("% 定义目标关系")
        # for rule in rules:
        #     if rule["type"] == "grid_extension":
        bias.append("head_pred(extends_to_grid,1).")
        #     elif rule["type"] == "vertical_fill":
        bias.append("head_pred(yellow_fills_vertical,1).")
        #     elif rule["type"] == "intersection_coloring":
        bias.append("head_pred(green_at_intersections,1).")

        # 确保至少有一个目标谓词
        # if not rules:
        #     bias.append("head_pred(transforms_grid,1).")

        # 背景谓词
        bias.append("\n% 背景知识谓词")
        bias.append("body_pred(grid_size,3).")
        bias.append("body_pred(color_value,2).")
        bias.append("body_pred(h_line,1).")
        bias.append("body_pred(v_line,1).")
        bias.append("body_pred(line_y_pos,2).")
        bias.append("body_pred(line_x_pos,2).")
        bias.append("body_pred(yellow_object,1).")
        bias.append("body_pred(x_min,2).")
        bias.append("body_pred(y_min,2).")
        bias.append("body_pred(color,2).")
        bias.append("body_pred(on_grid_line,2).")
        bias.append("body_pred(grid_intersection,2).")
        bias.append("body_pred(has_adjacent_yellow,2).")

        # 搜索约束
        bias.append("\n% 搜索约束")
        bias.append("max_vars(6).")
        bias.append("max_body(8).")
        bias.append("max_clauses(4).")

        return "\n".join(bias)

    def generate_popper_bias000(self):
        """生成Popper偏置文件 - 结合通用偏置和插件偏置"""
        bias = """
% % # 基本谓词
body(grid_size/3).
body(object/1).
body(input_object/1).
body(output_object/1).
body(color/2).
body(type/2).
body(x_min/2).
body(x_max/2).
body(y_min/2).
body(y_max/2).
body(width/2).
body(height/2).
body(size/2).
body(is_rectangle/1).
body(touches_edge/1).
body(left_of/2).
body(above/2).
body(same_color/2).

% % # 基本约束
max_vars(6).
max_body(10).
"""

        # 添加插件提供的特定偏置
        for plugin in self.applicable_plugins:
            plugin_bias = plugin.generate_bias()
            if plugin_bias:
                bias += f"\n# 从 {plugin.get_plugin_name()} 插件\n{plugin_bias}"

        return bias

    def save_popper_files(self, output_dir="."):
        """生成并保存Popper文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 准备数据
        background, positive, negative = self.prepare_popper_data()
        bias = self.generate_popper_bias()

        # 保存到文件
        with open(os.path.join(output_dir, "background.pl"), "w") as f:
            f.write("\n".join(background))

        with open(os.path.join(output_dir, "positive.pl"), "w") as f:
            f.write("\n".join(positive))

        with open(os.path.join(output_dir, "negative.pl"), "w") as f:
            f.write("\n".join(negative))

        with open(os.path.join(output_dir, "bias.pl"), "w") as f:
            f.write(bias)

        if self.debug:
            print(f"Popper文件已保存到 {output_dir} 目录")
            print(f"  - 背景知识: {len(background)} 条事实")
            print(f"  - 正例: {len(positive)} 条")
            print(f"  - 负例: {len(negative)} 条")



    def learn_rules_with_popper(self, output_dir="."):
        """使用Popper学习规则"""
        try:
            from popper.util import Settings #, print_prog_score
            from popper.loop import learn_solution

            # Popper现在期望具有特定命名的文件
            # 检查并确保文件名正确
            expected_files = {
                "bias.pl": "bias.pl",
                "positive.pl": "exs.pl",  # Popper现在期望examples在exs.pl
                "background.pl": "bk.pl"   # Popper现在期望背景知识在bk.pl
            }

            # 创建一个临时目录以符合Popper的期望
            tmp_popper_dir = os.path.join(output_dir, "popper_input")
            os.makedirs(tmp_popper_dir, exist_ok=True)

            # 复制并重命名文件
            for src_name, dest_name in expected_files.items():
                src_path = os.path.join(output_dir, src_name)
                dest_path = os.path.join(tmp_popper_dir, dest_name)

                # 确保源文件存在
                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                    print(f"已复制 {src_path} 到 {dest_path}")
                else:
                    print(f"警告: 找不到源文件 {src_path}")

            # 如果存在negative.pl，将其内容合并到exs.pl
            neg_file = os.path.join(output_dir, "negative.pl")
            if os.path.exists(neg_file):
                with open(neg_file, 'r') as neg_f:
                    neg_content = neg_f.read()

                with open(os.path.join(tmp_popper_dir, "exs.pl"), 'a') as exs_f:
                    exs_f.write("\n\n% Negative examples\n")
                    exs_f.write(neg_content)
                    print("已将负例合并到exs.pl")

            print(f"开始Popper学习，使用目录: {tmp_popper_dir}")

            # 使用新的Popper API
            settings = Settings(kbpath=tmp_popper_dir)
            prog, score, stats = learn_solution(settings)

            if prog != None:
                print("成功学习到规则:")
                Settings.print_prog_score(prog, score)

                # 将规则保存为文件
                with open(os.path.join(output_dir, "learned_rules.pl"), 'w') as f:
                    for rule in prog:
                        f.write(f"{rule}\n")

                return prog
            else:
                print("Popper未能找到有效规则")
                return []

        except ImportError:
            print("未能导入Popper。请确保已安装: pip install git+https://github.com/logic-and-learning-lab/Popper@main")
            print(f"详细错误信息:\n{traceback.format_exc()}")
            return []
        except Exception as e:
            print(f"学习规则时出错: {e}")
            print(f"详细堆栈跟踪:\n{traceback.format_exc()}")

            # 将错误信息写入日志文件
            with open(os.path.join(output_dir, "error_log.txt"), 'w') as f:
                f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误类型: {type(e).__name__}\n")
                f.write(f"错误信息: {str(e)}\n")
                f.write(f"堆栈跟踪:\n{traceback.format_exc()}")

            return []



    def learn_rules_with_popper00(self, output_dir="."):
        """使用Popper学习规则"""
        # self.save_popper_files(output_dir)
        try:
            from popper.util import Settings
            from popper.loop import learn_solution

            # 记录开始时间
            start_time = datetime.now()
            print(f"开始学习规则: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            settings = Settings(kbpath=output_dir)

            # settings = Settings(
            #     bias_file=os.path.join(output_dir, "bias.pl"),
            #     pos_file=os.path.join(output_dir, "positive.pl"),
            #     neg_file=os.path.join(output_dir, "negative.pl"),
            #     bk_file=os.path.join(output_dir, "background.pl"),
            #     timeout=60
            # )

            print("正在运行Popper学习器...")
            learned_rules = learn_solution(settings)

            if learned_rules:
                print("成功学习到规则:")
                for rule in learned_rules:
                    print(f"  {rule}")

                # 保存学习到的规则
                with open(os.path.join(output_dir, "learned_rules.pl"), 'w') as f:
                    for rule in learned_rules:
                        f.write(f"{rule}\n")

                return learned_rules
            else:
                print("Popper未能找到规则")
                return []

        except ImportError:
            print("未能导入Popper。请确保已安装: pip install popper-ilp")
            print(f"详细错误信息:\n{traceback.format_exc()}")
            return []
        except Exception as e:
            print(f"学习规则时出错: {e}")
            print(f"详细堆栈跟踪:\n{traceback.format_exc()}")

            # 将详细错误信息写入日志文件
            with open(os.path.join(output_dir, "error_log.txt"), 'w') as f:
                f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误类型: {type(e).__name__}\n")
                f.write(f"错误信息: {str(e)}\n")
                f.write(f"堆栈跟踪:\n{traceback.format_exc()}")

            return []

    def apply_learned_rules(self, input_grid, learned_rules=None):
        """应用学习到的规则解决任务"""
        if not learned_rules and not self.applicable_plugins:
            if self.debug:
                print("警告: 没有适用的规则或插件")
            return input_grid

        # 使用适用插件的特殊处理方法
        for plugin in self.applicable_plugins:
            output_grid = plugin.apply_solution(input_grid, learned_rules)
            if output_grid != input_grid:
                if self.debug:
                    print(f"使用插件 {plugin.get_plugin_name()} 应用解决方案")
                return output_grid

        # 默认情况下返回输入
        if self.debug:
            print("没有插件能够应用解决方案，返回原始输入")
        return input_grid

    def solve(self, task_path, output_dir="."):
        """解决ARC任务的主函数"""
        # 1. 加载任务
        self.load_task(task_path)

        # 2. 学习规则
        learned_rules = self.learn_rules_with_popper(output_dir)

        # 3. 解决测试案例
        solutions = []
        for input_grid, expected_output in self.test_pairs:
            output_grid = self.apply_learned_rules(input_grid, learned_rules)
            solutions.append((output_grid, expected_output))

        # 4. 评估结果
        if self.debug:
            for i, (output_grid, expected_output) in enumerate(solutions):
                correct = self._compare_grids(output_grid, expected_output)
                print(f"测试用例 {i+1}: {'正确' if correct else '不正确'}")

        return solutions

    def _compare_grids(self, grid1, grid2):
        """比较两个网格是否相同"""
        if len(grid1) != len(grid2):
            return False

        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False

        return True