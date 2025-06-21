from arc_solver_modular import PriorKnowledgePlugin
from typing import Dict, List, Any
import numpy as np

class GridExtensionPriorKnowledge(PriorKnowledgePlugin):
    """网格扩展任务的先验知识 - 专门处理05a7bcf2.json任务"""

    def get_plugin_name(self) -> str:
        return "grid_extension_knowledge"

    def is_applicable(self, task_data: Dict) -> bool:
        """检查是否为网格扩展类任务"""
        # 检查训练示例是否包含网格结构的特征
        blue_count = 0
        yellow_count = 0
        red_count = 0
        green_count = 0

        # 检查输入和输出
        for example in task_data['train']:
            input_grid = example['input']
            output_grid = example['output']

            # 检查颜色分布
            if self._count_color(input_grid, 6) > 0:  # 蓝色
                blue_count += 1
            if self._count_color(input_grid, 4) > 0:  # 黄色
                yellow_count += 1
            if self._count_color(input_grid, 1) > 0:  # 红色
                red_count += 1
            if self._count_color(output_grid, 2) > 0:  # 绿色
                green_count += 1

        # 特征匹配规则
        if (blue_count > 0 and yellow_count > 0 and
            self._count_color(task_data['train'][0]['output'], 6) >
            self._count_color(task_data['train'][0]['input'], 6)):
            return True

        return False

    def _count_color(self, grid, color_id):
        """计算网格中特定颜色的像素数量"""
        count = 0
        for row in grid:
            for cell in row:
                if cell == color_id:
                    count += 1
        return count

    def generate_facts(self, pair_id: int, input_objects: List, output_objects: List) -> List[str]:
        """生成网格扩展任务特定的事实"""
        facts = []

        # 提取特定对象
        input_h_lines = [obj for obj in input_objects if obj.get('type') == 'h_line' and obj.get('color') == 6]
        input_v_lines = [obj for obj in input_objects if obj.get('type') == 'v_line' and obj.get('color') == 6]
        input_yellow_objs = [obj for obj in input_objects if obj.get('color') == 4]  # 黄色
        input_red_objs = [obj for obj in input_objects if obj.get('color') == 1]  # 红色

        output_h_lines = [obj for obj in output_objects if obj.get('type') == 'h_line' and obj.get('color') == 6]
        output_v_lines = [obj for obj in output_objects if obj.get('type') == 'v_line' and obj.get('color') == 6]
        output_yellow_objs = [obj for obj in output_objects if obj.get('color') == 4]  # 黄色
        output_green_objs = [obj for obj in output_objects if obj.get('color') == 2]  # 绿色

        # 添加水平线事实
        for i, line in enumerate(input_h_lines):
            facts.append(f"h_blue_line(in_{pair_id}_{input_objects.index(line)}).")
            facts.append(f"line_y_pos(in_{pair_id}_{input_objects.index(line)}, {line['y']}).")

        # 添加垂直线事实
        for i, line in enumerate(input_v_lines):
            facts.append(f"v_blue_line(in_{pair_id}_{input_objects.index(line)}).")
            facts.append(f"line_x_pos(in_{pair_id}_{input_objects.index(line)}, {line['x']}).")

        # 添加黄色对象事实
        for i, obj in enumerate(input_yellow_objs):
            obj_id = f"in_{pair_id}_{input_objects.index(obj)}"
            facts.append(f"yellow_object({obj_id}).")

            # 计算与蓝线的关系
            for h_line in input_h_lines:
                h_line_id = f"in_{pair_id}_{input_objects.index(h_line)}"
                if h_line['y'] < obj['y_min']:
                    facts.append(f"line_above({h_line_id}, {obj_id}, {h_line['y']}).")
                elif h_line['y'] > obj['y_max']:
                    facts.append(f"line_below({h_line_id}, {obj_id}, {h_line['y']}).")

        # 添加网格特征
        if output_h_lines and output_v_lines:
            facts.append(f"forms_grid({pair_id}).")

            # 添加网格单元格信息
            h_positions = sorted(list(set([line['y'] for line in output_h_lines])))
            v_positions = sorted(list(set([line['x'] for line in output_v_lines])))

            for i in range(len(h_positions) + 1):
                top = 0 if i == 0 else h_positions[i-1]
                bottom = len(input_objects[0]['pixels'][0]) - 1 if i == len(h_positions) else h_positions[i]

                for j in range(len(v_positions) + 1):
                    left = 0 if j == 0 else v_positions[j-1]
                    right = len(input_objects[0]['pixels']) - 1 if j == len(v_positions) else v_positions[j]

                    facts.append(f"grid_cell({pair_id}, {i}, {j}, {left}, {top}, {right}, {bottom}).")

        # 添加垂直填充模式
        if len(output_yellow_objs) > len(input_yellow_objs):
            facts.append(f"vertical_fill_pattern({pair_id}).")

        # 添加交点着色模式
        if output_green_objs:
            facts.append(f"intersection_coloring({pair_id}).")

        return facts

    def generate_positive_examples(self):
        """生成正例文件"""
        return """% 05a7bcf2任务的目标概念
pos(extends_to_grid(0)).
pos(yellow_fills_vertical(0)).
pos(green_at_intersections(0)).
"""

    def generate_negative_examples(self):
        """生成负例文件"""
        return """% 05a7bcf2任务中不应该出现的概念
neg(rotates_objects(0)).
neg(mirrors_horizontally(0)).
neg(removes_all_objects(0)).
neg(inverts_colors(0)).
neg(random_color_change(0)).
"""


    def generate_positive_examples00(self, pair_id: int) -> List[str]:
        """生成网格扩展任务的正例"""
        return [
            f"extends_to_grid({pair_id}).",
            f"yellow_fills_vertical({pair_id}).",
            f"green_at_intersections({pair_id})."
        ]

    def generate_negative_examples00(self, pair_id: int) -> List[str]:
        """生成网格扩展任务的负例"""
        return [
            f"rotates_objects({pair_id}).",
            f"mirrors_horizontally({pair_id}).",
            f"changes_colors_randomly({pair_id})."
        ]

    def generate_bias(self) -> str:
        """生成网格扩展任务的Popper偏置"""
        return """
        # 目标关系
        head(extends_to_grid/1).
        head(yellow_fills_vertical/1).
        head(green_at_intersections/1).

        # 特定于网格任务的谓词
        body(h_blue_line/1).
        body(v_blue_line/1).
        body(yellow_object/1).
        body(line_y_pos/2).
        body(line_x_pos/2).
        body(line_above/3).
        body(line_below/3).
        body(forms_grid/1).
        body(grid_cell/7).
        body(vertical_fill_pattern/1).
        body(intersection_coloring/1).
        """

    def apply_solution(self, input_grid, learned_rules=None):
        """应用网格扩展解决方案"""
        # 创建输出网格的副本
        height, width = len(input_grid), len(input_grid[0])
        output_grid = [row[:] for row in input_grid]

        # 1. 识别蓝色线位置
        h_line_positions = []
        v_line_positions = []

        for y in range(height):
            if 6 in input_grid[y]:  # 蓝色线
                h_line_positions.append(y)

        for x in range(width):
            column = [input_grid[y][x] for y in range(height)]
            if 6 in column:
                v_line_positions.append(x)

        # 如果没有找到线，创建默认网格
        if not h_line_positions:
            h_line_positions = [height // 3, 2 * height // 3]
        if not v_line_positions:
            v_line_positions = [width // 3, 2 * width // 3]

        # 2. 创建完整的网格
        # 水平线
        for y in h_line_positions:
            for x in range(width):
                output_grid[y][x] = 6  # 蓝色

        # 垂直线
        for x in v_line_positions:
            for y in range(height):
                output_grid[y][x] = 6  # 蓝色

        # 3. 识别黄色对象位置
        yellow_positions = []
        for y in range(height):
            for x in range(width):
                if input_grid[y][x] == 4:  # 黄色
                    yellow_positions.append((x, y))

        # 4. 垂直扩展黄色对象
        yellow_columns = set()
        for x, y in yellow_positions:
            yellow_columns.add(x)

        # 找到每个单元格
        cells = []
        for i in range(len(h_line_positions) + 1):
            top = 0 if i == 0 else h_line_positions[i-1] + 1
            bottom = height - 1 if i == len(h_line_positions) else h_line_positions[i] - 1

            for j in range(len(v_line_positions) + 1):
                left = 0 if j == 0 else v_line_positions[j-1] + 1
                right = width - 1 if j == len(v_line_positions) else v_line_positions[j] - 1

                cells.append((left, top, right, bottom))

        # 5. 垂直扩展每个黄色对象
        for x, y in yellow_positions:
            # 找到哪个单元格包含这个黄色像素
            for left, top, right, bottom in cells:
                if left <= x <= right and top <= y <= bottom:
                    # 垂直填充单元格
                    for fill_y in range(top, bottom + 1):
                        if output_grid[fill_y][x] == 0:  # 只填充空白区域
                            output_grid[fill_y][x] = 4  # 黄色

        # 6. 在黄色对象与蓝线交点处添加绿色
        for x in yellow_columns:
            for y in h_line_positions:
                # 检查交点的上下左右是否有黄色
                has_yellow_nearby = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < width and 0 <= ny < height and
                        output_grid[ny][nx] == 4):
                        has_yellow_nearby = True
                        break

                if has_yellow_nearby and output_grid[y][x] == 6:  # 蓝线
                    output_grid[y][x] = 2  # 绿色

        return output_grid