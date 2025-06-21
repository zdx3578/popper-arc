# from arc_solver_modular_integration import PriorKnowledgePlugin
from base_classes import PriorKnowledgePlugin
from typing import Dict, List, Any

# class GridExtensionPriorKnowledge(PriorKnowledgePlugin):
#     """网格扩展任务的先验知识 - 基于WeightedARCDiffAnalyzer分析结果"""
class GridExtensionPriorKnowledge(PriorKnowledgePlugin):
    def __init__(self):
        self.debug_print = print

    def is_applicable(self, task_data: Dict) -> bool:
        # 直接针对05a7bcf2特征
        has_blue = False
        has_yellow = False
        has_grid_pattern = False

        for example in task_data['train']:
            input_grid = example['input']
            output_grid = example['output']

            # 蓝色线条检测
            blue_in = self._count_color(input_grid, 6)
            blue_out = self._count_color(output_grid, 6)
            if blue_in > 0 and blue_out > blue_in:
                has_blue = True

            # 黄色对象检测
            if self._has_color(input_grid, 4):
                has_yellow = True

            # 网格模式检测 - 至少有两条蓝线相交
            if self._has_grid_structure(output_grid):
                has_grid_pattern = True

        # 打印调试信息
        print(f"任务特征检测: 蓝线={has_blue}, 黄色对象={has_yellow}, 网格模式={has_grid_pattern}")

        # 05a7bcf2任务的典型特征
        matches_05a7bcf2 = has_blue and has_yellow and has_grid_pattern

        # 特殊情况: 如果任务ID直接包含在文件名中
        if any('05a7bcf2' in str(key) for key in task_data.keys()):
            print("直接检测到05a7bcf2任务ID!")
            return True

        return matches_05a7bcf2

    def _has_grid_structure(self, grid):
        """检查网格是否有网格结构(至少两条蓝线相交)"""
        h_lines = []
        v_lines = []

        # 查找横线
        for y in range(len(grid)):
            if self._count_color_in_row(grid, y, 6) > 1:
                h_lines.append(y)

        # 查找竖线
        for x in range(len(grid[0])):
            if self._count_color_in_column(grid, x, 6) > 1:
                v_lines.append(x)

        # 至少有一条横线和一条竖线
        return len(h_lines) > 0 and len(v_lines) > 0

    def _count_color_in_row(self, grid, row, color):
        """计算行中特定颜色的数量"""
        return sum(1 for cell in grid[row] if cell == color)

    def _count_color_in_column(self, grid, col, color):
        """计算列中特定颜色的数量"""
        return sum(1 for row in grid if row[col] == color)


    def get_plugin_name(self) -> str:
        return "grid_extension_knowledge"

    def is_applicable0(self, task_data: Dict) -> bool:
        """检查是否为网格扩展类任务"""
        # 检查训练示例是否包含网格结构
        for example in task_data['train']:
            input_grid = example['input']
            output_grid = example['output']

            # 检查蓝线（颜色索引6）是否存在
            if self._has_color(input_grid, 6) and self._has_color(output_grid, 6):
                # 检查蓝线是否在输出中形成更完整的网格
                if self._count_color(output_grid, 6) > self._count_color(input_grid, 6):
                    return True

        return False
    def is_applicable00(self, task_data: Dict) -> bool:
        """检查是否为网格扩展类任务"""
        # 针对05a7bcf2任务的特定检测
        for example in task_data['train']:
            input_grid = example['input']
            output_grid = example['output']

            # 特征1: 检查蓝色线条(颜色6)
            blue_in_input = any(6 in row for row in input_grid)
            blue_in_output = any(6 in row for row in output_grid)

            # 特征2: 检查黄色对象(颜色4)
            yellow_in_input = any(4 in row for row in input_grid)

            # 特征3: 检查绿色(颜色2)，通常在输出的交点处
            green_in_output = any(2 in row for row in output_grid)

            # 如果满足这些特征，很可能是网格扩展任务
            if blue_in_input and blue_in_output and yellow_in_input:
                if self.debug_print:
                    self.debug_print("检测到网格扩展任务模式!")
                return True

        return False

    def _has_color(self, grid, color_id):
        """检查网格中是否有特定颜色"""
        return any(color_id in row for row in grid)

    def _count_color(self, grid, color_id):
        """计算网格中特定颜色的像素数量"""
        return sum(row.count(color_id) for row in grid)

    def generate_facts(self, pair_id: int, input_objects: List, output_objects: List) -> List[str]:
        """生成网格扩展任务特定的事实"""
        facts = []

        # 提取线条和黄色对象
        input_h_lines = [obj for obj in input_objects if self._is_h_line(obj) and obj.main_color == 6]
        input_v_lines = [obj for obj in input_objects if self._is_v_line(obj) and obj.main_color == 6]
        input_yellow_objs = [obj for obj in input_objects if obj.main_color == 4]  # 黄色

        output_h_lines = [obj for obj in output_objects if self._is_h_line(obj) and obj.main_color == 6]
        output_v_lines = [obj for obj in output_objects if self._is_v_line(obj) and obj.main_color == 6]
        output_yellow_objs = [obj for obj in output_objects if obj.main_color == 4]  # 黄色
        output_green_objs = [obj for obj in output_objects if obj.main_color == 2]  # 绿色

        # 添加水平线事实
        for i, line in enumerate(input_h_lines):
            obj_id = f"in_{pair_id}_{self._find_obj_index(line, input_objects)}"
            facts.append(f"h_line({obj_id}).")
            facts.append(f"line_y_pos({obj_id}, {line.top}).")

        # 添加垂直线事实
        for i, line in enumerate(input_v_lines):
            obj_id = f"in_{pair_id}_{self._find_obj_index(line, input_objects)}"
            facts.append(f"v_line({obj_id}).")
            facts.append(f"line_x_pos({obj_id}, {line.left}).")

        # 添加黄色对象事实
        for i, obj in enumerate(input_yellow_objs):
            obj_id = f"in_{pair_id}_{self._find_obj_index(obj, input_objects)}"
            facts.append(f"yellow_object({obj_id}).")

            # 计算与蓝线的关系
            for h_line in input_h_lines:
                h_line_id = f"in_{pair_id}_{self._find_obj_index(h_line, input_objects)}"
                if h_line.top < obj.top:
                    facts.append(f"line_above({h_line_id}, {obj_id}, {h_line.top}).")
                elif h_line.top > obj.top + obj.height:
                    facts.append(f"line_below({h_line_id}, {obj_id}, {h_line.top}).")

        # 添加输出中的网格特征
        if output_h_lines and output_v_lines:
            facts.append(f"forms_grid({pair_id}).")

            # 添加输出中的交叉点信息
            for h_line in output_h_lines:
                for v_line in output_v_lines:
                    h_id = f"out_{pair_id}_{self._find_obj_index(h_line, output_objects)}"
                    v_id = f"out_{pair_id}_{self._find_obj_index(v_line, output_objects)}"
                    facts.append(f"grid_intersection({pair_id}, {h_id}, {v_id}, {v_line.left}, {h_line.top}).")

            # 添加颜色变化信息
            if output_green_objs:
                facts.append(f"has_green_intersections({pair_id}).")

        # 垂直填充模式
        if len(output_yellow_objs) > len(input_yellow_objs):
            facts.append(f"yellow_fills_vertical({pair_id}).")

        return facts

    def generate_positive_examples(self, pair_id: int) -> List[str]:
        """生成网格扩展任务的正例"""
        return [
            f"extends_to_grid({pair_id}).",
            f"yellow_fills_vertical({pair_id}).",
            f"green_at_intersections({pair_id})."
        ]

    def generate_negative_examples(self, pair_id: int) -> List[str]:
        """生成网格扩展任务的负例"""
        return [
            f"rotates_objects({pair_id}).",
            f"mirrors_horizontally({pair_id}).",
            f"random_color_change({pair_id})."
        ]

    def generate_bias(self) -> str:
        """生成网格扩展任务的Popper偏置"""
        return """
# 目标关系
head(extends_to_grid/1).
head(yellow_fills_vertical/1).
head(green_at_intersections/1).

# 特定于网格任务的谓词
body(h_line/1).
body(v_line/1).
body(yellow_object/1).
body(line_y_pos/2).
body(line_x_pos/2).
body(line_above/3).
body(line_below/3).
body(forms_grid/1).
body(grid_intersection/5).
body(has_green_intersections/1).
"""

    def _is_h_line(self, obj):
        """检查对象是否是水平线"""
        # 根据WeightedObjInfo结构检查
        return hasattr(obj, 'width') and hasattr(obj, 'height') and obj.width > 1 and obj.height == 1

    def _is_v_line(self, obj):
        """检查对象是否是垂直线"""
        # 根据WeightedObjInfo结构检查
        return hasattr(obj, 'width') and hasattr(obj, 'height') and obj.width == 1 and obj.height > 1

    def _find_obj_index(self, obj, obj_list):
        """找到对象在列表中的索引"""
        for i, o in enumerate(obj_list):
            if o.obj_id == obj.obj_id:
                return i
        return -1

    def apply_solution(self, input_grid, learned_rules=None):
        """应用网格扩展解决方案"""
        # 创建输出网格
        output_grid = [row[:] for row in input_grid]
        height, width = len(input_grid), len(input_grid[0])

        # 1. 提取网格线位置
        h_lines = []
        v_lines = []
        yellow_positions = []

        for y in range(height):
            if 6 in input_grid[y]:
                h_lines.append(y)

        for x in range(width):
            if 6 in [input_grid[y][x] for y in range(height)]:
                v_lines.append(x)

        for y in range(height):
            for x in range(width):
                if input_grid[y][x] == 4:  # 黄色
                    yellow_positions.append((x, y))

        # 2. 创建完整网格
        # 如果没有足够的线，添加更多
        if len(h_lines) < 2:
            h_spacing = height // 3
            h_lines = [h_spacing, 2 * h_spacing]

        if len(v_lines) < 2:
            v_spacing = width // 3
            v_lines = [v_spacing, 2 * v_spacing]

        # 填充所有水平线
        for y in h_lines:
            for x in range(width):
                output_grid[y][x] = 6  # 蓝色

        # 填充所有垂直线
        for x in v_lines:
            for y in range(height):
                output_grid[y][x] = 6  # 蓝色

        # 3. 垂直延伸黄色对象
        yellow_columns = {x for x, _ in yellow_positions}

        for x in yellow_columns:
            # 查找包含此黄色对象的单元格
            containing_cells = []

            for i in range(len(h_lines) + 1):
                top = 0 if i == 0 else h_lines[i-1] + 1
                bottom = height - 1 if i == len(h_lines) else h_lines[i] - 1

                for j in range(len(v_lines) + 1):
                    left = 0 if j == 0 else v_lines[j-1] + 1
                    right = width - 1 if j == len(v_lines) else v_lines[j] - 1

                    # 检查该单元格是否包含黄色像素
                    has_yellow = False
                    for y in range(top, bottom + 1):
                        if left <= x <= right and input_grid[y][x] == 4:
                            has_yellow = True
                            break

                    if has_yellow:
                        # 填充整个单元格的黄色
                        for y in range(top, bottom + 1):
                            if output_grid[y][x] == 0:  # 只填充空白区域
                                output_grid[y][x] = 4  # 黄色

        # 4. 在交点处添加绿色
        for x in v_lines:
            for y in h_lines:
                if output_grid[y][x] == 6:  # 蓝色交叉点
                    # 检查周围是否有黄色
                    has_adjacent_yellow = False
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and output_grid[ny][nx] == 4:
                            has_adjacent_yellow = True
                            break

                    # 如果有相邻黄色，将交点变为绿色
                    if has_adjacent_yellow:
                        output_grid[y][x] = 2  # 绿色

        return output_grid




