import numpy as np
from typing import List, Dict, Tuple, Any, Set, FrozenSet
from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
import copy

# 导入现有函数
from objutil import all_pureobjects_from_grid, objects_fromone_params
from weightgird import grid2grid_fromgriddiff
from objutil import shift_pure_obj_to_0_0_0, uppermost, leftmost, lowermost, rightmost, palette, extend_obj


def shift_to_origin(obj, preserve_colors=True):
    """
    将对象集合移到左上角(0,0)，并可选择是否保留颜色信息。
    将输入转换为可哈希的frozenset。
    """
    # 如果输入为空，直接返回空frozenset
    if not obj:
        return frozenset()

    # 计算最小行列值
    min_row = float('inf')
    min_col = float('inf')

    for _, (r, c) in obj:
        min_row = min(min_row, r)
        min_col = min(min_col, c)

    # 创建移动到原点的frozenset
    if preserve_colors:
        return frozenset([(color, (r - min_row, c - min_col))
                         for color, (r, c) in obj])
    else:
        return frozenset([(0, (r - min_row, c - min_col))
                         for color, (r, c) in obj])

class ObjInfo:
    """增强型对象信息类，存储对象的所有相关信息和变换"""

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
        self.pair_id = pair_id
        self.in_or_out = in_or_out
        self.original_obj = obj
        self.obj = obj
        self.obj_params = obj_params if obj_params else (True, True, False)
        self.grid_hw = grid_hw
        self.background = background
        self.debug = True

        # 计算标准化对象
        self.obj_00 = self._shift_to_origin(obj)  # 移到左上角但保留颜色
        self.obj_000 = self._shift_to_origin(obj, preserve_colors=False)  # 完全标准化（去色）
        self.obj_weight = 0

        # 计算边界框
        self.bounding_box = (
            uppermost(obj), leftmost(obj),
            lowermost(obj), rightmost(obj)
        )
        self.top = uppermost(obj)
        self.left = leftmost(obj)

        # 颜色信息
        self.color_ranking = palette(obj)

        # 计算扩展操作
        self.obj000_ops = extend_obj(self.obj_000)
        self.obj_ops = extend_obj(obj)

        # 基本属性
        self.size = len(obj)
        self.height = self.bounding_box[2] - self.bounding_box[0] + 1
        self.width = self.bounding_box[3] - self.bounding_box[1] + 1
        self.main_color = self._get_main_color()

        # 生成唯一ID
        # self.obj_id = f"{pair_id}_{in_or_out}_{hash(self.obj_000)}"
        self.obj_id = f"{pair_id}_{in_or_out}_{hash(self.obj_000)}_{self.top}_{self.left}"

        # 转换后的对象变体
        self.rotated_variants = self._generate_rotated_variants()
        self.mirrored_variants = self.obj000_ops
        # self.mirrored_variants = self._generate_mirrored_variants()

        # 对象关系
        self.is_part_of = []  # 存储该对象是哪些对象的一部分
        self.has_parts = []   # 存储该对象包含哪些子部分

    def _get_main_color(self):
        """获取对象的主要颜色"""
        if not self.original_obj:
            return None

        # 计算每种颜色的像素数
        color_counts = defaultdict(int)
        for value, _ in self.original_obj:
            color_counts[value] += 1

        # 返回出现最多的颜色
        if color_counts:
            return max(color_counts.items(), key=lambda x: x[1])[0]
        return None

    def _shift_to_origin(self, obj, preserve_colors=True):
        """类方法版本，调用独立函数实现"""
        return shift_to_origin(obj, preserve_colors)

    def _generate_rotated_variants(self):
        """生成对象的所有旋转变体"""
        variants = []

        # 原始对象（0度旋转）
        obj_00 = self.obj_00
        variants.append(("rot_0", obj_00))

        # 90度旋转
        obj_90 = self._rotate_object(obj_00, 90)
        obj_90_shifted = self._shift_to_origin(obj_90)
        variants.append(("rot_90", obj_90_shifted))

        # 180度旋转
        obj_180 = self._rotate_object(obj_00, 180)
        obj_180_shifted = self._shift_to_origin(obj_180)
        variants.append(("rot_180", obj_180_shifted))

        # 270度旋转
        obj_270 = self._rotate_object(obj_00, 270)
        obj_270_shifted = self._shift_to_origin(obj_270)
        variants.append(("rot_270", obj_270_shifted))

        return variants

    def _generate_mirrored_variants(self):
        """生成对象的所有镜像变体"""
        variants = []

        # 原始对象
        obj_00 = self.obj_00
        variants.append(("original", obj_00))

        # 水平镜像
        obj_h_mirror = self._mirror_object(obj_00, "horizontal")
        obj_h_mirror_shifted = self._shift_to_origin(obj_h_mirror)
        variants.append(("h_mirror", obj_h_mirror_shifted))

        # 垂直镜像
        obj_v_mirror = self._mirror_object(obj_00, "vertical")
        obj_v_mirror_shifted = self._shift_to_origin(obj_v_mirror)
        variants.append(("v_mirror", obj_v_mirror_shifted))

        return variants

    def _rotate_object(self, obj, degrees):
        """旋转对象指定角度"""
        # 获取对象边界
        min_r = min([r for _, (r, c) in obj]) if obj else 0
        min_c = min([c for _, (r, c) in obj]) if obj else 0
        max_r = max([r for _, (r, c) in obj]) if obj else 0
        max_c = max([c for _, (r, c) in obj]) if obj else 0

        # 对象高度和宽度
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        # 创建一个网格来表示对象
        grid = np.zeros((height, width), dtype=int)
        color_map = {}  # 保存位置和颜色的映射

        # 填充网格
        for val, (r, c) in obj:
            grid[r - min_r, c - min_c] = 1  # 标记位置
            color_map[(r - min_r, c - min_c)] = val  # 保存颜色

        # 旋转网格
        if degrees == 90:
            rotated_grid = np.rot90(grid, k=1)
        elif degrees == 180:
            rotated_grid = np.rot90(grid, k=2)
        elif degrees == 270:
            rotated_grid = np.rot90(grid, k=3)
        else:
            rotated_grid = grid.copy()

        # 创建旋转后的对象
        rotated_obj = []
        rot_height, rot_width = rotated_grid.shape

        for r in range(rot_height):
            for c in range(rot_width):
                if rotated_grid[r, c] == 1:
                    # 找出该位置对应于原网格中的哪个位置
                    if degrees == 90:
                        orig_r, orig_c = c, height - 1 - r
                    elif degrees == 180:
                        orig_r, orig_c = height - 1 - r, width - 1 - c
                    elif degrees == 270:
                        orig_r, orig_c = width - 1 - c, r
                    else:
                        orig_r, orig_c = r, c

                    # 获取原始颜色
                    color = color_map.get((orig_r, orig_c), 1)  # 默认为1
                    rotated_obj.append((color, (r, c)))

        return frozenset(rotated_obj)

    def _mirror_object(self, obj, direction):
        """镜像对象"""
        # 获取对象边界
        min_r = min([r for _, (r, c) in obj]) if obj else 0
        min_c = min([c for _, (r, c) in obj]) if obj else 0
        max_r = max([r for _, (r, c) in obj]) if obj else 0
        max_c = max([c for _, (r, c) in obj]) if obj else 0

        # 对象高度和宽度
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        # 创建镜像对象
        mirrored_obj = []

        for val, (r, c) in obj:
            if direction == "horizontal":
                # 水平镜像 (x坐标不变，y坐标镜像)
                new_r = r
                new_c = max_c - (c - min_c)
            elif direction == "vertical":
                # 垂直镜像 (y坐标不变，x坐标镜像)
                new_r = max_r - (r - min_r)
                new_c = c
            else:
                new_r, new_c = r, c

            mirrored_obj.append((val, (new_r, new_c)))

        return frozenset(mirrored_obj)

    def is_same_shape(self, other_obj_info):
        """判断是否与另一个对象具有相同的形状（忽略颜色和位置）"""
        return self.obj_000 == other_obj_info.obj_000

    def contains_object(self, other_obj_info):
        """判断是否包含另一个对象（作为子部分）"""
        # 检查other_obj是否为self的子集
        other_positions = set((r, c) for _, (r, c) in other_obj_info.original_obj)
        self_positions = set((r, c) for _, (r, c) in self.original_obj)

        return other_positions.issubset(self_positions)

    def matches_with_transformation(self, other_obj_info):
        """检查是否匹配另一个对象的任何变换形式（旋转或镜像）"""
        # 检查是否是相同形状
        if self.is_same_shape(other_obj_info):
            return True, "same_shape", None

        # 检查旋转变体
        for rot_name, rot_obj in self.rotated_variants:
            if rot_obj and shift_pure_obj_to_0_0_0(rot_obj) == other_obj_info.obj_000:
                return True, "rotation", rot_name

        # 检查镜像变体
        for mirror_name, mirror_obj in self.mirrored_variants:
            if mirror_obj and shift_pure_obj_to_0_0_0(mirror_obj) == other_obj_info.obj_000:
                return True, "mirror", mirror_name

        # 检查镜像+旋转的复合变换
        for mirror_name, mirror_obj in self.mirrored_variants:
            if not mirror_obj:
                continue

            mirror_obj_info = ObjInfo(
                self.pair_id, self.in_or_out,
                mirror_obj, self.obj_params, self.grid_hw, self.background
            )

            for rot_name, rot_obj in mirror_obj_info.rotated_variants:
                if rot_obj and shift_pure_obj_to_0_0_0(rot_obj) == other_obj_info.obj_000:
                    return True, "mirror+rotation", f"{mirror_name}+{rot_name}"

        return False, None, None

    def get_color_transformation(self, other_obj_info):
        """计算到另一个对象的颜色变换"""
        # 检查对象是否有相同的形状模式
        matches, transform_type, transform_name = self.matches_with_transformation(other_obj_info)

        if not matches:
            return None

        # 计算颜色映射
        color_mapping = {}

        # 考虑可能的旋转/镜像情况
        if transform_type == "same_shape":
            source_obj = self.obj_00
            target_obj = other_obj_info.obj_00
        elif transform_type == "rotation":
            source_obj = next((obj for name, obj in self.rotated_variants if name == transform_name), None)
            target_obj = other_obj_info.obj_00
        elif transform_type == "mirror":
            source_obj = next((obj for name, obj in self.mirrored_variants if name == transform_name), None)
            target_obj = other_obj_info.obj_00
        elif transform_type == "mirror+rotation":
            mirror_name, rot_name = transform_name.split('+')
            mirror_obj = next((obj for name, obj in self.mirrored_variants if name == mirror_name), None)
            if mirror_obj:
                mirror_obj_info = ObjInfo(
                    self.pair_id, self.in_or_out,
                    mirror_obj, self.obj_params, self.grid_hw, self.background
                )
                source_obj = next((obj for name, obj in mirror_obj_info.rotated_variants if name == rot_name), None)
            else:
                source_obj = None
            target_obj = other_obj_info.obj_00
        else:
            return None

        # 如果无法获取变换后的对象，返回None
        if not source_obj or not target_obj:
            return None

        # 构建颜色映射
        src_positions = {(r, c): val for val, (r, c) in source_obj}
        tgt_positions = {(r, c): val for val, (r, c) in target_obj}

        for pos in src_positions:
            if pos in tgt_positions:
                src_color = src_positions[pos]
                tgt_color = tgt_positions[pos]

                if src_color not in color_mapping:
                    color_mapping[src_color] = []

                color_mapping[src_color].append(tgt_color)

        # 计算主要颜色映射（众数）
        final_mapping = {}
        for src_color, tgt_colors in color_mapping.items():
            if tgt_colors:
                # 找出最常见的目标颜色
                color_counts = defaultdict(int)
                for c in tgt_colors:
                    color_counts[c] += 1

                most_common = max(color_counts.items(), key=lambda x: x[1])
                final_mapping[src_color] = most_common[0]

        return {
            "transform_type": transform_type,
            "transform_name": transform_name,
            "color_mapping": final_mapping
        }

    def get_positional_change(self, other_obj_info):
        """计算到另一个对象的位置变化"""
        # 获取中心点
        self_center = (
            (self.bounding_box[0] + self.bounding_box[2]) / 2,
            (self.bounding_box[1] + self.bounding_box[3]) / 2
        )

        other_center = (
            (other_obj_info.bounding_box[0] + other_obj_info.bounding_box[2]) / 2,
            (other_obj_info.bounding_box[1] + other_obj_info.bounding_box[3]) / 2
        )

        # 计算位移
        delta_row = other_center[0] - self_center[0]
        delta_col = other_center[1] - self_center[1]

        # 计算是否有规律的位移
        # 例如，是否向右移动了整个宽度
        width_shift = abs(delta_col) / self.width if self.width > 0 else 0
        height_shift = abs(delta_row) / self.height if self.height > 0 else 0

        # 检查是否是整数倍的位移
        is_width_multiple = abs(width_shift - round(width_shift)) < 0.1
        is_height_multiple = abs(height_shift - round(height_shift)) < 0.1

        # 判断位移类型
        position_change = {
            "delta_row": delta_row,
            "delta_col": delta_col,
            "width_shift": width_shift,
            "height_shift": height_shift,
            "is_width_multiple": is_width_multiple,
            "is_height_multiple": is_height_multiple
        }

        # 添加方向描述
        if abs(delta_row) > abs(delta_col) * 2:
            position_change["direction"] = "垂直"
            position_change["orientation"] = "上" if delta_row < 0 else "下"
        elif abs(delta_col) > abs(delta_row) * 2:
            position_change["direction"] = "水平"
            position_change["orientation"] = "左" if delta_col < 0 else "右"
        else:
            position_change["direction"] = "对角线"
            if delta_row < 0 and delta_col < 0:
                position_change["orientation"] = "左上"
            elif delta_row < 0 and delta_col > 0:
                position_change["orientation"] = "右上"
            elif delta_row > 0 and delta_col < 0:
                position_change["orientation"] = "左下"
            else:
                position_change["orientation"] = "右下"

        return position_change

    def to_dict(self):
        """转换为可序列化的字典表示"""
        return {
            "id": self.obj_id,
            "pair_id": self.pair_id,
            "in_or_out": self.in_or_out,
            "size": self.size,
            "height": self.height,
            "width": self.width,
            "main_color": self.main_color,
            "bounding_box": self.bounding_box,
            "color_ranking": self.color_ranking
        }


class ARCDiffAnalyzer:
    """
    增强型ARC差异网格分析器
    专注于基于对象形状的分析和变换识别
    """

    def __init__(self, debug=True, debug_dir="debug_output"):
        """
        初始化分析器

        Args:
            debug: 是否启用调试模式
            debug_dir: 调试信息输出目录
        """
        self.train_pairs = []  # 存储(input_grid, output_grid)对
        self.diff_pairs = []   # 存储(diff_in, diff_out)对
        self.oneInOut_mapping_rules = []  # 每对数据的映射规则
        self.common_patterns = {}  # 共有模式

        self.test_pairs = []

        # 调试相关
        self.debug = debug
        self.debug_dir = debug_dir
        if debug:
            os.makedirs(debug_dir, exist_ok=True)

        # 参数组合，用于对象提取
        self.param_combinations = [
            (True, True, False),
            (True, False, False),
            (False, False, False),
            (False, True, False)
        ]

        # 所有提取的对象，使用ObjInfo增强型数据结构
        self.all_objects = {
            'input': [],  # [(pair_id, ObjInfo), ...]
            'output': [], # [(pair_id, ObjInfo), ...]
            'diff_in': [], # [(pair_id, ObjInfo), ...]
            'diff_out': []  # [(pair_id, ObjInfo), ...]
        }

        # 对象形状库 - 用于识别常见形状
        self.shape_library = {}  # shape_id -> normalized_shape

    def add_train_pair(self, pair_id, input_grid, output_grid):
        """
        添加一对训练数据并计算差异网格

        Args:
            pair_id: 训练对ID
            input_grid: 输入网格
            output_grid: 输出网格
        """
        if self.debug:
            self._debug_print(f"处理 task pair ！训练对！ {pair_id}")
            self._debug_save_grid(input_grid, f"input_{pair_id}")
            self._debug_save_grid(output_grid, f"output_{pair_id}")

        # 确保网格是元组的元组格式
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)
        if isinstance(output_grid, list):
            output_grid = tuple(tuple(row) for row in output_grid)

        # 保存原始网格对
        self.train_pairs.append((input_grid, output_grid))

        # 计算差异网格，直接使用现有函数
        diff_in, diff_out = grid2grid_fromgriddiff(input_grid, output_grid)
        self.diff_pairs.append((diff_in, diff_out))

        if self.debug:
            self._debug_save_grid(diff_in, f"diff_in_{pair_id}")
            self._debug_save_grid(diff_out, f"diff_out_{pair_id}")
            self._debug_print(f"成功计算差异网格，pair_id={pair_id}")

        # 获取网格尺寸
        height_in, width_in = len(input_grid), len(input_grid[0])
        height_out, width_out = len(output_grid), len(output_grid[0])

        # 提取对象，使用现有函数
        input_objects = all_pureobjects_from_grid(
            self.param_combinations, pair_id, 'in', input_grid, [height_in, width_in]
        )
        output_objects = all_pureobjects_from_grid(
            self.param_combinations, pair_id, 'out', output_grid, [height_out, width_out]
        )

        # 转换为增强型对象信息
        input_obj_infos = [
            ObjInfo(pair_id, 'in', obj, obj_params=None, grid_hw=[height_in, width_in])
            for obj in input_objects
        ]

        output_obj_infos = [
            ObjInfo(pair_id, 'out', obj, obj_params=None, grid_hw=[height_out, width_out])
            for obj in output_objects
        ]

        if self.debug:
            self._debug_print(f"从输入网格提取了 {len(input_obj_infos)} 个对象")
            self._debug_print(f"从输出网格提取了 {len(output_obj_infos)} 个对象")
            self._debug_save_obj_infos(input_obj_infos, f"input_objects_{pair_id}")
            self._debug_save_obj_infos(output_obj_infos, f"output_objects_{pair_id}")

        # 为diff网格也提取对象
        if diff_in is not None and diff_out is not None:
            height_diff, width_diff = len(diff_in), len(diff_in[0])
            diff_in_objects = all_pureobjects_from_grid(
                self.param_combinations, pair_id, 'diff_in', diff_in, [height_diff, width_diff]
            )
            diff_out_objects = all_pureobjects_from_grid(
                self.param_combinations, pair_id, 'diff_out', diff_out, [height_diff, width_diff]
            )

            # 转换为增强型对象信息
            diff_in_obj_infos = [
                ObjInfo(pair_id, 'diff_in', obj, obj_params=None, grid_hw=[height_diff, width_diff])
                for obj in diff_in_objects
            ]

            diff_out_obj_infos = [
                ObjInfo(pair_id, 'diff_out', obj, obj_params=None, grid_hw=[height_diff, width_diff])
                for obj in diff_out_objects
            ]

            if self.debug:
                self._debug_print(f"从差异输入网格提取了 {len(diff_in_obj_infos)} 个对象")
                self._debug_print(f"从差异输出网格提取了 {len(diff_out_obj_infos)} 个对象")
                self._debug_save_obj_infos(diff_in_obj_infos, f"diff_in_objects_{pair_id}")
                self._debug_save_obj_infos(diff_out_obj_infos, f"diff_out_objects_{pair_id}")
        else:
            diff_in_obj_infos = []
            diff_out_obj_infos = []
            if self.debug:
                self._debug_print("差异网格为空")

        # 分析对象间的部分-整体关系
        self._analyze_part_whole_relationships(input_obj_infos)
        self._analyze_part_whole_relationships(output_obj_infos)

        # 存储提取的对象
        self.all_objects['input'].append((pair_id, input_obj_infos))
        self.all_objects['output'].append((pair_id, output_obj_infos))
        self.all_objects['diff_in'].append((pair_id, diff_in_obj_infos))
        self.all_objects['diff_out'].append((pair_id, diff_out_obj_infos))

        # 更新形状库
        self._update_shape_library(input_obj_infos + output_obj_infos)

        # 分析diff映射关系
        mapping_rule = self.analyze_diff_mapping(
            pair_id, input_grid, output_grid, diff_in, diff_out,
            input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos
        )

        self.oneInOut_mapping_rules.append(mapping_rule)

        if self.debug:
            self._debug_save_json(mapping_rule, f"mapping_rule_{pair_id}")
            self._debug_print(f"完成训练对 {pair_id} 的分析")

    def _analyze_part_whole_relationships(self, obj_infos):
        """分析对象间的部分-整体关系"""
        # 对象按大小排序
        sorted_objs = sorted(obj_infos, key=lambda x: x.size, reverse=True)

        # 分析包含关系
        for i, larger_obj in enumerate(sorted_objs):
            for j, smaller_obj in enumerate(sorted_objs):
                if i != j and smaller_obj.size < larger_obj.size:
                    if larger_obj.contains_object(smaller_obj):
                        larger_obj.has_parts.append(smaller_obj.obj_id)
                        smaller_obj.is_part_of.append(larger_obj.obj_id)

    def _update_shape_library(self, obj_infos):
        """更新形状库，记录所有唯一形状"""
        for obj_info in obj_infos:
            shape_key = hash(obj_info.obj_000)
            if shape_key not in self.shape_library:
                self.shape_library[shape_key] = {
                    "normalized_shape": obj_info.obj_000,
                    "first_seen": obj_info.obj_id,
                    "count": 1,
                    "occurrences": [obj_info.obj_id]
                }
            else:
                self.shape_library[shape_key]["count"] += 1
                self.shape_library[shape_key]["occurrences"].append(obj_info.obj_id)

    def analyze_diff_mapping(self, pair_id, input_grid, output_grid, diff_in, diff_out,
                             input_obj_infos, output_obj_infos, diff_in_obj_infos, diff_out_obj_infos):
        """
        分析差异网格之间的映射关系，基于形状匹配和变换识别

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
            self._debug_print(f"开始分析差异映射关系，pair_id={pair_id}")

        mapping_rule = {
            "pair_id": pair_id,
            "object_mappings": [],
            "shape_transformations": [],
            "color_mappings": {},
            "position_changes": [],
            "part_whole_relationships": []
        }

        # 准备分析diff对象间的映射
        if not diff_in_obj_infos or not diff_out_obj_infos:
            if self.debug:
                self._debug_print("差异对象为空，无法分析映射")
            return mapping_rule

        # 1. 基于形状匹配寻找对象映射
        object_mappings = self._find_object_mappings_by_shape(diff_in_obj_infos, diff_out_obj_infos)

        if self.debug:
            self._debug_print(f"找到 {len(object_mappings)} 个基于形状的对象匹配")
            self._debug_save_json(object_mappings, f"shape_mappings_{pair_id}")

        # 2. 分析每个映射的变换
        for in_obj, out_obj, match_info in object_mappings:
            # 分析颜色变换
            color_transformation = in_obj.get_color_transformation(out_obj)

            # 分析位置变换
            position_change = in_obj.get_positional_change(out_obj)

            # 分析部分-整体关系变换
            part_whole_change = self._analyze_part_whole_change(in_obj, out_obj)

            if self.debug:
                self._debug_print(f"对象映射: {in_obj.obj_id} -> {out_obj.obj_id}, "
                               f"变换类型={match_info['transform_type']}")

            # 添加到映射规则
            mapping_rule["object_mappings"].append({
                "diff_in_object": in_obj.to_dict(),
                "diff_out_object": out_obj.to_dict(),
                "match_info": match_info
            })

            # 记录形状变换
            mapping_rule["shape_transformations"].append({
                "in_obj_id": in_obj.obj_id,
                "out_obj_id": out_obj.obj_id,
                "transform_type": match_info["transform_type"],
                "transform_name": match_info["transform_name"],
                "confidence": match_info["confidence"]
            })

            # 记录颜色映射
            if color_transformation and color_transformation.get("color_mapping"):
                for from_color, to_color in color_transformation["color_mapping"].items():
                    if from_color not in mapping_rule["color_mappings"]:
                        mapping_rule["color_mappings"][from_color] = to_color

                    if self.debug:
                        self._debug_print(f"颜色映射: {from_color} -> {to_color}")

            # 记录位置变化
            mapping_rule["position_changes"].append({
                "in_obj_id": in_obj.obj_id,
                "out_obj_id": out_obj.obj_id,
                "delta_row": position_change["delta_row"],
                "delta_col": position_change["delta_col"],
                "direction": position_change.get("direction"),
                "orientation": position_change.get("orientation")
            })

            # 记录部分-整体关系变化
            if part_whole_change:
                mapping_rule["part_whole_relationships"].append(part_whole_change)

        if self.debug:
            self._debug_print(f"完成差异映射分析，发现 {len(mapping_rule['color_mappings'])} 个颜色映射，"
                           f"{len(mapping_rule['position_changes'])} 个位置变化")

        return mapping_rule

    def _find_object_mappings_by_shape(self, diff_in_obj_infos, diff_out_obj_infos):
        """
        基于形状匹配寻找对象映射

        Returns:
            列表 [(in_obj, out_obj, match_info), ...]
        """
        mappings = []

        # 为每个输入对象找到最匹配的输出对象
        for in_obj in diff_in_obj_infos:
            best_match = None
            best_match_info = None
            best_confidence = -1

            for out_obj in diff_out_obj_infos:
                # 检查是否匹配任何变换形式
                matches, transform_type, transform_name = in_obj.matches_with_transformation(out_obj)

                if matches:
                    # 计算匹配置信度
                    confidence = 1 #self._calculate_transformation_confidence(
                        # in_obj, out_obj, transform_type, transform_name                    )

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = out_obj
                        best_match_info = {
                            "transform_type": transform_type,
                            "transform_name": transform_name,
                            "confidence": confidence
                        }

            if best_match and best_confidence > 0:
                mappings.append((in_obj, best_match, best_match_info))

        return mappings

    def _calculate_transformation_confidence(self, in_obj, out_obj, transform_type, transform_name):
        """计算变换匹配的置信度"""
        # 基础置信度
        confidence = 0.7  # 初始置信度

        # 对于完全相同形状，增加置信度
        if transform_type == "same_shape":
            confidence += 0.3

        # 根据对象大小调整置信度
        # 较大对象的匹配更可靠
        size_factor = min(0.2, in_obj.size / 100)  # 最多增加0.2
        confidence += size_factor

        # 根据颜色复杂度调整置信度
        # 具有多种颜色的对象匹配更可靠
        color_count = len(in_obj.color_ranking)
        color_factor = min(0.1, (color_count - 1) * 0.02)  # 最多增加0.1
        confidence += color_factor

        # 确保置信度在[0,1]范围内
        return min(1.0, max(0.0, confidence))

    def _analyze_part_whole_change(self, in_obj, out_obj):
        """分析对象间部分-整体关系的变化"""
        # 对于输入对象，获取其所属的整体对象
        in_whole_objects = in_obj.is_part_of

        # 对于输出对象，获取其所属的整体对象
        out_whole_objects = out_obj.is_part_of

        # 对于输入对象，获取其包含的部分对象
        in_part_objects = in_obj.has_parts

        # 对于输出对象，获取其包含的部分对象
        out_part_objects = out_obj.has_parts

        # 分析关系变化
        return {
            "in_obj_id": in_obj.obj_id,
            "out_obj_id": out_obj.obj_id,
            "in_is_part_of": in_whole_objects,
            "out_is_part_of": out_whole_objects,
            "in_has_parts": in_part_objects,
            "out_has_parts": out_part_objects,
            "gained_parts": [p for p in out_part_objects if p not in in_part_objects],
            "lost_parts": [p for p in in_part_objects if p not in out_part_objects],
            "became_part_of": [w for w in out_whole_objects if w not in in_whole_objects],
            "no_longer_part_of": [w for w in in_whole_objects if w not in out_whole_objects]
        }

    def analyze_common_patterns(self):
        """
        分析多对训练数据的共有模式

        Returns:
            共有模式字典
        """
        if self.debug:
            self._debug_print("开始分析共有模式")

        if not self.oneInOut_mapping_rules:
            return {}

        # 分析共有的形状变换模式
        common_shape_transformations = self._find_common_shape_transformations()

        # 分析共有的颜色映射模式
        common_color_mappings = self._find_common_color_mappings()

        # 分析共有的位置变化模式
        common_position_changes = self._find_common_position_changes()

        # 分析共有的部分-整体关系模式
        common_part_whole_patterns = self._find_common_part_whole_patterns()

        self.common_patterns = {
            "shape_transformations": common_shape_transformations,
            "color_mappings": common_color_mappings,
            "position_changes": common_position_changes,
            "part_whole_patterns": common_part_whole_patterns
        }

        if self.debug:
            self._debug_save_json(self.common_patterns, "common_patterns")
            self._debug_print(f"找到 {len(common_shape_transformations)} 个共有形状变换模式")
            self._debug_print(f"找到 {len(common_color_mappings.get('mappings', {}))} 个共有颜色映射")
            self._debug_print(f"找到 {len(common_position_changes)} 个共有位置变化模式")

        return self.common_patterns

    def _find_common_shape_transformations(self):
        """寻找共有的形状变换模式"""
        # 收集所有形状变换
        all_transformations = []
        for rule in self.oneInOut_mapping_rules:
            for transform in rule.get("shape_transformations", []):
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
                avg_confidence = sum(t["confidence"] for t in transforms) / len(transforms)

                common_transforms.append({
                    "transform_type": t_type,
                    "transform_name": t_name,
                    "count": len(transforms),
                    "confidence": avg_confidence,
                    "examples": [t["in_obj_id"] + "->" + t["out_obj_id"] for t in transforms]
                })

                if self.debug:
                    self._debug_print(f"发现共有形状变换模式: {t_type}({t_name}), 出现 {len(transforms)} 次, "
                                   f"置信度: {avg_confidence:.2f}")

        # 按出现次数排序
        return sorted(common_transforms, key=lambda x: x["count"], reverse=True)

    def _find_common_color_mappings(self):
        """寻找共有的颜色映射模式"""
        all_mappings = {}

        # 收集所有颜色映射
        for rule in self.oneInOut_mapping_rules:
            for from_color, to_color in rule["color_mappings"].items():
                key = (from_color, to_color)
                if key not in all_mappings:
                    all_mappings[key] = 0
                all_mappings[key] += 1

        # 找出共有的映射
        common_mappings = {}
        total_examples = len(self.oneInOut_mapping_rules)

        for (from_color, to_color), count in all_mappings.items():
            if count > 1:  # 至少在两个示例中出现
                confidence = count / total_examples
                common_mappings[from_color] = {
                    "to_color": to_color,
                    "count": count,
                    "confidence": confidence
                }

                if self.debug:
                    self._debug_print(f"发现共有颜色映射: {from_color} -> {to_color}, 置信度: {confidence:.2f}")

        # 分析颜色变化模式
        color_patterns = []

        # 检查是否有统一的颜色偏移
        offsets = [to["to_color"] - from_color for from_color, to in common_mappings.items()]
        if offsets and len(set(offsets)) == 1:  # 所有颜色有相同的偏移
            color_patterns.append({
                "type": "color_offset",
                "offset": offsets[0],
                "confidence": 1.0
            })

            if self.debug:
                self._debug_print(f"发现统一颜色偏移模式: +{offsets[0]}")

        return {
            "mappings": common_mappings,
            "patterns": color_patterns
        }

    def _find_common_position_changes(self):
        """寻找共有的位置变化模式"""
        # 收集所有位置变化
        all_changes = []
        for rule in self.oneInOut_mapping_rules:
            for change in rule.get("position_changes", []):
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

        # 分析位移组
        for (dr, dc), changes in delta_groups.items():
            if len(changes) >= 2:  # 至少出现两次
                common_changes.append({
                    "type": "absolute_position",
                    "delta_row": dr,
                    "delta_col": dc,
                    "count": len(changes),
                    "confidence": len(changes) / len(all_changes)
                })

        # 分析方向组
        for (direction, orientation), changes in direction_groups.items():
            if len(changes) >= 2:  # 至少出现两次
                common_changes.append({
                    "type": "directional",
                    "direction": direction,
                    "orientation": orientation,
                    "count": len(changes),
                    "confidence": len(changes) / len(all_changes)
                })

        # 按出现次数排序
        return sorted(common_changes, key=lambda x: x["count"], reverse=True)

    def _find_common_part_whole_patterns(self):
        """寻找共有的部分-整体关系模式"""
        # 收集所有部分-整体关系变化
        all_changes = []
        for rule in self.oneInOut_mapping_rules:
            for change in rule.get("part_whole_relationships", []):
                all_changes.append(change)

        if not all_changes:
            return []

        # 分析常见变化模式
        patterns = []

        # 检查对象是否常常获得或失去部分
        gain_counts = defaultdict(int)
        loss_counts = defaultdict(int)

        for change in all_changes:
            if change.get("gained_parts"):
                gain_counts["gained_parts"] += 1
            if change.get("lost_parts"):
                loss_counts["lost_parts"] += 1
            if change.get("became_part_of"):
                gain_counts["became_part_of"] += 1
            if change.get("no_longer_part_of"):
                loss_counts["no_longer_part_of"] += 1

        # 添加常见模式
        total = len(all_changes)
        for change_type, count in gain_counts.items():
            if count >= 2:  # 至少出现两次
                patterns.append({
                    "type": change_type,
                    "count": count,
                    "confidence": count / total
                })

        for change_type, count in loss_counts.items():
            if count >= 2:  # 至少出现两次
                patterns.append({
                    "type": change_type,
                    "count": count,
                    "confidence": count / total
                })

        return sorted(patterns, key=lambda x: x["count"], reverse=True)

    def apply_common_patterns(self, input_grid):
        """
        将共有模式应用到新的输入网格

        Args:
            input_grid: 输入网格

        Returns:
            预测的输出网格
        """
        if self.debug:
            self._debug_print("开始应用共有模式到测试输入")
            self._debug_save_grid(input_grid, "test_input")

        if not self.common_patterns:
            self.analyze_common_patterns()

        # 确保输入网格是元组格式
        if isinstance(input_grid, list):
            input_grid = tuple(tuple(row) for row in input_grid)

        # 获取网格尺寸
        height, width = len(input_grid), len(input_grid[0])

        # 提取输入网格中的对象
        input_objects = all_pureobjects_from_grid(
            self.param_combinations, -1, 'test_in', input_grid, [height, width]
        )

        # 转换为增强型对象信息
        input_obj_infos = [
            ObjInfo(-1, 'test_in', obj, obj_params=None, grid_hw=[height, width])
            for obj in input_objects
        ]

        if self.debug:
            self._debug_print(f"从测试输入提取了 {len(input_obj_infos)} 个对象")
            self._debug_save_obj_infos(input_obj_infos, "test_input_objects")

        # 创建输出网格（初始为输入的副本）
        output_grid = [list(row) for row in input_grid]

        # 应用颜色映射
        if "color_mappings" in self.common_patterns:
            color_mappings = self.common_patterns["color_mappings"].get("mappings", {})
            for from_color, mapping in color_mappings.items():
                to_color = mapping["to_color"]
                cells_changed = 0
                for i in range(height):
                    for j in range(width):
                        if input_grid[i][j] == from_color:
                            output_grid[i][j] = to_color
                            cells_changed += 1

                if self.debug and cells_changed > 0:
                    self._debug_print(f"应用颜色映射: {from_color} -> {to_color}, 改变了 {cells_changed} 个单元格")

        # 应用颜色模式
        if "color_mappings" in self.common_patterns:
            for pattern in self.common_patterns["color_mappings"].get("patterns", []):
                if pattern["type"] == "color_offset" and pattern["confidence"] > 0.5:
                    offset = pattern["offset"]
                    cells_changed = 0
                    for i in range(height):
                        for j in range(width):
                            if input_grid[i][j] != 0:  # 不处理背景
                                output_grid[i][j] = (input_grid[i][j] + offset) % 10  # 假设颜色范围是0-9
                                cells_changed += 1

                    if self.debug and cells_changed > 0:
                        self._debug_print(f"应用颜色偏移: +{offset}, 改变了 {cells_changed} 个单元格")

        # 应用位置变化
        position_changes = self.common_patterns.get("position_changes", [])
        if position_changes:
            # 找到最高置信度的位置变化
            best_position_change = max(position_changes, key=lambda x: x["confidence"])

            if best_position_change["type"] == "absolute_position" and best_position_change["confidence"] > 0.5:
                # 应用绝对位置变化
                dr, dc = best_position_change["delta_row"], best_position_change["delta_col"]

                # 创建临时网格保存结果
                temp_grid = [[0 for _ in range(width)] for _ in range(height)]
                cells_moved = 0

                # 对每个对象应用位置变化
                for obj_info in input_obj_infos:
                    obj_color = obj_info.main_color
                    # 对象中的每个像素
                    for _, (r, c) in obj_info.original_obj:
                        nr, nc = int(r + dr), int(c + dc)
                        if 0 <= nr < height and 0 <= nc < width:
                            temp_grid[nr][nc] = obj_color
                            cells_moved += 1

                # 合并结果
                for i in range(height):
                    for j in range(width):
                        if temp_grid[i][j] != 0:  # 只覆盖非零值
                            output_grid[i][j] = temp_grid[i][j]

                if self.debug and cells_moved > 0:
                    self._debug_print(f"应用位置变化: ({dr}, {dc}), 移动了 {cells_moved} 个单元格")

        # 应用形状变换
        shape_transformations = self.common_patterns.get("shape_transformations", [])
        if shape_transformations:
            # 找到最高置信度的形状变换
            best_shape_transform = max(shape_transformations, key=lambda x: x["confidence"])

            if best_shape_transform["confidence"] > 0.5:
                # 根据最佳形状变换类型实施变换
                transform_type = best_shape_transform["transform_type"]
                transform_name = best_shape_transform["transform_name"]

                if transform_type in ["rotation", "mirror"]:
                    temp_grid = [[0 for _ in range(width)] for _ in range(height)]
                    cells_transformed = 0

                    for obj_info in input_obj_infos:
                        # 应用相应的变换
                        transformed_obj = None

                        if transform_type == "rotation":
                            # 查找匹配的旋转变体
                            for name, obj in obj_info.rotated_variants:
                                if name == transform_name:
                                    transformed_obj = obj
                                    break
                        elif transform_type == "mirror":
                            # 查找匹配的镜像变体
                            for name, obj in obj_info.mirrored_variants:
                                if name == transform_name:
                                    transformed_obj = obj
                                    break

                        if transformed_obj:
                            # 在原始位置应用变换
                            for val, (r, c) in transformed_obj:
                                # 调整到原始对象位置
                                nr = r + obj_info.min_row
                                nc = c + obj_info.min_col
                                if 0 <= nr < height and 0 <= nc < width:
                                    temp_grid[nr][nc] = val
                                    cells_transformed += 1

                    # 合并结果
                    for i in range(height):
                        for j in range(width):
                            if temp_grid[i][j] != 0:  # 只覆盖非零值
                                output_grid[i][j] = temp_grid[i][j]

                    if self.debug and cells_transformed > 0:
                        self._debug_print(f"应用形状变换: {transform_type}({transform_name}), "
                                      f"变换了 {cells_transformed} 个单元格")

        if self.debug:
            self._debug_save_grid(output_grid, "test_output_predicted")
            self._debug_print("完成测试预测")

        return output_grid

    # ==== 调试辅助方法 ====



    def _debug_print(self, message):
        """输出调试信息"""
        if self.debug:
            print(f"[调试] {message}")

    def _debug_save_grid(self, grid, name):
        """保存网格到文件并可视化"""
        if not self.debug:
            return

        # 保存网格数据
        grid_file = os.path.join(self.debug_dir, f"{name}.json")
        with open(grid_file, 'w') as f:
            if isinstance(grid, tuple):
                # 转换元组为列表
                grid_list = [list(row) for row in grid]
                json.dump(grid_list, f, indent=2)
            else:
                json.dump(grid, f, indent=2)

        # 可视化网格
        if grid is not None:
            plt.figure(figsize=(5, 5))
            grid_array = np.array(grid)
            # 处理None值
            if isinstance(grid_array[0, 0], type(None)) or grid_array.dtype == object:
                # 创建一个新数组，将None替换为-1
                vis_array = np.zeros(grid_array.shape, dtype=np.int8)
                for i in range(grid_array.shape[0]):
                    for j in range(grid_array.shape[1]):
                        if grid_array[i, j] is not None:
                            vis_array[i, j] = grid_array[i, j]
                        else:
                            vis_array[i, j] = -1
                plt.imshow(vis_array, cmap='tab10')
            else:
                plt.imshow(grid_array, cmap='tab10')
            plt.colorbar()
            plt.title(name)
            plt.savefig(os.path.join(self.debug_dir, f"{name}.png"))
            plt.close()

    def _debug_save_obj_infos(self, obj_infos, name):
        """保存对象信息集合"""
        if not self.debug:
            return

        obj_file = os.path.join(self.debug_dir, f"{name}.json")

        # 转换对象为可序列化形式
        serializable_objects = []
        for i, obj_info in enumerate(obj_infos):
            obj_dict = obj_info.to_dict()
            obj_dict["index"] = i
            serializable_objects.append(obj_dict)

        # 使用统一工具保存JSON
        JSONSerializer.save_json(serializable_objects, obj_file)

    def _debug_save_json(self, data, name):
        """保存JSON数据"""
        if not self.debug:
            return

        json_file = os.path.join(self.debug_dir, f"{name}.json")
        JSONSerializer.save_json(data, json_file)

    def _debug_append_json(self, data, name):
        """追加JSON数据到文件"""
        if not self.debug:
            return

        json_file = os.path.join(self.debug_dir, f"{name}.json")
        JSONSerializer.append_json(data, json_file)




class JSONSerializer:
    """统一的JSON序列化工具类"""

    @staticmethod
    def convert_to_serializable(obj: Any) -> Any:
        """将任何对象递归转换为JSON可序列化格式"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (set, frozenset)):
            return list(JSONSerializer.convert_to_serializable(item) for item in obj)
        elif isinstance(obj, (list, tuple)):
            return [JSONSerializer.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): JSONSerializer.convert_to_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            # 如果对象有to_dict方法，调用它
            return JSONSerializer.convert_to_serializable(obj.to_dict())
        elif hasattr(obj, '__dict__'):
            # 处理自定义类
            return JSONSerializer.convert_to_serializable(obj.__dict__)
        else:
            # 对于其他类型，转为字符串
            try:
                json.dumps(obj)  # 测试是否可序列化
                return obj
            except (TypeError, OverflowError):
                return str(obj)

    @staticmethod
    def save_json(data: Any, filepath: str, indent: int = 2) -> None:
        """将数据保存为JSON文件"""
        serializable_data = JSONSerializer.convert_to_serializable(data)

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=indent)

    @staticmethod
    def append_json(data: Any, filepath: str, indent: int = 2) -> None:
        """追加数据到JSON文件"""
        serializable_data = JSONSerializer.convert_to_serializable(data)

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # 读取现有数据
        existing_data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []

        # 追加新数据
        if isinstance(existing_data, list):
            existing_data.append(serializable_data)
        else:
            existing_data = [existing_data, serializable_data]

        # 保存更新后的数据
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=indent)

    @staticmethod
    def load_json(filepath: str) -> Any:
        """从JSON文件加载数据"""
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            return json.load(f)
