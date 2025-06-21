"""
ARC多维度关系库系统

用于构建和管理多个维度的对象属性和关系库，支持跨数据对的模式挖掘。
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Any, Optional, Union
import hashlib
import json

class ARCRelationshipLibraries:
    """多维度ARC关系库管理系统"""

    def __init__(self, debug=True, debug_print=None):
        """
        初始化关系库系统

        Args:
            debug: 是否启用调试模式
            debug_print: 调试打印函数
        """
        self.debug = True # debug
        self.debug_print = debug_print
        self._cached_all_objects = None  # 添加缓存属性

        # 基础属性库
        self.shape_library = {}  # shape_hash -> {详细信息}
        self.color_library = {}  # color -> {详细信息，出现次数，相关对象}

        # 对象索引库
        self.objects_by_pair = defaultdict(lambda: {"input": [], "output": [], "diff_in": [], "diff_out": []})
        # self.objects_by_pair = defaultdict(lambda: {"input": [], "output": []})  # pair_id -> {input/output -> [obj_ids]}
        self.objects_by_shape = defaultdict(list)  # shape_hash -> [(pair_id, io_type, obj_id), ...]
        self.objects_by_color = defaultdict(list)  # color -> [(pair_id, io_type, obj_id), ...]
        self.objects_by_size = defaultdict(list)   # size -> [(pair_id, io_type, obj_id), ...]
        self.objects_by_position = defaultdict(list) # (row, col) -> [(pair_id, io_type, obj_id), ...]
        #!
        self.objects_by_if_adj_border = defaultdict(list)
        self.objects_by_if_color_is_bg = defaultdict(list)
        self.objects_by_if_color_is_bg_if_sizeMax = defaultdict(list)


        # 操作库
        self.removed_objects = []  # [(pair_id, obj_id, object_info), ...]
        self.added_objects = []    # [(pair_id, obj_id, object_info), ...]
        self.preserved_objects = []  # [(pair_id, input_obj_id, output_obj_id), ...]
        self.transformed_objects = []  # [(pair_id, input_obj_id, output_obj_id, transform_info), ...]

        # 关系映射库
        self.shape_to_operation_map = defaultdict(list)  # shape_hash -> [(pair_id, operation, obj_id), ...]
        self.color_to_operation_map = defaultdict(list)  # color -> [(pair_id, operation, obj_id), ...]
        self.position_to_operation_map = defaultdict(list)  # position -> [(pair_id, operation, obj_id), ...]

        # 转换规则库
        self.shape_transformation_rules = []  # [{rule_info}, ...]
        self.color_transformation_rules = []  # [{rule_info}, ...]
        self.position_transformation_rules = [] # [{rule_info}, ...]
        self.complex_transformation_rules = []  # [{rule_info}, ...]

        # 条件映射库
        self.condition_to_effect_map = []  # [{condition, effect, confidence, pairs}, ...]

        # 相似对象分组
        self.similar_objects_groups = []  # [[(pair_id, io_type, obj_id), ...], ...]

        # 属性关联分析
        self.attribute_correlations = defaultdict(lambda: defaultdict(int))  # (attr1, value1) -> {(attr2, value2) -> count}

        # 统计数据
        self.total_pairs = 0
        self.total_objects = 0
        self.statistics = {
            "objects_per_pair_avg": 0,
            "most_common_shapes": [],
            "most_common_colors": [],
            "most_common_operations": {}
        }

    def reset(self):
        """重置所有库"""
        self._cached_all_objects = None  # 清除缓存
        self.__init__(self.debug, self.debug_print)
        # self._cached_all_objects = None  # 清除缓存

    def build_libraries_from_data(self, oneInOut_mapping_rules, all_objects):
        """
        从分析结果构建所有关系库

        Args:
            oneInOut_mapping_rules: 映射规则列表
            all_objects: 所有对象信息
        """
        if self.debug:
            self.debug_print("\n\n\n\n开始构建多维度关系库...\n\n\n")

        self.total_pairs = len(oneInOut_mapping_rules)

        # 1. 首先处理所有对象，建立基础属性库
        self._build_object_attribute_libraries(all_objects)

        # 2. 处理所有映射规则，建立操作库
        self._build_operation_libraries(oneInOut_mapping_rules)

        # 3. 建立属性-操作关系库
        self._build_attribute_operation_maps()

        # 4. 建立相似对象分组
        self._group_similar_objects()

        # 5. 分析属性相关性
        self._analyze_attribute_correlations()

        # 6. 计算统计数据
        self._calculate_statistics()

        if self.debug:
            self._log_library_statistics()

    def _build_object_attribute_libraries(self, all_objects):
        """
        构建对象属性库

        Args:
            all_objects: 所有对象信息 {'input': [(pair_id, [obj_infos])], 'output': [...]}
        """
        # # 处理输入对象
        # for pair_id, obj_infos in all_objects.get('input', []):
        #     for obj_info in obj_infos:
        #         self._process_single_object(pair_id, 'input', obj_info)

        # # 处理输出对象
        # for pair_id, obj_infos in all_objects.get('output', []):
        #     for obj_info in obj_infos:
        #         self._process_single_object(pair_id, 'output', obj_info)

        # # 处理差异网格对象（如果有）
        # for io_type in ['diff_in', 'diff_out']:
        #     for pair_id, obj_infos in all_objects.get(io_type, []):
        #         for obj_info in obj_infos:
        #             self._process_single_object(pair_id, io_type, obj_info)

        # 2. 处理所有对象，建立基础属性库
        valid_io_types = ['input', 'output', 'diff_in', 'diff_out']
        for io_type in valid_io_types:
            if io_type in all_objects:
                for pair_id, obj_infos in all_objects.get(io_type, []):
                    if self.debug and len(obj_infos) > 0:
                        self.debug_print(f"处理 {io_type} 类型的对象，pair_id={pair_id}，对象数量={len(obj_infos)}")
                    for obj_info in obj_infos:
                        self._process_single_object(pair_id, io_type, obj_info)


    def _process_single_object(self, pair_id, io_type, obj_info):
        """
        处理单个对象，更新相关库

        Args:
            pair_id: 数据对ID
            io_type: 'input'或'output'
            obj_info: 对象信息
        """
        # 获取对象ID和基本属性
        obj_id = obj_info.obj_id
        self.total_objects += 1

        if io_type not in self.objects_by_pair[pair_id]:
            if self.debug:
                self.debug_print(f"警告: 创建对象索引库中未预期的io_type: {io_type}")
            self.objects_by_pair[pair_id][io_type] = []

        # 添加到对象索引库
        self.objects_by_pair[pair_id][io_type].append(obj_id)

        # 处理形状
        try:
            shape_hash = self._get_shape_hash(obj_info)

            # 更新形状库
            if shape_hash not in self.shape_library:
                self.shape_library[shape_hash] = {
                    "normalized_shape": obj_info.obj_000,
                    "first_seen": obj_id,
                    "count": 1,
                    "occurrences": [(pair_id, io_type, obj_id)]
                }
            else:
                self.shape_library[shape_hash]["count"] += 1
                self.shape_library[shape_hash]["occurrences"].append((pair_id, io_type, obj_id))

            # 添加到形状索引
            self.objects_by_shape[shape_hash].append((pair_id, io_type, obj_id))

        except (AttributeError, TypeError) as e:
            if self.debug:
                self.debug_print(f"处理对象形状时出错: {e}, obj_id={obj_id}")

        # 处理颜色
        try:
            color = obj_info.main_color

            # 更新颜色库
            if color not in self.color_library:
                self.color_library[color] = {
                    "count": 1,
                    "occurrences": [(pair_id, io_type, obj_id)]
                }
            else:
                self.color_library[color]["count"] += 1
                self.color_library[color]["occurrences"].append((pair_id, io_type, obj_id))

            # 添加到颜色索引
            self.objects_by_color[color].append((pair_id, io_type, obj_id))

        except AttributeError:
            pass

        # 处理大小
        try:
            size = obj_info.size
            self.objects_by_size[size].append((pair_id, io_type, obj_id))
        except AttributeError:
            pass

        # 处理位置#!是否背景对象，是否靠近边界
        try:
            position = (obj_info.top, obj_info.left)
            self.objects_by_position[position].append((pair_id, io_type, obj_id))
        except AttributeError:
            pass

    def _build_operation_libraries(self, oneInOut_mapping_rules):
        """
        构建操作库

        Args:
            oneInOut_mapping_rules: 映射规则列表
        """
        for rule in oneInOut_mapping_rules:
            pair_id = rule.get("pair_id")

            # 检查是否有输入到输出的转换规则
            if "input_to_output_transformation" in rule:
                transform_rule = rule["input_to_output_transformation"]

                # 处理移除的对象
                for removed in transform_rule.get("removed_objects", []):
                    obj_id = removed.get("input_obj_id")
                    obj_info = removed.get("object", {})
                    self.removed_objects.append((pair_id, obj_id, obj_info))

                    # 更新形状-操作映射
                    shape_hash = self._get_shape_hash_from_dict(obj_info)
                    if shape_hash:
                        self.shape_to_operation_map[shape_hash].append((pair_id, "removed", obj_id))

                    # 更新颜色-操作映射
                    color = obj_info.get("main_color")
                    if color is not None:
                        self.color_to_operation_map[color].append((pair_id, "removed", obj_id))

                # 处理添加的对象
                for added in transform_rule.get("added_objects", []):
                    obj_id = added.get("output_obj_id")
                    obj_info = added.get("object", {})
                    self.added_objects.append((pair_id, obj_id, obj_info))

                    # 更新形状-操作映射
                    shape_hash = self._get_shape_hash_from_dict(obj_info)
                    if shape_hash:
                        self.shape_to_operation_map[shape_hash].append((pair_id, "added", obj_id))

                    # 更新颜色-操作映射
                    color = obj_info.get("main_color")
                    if color is not None:
                        self.color_to_operation_map[color].append((pair_id, "added", obj_id))

                # 处理保留的对象
                for preserved in transform_rule.get("preserved_objects", []):
                    input_obj_id = preserved.get("input_obj_id")
                    output_obj_id = preserved.get("output_obj_id")
                    obj_info = preserved.get("object", {})

                    self.preserved_objects.append((pair_id, input_obj_id, output_obj_id))

                    # 更新形状-操作映射
                    shape_hash = self._get_shape_hash_from_dict(obj_info)
                    if shape_hash:
                        self.shape_to_operation_map[shape_hash].append((pair_id, "preserved", input_obj_id))

                    # 更新颜色-操作映射
                    color = obj_info.get("main_color")
                    if color is not None:
                        self.color_to_operation_map[color].append((pair_id, "preserved", input_obj_id))

                # 处理变换的对象
                for modified in transform_rule.get("modified_objects", []):
                    input_obj_id = modified.get("input_obj_id")
                    output_obj_id = modified.get("output_obj_id")
                    transform_info = modified.get("transformation", {})

                    self.transformed_objects.append(
                        (pair_id, input_obj_id, output_obj_id, transform_info)
                    )

                    # 从rule中寻找相应的input对象信息
                    for obj in rule.get("weighted_objects", []):
                        if obj.get("obj_id") == input_obj_id and obj.get("type") == "in":
                            # 更新形状-操作映射
                            if "object" in obj:
                                shape_hash = self._get_shape_hash_from_dict(obj["object"])
                                if shape_hash:
                                    transform_type = transform_info.get("type", "modified")
                                    self.shape_to_operation_map[shape_hash].append((pair_id, transform_type, input_obj_id))
                            break

    def _build_attribute_operation_maps(self):
        """构建属性与操作之间的关系映射"""
        # 已经在_build_operation_libraries中构建了一部分
        # 这里可以添加更复杂的关系映射，如形状与颜色变化的关联

        # 分析形状与颜色变化之间的关系
        for pair_id, input_obj_id, output_obj_id, transform_info in self.transformed_objects:
            # 检查是否有颜色变换
            if "color_transform" in transform_info and "color_mapping" in transform_info["color_transform"]:
                color_mapping = transform_info["color_transform"]["color_mapping"]

                # 为每个颜色变化创建规则
                for from_color, to_color in color_mapping.items():
                    # 寻找输入对象的形状
                    input_shape = None
                    for shape_hash, objects in self.objects_by_shape.items():
                        if (pair_id, "input", input_obj_id) in objects:
                            input_shape = shape_hash
                            break

                    if input_shape:
                        # 记录颜色转换规则
                        rule = {
                            "pair_id": pair_id,
                            "input_obj_id": input_obj_id,
                            "output_obj_id": output_obj_id,
                            "shape_hash": input_shape,
                            "from_color": from_color,
                            "to_color": to_color,
                            "context": {"transform_type": transform_info.get("type")}
                        }
                        self.color_transformation_rules.append(rule)

    def _get_all_objects(self):
        """获取所有对象，使用缓存避免重复计算"""
        # 如果缓存不存在，则计算并缓存
        if self._cached_all_objects is None:
            if self.debug:
                self.debug_print("首次构建所有对象列表...")

            self._cached_all_objects = []
            for pair_id, pair_data in self.objects_by_pair.items():
                for io_type, obj_ids in pair_data.items():
                    for obj_id in obj_ids:
                        self._cached_all_objects.append((pair_id, io_type, obj_id))

            if self.debug:
                self.debug_print(f"共收集到 {len(self._cached_all_objects)} 个对象")
        else:
            if self.debug:
                self.debug_print(f"使用缓存的对象列表 (对象数: {len(self._cached_all_objects)})")

        return self._cached_all_objects

    def _group_similar_objects(self):
        """将相似的对象分组"""
        if self.debug:
            self.debug_print("开始对象分组...")

        try:
            # 基于形状分组
            grouped_by_shape = defaultdict(list)

            for shape_hash, objects in self.objects_by_shape.items():
                if len(objects) >= 2:  # 只考虑至少出现两次的形状
                    grouped_by_shape[shape_hash] = objects

            # 添加到相似对象分组
            for shape_hash, objects in grouped_by_shape.items():
                if len(objects) >= 2:
                    self.similar_objects_groups.append({
                        "type": "shape_based",
                        "shape_hash": shape_hash,
                        "shape_library_key": shape_hash,
                        "objects": objects,
                        "count": len(objects)
                    })

            if self.debug:
                self.debug_print(f"已创建 {len(grouped_by_shape)} 个基于形状的分组")

            # 基于颜色和形状组合分组
            grouped_by_color_shape = defaultdict(list)

            # 安全地构建所有对象的列表
            all_objects = self._get_all_objects()
            # all_objects = []
            # for pair_id, pair_data in self.objects_by_pair.items():
            #     for io_type, obj_ids in pair_data.items():
            #         for obj_id in obj_ids:
            #             all_objects.append((pair_id, io_type, obj_id))

            if self.debug:
                self.debug_print(f"处理 {len(all_objects)} 个对象的颜色和形状分组")

            # 添加进度报告
            total = len(all_objects)
            for idx, (pair_id, io_type, obj_id) in enumerate(all_objects):
                if self.debug and idx % 100 == 0 and idx > 0:
                    self.debug_print(f"处理进度: {idx}/{total} ({idx/total*100:.1f}%)")

                # 获取对象的形状和颜色
                shape_hash = None
                color = None

                # 查找形状
                for sh, objects in self.objects_by_shape.items():
                    if (pair_id, io_type, obj_id) in objects:
                        shape_hash = sh
                        break

                # 查找颜色
                for c, objects in self.objects_by_color.items():
                    if (pair_id, io_type, obj_id) in objects:
                        color = c
                        break

                if shape_hash is not None and color is not None:
                    key = (shape_hash, color)
                    grouped_by_color_shape[key].append((pair_id, io_type, obj_id))

            # 添加到相似对象分组
            for (shape_hash, color), objects in grouped_by_color_shape.items():
                if len(objects) >= 2:
                    self.similar_objects_groups.append({
                        "type": "shape_color_based",
                        "shape_hash": shape_hash,
                        "color": color,
                        "objects": objects,
                        "count": len(objects)
                    })

            if self.debug:
                self.debug_print(f"已创建 {len(grouped_by_color_shape)} 个基于形状和颜色的分组")
                self.debug_print(f"对象分组完成，总共 {len(self.similar_objects_groups)} 个分组")

        except Exception as e:
            if self.debug:
                self.debug_print(f"对象分组时发生错误: {e}")
                import traceback
                self.debug_print(traceback.format_exc())

    def _analyze_attribute_correlations(self):
        """分析不同属性之间的相关性"""

        if self.debug:
            self.debug_print("开始分析属性相关性...")

        try:
            # 收集所有对象
            all_objects = self._get_all_objects()
            # all_objects = []
            # for pair_id, pair_data in self.objects_by_pair.items():
            #     for io_type, obj_ids in pair_data.items():
            #         for obj_id in obj_ids:
            #             all_objects.append((pair_id, io_type, obj_id))

            total_objects = len(all_objects)
            if self.debug:
                self.debug_print(f"找到 {total_objects} 个对象需要分析")

            # 创建操作类型查找索引，提高性能
            removed_index = {(p, o): True for p, o, _ in self.removed_objects}
            added_index = {(p, o): True for p, o, _ in self.added_objects}
            preserved_in_index = {(p, i_o): o_o for p, i_o, o_o in self.preserved_objects}
            preserved_out_index = {(p, o_o): i_o for p, i_o, o_o in self.preserved_objects}
            transformed_in_index = {(p, i_o): o_o for p, i_o, o_o, _ in self.transformed_objects}
            transformed_out_index = {(p, o_o): i_o for p, i_o, o_o, _ in self.transformed_objects}

            # 处理计数器
            processed = 0

            # 对每个对象，分析属性关系
            for pair_id, io_type, obj_id in all_objects:
                processed += 1
                if self.debug and processed % 1000 == 0:
                    self.debug_print(f"已处理: {processed}/{total_objects} ({processed/total_objects*100:.1f}%)")

                # 获取对象的各种属性
                attributes = {}

                # 查找形状 (优化: 直接从形状库中查找)
                for shape_hash, objects in self.objects_by_shape.items():
                    if (pair_id, io_type, obj_id) in objects:
                        attributes["shape"] = shape_hash
                        break

                # 查找颜色
                for color, objects in self.objects_by_color.items():
                    if (pair_id, io_type, obj_id) in objects:
                        attributes["color"] = color
                        break

                # 查找大小
                for size, objects in self.objects_by_size.items():
                    if (pair_id, io_type, obj_id) in objects:
                        attributes["size"] = size
                        break

                # 使用索引快速查找操作类型
                operation_type = None
                obj_key = (pair_id, obj_id)

                if io_type == "input":
                    if obj_key in removed_index:
                        operation_type = "removed"
                    elif obj_key in preserved_in_index:
                        operation_type = "preserved"
                    elif obj_key in transformed_in_index:
                        operation_type = "transformed"
                elif io_type == "output":
                    if obj_key in added_index:
                        operation_type = "added"
                    elif obj_key in preserved_out_index:
                        operation_type = "preserved_result"
                    elif obj_key in transformed_out_index:
                        operation_type = "transformed_result"

                if operation_type:
                    attributes["operation"] = operation_type

                # 记录属性相关性
                attribs = list(attributes.items())
                for i, (attr1, val1) in enumerate(attribs):
                    for attr2, val2 in attribs[i+1:]:
                        self.attribute_correlations[(attr1, val1)][(attr2, val2)] += 1
                        self.attribute_correlations[(attr2, val2)][(attr1, val1)] += 1

            if self.debug:
                correlation_count = sum(len(corrs) for corrs in self.attribute_correlations.values())
                self.debug_print(f"属性相关性分析完成，共发现 {correlation_count} 个属性相关")

        except Exception as e:
            if self.debug:
                self.debug_print(f"分析属性相关性时出错: {str(e)}")
                import traceback
                self.debug_print(traceback.format_exc())




    def _calculate_statistics(self):
        """计算库统计数据"""
        try:
            if self.debug:
                self.debug_print("开始计算统计数据...")

            # 计算每个数据对的平均对象数
            if self.total_pairs > 0:
                # 首先检查 objects_by_pair 的结构
                if self.debug:
                    pair_ids = list(self.objects_by_pair.keys())
                    if pair_ids:
                        first_pair_id = pair_ids[0]
                        sample_value = self.objects_by_pair[first_pair_id]
                        self.debug_print(f"objects_by_pair 样本值类型: {type(sample_value)}")
                        if hasattr(sample_value, '__iter__'):
                            self.debug_print(f"样本值内容: {sample_value}")

                total_objects = 0
                for pair_id, pair_data in self.objects_by_pair.items():
                    # 确保 pair_data 是字典
                    if isinstance(pair_data, dict):
                        # 正确处理字典值
                        for io_type in ["input", "output", "diff_in", "diff_out"]:
                            if io_type in pair_data:
                                total_objects += len(pair_data[io_type])
                    else:
                        # 如果不是字典，尝试其他方式处理
                        if self.debug:
                            self.debug_print(f"警告: pair_id={pair_id} 的值类型为 {type(pair_data)}，而不是字典")

                        # 尝试将它作为包含键值对的元组或列表处理
                        try:
                            if isinstance(pair_data, (tuple, list)) and len(pair_data) >= 2:
                                # 可能是 (key, value) 格式
                                for item in pair_data:
                                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                                        key, objs = item[0], item[1]
                                        if key in ["input", "output", "diff_in", "diff_out"] and hasattr(objs, '__len__'):
                                            total_objects += len(objs)
                        except Exception as e:
                            if self.debug:
                                self.debug_print(f"尝试处理pair_data时出错: {e}")

                # 避免除零错误
                if self.total_pairs > 0:
                    self.statistics["objects_per_pair_avg"] = total_objects / (self.total_pairs * 2)
                else:
                    self.statistics["objects_per_pair_avg"] = 0

                if self.debug:
                    self.debug_print(f"计算得到总对象数: {total_objects}, 平均每对: {self.statistics['objects_per_pair_avg']:.2f}")

            # 最常见的形状
            try:
                shapes_by_freq = sorted(
                    [(sh, data["count"]) for sh, data in self.shape_library.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                self.statistics["most_common_shapes"] = shapes_by_freq[:10]  # 前10个
            except Exception as e:
                if self.debug:
                    self.debug_print(f"计算最常见形状时出错: {e}")
                self.statistics["most_common_shapes"] = []

            # 最常见的颜色
            try:
                colors_by_freq = sorted(
                    [(c, data["count"]) for c, data in self.color_library.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                self.statistics["most_common_colors"] = colors_by_freq[:10]  # 前10个
            except Exception as e:
                if self.debug:
                    self.debug_print(f"计算最常见颜色时出错: {e}")
                self.statistics["most_common_colors"] = []

            # 操作统计
            operations = {
                "removed": len(self.removed_objects),
                "added": len(self.added_objects),
                "preserved": len(self.preserved_objects),
                "transformed": len(self.transformed_objects)
            }
            self.statistics["most_common_operations"] = operations

            if self.debug:
                self.debug_print("统计数据计算完成")

        except Exception as e:
            if self.debug:
                self.debug_print(f"计算统计数据时出现错误: {e}")
                import traceback
                self.debug_print(traceback.format_exc())

            # 设置默认统计值
            self.statistics = {
                "objects_per_pair_avg": 0,
                "most_common_shapes": [],
                "most_common_colors": [],
                "most_common_operations": {
                    "removed": 0,
                    "added": 0,
                    "preserved": 0,
                    "transformed": 0
                }
            }

    def _log_library_statistics(self):
        """输出库统计数据"""
        if not self.debug_print:
            return

        self.debug_print("\n==== 多维度关系库统计 ====")
        self.debug_print(f"总数据对: {self.total_pairs}")
        self.debug_print(f"总对象数: {self.total_objects}")
        self.debug_print(f"平均每对数据的对象数: {self.statistics['objects_per_pair_avg']:.2f}")
        self.debug_print(f"形状库大小: {len(self.shape_library)}")
        self.debug_print(f"颜色库大小: {len(self.color_library)}")
        self.debug_print("\n最常见的形状:")
        for i, (shape_hash, count) in enumerate(self.statistics["most_common_shapes"][:5]):
            self.debug_print(f"  {i+1}. 形状哈希 {shape_hash}: 出现 {count} 次")

        self.debug_print("\n最常见的颜色:")
        for i, (color, count) in enumerate(self.statistics["most_common_colors"][:5]):
            self.debug_print(f"  {i+1}. 颜色 {color}: 出现 {count} 次")

        self.debug_print("\n操作统计:")
        for op, count in self.statistics["most_common_operations"].items():
            self.debug_print(f"  {op}: {count}")

        self.debug_print(f"\n相似对象分组: {len(self.similar_objects_groups)}")
        self.debug_print(f"颜色转换规则: {len(self.color_transformation_rules)}")
        self.debug_print("============================\n")

    def find_patterns_across_pairs(self):
        """
        在所有库中寻找跨数据对的通用模式

        Returns:
            跨数据对的模式列表
        """
        patterns = []

        # 1. 寻找形状与操作之间的关系模式
        patterns.extend(self._find_shape_operation_patterns())

        # 2. 寻找颜色与操作之间的关系模式
        patterns.extend(self._find_color_operation_patterns())

        # 3. 寻找形状与颜色变化之间的关系模式
        patterns.extend(self._find_shape_color_patterns())

        # 4. 寻找跨数据对的条件模式
        patterns.extend(self._find_conditional_patterns())

        # 5. 寻找属性相关性模式
        patterns.extend(self._find_attribute_correlation_patterns())

        # 按置信度排序
        patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return patterns

    def _find_shape_operation_patterns(self):
        """寻找形状与操作的关系模式"""
        patterns = []

        # 分析每种形状最常见的操作
        for shape_hash, operations in self.shape_to_operation_map.items():
            # if len(operations) < 2:
            #     continue

            # 按操作类型分组
            op_types = {}
            for pair_id, op_type, obj_id in operations:
                if op_type not in op_types:
                    op_types[op_type] = []
                op_types[op_type].append((pair_id, obj_id))

            # 找出最常见的操作
            if op_types:
                most_common_op, occurrences = max(op_types.items(), key=lambda x: len(x[1]))

                # 如果这种操作占比较高，添加为模式
                consistency = len(occurrences) / len(operations)
                if consistency >= 0.6 and len(occurrences) >= 1:
                    pattern = {
                        "type": "shape_operation_pattern",
                        "shape_hash": shape_hash,
                        "operation": most_common_op,
                        "consistency": consistency,
                        "occurrences": len(occurrences),
                        "total": len(operations),
                        "confidence": consistency,
                        "supporting_pairs": list(set(p for p, _ in occurrences))
                    }
                    patterns.append(pattern)

        return patterns

    def _find_color_operation_patterns(self):
        """寻找颜色与操作的关系模式"""
        patterns = []

        # 分析每种颜色最常见的操作
        for color, operations in self.color_to_operation_map.items():
            # if len(operations) < 2:
            #     continue

            # 按操作类型分组
            op_types = {}
            for pair_id, op_type, obj_id in operations:
                if op_type not in op_types:
                    op_types[op_type] = []
                op_types[op_type].append((pair_id, obj_id))

            # 找出最常见的操作
            if op_types:
                most_common_op, occurrences = max(op_types.items(), key=lambda x: len(x[1]))

                # 如果这种操作占比较高，添加为模式
                consistency = len(occurrences) / len(operations)
                if consistency >= 0.6 and len(occurrences) >= 1:
                    pattern = {
                        "type": "color_operation_pattern",
                        "color": color,
                        "operation": most_common_op,
                        "consistency": consistency,
                        "occurrences": len(occurrences),
                        "total": len(operations),
                        "confidence": consistency,
                        "supporting_pairs": list(set(p for p, _ in occurrences))
                    }
                    patterns.append(pattern)

        return patterns

    def _find_shape_color_patterns(self):
        """寻找形状与颜色变化的关系模式"""
        patterns = []

        # 按形状分组颜色变化
        shape_color_changes = defaultdict(list)

        for rule in self.color_transformation_rules:
            shape_hash = rule.get("shape_hash")
            from_color = rule.get("from_color")
            to_color = rule.get("to_color")
            pair_id = rule.get("pair_id")

            if shape_hash and from_color is not None and to_color is not None:
                key = (shape_hash, from_color, to_color)
                shape_color_changes[key].append(pair_id)

        # 寻找一致的颜色变化模式
        for (shape_hash, from_color, to_color), pair_ids in shape_color_changes.items():
            if len(pair_ids) >= 2:  # 至少在两个数据对中出现
                # 估算置信度
                # 计算这种形状的总体出现次数
                shape_total_occurrences = len(set(pair_id for pair_id, _, _ in self.shape_library.get(shape_hash, {}).get("occurrences", [])))

                if shape_total_occurrences > 0:
                    confidence = len(set(pair_ids)) / shape_total_occurrences

                    pattern = {
                        "type": "shape_color_change_pattern",
                        "shape_hash": shape_hash,
                        "from_color": from_color,
                        "to_color": to_color,
                        "occurrences": len(set(pair_ids)),
                        "shape_total": shape_total_occurrences,
                        "confidence": confidence,
                        "supporting_pairs": list(set(pair_ids))
                    }
                    patterns.append(pattern)

        return patterns


    def _find_conditional_patterns(self):
        """寻找条件性模式，例如：如果移除了形状X，那么形状Y会变色"""
        patterns = []
        enhanced_data = {}  # 存储增强的信息，但不影响原有流程

        if self.debug:
            self.debug_print("\n\n开始寻找条件性模式 (兼容模式)...")

        try:
            # 收集每个数据对中被移除的形状 - 保持原始结构，但保存增强信息
            removals_by_pair = defaultdict(list)
            removals_enhanced = defaultdict(list)  # 增强版本
            for pair_id, obj_id, obj_info in self.removed_objects:
                shape_hash = self._get_shape_hash_from_dict(obj_info)
                if shape_hash:
                    # 原始结构保持不变
                    removals_by_pair[pair_id].append((shape_hash, obj_id))
                    # 增强信息单独存储
                    removals_enhanced[pair_id].append((shape_hash, obj_id, obj_info))

            if self.debug:
                self.debug_print(f"收集了 {len(removals_by_pair)} 个数据对的移除对象信息")

            # 收集每个数据对中的颜色变化 - 保持原始结构，但保存增强信息
            color_changes_by_pair = defaultdict(list)
            color_changes_enhanced = defaultdict(list)  # 增强版本
            for rule in self.color_transformation_rules:
                try:
                    pair_id = rule.get("pair_id")
                    from_color = rule.get("from_color")
                    to_color = rule.get("to_color")

                    if isinstance(pair_id, (int, str)) and from_color is not None and to_color is not None:
                        # 原始结构保持不变
                        color_changes_by_pair[pair_id].append((from_color, to_color, rule))

                        # 尝试获取增强信息
                        input_obj_id = rule.get("input_obj_id", "unknown")
                        output_obj_id = rule.get("output_obj_id", "unknown")
                        shape_hash = rule.get("shape_hash")
                        if shape_hash is None:
                            # 尝试从规则中推断形状
                            if self.debug:
                                self.debug_print(f"规则中缺少shape_hash，尝试推断... rule_id: {pair_id}")

                        # 存储增强信息
                        enhanced_info = {
                            "from_color": from_color,
                            "to_color": to_color,
                            "input_obj_id": input_obj_id,
                            "output_obj_id": output_obj_id,
                            "shape_hash": shape_hash,
                            "rule": rule
                        }
                        color_changes_enhanced[pair_id].append(enhanced_info)
                except Exception as e:
                    if self.debug:
                        self.debug_print(f"处理颜色变化规则时出错: {e}, rule: {rule}")

            if self.debug:
                self.debug_print(f"收集了 {len(color_changes_by_pair)} 个数据对的颜色变化信息")

            # 寻找移除形状与颜色变化之间的关联 - 保持原始流程
            removal_color_associations = defaultdict(lambda: defaultdict(list))

            for pair_id, removals in removals_by_pair.items():
                if pair_id in color_changes_by_pair:
                    for shape_hash, obj_id in removals:
                        for from_color, to_color, rule in color_changes_by_pair[pair_id]:
                            key = (shape_hash, from_color, to_color)
                            removal_color_associations[key][pair_id].append(rule)

            # 尝试构建增强版关联，但不影响原流程
            enhanced_associations = {}
            try:
                if self.debug:
                    self.debug_print("尝试构建增强版关联数据...")

                enhanced_associations = defaultdict(lambda: defaultdict(list))
                for pair_id, removals in removals_enhanced.items():
                    if pair_id in color_changes_enhanced:
                        for removed_shape_hash, removed_obj_id, removed_obj_info in removals:
                            for change_info in color_changes_enhanced[pair_id]:
                                from_color = change_info["from_color"]
                                to_color = change_info["to_color"]
                                affected_shape_hash = change_info.get("shape_hash", "UNKNOWN")

                                # 记录增强信息但不用于主要流程
                                key = (removed_shape_hash, from_color, to_color, affected_shape_hash)
                                enhanced_associations[key][pair_id].append({
                                    "change_info": change_info,
                                    "removal_info": {
                                        "obj_id": removed_obj_id,
                                        "shape_hash": removed_shape_hash,
                                        "obj_info": removed_obj_info
                                    }
                                })
            except Exception as e:
                if self.debug:
                    self.debug_print(f"构建增强版关联时出错 (不影响主流程): {e}")

            # 过滤和转换为模式 - 保持原始流程
            for (shape_hash, from_color, to_color), pair_data in removal_color_associations.items():
                if len(pair_data) >= 1:  # 至少在1个数据对中出现
                    # 收集所有支持的数据对和规则
                    supporting_pairs = []
                    supporting_rules = []

                    for pair_id, rules in pair_data.items():
                        supporting_pairs.append(pair_id)
                        supporting_rules.extend(rules)

                    # 估算置信度
                    # 计算同时有该形状移除和该颜色变化的比例
                    pairs_with_shape_removal = set(pair_id for pair_id, removals in removals_by_pair.items()
                                            if any(sh == shape_hash for sh, _ in removals))
                    pairs_with_color_change = set(pair_id for pair_id, changes in color_changes_by_pair.items()
                                            if any(fc == from_color and tc == to_color for fc, tc, _ in changes))

                    if pairs_with_shape_removal and pairs_with_color_change:
                        total_relevant_pairs = len(pairs_with_shape_removal.union(pairs_with_color_change))
                        confidence = len(supporting_pairs) / total_relevant_pairs

                        # 尝试从增强数据中提取更多信息
                        output_objects = []
                        try:
                            for pair_id in supporting_pairs:
                                # 查找对应的增强信息
                                for key, pair_dict in enhanced_associations.items():
                                    if key[0] == shape_hash and key[1] == from_color and key[2] == to_color and pair_id in pair_dict:
                                        for assoc in pair_dict[pair_id]:
                                            change = assoc.get("change_info", {})
                                            output_obj_id = change.get("output_obj_id")
                                            if output_obj_id and output_obj_id != "unknown":
                                                output_objects.append(output_obj_id)
                        except Exception as e:
                            if self.debug:
                                self.debug_print(f"提取输出对象时出错 (不影响模式生成): {e}")

                        # 创建模式 - 基本结构保持不变
                        pattern = {
                            "type": "conditional_pattern",
                            "subtype": "removal_color_change",
                            "condition": {
                                "operation": "remove",
                                "shape_hash": shape_hash
                            },
                            "effect": {
                                "color_change": {
                                    "from_color": from_color,
                                    "to_color": to_color
                                }
                            },
                            "supporting_pairs": supporting_pairs,
                            "confidence": confidence,
                            "description": f"当形状{shape_hash}被移除时，颜色从{from_color}变为{to_color}"
                        }

                        # 尝试添加增强信息，但不影响原有结构
                        if output_objects:
                            pattern["output_objects"] = output_objects

                        patterns.append(pattern)

            if self.debug:
                self.debug_print(f"找到 {len(patterns)} 个条件模式")

                # 记录详细的增强数据统计
                enhanced_patterns_count = 0
                for key, pair_dict in enhanced_associations.items():
                    if len(pair_dict) >= 2:
                        enhanced_patterns_count += 1

                self.debug_print(f"增强版关联数据中有 {enhanced_patterns_count} 个可能的模式")
                if len(patterns) == 0 and enhanced_patterns_count > 0:
                    self.debug_print("警告: 原始模式为0但增强版有模式，可能存在兼容性问题")
                    # 记录几个样本增强模式以供调试
                    sample_count = 0
                    for key, pair_dict in enhanced_associations.items():
                        if len(pair_dict) >= 2 and sample_count < 2:
                            removed_shape, from_col, to_col, affected_shape = key
                            self.debug_print(f"样本增强模式: 当形状{removed_shape}被移除时，形状{affected_shape}的颜色从{from_col}变为{to_col}")
                            sample_count += 1

        except Exception as e:
            if self.debug:
                self.debug_print(f"寻找条件模式时发生异常 (回退到基本模式): {e}")
                import traceback
                self.debug_print(traceback.format_exc())

        if len(patterns) == 0 and self.debug:
            self.debug_print("未找到条件模式，检查可能的问题...")

            # 提供额外的诊断信息
            if not self.color_transformation_rules:
                self.debug_print("- 未发现颜色转换规则")
            else:
                self.debug_print(f"- 有 {len(self.color_transformation_rules)} 条颜色转换规则")

            if not self.removed_objects:
                self.debug_print("- 未发现被移除的对象")
            else:
                self.debug_print(f"- 有 {len(self.removed_objects)} 个被移除的对象")

        # 存储增强数据以供将来使用
        enhanced_data["removal_color_enhanced"] = enhanced_associations

        return patterns


    def _find_conditional_patterns0(self):
        """寻找条件性模式，例如：如果移除了形状X，那么形状Y会变色"""
        patterns = []

        # 收集每个数据对中被移除的形状
        removals_by_pair = defaultdict(list)
        for pair_id, obj_id, obj_info in self.removed_objects:
            shape_hash = self._get_shape_hash_from_dict(obj_info)
            if shape_hash:
                removals_by_pair[pair_id].append((shape_hash, obj_id))

        # 收集每个数据对中的颜色变化
        color_changes_by_pair = defaultdict(list)
        for rule in self.color_transformation_rules:
            pair_id = rule.get("pair_id")
            from_color = rule.get("from_color")
            to_color = rule.get("to_color")
            if isinstance(pair_id, (int, str)) and from_color is not None and to_color is not None:
                color_changes_by_pair[pair_id].append((from_color, to_color, rule))

        # 寻找移除形状与颜色变化之间的关联
        removal_color_associations = defaultdict(lambda: defaultdict(list))

        for pair_id, removals in removals_by_pair.items():
            if pair_id in color_changes_by_pair:
                for shape_hash, _ in removals:
                    for from_color, to_color, rule in color_changes_by_pair[pair_id]:
                        key = (shape_hash, from_color, to_color)
                        removal_color_associations[key][pair_id].append(rule)

        # 过滤和转换为模式
        for (shape_hash, from_color, to_color), pair_data in removal_color_associations.items():
            if len(pair_data) >= 1:  # 至少在两个数据对中出现
                # 收集所有支持的数据对和规则
                supporting_pairs = []
                supporting_rules = []

                for pair_id, rules in pair_data.items():
                    supporting_pairs.append(pair_id)
                    supporting_rules.extend(rules)

                # 估算置信度
                # 计算同时有该形状移除和该颜色变化的比例
                pairs_with_shape_removal = set(pair_id for pair_id, removals in removals_by_pair.items()
                                           if any(sh == shape_hash for sh, _ in removals))
                pairs_with_color_change = set(pair_id for pair_id, changes in color_changes_by_pair.items()
                                           if any(fc == from_color and tc == to_color for fc, tc, _ in changes))

                if pairs_with_shape_removal and pairs_with_color_change:
                    total_relevant_pairs = len(pairs_with_shape_removal.union(pairs_with_color_change))
                    confidence = len(supporting_pairs) / total_relevant_pairs

                    pattern = {
                        "type": "conditional_pattern",
                        "subtype": "removal_color_change",
                        "condition": {
                            "operation": "remove",
                            "shape_hash": shape_hash
                        },
                        "effect": {
                            "color_change": {
                                #! what id
                                "from_color": from_color,
                                "to_color": to_color
                            }
                        },
                        "supporting_pairs": supporting_pairs,
                        "confidence": confidence,
                        "description": f"当形状{shape_hash}被移除时，颜色从{from_color}变为{to_color}"
                    }
                    patterns.append(pattern)

        return patterns

    def _find_attribute_correlation_patterns(self):
        """寻找属性相关性模式"""
        patterns = []

        # 分析属性相关性
        for (attr1, val1), correlations in self.attribute_correlations.items():
            # 根据相关频率排序
            sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

            # 只考虑前几个高度相关的属性
            for (attr2, val2), count in sorted_correlations[:3]:
                if count >= 3:  # 至少出现3次
                    # 估算置信度
                    attr1_total = sum(correlations.values())
                    confidence = count / attr1_total if attr1_total > 0 else 0

                    if confidence >= 0.7:  # 至少70%一致性
                        pattern = {
                            "type": "attribute_correlation",
                            "attribute1": attr1,
                            "value1": val1,
                            "attribute2": attr2,
                            "value2": val2,
                            "count": count,
                            "total": attr1_total,
                            "confidence": confidence,
                            "description": f"属性{attr1}={val1}与属性{attr2}={val2}高度关联"
                        }
                        patterns.append(pattern)

        return patterns

    def query_objects(self, **criteria):
        """
        根据多条件查询对象

        Args:
            criteria: 查询条件，可包含shape_hash, color, operation等

        Returns:
            匹配的对象列表 [(pair_id, io_type, obj_id), ...]
        """
        # 对每种条件筛选对象
        candidate_objects = None

        if "shape_hash" in criteria:
            shape_objects = set(self.objects_by_shape.get(criteria["shape_hash"], []))
            candidate_objects = shape_objects if candidate_objects is None else candidate_objects.intersection(shape_objects)

        if "color" in criteria:
            color_objects = set(self.objects_by_color.get(criteria["color"], []))
            candidate_objects = color_objects if candidate_objects is None else candidate_objects.intersection(color_objects)

        if "size" in criteria:
            size_objects = set(self.objects_by_size.get(criteria["size"], []))
            candidate_objects = size_objects if candidate_objects is None else candidate_objects.intersection(size_objects)

        if "position" in criteria:
            position_objects = set(self.objects_by_position.get(criteria["position"], []))
            candidate_objects = position_objects if candidate_objects is None else candidate_objects.intersection(position_objects)

        if "pair_id" in criteria:
            pair_objects = set()
            for io_type in ["input", "output"]:
                for obj_id in self.objects_by_pair.get(criteria["pair_id"], {}).get(io_type, []):
                    pair_objects.add((criteria["pair_id"], io_type, obj_id))
            candidate_objects = pair_objects if candidate_objects is None else candidate_objects.intersection(pair_objects)

        if "io_type" in criteria:
            io_objects = set()
            for pair_id, pair_data in self.objects_by_pair.items():
                for obj_id in pair_data.get(criteria["io_type"], []):
                    io_objects.add((pair_id, criteria["io_type"], obj_id))
            candidate_objects = io_objects if candidate_objects is None else candidate_objects.intersection(io_objects)

        if "operation" in criteria:
            op_objects = set()

            if criteria["operation"] == "removed":
                for pair_id, obj_id, _ in self.removed_objects:
                    op_objects.add((pair_id, "input", obj_id))

            elif criteria["operation"] == "added":
                for pair_id, obj_id, _ in self.added_objects:
                    op_objects.add((pair_id, "output", obj_id))

            elif criteria["operation"] == "preserved":
                for pair_id, in_obj_id, out_obj_id in self.preserved_objects:
                    op_objects.add((pair_id, "input", in_obj_id))
                    op_objects.add((pair_id, "output", out_obj_id))

            elif criteria["operation"] == "transformed":
                for pair_id, in_obj_id, out_obj_id, _ in self.transformed_objects:
                    op_objects.add((pair_id, "input", in_obj_id))
                    op_objects.add((pair_id, "output", out_obj_id))

            candidate_objects = op_objects if candidate_objects is None else candidate_objects.intersection(op_objects)

        return list(candidate_objects) if candidate_objects else []

    def get_object_details(self, pair_id, io_type, obj_id):
        """
        获取对象的详细信息

        Args:
            pair_id: 数据对ID
            io_type: 'input'或'output'
            obj_id: 对象ID

        Returns:
            对象的详细信息字典
        """
        details = {
            "pair_id": pair_id,
            "io_type": io_type,
            "obj_id": obj_id,
            "attributes": {}
        }

        # 查找形状
        for shape_hash, objects in self.objects_by_shape.items():
            if (pair_id, io_type, obj_id) in objects:
                details["attributes"]["shape_hash"] = shape_hash
                break

        # 查找颜色
        for color, objects in self.objects_by_color.items():
            if (pair_id, io_type, obj_id) in objects:
                details["attributes"]["color"] = color
                break

        # 查找大小
        for size, objects in self.objects_by_size.items():
            if (pair_id, io_type, obj_id) in objects:
                details["attributes"]["size"] = size
                break

        # 查找位置
        for position, objects in self.objects_by_position.items():
            if (pair_id, io_type, obj_id) in objects:
                details["attributes"]["position"] = position
                break

        # 查找操作信息
        if io_type == "input":
            # 检查是否被移除
            for p, o, obj_info in self.removed_objects:
                if p == pair_id and o == obj_id:
                    details["operation"] = {
                        "type": "removed",
                        "info": obj_info
                    }
                    break

            # 检查是否被保留
            if "operation" not in details:
                for p, i_o, o_o in self.preserved_objects:
                    if p == pair_id and i_o == obj_id:
                        details["operation"] = {
                            "type": "preserved",
                            "output_obj_id": o_o
                        }
                        break

            # 检查是否被变换
            if "operation" not in details:
                for p, i_o, o_o, transform_info in self.transformed_objects:
                    if p == pair_id and i_o == obj_id:
                        details["operation"] = {
                            "type": "transformed",
                            "output_obj_id": o_o,
                            "transformation": transform_info
                        }
                        break
        elif io_type == "output":
            # 检查是否被添加
            for p, o, obj_info in self.added_objects:
                if p == pair_id and o == obj_id:
                    details["operation"] = {
                        "type": "added",
                        "info": obj_info
                    }
                    break

            # 检查是否是保留的结果
            if "operation" not in details:
                for p, i_o, o_o in self.preserved_objects:
                    if p == pair_id and o_o == obj_id:
                        details["operation"] = {
                            "type": "preserved_result",
                            "input_obj_id": i_o
                        }
                        break

            # 检查是否是变换的结果
            if "operation" not in details:
                for p, i_o, o_o, transform_info in self.transformed_objects:
                    if p == pair_id and o_o == obj_id:
                        details["operation"] = {
                            "type": "transformed_result",
                            "input_obj_id": i_o,
                            "transformation": transform_info
                        }
                        break

        return details

    def export_libraries_to_json(self, filename=None):
        """
        将关系库导出为JSON文件

        Args:
            filename: 输出文件名，如果为None则返回JSON字符串

        Returns:
            如果filename为None，返回JSON字符串；否则返回None
        """
        # 构建导出数据
        export_data = {
            "statistics": self.statistics,
            "summary": {
                "total_pairs": self.total_pairs,
                "total_objects": self.total_objects,
                "shape_library_size": len(self.shape_library),
                "color_library_size": len(self.color_library),
                "removed_objects": len(self.removed_objects),
                "added_objects": len(self.added_objects),
                "preserved_objects": len(self.preserved_objects),
                "transformed_objects": len(self.transformed_objects),
                "color_transformation_rules": len(self.color_transformation_rules)
            },
            "patterns": self.find_patterns_across_pairs()
        }

        # 将hash键转换为字符串，以便JSON序列化
        export_json = json.dumps(export_data, default=self._json_serializer)

        if filename:
            with open(filename, 'w') as f:
                f.write(export_json)
            return None
        else:
            return export_json

    def _json_serializer(self, obj):
        """自定义JSON序列化器，处理不可序列化的对象"""
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, (tuple)):
            return list(obj)
        try:
            return str(obj)
        except:
            return "非序列化对象"

    def _get_shape_hash(self, obj_info):
        """从对象信息中获取形状哈希值"""
        try:
            # 首先尝试从ID中提取哈希值
            if hasattr(obj_info, 'obj_id'):
                # 提取ID中的哈希部分
                id_parts = obj_info.obj_id.split('_')
                if len(id_parts) >= 3:
                    return id_parts[2]  # 返回哈希部分

            # 如果从ID中获取失败，尝试传统方法
            if hasattr(obj_info, 'obj_000'):
                return hash(tuple(map(tuple, obj_info.obj_000)))
            return None
        except (TypeError, AttributeError):
            return None

    def _get_shape_hash_from_dict(self, obj_info_dict):
        """从对象信息字典中获取形状哈希值"""
        try:
            # 首先尝试从ID中提取哈希值
            if 'id' in obj_info_dict:
                # 提取ID中的哈希部分
                id_parts = obj_info_dict['id'].split('_')
                if len(id_parts) >= 3:
                    return id_parts[2]  # 返回哈希部分

            # 如果从ID中获取失败，尝试传统方法
            if 'obj_000' in obj_info_dict:
                shape_data = obj_info_dict['obj_000']
                return hash(tuple(map(tuple, shape_data)))
            return None
        except (TypeError, KeyError):
            return None