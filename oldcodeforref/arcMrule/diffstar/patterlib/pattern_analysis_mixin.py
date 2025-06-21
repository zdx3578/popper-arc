
class PatternAnalysisMixin:

    def _analyze_underlying_pattern_for_addition(self, target_color, rule):
        """分析添加操作背后可能存在的模式，以对象为单位分析，评估模式通用性"""
        if not hasattr(self, 'task') or not self.task:
            return None  # 如果没有训练任务数据，无法分析

        # 收集所有支持该规则的训练示例
        supporting_examples = []
        for pair_idx in rule.get('supporting_pairs', []):
            if pair_idx < len(self.task['train']):
                supporting_examples.append({
                    'example': self.task['train'][pair_idx],
                    'pair_idx': pair_idx
                })

        if not supporting_examples:
            return None

        # 模式类型及其权重定义
        pattern_types = {
            'four_box_pattern': {'weight': 1.5, 'threshold': 0.7},  # 4Box模式权重更高，识别阈值更低
            'symmetry_pattern': {'weight': 1.2, 'threshold': 0.8},
            'proximity_pattern': {'weight': 1.0, 'threshold': 0.8},
            'alignment_pattern': {'weight': 1.0, 'threshold': 0.8}
        }

        # 初始化模式分析结果
        pattern_candidates = {
            ptype: {
                'instances': [],
                'confidence': 0,
                'example_coverage': {}, # 记录每个示例的覆盖情况
                'weight': data['weight'],
                'threshold': data['threshold']
            } for ptype, data in pattern_types.items()
        }

        # 对每个示例，分析添加对象与可能的模式
        for idx, example_data in enumerate(supporting_examples):
            example = example_data['example']
            pair_idx = example_data['pair_idx']

            input_grid = example['input']
            output_grid = example['output']

            # 找出所有新添加的目标颜色位置
            added_positions = self._find_added_positions(input_grid, output_grid, target_color)
            if not added_positions:
                continue

            # 将相连的添加像素识别为对象
            added_objects = self._find_connected_objects(added_positions)

            if self.debug:
                self.debug_print(f"示例 {idx} (pair_idx={pair_idx}): 发现 {len(added_positions)} 个添加像素，形成 {len(added_objects)} 个对象")

            # 1. 检查4Box模式
            fourbox_results = self._check_objects_for_4box_patterns(input_grid, added_objects, added_positions)
            if fourbox_results:
                for result in fourbox_results:
                    result['example_idx'] = idx
                    result['pair_idx'] = pair_idx
                pattern_candidates['four_box_pattern']['instances'].extend(fourbox_results)
                pattern_candidates['four_box_pattern']['example_coverage'][idx] = len(fourbox_results)

            # 可以类似地添加其他模式检查...

        # 评估模式通用性和置信度
        for pattern_type, data in pattern_candidates.items():
            # 如果没有实例，跳过
            if not data['instances']:
                continue

            # 计算示例覆盖率 - 多少比例的示例支持此模式
            examples_count = len(supporting_examples)
            examples_with_pattern = len(data['example_coverage'])
            example_coverage_ratio = examples_with_pattern / examples_count if examples_count > 0 else 0

            # 计算对象覆盖率 - 多少比例的添加对象符合此模式
            total_objects = sum(len(self._find_connected_objects(self._find_added_positions(ex['example']['input'],
                                                                                        ex['example']['output'],
                                                                                        target_color)))
                            for ex in supporting_examples)
            pattern_objects = len(data['instances'])
            object_coverage_ratio = pattern_objects / total_objects if total_objects > 0 else 0

            # 计算加权置信度
            raw_confidence = (example_coverage_ratio * 0.7) + (object_coverage_ratio * 0.3)
            data['confidence'] = raw_confidence * data['weight']

            # 记录覆盖指标
            data['example_coverage_ratio'] = example_coverage_ratio
            data['object_coverage_ratio'] = object_coverage_ratio

            if self.debug:
                self.debug_print(f"模式 {pattern_type}: 示例覆盖率 {example_coverage_ratio:.2f}, "
                            f"对象覆盖率 {object_coverage_ratio:.2f}, "
                            f"加权置信度 {data['confidence']:.2f}")

        # 找出置信度最高的模式
        valid_patterns = [(ptype, data) for ptype, data in pattern_candidates.items()
                        if data['instances'] and data['confidence'] >= data['threshold']]

        if not valid_patterns:
            return None

        best_pattern = max(valid_patterns, key=lambda x: x[1]['confidence'])
        pattern_type, pattern_data = best_pattern

        return self._create_enhanced_pattern_description(pattern_type, pattern_data, target_color)

    def _check_objects_for_4box_patterns(self, grid, objects, all_positions_set):
        """
        检查添加对象是否与4Box模式相关

        Args:
            grid: 输入网格
            objects: 添加对象的列表，每个元素是一组相连的位置坐标
            all_positions_set: 所有添加位置的集合

        Returns:
            4Box模式实例列表
        """
        if not isinstance(all_positions_set, set):
            all_positions_set = set(all_positions_set)

        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        fourbox_instances = []

        # 对每个对象分析其边界周围的颜色分布
        for obj_idx, obj_positions in enumerate(objects):
            # 找出对象边界
            obj_boundary = self._find_object_boundary(obj_positions, width, height)

            # 分析边界周围的颜色分布
            surrounding_colors_count = self._analyze_surrounding_colors(obj_boundary, all_positions_set, grid)

            # 检查是否有颜色在多个方向上包围对象
            best_surrounding_color = None
            max_direction_count = 0
            max_surrounding_count = 0
            direction_counts = None

            for color, counts in surrounding_colors_count.items():
                # 计算这个颜色出现在多少个方向
                directions_with_color = sum(1 for count in counts.values() if count > 0)
                total_count = sum(counts.values())

                # 选择出现在最多方向的颜色
                if directions_with_color > max_direction_count or (directions_with_color == max_direction_count and total_count > max_surrounding_count):
                    max_direction_count = directions_with_color
                    max_surrounding_count = total_count
                    best_surrounding_color = color
                    direction_counts = counts.copy()

            # 如果至少有2个方向被同一颜色包围，认为符合4Box模式
            if max_direction_count >= 2 and best_surrounding_color is not None:
                # 确定对象的中心位置和主要颜色
                center_x = sum(x for x, y in obj_positions) / len(obj_positions)
                center_y = sum(y for x, y in obj_positions) / len(obj_positions)
                center_pos = (int(center_x), int(center_y))

                # 查找对象所在区域在原网格中的颜色
                original_colors = {}
                for x, y in obj_positions:
                    if 0 <= x < width and 0 <= y < height:
                        orig_color = grid[y][x]
                        if orig_color not in original_colors:
                            original_colors[orig_color] = 0
                        original_colors[orig_color] += 1

                # 确定主要颜色
                main_color = max(original_colors.items(), key=lambda x: x[1])[0] if original_colors else None

                # 计算包围度 - 被包围边界点占总边界点的比例
                boundary_coverage = max_surrounding_count / len(obj_boundary) if obj_boundary else 0

                # 根据方向计算4Box完整度系数
                completeness_factor = max_direction_count / 4.0  # 满分为1.0

                # 添加发现的实例
                fourbox_instances.append({
                    'object_index': obj_idx,
                    'object_size': len(obj_positions),
                    'center_position': center_pos,
                    'center_color': main_color,
                    'surrounding_color': best_surrounding_color,
                    'direction_counts': direction_counts,
                    'total_directions': max_direction_count,
                    'is_complete': max_direction_count == 4,
                    'boundary_coverage': boundary_coverage,
                    'completeness_factor': completeness_factor,
                    'pattern_strength': boundary_coverage * completeness_factor,  # 综合强度评分
                    'object_positions': list(obj_positions)  # 记录对象位置
                })

        # 按模式强度排序
        fourbox_instances.sort(key=lambda x: x['pattern_strength'], reverse=True)
        return fourbox_instances

    def _create_enhanced_pattern_description(self, pattern_type, pattern_data, target_color):
        """创建增强的模式描述，包括通用性和置信度指标，以及可执行规则"""
        instances = pattern_data['instances']
        confidence = pattern_data['confidence']
        example_coverage = pattern_data.get('example_coverage_ratio', 0)
        object_coverage = pattern_data.get('object_coverage_ratio', 0)

        result = {
            'pattern_type': pattern_type,
            'confidence': confidence,
            'instance_count': len(instances),
            'example_coverage': example_coverage,
            'object_coverage': object_coverage,
            'is_universal': example_coverage > 0.9  # 90%以上的示例支持则视为通用规则
        }

        # 根据模式类型创建描述
        if pattern_type == 'four_box_pattern':
            # 分析所有实例中的主要颜色和环绕颜色
            center_colors = {}
            surr_colors = {}
            complete_count = 0

            for instance in instances:
                # 统计中心颜色
                center_color = instance.get('center_color')
                if center_color is not None:
                    if center_color not in center_colors:
                        center_colors[center_color] = 0
                    center_colors[center_color] += 1

                # 统计环绕颜色
                surr_color = instance.get('surrounding_color')
                if surr_color is not None:
                    if surr_color not in surr_colors:
                        surr_colors[surr_color] = 0
                    surr_colors[surr_color] += 1

                # 统计完整4Box的数量
                if instance.get('is_complete', False):
                    complete_count += 1

            # 找出最常见的颜色
            main_center_color = max(center_colors.items(), key=lambda x: x[1])[0] if center_colors else None
            main_surr_color = max(surr_colors.items(), key=lambda x: x[1])[0] if surr_colors else None

            # 记录完整性比例
            result['complete_ratio'] = complete_count / len(instances) if instances else 0
            result['center_color'] = main_center_color
            result['surrounding_color'] = main_surr_color

            # 创建描述
            pattern_quality = "完全" if result['complete_ratio'] > 0.8 else "部分"
            universality = "通用" if result['is_universal'] else "部分适用的"

            result['description'] = (
                f"{universality}4Box模式：中心颜色{main_center_color}的对象被颜色{main_surr_color}"
                f"的对象{pattern_quality}包围时，添加颜色为{target_color}的新对象"
            )
            result['formal_name'] = '4Box模式'

            # 添加可执行规则
            result['executable_rule'] = {
                'rule_type': 'four_box_pattern',
                'action': 'add_objects',
                'center_color': main_center_color,
                'surrounding_color': main_surr_color,
                'target_color': target_color,
                'min_directions': 4 if pattern_quality == "部分" else 4,
                'required_completion': 0.7 if pattern_quality == "部分" else 0.9,
                'priority': 0.8 if universality == "通用" else 0.6
            }

            # 添加该模式的执行函数引用
            result['detect_fun'] = 'detect_four_box_pattern'
            result['execute_function'] = 'apply_four_box_pattern_rule'

        # 这里可以添加其他模式类型的处理...

        return result


    def _create_enhanced_pattern_description0(self, pattern_type, pattern_data, target_color):
        """创建增强的模式描述，包括通用性和置信度指标"""
        instances = pattern_data['instances']
        confidence = pattern_data['confidence']
        example_coverage = pattern_data.get('example_coverage_ratio', 0)
        object_coverage = pattern_data.get('object_coverage_ratio', 0)

        result = {
            'pattern_type': pattern_type,
            'confidence': confidence,
            'instance_count': len(instances),
            'example_coverage': example_coverage,
            'object_coverage': object_coverage,
            'is_universal': example_coverage > 0.9  # 90%以上的示例支持则视为通用规则
        }

        # 根据模式类型创建描述
        if pattern_type == 'four_box_pattern':
            # 分析所有实例中的主要颜色和环绕颜色
            center_colors = {}
            surr_colors = {}
            complete_count = 0

            for instance in instances:
                # 统计中心颜色
                center_color = instance.get('center_color')
                if center_color is not None:
                    if center_color not in center_colors:
                        center_colors[center_color] = 0
                    center_colors[center_color] += 1

                # 统计环绕颜色
                surr_color = instance.get('surrounding_color')
                if surr_color is not None:
                    if surr_color not in surr_colors:
                        surr_colors[surr_color] = 0
                    surr_colors[surr_color] += 1

                # 统计完整4Box的数量
                if instance.get('is_complete', False):
                    complete_count += 1

            # 找出最常见的颜色
            main_center_color = max(center_colors.items(), key=lambda x: x[1])[0] if center_colors else None
            main_surr_color = max(surr_colors.items(), key=lambda x: x[1])[0] if surr_colors else None

            # 记录完整性比例
            result['complete_ratio'] = complete_count / len(instances) if instances else 0
            result['center_color'] = main_center_color
            result['surrounding_color'] = main_surr_color

            # 创建描述
            pattern_quality = "完全" if result['complete_ratio'] > 0.8 else "部分"
            universality = "通用" if result['is_universal'] else "部分适用的"

            result['description'] = (
                f"{universality}4Box模式：中心颜色{main_center_color}的对象被颜色{main_surr_color}"
                f"的对象{pattern_quality}包围时，添加颜色为{target_color}的新对象"
            )
            result['formal_name'] = '4Box模式'

        # 这里可以添加其他模式类型的处理...

        return result



    def _analyze_underlying_pattern_for_addition00(self, target_color, rule):
        """分析添加操作背后可能存在的模式

        Args:
            target_color: 被添加的目标颜色
            rule: 添加操作的规则

        Returns:
            如果找到可能的模式，返回模式信息；否则返回None
        """
        if not hasattr(self, 'task') or not self.task:
            return None  # 如果没有训练任务数据，无法分析

        # 收集所有支持该规则的训练示例
        supporting_examples = []
        for pair_idx in rule.get('supporting_pairs', []):
            if pair_idx < len(self.task['train']):
                supporting_examples.append(self.task['train'][pair_idx])

        if not supporting_examples:
            return None

        # 初始化模式分析结果
        pattern_candidates = {
            'four_box_pattern': {'instances': [], 'confidence': 0},
            'symmetry_pattern': {'instances': [], 'confidence': 0},
            'proximity_pattern': {'instances': [], 'confidence': 0},
            'alignment_pattern': {'instances': [], 'confidence': 0}
        }

        # 对每个示例，分析添加位置与可能的模式
        # for example in supporting_examples:
        for idx, example in enumerate(supporting_examples):
            print(f"Index {idx}, ")
            input_grid = example['input']
            output_grid = example['output']

            # 找出所有新添加的目标颜色位置
            added_positions = self._find_added_positions(input_grid, output_grid, target_color)
            if not added_positions:
                continue

            # 检查这些添加位置是否与特定模式相关联
            # 1. 检查4Box模式
            fourbox_instances = self._check_for_4box_patterns(input_grid, added_positions)
            if fourbox_instances:
                pattern_candidates['four_box_pattern']['instances'].extend(fourbox_instances)

            # # 2. 检查对称性模式#! 需要完善，暂时不用
            # symmetry_instances = self._check_for_symmetry_patterns(input_grid, added_positions)
            # if symmetry_instances:
            #     pattern_candidates['symmetry_pattern']['instances'].extend(symmetry_instances)

            # # 3. 检查邻近性模式
            # proximity_instances = self._check_for_proximity_patterns(input_grid, added_positions)
            # if proximity_instances:
            #     pattern_candidates['proximity_pattern']['instances'].extend(proximity_instances)

            # # 4. 检查对齐模式
            # alignment_instances = self._check_for_alignment_patterns(input_grid, added_positions)
            # if alignment_instances:
            #     pattern_candidates['alignment_pattern']['instances'].extend(alignment_instances)

        # 计算每种模式的置信度
        total_added_positions = sum(len(self._find_added_positions(ex['input'], ex['output'], target_color))
                                for ex in supporting_examples)

        for pattern_type, data in pattern_candidates.items():
            if total_added_positions > 0:
                data['confidence'] = len(data['instances']) / total_added_positions

        # 找出置信度最高的模式
        best_pattern = max(pattern_candidates.items(), key=lambda x: x[1]['confidence'])
        pattern_type, pattern_data = best_pattern

        # 仅当置信度足够高时才返回模式
        if pattern_data['confidence'] > 0.5 and pattern_data['instances']:
            return self._create_pattern_description(pattern_type, pattern_data, target_color)

        return None

    def _find_added_positions(self, input_grid, output_grid, color):
        """找出所有新添加的指定颜色的位置"""
        added_positions = []

        # 确保网格维度一致
        input_height = len(input_grid)
        input_width = len(input_grid[0]) if input_height > 0 else 0
        output_height = len(output_grid)
        output_width = len(output_grid[0]) if output_height > 0 else 0

        # 检查现有范围内的变化
        for y in range(min(output_height, input_height)):
            for x in range(min(output_width, input_width)):
                # 检查在输出中是指定颜色，但在输入中不是
                if output_grid[y][x] == color and input_grid[y][x] != color:
                    added_positions.append((x, y))

        # 检查输出比输入更大的区域
        for y in range(output_height):
            for x in range(output_width):
                if y >= input_height or x >= input_width:
                    if output_grid[y][x] == color:
                        added_positions.append((x, y))

        return added_positions

    def _check_for_4box_patterns0(self, grid, positions):
        """检查位置是否与4Box模式相关，只考虑添加对象的边界位置"""
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 将位置列表转换为集合，便于快速查找
        positions_set = set(positions)

        # 找出所有边界位置
        boundary_positions = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

        for x, y in positions:
            is_boundary = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # 如果相邻位置不是添加位置(不在positions_set中)，
                # 或者在网格边界之外，那么当前位置是边界位置
                if (nx, ny) not in positions_set or nx < 0 or ny < 0 or nx >= width or ny >= height:
                    is_boundary = True
                    break

            if is_boundary:
                boundary_positions.append((x, y))

        if self.debug:
            self.debug_print(f"发现{len(positions)}个添加位置中有{len(boundary_positions)}个边界位置")

        # 只对边界位置进行4Box模式检查
        fourbox_instances = []

        for x, y in boundary_positions:
            surrounding_colors = {}

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in positions_set:
                    # 只检查非添加位置的周围颜色
                    color = grid[ny][nx]
                    if color not in surrounding_colors:
                        surrounding_colors[color] = 0
                    surrounding_colors[color] += 1

            # 如果有一种颜色出现在多个方向，可能是4Box模式
            for color, count in surrounding_colors.items():
                if count >= 2:  # 要求至少两个方向有相同颜色
                    fourbox_instances.append({
                        'center_position': (x, y),
                        'center_color': grid[y][x] if 0 <= y < height and 0 <= x < width else None,
                        'surrounding_color': color,
                        'surrounding_count': count,
                        'complete': count == 4  # 完全的4Box需要四个方向都有
                    })

        return fourbox_instances

    def _check_for_symmetry_patterns(self, grid, positions):
        """检查位置是否与对称性模式相关"""
        # 简化实现，检查位置是否形成对称图案
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 检查中心对称
        center_x = width // 2
        center_y = height // 2

        symmetric_instances = []
        for x, y in positions:
            # 计算相对于中心的对称点
            sym_x = 2 * center_x - x
            sym_y = 2 * center_y - y

            # 检查对称点是否也在添加位置列表中
            if (sym_x, sym_y) in positions:
                symmetric_instances.append({
                    'position': (x, y),
                    'symmetric_position': (sym_x, sym_y),
                    'symmetry_type': 'central',
                    'center': (center_x, center_y)
                })

        return symmetric_instances

    def _check_for_proximity_patterns(self, grid, positions):
        """检查位置是否与邻近性模式相关"""
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        proximity_instances = []

        for x, y in positions:
            # 检查周围是否有特定颜色的对象
            nearby_objects = {}
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and (dx != 0 or dy != 0):
                        color = grid[ny][nx]
                        if color != 0:  # 假设0是背景
                            if color not in nearby_objects:
                                nearby_objects[color] = []
                            nearby_objects[color].append((nx, ny, dx, dy))

            # 如果周围有对象，记录邻近性关系
            if nearby_objects:
                proximity_instances.append({
                    'position': (x, y),
                    'nearby_objects': nearby_objects
                })

        return proximity_instances

    def _check_for_alignment_patterns(self, grid, positions):
        """检查位置是否与对齐模式相关"""
        # 检查添加位置是否与现有对象在同一行或同一列
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        alignment_instances = []

        # 对每个位置找出在同行或同列的非零值
        for x, y in positions:
            row_alignments = []
            col_alignments = []

            # 检查同行对象
            for nx in range(width):
                if nx != x and grid[y][nx] != 0:
                    row_alignments.append((nx, y, grid[y][nx]))

            # 检查同列对象
            for ny in range(height):
                if ny != y and grid[ny][x] != 0:
                    col_alignments.append((x, ny, grid[ny][x]))

            if row_alignments or col_alignments:
                alignment_instances.append({
                    'position': (x, y),
                    'row_alignments': row_alignments,
                    'col_alignments': col_alignments
                })

        return alignment_instances

    def _create_pattern_description0(self, pattern_type, pattern_data, target_color):
        """根据模式类型和数据创建模式描述"""
        instances = pattern_data['instances']
        confidence = pattern_data['confidence']

        # 根据模式类型创建不同的描述
        if pattern_type == 'four_box_pattern':
            # 分析4Box模式的主要特征
            center_colors = {}
            surr_colors = {}

            for instance in instances:
                center_color = instance.get('center_color')
                if center_color is not None:
                    if center_color not in center_colors:
                        center_colors[center_color] = 0
                    center_colors[center_color] += 1

                surr_color = instance.get('surrounding_color')
                if surr_color is not None:
                    if surr_color not in surr_colors:
                        surr_colors[surr_color] = 0
                    surr_colors[surr_color] += 1

            # 找出最常见的中心颜色和环绕颜色
            main_center_color = max(center_colors.items(), key=lambda x: x[1])[0] if center_colors else None
            main_surr_color = max(surr_colors.items(), key=lambda x: x[1])[0] if surr_colors else None

            description = f"中心颜色{main_center_color}被颜色{main_surr_color}包围的4Box模式触发添加颜色{target_color}的对象"

            return {
                'pattern_type': '4Box模式',
                'center_color': main_center_color,
                'surrounding_color': main_surr_color,
                'confidence': confidence,
                'instances_count': len(instances),
                'description': description
            }

        elif pattern_type == 'symmetry_pattern':
            # 分析对称模式
            symm_types = {}
            for instance in instances:
                stype = instance.get('symmetry_type', 'unknown')
                if stype not in symm_types:
                    symm_types[stype] = 0
                symm_types[stype] += 1

            main_symm_type = max(symm_types.items(), key=lambda x: x[1])[0] if symm_types else 'unknown'

            return {
                'pattern_type': '对称模式',
                'symmetry_type': main_symm_type,
                'confidence': confidence,
                'instances_count': len(instances),
                'description': f"基于{main_symm_type}对称添加颜色{target_color}的对象"
            }

        elif pattern_type == 'proximity_pattern':
            # 分析邻近性模式
            nearby_colors = {}
            for instance in instances:
                for color, positions in instance.get('nearby_objects', {}).items():
                    if color not in nearby_colors:
                        nearby_colors[color] = 0
                    nearby_colors[color] += len(positions)

            main_nearby_color = max(nearby_colors.items(), key=lambda x: x[1])[0] if nearby_colors else None

            return {
                'pattern_type': '邻近模式',
                'nearby_color': main_nearby_color,
                'confidence': confidence,
                'instances_count': len(instances),
                'description': f"在颜色{main_nearby_color}对象附近添加颜色{target_color}的对象"
            }

        elif pattern_type == 'alignment_pattern':
            # 分析对齐模式
            row_alignment = sum(1 for i in instances if i.get('row_alignments', []))
            col_alignment = sum(1 for i in instances if i.get('col_alignments', []))

            alignment_type = "行对齐" if row_alignment > col_alignment else "列对齐"

            return {
                'pattern_type': '对齐模式',
                'alignment_type': alignment_type,
                'confidence': confidence,
                'instances_count': len(instances),
                'description': f"基于{alignment_type}添加颜色{target_color}的对象"
            }

        # 默认情况
        return {
            'pattern_type': pattern_type,
            'confidence': confidence,
            'instances_count': len(instances),
            'description': f"基于{pattern_type}添加颜色{target_color}的对象"
        }







    def _check_for_4box_patterns00(self, grid, positions):
        """
        检查整体对象是否与4Box模式相关，将相连的添加像素视为一个整体对象
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 将位置列表转换为集合，便于快速查找
        positions_set = set(positions)

        # 标识连通区域（对象）
        objects = self._find_connected_objects(positions)

        if self.debug:
            self.debug_print(f"发现{len(positions)}个添加位置，组成了{len(objects)}个连通对象")

        fourbox_instances = []

        # 对每个对象分析其边界周围的颜色分布
        for obj_idx, obj_positions in enumerate(objects):
            obj_boundary = self._find_object_boundary(obj_positions, width, height)

            # 分析对象边界周围的颜色
            surrounding_colors_count = self._analyze_surrounding_colors(obj_boundary, positions_set, grid)

            # 检查是否存在主要的围绕颜色
            for color, direction_counts in surrounding_colors_count.items():
                # 计算这个颜色在不同方向出现的数量
                total_directions = sum(1 for count in direction_counts.values() if count > 0)

                # 如果一个颜色出现在至少2个方向，可能是4Box模式
                if total_directions >= 2:
                    # 计算对象的中心点（用于记录）
                    center_x = sum(x for x, y in obj_positions) / len(obj_positions)
                    center_y = sum(y for x, y in obj_positions) / len(obj_positions)
                    center_pos = (int(center_x), int(center_y))

                    # 查找对象内部的颜色（如果对象覆盖了原始网格上的区域）
                    original_colors = {}
                    for x, y in obj_positions:
                        if 0 <= x < width and 0 <= y < height:
                            orig_color = grid[y][x]
                            if orig_color not in original_colors:
                                original_colors[orig_color] = 0
                            original_colors[orig_color] += 1

                    # 确定对象的主要颜色
                    main_color = max(original_colors.items(), key=lambda x: x[1])[0] if original_colors else None

                    # 添加发现的实例
                    fourbox_instances.append({
                        'object_index': obj_idx,
                        'object_positions': list(obj_positions),
                        'center_position': center_pos,
                        'center_color': main_color,
                        'surrounding_color': color,
                        'direction_counts': direction_counts,
                        'total_directions': total_directions,
                        'complete': total_directions == 4,  # 完全的4Box需要四个方向都有
                        'surrounding_ratio': sum(direction_counts.values()) / len(obj_boundary)  # 包围程度
                    })

        return fourbox_instances

    def _find_connected_objects(self, positions):
        """
        将相连的像素分组为对象

        Args:
            positions: 添加位置的列表

        Returns:
            列表，每个元素是一组相连的位置坐标
        """
        if not positions:
            return []

        # 将位置列表转换为集合，便于快速查找
        remaining = set(positions)
        objects = []

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上下左右四个方向

        while remaining:
            # 从剩余位置中取出一个作为起点
            start = next(iter(remaining))

            # 使用BFS查找所有相连的位置
            current_object = set()
            queue = [start]
            current_object.add(start)
            remaining.remove(start)

            while queue:
                x, y = queue.pop(0)

                # 检查四个方向的邻居
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    neighbor = (nx, ny)

                    # 如果邻居在剩余位置中，添加到当前对象
                    if neighbor in remaining:
                        queue.append(neighbor)
                        current_object.add(neighbor)
                        remaining.remove(neighbor)

            # 将当前对象添加到对象列表
            objects.append(current_object)

        return objects

    def _find_object_boundary(self, obj_positions, width, height):
        """
        找出对象的边界像素

        Args:
            obj_positions: 对象的位置集合
            width, height: 网格的宽度和高度

        Returns:
            对象边界位置的列表
        """
        obj_positions_set = set(obj_positions)
        boundary_positions = []

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上下左右四个方向

        for x, y in obj_positions:
            # 检查是否是边界像素
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 如果邻居不在对象中，当前像素是边界
                if (nx, ny) not in obj_positions_set:
                    boundary_positions.append((x, y, dx, dy))  # 记录方向

        return boundary_positions

    def _analyze_surrounding_colors(self, boundary_positions, all_added_positions, grid, ignore_grid_boundary=True):
        """
        分析对象边界周围的颜色分布

        Args:
            boundary_positions: 对象边界位置列表，每个元素是(x, y, dx, dy)
            all_added_positions: 所有添加位置的集合
            grid: 输入网格
            ignore_grid_boundary: 是否忽略网格边界作为包围颜色（True表示边界不算包围）

        Returns:
            字典，键为颜色，值为该颜色在不同方向出现的次数
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 按颜色记录在不同方向出现的次数
        surrounding_colors_count = {}  # 颜色 -> {方向 -> 计数}

        # 方向映射
        direction_map = {
            (0, 1): 'top',
            (1, 0): 'right',
            (0, -1): 'bottom',
            (-1, 0): 'left'
        }

        for x, y, dx, dy in boundary_positions:
            nx, ny = x + dx, y + dy

            # 检查是否是网格边界
            is_boundary = nx < 0 or ny < 0 or nx >= width or ny >= height

            # 如果是边界且设置了忽略边界，则跳过
            if is_boundary and ignore_grid_boundary:
                continue

            # 网格内且不是添加位置的点
            if not is_boundary and (nx, ny) not in all_added_positions:
                color = grid[ny][nx]
                direction = direction_map.get((dx, dy), 'unknown')

                if color not in surrounding_colors_count:
                    surrounding_colors_count[color] = {
                        'top': 0,
                        'right': 0,
                        'bottom': 0,
                        'left': 0
                    }

                surrounding_colors_count[color][direction] += 1

        return surrounding_colors_count


    def detect_four_box_pattern(self, grid, rule):
        """
        在输入网格中检测4Box模式，排除背景对象

        Args:
            grid: 输入网格
            rule: 执行规则

        Returns:
            检测到的四字形模式列表
        """
        center_color = rule.get('center_color')
        surrounding_color = rule.get('surrounding_color')
        min_directions = rule.get('min_directions', 4)

        # 找出所有中心颜色的像素位置
        center_positions = []
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # 背景色通常是0
        background_color = 0

        for y in range(height):
            for x in range(width):
                if grid[y][x] == center_color:
                    center_positions.append((x, y))

        if not center_positions:
            return []

        # 将中心颜色的像素分组成对象
        center_objects = self._find_connected_objects(center_positions)

        # 过滤掉太大的对象（可能是背景）
        filtered_objects = []
        for obj in center_objects:
            # 如果对象太大（比如超过总网格的50%），可能是背景，跳过
            if len(obj) > (width * height * 0.5):
                if hasattr(self, 'debug') and self.debug and hasattr(self, 'debug_print'):
                    self.debug_print(f"跳过大型背景对象，大小: {len(obj)}像素")
                continue

            # 如果对象多处接触网格边界，可能是背景对象
            border_touches = 0
            for x, y in obj:
                if x == 0 or y == 0 or x == width-1 or y == height-1:
                    border_touches += 1

            # 如果接触边界的像素超过对象大小的50%，可能是背景
            if border_touches > len(obj) * 0.5:
                if hasattr(self, 'debug') and self.debug and hasattr(self, 'debug_print'):
                    self.debug_print(f"跳过边界对象，接触边界: {border_touches}像素")
                continue

            filtered_objects.append(obj)

        # 使用对象级分析函数检查4Box模式，忽略网格边界
        fourbox_instances = []
        for obj in filtered_objects:
            instances = self._check_objects_for_4box_patterns(grid, [obj], set(center_positions))
            fourbox_instances.extend(instances)

        # 过滤符合规则要求的实例
        filtered_instances = []
        for instance in fourbox_instances:
            # 检查是否符合规则中的颜色和方向要求
            if (instance['surrounding_color'] == surrounding_color and
                instance['total_directions'] >= min_directions):
                filtered_instances.append(instance)

        return filtered_instances



    def apply_four_box_pattern_rule(self, grid, rule):
        """
        应用4Box模式规则，考虑整个对象的需求

        Args:
            grid: 输入网格
            rule: 执行规则

        Returns:
            更新后的网格
        """
        # 确保输入网格是可修改的列表
        output_grid = [list(row) for row in grid]

        # 检测4Box模式
        detected_patterns = self.detect_four_box_pattern(grid, rule)

        # 没有检测到模式，直接返回原网格
        if not detected_patterns:
            return output_grid

        # 对每个检测到的模式，根据对象大小确定处理方式
        target_color = rule.get('target_color')

        for pattern in detected_patterns:
            # 如果是对象级结果，包含object_positions
            if 'object_positions' in pattern and pattern['object_positions']:
                # 对整个对象应用规则
                for x, y in pattern['object_positions']:
                    output_grid[y][x] = target_color

                if hasattr(self, 'debug') and self.debug and hasattr(self, 'debug_print'):
                    obj_size = len(pattern['object_positions'])
                    self.debug_print(f"应用4Box对象规则：替换{obj_size}个像素的对象为颜色{target_color}")
            else:
                # 后备：只修改中心位置
                x, y = pattern['center_position']
                output_grid[y][x] = target_color

                if hasattr(self, 'debug') and self.debug and hasattr(self, 'debug_print'):
                    self.debug_print(f"应用4Box像素规则：将位置({x},{y})改为颜色{target_color}")

        return output_grid





