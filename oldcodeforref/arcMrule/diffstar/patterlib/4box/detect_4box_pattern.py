def detect_four_box_pattern_new2(self, grid, rule):
    """
    在输入网格中检测4Box模式（使用对象级分析）

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

    for y in range(height):
        for x in range(width):
            if grid[y][x] == center_color:
                center_positions.append((x, y))

    if not center_positions:
        return []

    # 将中心颜色的像素分组成对象
    center_objects = self._find_connected_objects(center_positions)

    # 使用现有的对象分析函数检查4Box模式
    fourbox_instances = self._check_objects_for_4box_patterns(grid, center_objects, set(center_positions))

    # 过滤符合规则要求的实例
    filtered_instances = []
    for instance in fourbox_instances:
        # 检查是否符合规则中的颜色和方向要求
        if (instance['surrounding_color'] == surrounding_color and
            instance['total_directions'] >= min_directions):
            filtered_instances.append(instance)

    return filtered_instances


def _detect_four_box_pattern0(self, params=None):
    """
    检测4-Box围绕模式：中心像素/对象被另一种颜色在上下左右四个方向完全包围

    Args:
        params: 参数字典，包含检测的配置选项

    Returns:
        包含检测到的4-Box模式列表
    """
    if params is None:
        params = {}

    require_exact_match = params.get('require_exact_match', True)
    allow_diagonal = params.get('allow_diagonal', False)
    boundary_counts = params.get('boundary_counts', False)

    patterns = []

    # 遍历所有训练数据对
    for pair_id in self.objects_by_pair:
        # 分别检查输入和输出网格
        for io_type in ['input', 'output']:
            if io_type not in self.objects_by_pair[pair_id]:
                continue

            # 重建网格（如果没有存储完整网格）
            grid = self._reconstruct_grid_from_objects(pair_id, io_type)
            if not grid:
                continue

            height, width = len(grid), len(grid[0])

            # 遍历网格中的每个位置
            for i in range(height):
                for j in range(width):
                    center_color = grid[i][j]
                    if center_color == 0:  # 跳过背景
                        continue

                    # 检查四个方向
                    directions = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

                    # 如果允许对角线，添加对角线方向
                    if allow_diagonal:
                        directions.extend([(i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)])

                    # 统计周围颜色
                    surrounding_colors = {}
                    valid_directions = 0

                    for ni, nj in directions:
                        # 检查边界
                        if 0 <= ni < height and 0 <= nj < width:
                            valid_directions += 1
                            neighbor_color = grid[ni][nj]

                            if neighbor_color not in surrounding_colors:
                                surrounding_colors[neighbor_color] = 0
                            surrounding_colors[neighbor_color] += 1
                        elif boundary_counts:
                            # 如果边界也算作包围
                            boundary_color = -1  # 特殊值表示边界
                            if boundary_color not in surrounding_colors:
                                surrounding_colors[boundary_color] = 0
                            surrounding_colors[boundary_color] += 1
                            valid_directions += 1

                    # 找出最主要的包围颜色
                    if surrounding_colors:
                        main_surrounding = max(surrounding_colors.items(), key=lambda x: x[1])
                        main_color, main_count = main_surrounding

                        # 严格模式：必须完全被同一种颜色包围
                        if require_exact_match and main_count < valid_directions:
                            continue

                        # 不能被自己的颜色包围
                        if main_color == center_color:
                            continue

                        # 边界不算作有效的包围色
                        if main_color == -1:
                            continue

                        # 计算包围比例
                        surround_ratio = main_count / valid_directions

                        # 创建模式对象
                        pattern = {
                            'pattern_type': 'four_box_pattern',
                            'center_position': (i, j),
                            'center_color': center_color,
                            'surrounding_color': main_color,
                            'surrounding_ratio': surround_ratio,
                            'pair_id': pair_id,
                            'io_type': io_type,
                            'supporting_pairs': [pair_id],
                            'confidence': surround_ratio,
                            'description': f"颜色{center_color}的对象被颜色{main_color}以{surround_ratio:.2f}的比例包围"
                        }

                        patterns.append(pattern)

    # 聚合相似模式（相同颜色组合）
    aggregated_patterns = {}
    for pattern in patterns:
        key = (pattern['center_color'], pattern['surrounding_color'])

        if key not in aggregated_patterns:
            aggregated_patterns[key] = {
                'pattern_type': 'four_box_pattern',
                'center_color': pattern['center_color'],
                'surrounding_color': pattern['surrounding_color'],
                'instances': [],
                'supporting_pairs': set(),
                'avg_ratio': 0,
                'confidence': 0,
                'description': ""
            }

        aggregated_patterns[key]['instances'].append(pattern)
        aggregated_patterns[key]['supporting_pairs'].add(pattern['pair_id'])

    # 计算聚合模式的统计信息
    result_patterns = []
    for key, agg_pattern in aggregated_patterns.items():
        instances = agg_pattern['instances']
        agg_pattern['avg_ratio'] = sum(p['surrounding_ratio'] for p in instances) / len(instances)
        agg_pattern['supporting_pairs'] = list(agg_pattern['supporting_pairs'])
        agg_pattern['confidence'] = len(agg_pattern['supporting_pairs']) / len(self.objects_by_pair)
        agg_pattern['description'] = f"颜色{agg_pattern['center_color']}的对象常被颜色{agg_pattern['surrounding_color']}包围"

        # 只保留较有信心的模式
        if len(agg_pattern['supporting_pairs']) >= 2 or agg_pattern['avg_ratio'] > 0.8:
            result_patterns.append(agg_pattern)

    return result_patterns

def _reconstruct_grid_from_objects(self, pair_id, io_type):
    """
    从对象列表重建网格（如果需要的话）

    Args:
        pair_id: 数据对ID
        io_type: 输入或输出

    Returns:
        重建的网格，或者None如果无法重建
    """
    # 如果系统中存储了原始网格，可以直接返回
    if hasattr(self, 'original_grids') and pair_id in self.original_grids and io_type in self.original_grids[pair_id]:
        return self.original_grids[pair_id][io_type]

    # 否则尝试从对象列表重建
    objects = []
    if pair_id in self.objects_by_pair and io_type in self.objects_by_pair[pair_id]:
        for obj_id in self.objects_by_pair[pair_id][io_type]:
            if obj_id in self.object_attributes:
                objects.append(self.object_attributes[obj_id])

    if not objects:
        return None

    # 找出网格大小
    max_x, max_y = 0, 0
    for obj in objects:
        if 'position' in obj:
            x, y = obj['position']
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        elif 'bounding_box' in obj:
            x1, y1, x2, y2 = obj['bounding_box']
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

    # 创建空白网格
    grid = [[0 for _ in range(max_y + 1)] for _ in range(max_x + 1)]

    # 填充对象
    for obj in objects:
        if 'color' not in obj:
            continue

        color = obj['color']

        if 'pixels' in obj:
            for x, y in obj['pixels']:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                    grid[x][y] = color
        elif 'bounding_box' in obj and 'full_obj_mask' in obj:
            x1, y1, x2, y2 = obj['bounding_box']
            mask = obj['full_obj_mask']

            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j]:
                        x, y = x1 + i, y1 + j
                        if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                            grid[x][y] = color

    return grid