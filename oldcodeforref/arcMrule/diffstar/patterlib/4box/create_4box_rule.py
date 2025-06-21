def _create_four_box_pattern_rule(self, pattern):
    """
    从4-Box模式创建转换规则
    
    Args:
        pattern: 检测到的4-Box模式
        
    Returns:
        创建的规则对象
    """
    # 基本规则信息
    rule = {
        'rule_type': 'spatial_pattern_rule',
        'pattern_type': 'four_box_pattern',
        'center_color': pattern.get('center_color'),
        'surrounding_color': pattern.get('surrounding_color'),
        'supporting_pairs': pattern.get('supporting_pairs', []),
        'confidence': pattern.get('confidence', 0.5),
        'description': pattern.get('description', '')
    }
    
    # 分析训练数据中这种模式的输入-输出变换
    transformations = self._analyze_pattern_transformations(pattern)
    
    # 添加变换规则
    if transformations:
        most_common_transform = max(transformations.items(), key=lambda x: x[1]['count'])
        transform_type, transform_data = most_common_transform
        
        rule['transformation'] = {
            'type': transform_type,
            'parameters': transform_data['parameters'],
            'confidence': transform_data['count'] / sum(t['count'] for t in transformations.values())
        }
        
        # 更新描述
        if transform_type == 'color_change':
            rule['description'] = f"当颜色{rule['center_color']}被颜色{rule['surrounding_color']}包围时，将中心颜色变为{transform_data['parameters'].get('new_color')}"
        elif transform_type == 'remove':
            rule['description'] = f"当颜色{rule['center_color']}被颜色{rule['surrounding_color']}包围时，移除中心对象"
        elif transform_type == 'expand':
            rule['description'] = f"当颜色{rule['center_color']}被颜色{rule['surrounding_color']}包围时，中心对象扩展并替换周围对象"
    
    # 添加到复合规则列表
    self.composite_rules.append(rule)
    
    return rule

def _analyze_pattern_transformations(self, pattern):
    """
    分析模式在训练数据中的变换
    
    Args:
        pattern: 检测到的模式
        
    Returns:
        变换类型及其统计信息的字典
    """
    transformations = {}
    center_color = pattern.get('center_color')
    surrounding_color = pattern.get('surrounding_color')
    
    # 遍历支持该模式的所有数据对
    for pair_id in pattern.get('supporting_pairs', []):
        # 检查输入中的模式实例
        input_instances = []
        if pair_id in self.objects_by_pair and 'input' in self.objects_by_pair[pair_id]:
            input_grid = self._reconstruct_grid_from_objects(pair_id, 'input')
            if input_grid:
                input_instances = self._find_pattern_instances(input_grid, center_color, surrounding_color)
        
        # 检查输出中的模式实例
        output_instances = []
        if pair_id in self.objects_by_pair and 'output' in self.objects_by_pair[pair_id]:
            output_grid = self._reconstruct_grid_from_objects(pair_id, 'output')
            if output_grid:
                output_instances = self._find_pattern_instances(output_grid, center_color, surrounding_color)
        
        # 分析变换
        if input_instances:
            # 情况1: 检查颜色变化
            if len(input_instances) == len(output_instances):
                # 可能是颜色改变
                for i, (ix, iy) in enumerate(input_instances):
                    if i < len(output_instances):
                        ox, oy = output_instances[i]
                        input_color = input_grid[ix][iy]
                        
                        # 检查输出网格中相应位置的颜色
                        if 0 <= ox < len(output_grid) and 0 <= oy < len(output_grid[0]):
                            output_color = output_grid[ox][oy]
                            
                            if input_color != output_color:
                                transform_type = 'color_change'
                                if transform_type not in transformations:
                                    transformations[transform_type] = {
                                        'count': 0,
                                        'parameters': {'new_color': output_color}
                                    }
                                transformations[transform_type]['count'] += 1
            
            # 情况2: 检查移除
            if len(input_instances) > len(output_instances):
                removed_count = 0
                for ix, iy in input_instances:
                    found_in_output = False
                    for ox, oy in output_instances:
                        if ix == ox and iy == oy:
                            found_in_output = True
                            break
                    
                    if not found_in_output:
                        removed_count += 1
                
                if removed_count > 0:
                    transform_type = 'remove'
                    if transform_type not in transformations:
                        transformations[transform_type] = {'count': 0, 'parameters': {}}
                    transformations[transform_type]['count'] += removed_count
            
            # 情况3: 检查扩展
            if len(input_instances) < len(output_instances):
                transform_type = 'expand'
                if transform_type not in transformations:
                    transformations[transform_type] = {'count': 0, 'parameters': {}}
                transformations[transform_type]['count'] += 1
    
    return transformations

def _find_pattern_instances(self, grid, center_color, surrounding_color):
    """
    在网格中找出所有符合指定颜色配置的4-Box模式实例
    
    Args:
        grid: 要检查的网格
        center_color: 中心颜色
        surrounding_color: 周围颜色
        
    Returns:
        模式实例位置列表 [(x1,y1), (x2,y2), ...]
    """
    instances = []
    height, width = len(grid), len(grid[0])
    
    for i in range(height):
        for j in range(width):
            if grid[i][j] == center_color:
                # 检查四个方向
                directions = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                surrounded = True
                
                for ni, nj in directions:
                    if not (0 <= ni < height and 0 <= nj < width and grid[ni][nj] == surrounding_color):
                        surrounded = False
                        break
                
                if surrounded:
                    instances.append((i, j))
    
    return instances