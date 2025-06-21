def _apply_4box_pattern_rule(self, rule, grid):
    """
    应用4-Box模式规则到网格
    
    Args:
        rule: 要应用的规则
        grid: 要修改的网格
        
    Returns:
        修改后的网格
    """
    # 提取规则参数
    center_color = rule.get('center_color')
    surrounding_color = rule.get('surrounding_color')
    transformation = rule.get('transformation', {})
    transform_type = transformation.get('type')
    parameters = transformation.get('parameters', {})
    
    # 找出所有匹配的模式实例
    instances = self._find_pattern_instances(grid, center_color, surrounding_color)
    
    # 应用变换
    if instances:
        if transform_type == 'color_change' and 'new_color' in parameters:
            new_color = parameters['new_color']
            for i, j in instances:
                grid[i][j] = new_color
        
        elif transform_type == 'remove':
            for i, j in instances:
                grid[i][j] = 0  # 假设0是背景色
        
        elif transform_type == 'expand':
            # 扩展中心颜色到周围
            for i, j in instances:
                # 四个方向
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                    if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and grid[ni][nj] == surrounding_color:
                        grid[ni][nj] = center_color
    
    return grid