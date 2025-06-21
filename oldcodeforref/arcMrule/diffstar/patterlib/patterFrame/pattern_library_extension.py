"""
模式库扩展示例：添加自定义模式
"""

def extend_pattern_framework(framework):
    """向现有框架添加自定义模式"""
    # 注册自定义的网格线模式
    framework.register_pattern(
        pattern_id='grid_line_pattern',
        category='spatial',
        name='网格线模式',
        description='等距离的水平或垂直线形成网格结构',
        detector=detect_grid_line_pattern,
        applier=apply_grid_line_pattern,
        similarity_func=calculate_grid_line_similarity,
        parameters={
            'min_lines': 2,
            'max_gap': 3,
            'min_length_ratio': 0.5
        },
        priority=65
    )
    
    # 注册自定义的角落模式
    framework.register_pattern(
        pattern_id='corner_pattern',
        category='spatial',
        name='角落模式',
        description='对象位于网格的四个角落位置',
        detector=detect_corner_pattern,
        applier=apply_corner_pattern,
        similarity_func=calculate_corner_similarity,
        parameters={
            'corner_distance': 1,
            'require_all_corners': False,
            'require_same_color': True
        },
        priority=60
    )
    
    return framework

def detect_grid_line_pattern(input_data, params):
    """检测网格线模式"""
    grid = input_data.get('grid')
    if not grid:
        return []
    
    min_lines = params.get('min_lines', 2)
    max_gap = params.get('max_gap', 3)
    min_length_ratio = params.get('min_length_ratio', 0.5)
    
    patterns = []
    height, width = len(grid), len(grid[0])
    
    # 检测水平线
    horizontal_lines = []
    for i in range(height):
        lines_in_row = []
        current_line = {'start': -1, 'end': -1, 'color': None}
        
        for j in range(width):
            color = grid[i][j]
            
            # 如果不是背景，检查是否是线的一部分
            if color != 0:
                # 开始新线或继续当前线
                if current_line['start'] == -1:
                    current_line = {'start': j, 'end': j, 'color': color}
                elif color == current_line['color']:
                    current_line['end'] = j
                else:
                    # 颜色变化，结束当前线
                    if current_line['end'] - current_line['start'] >= min_length_ratio * width:
                        lines_in_row.append(current_line)
                    current_line = {'start': j, 'end': j, 'color': color}
        
        # 检查最后一条线
        if current_line['start'] != -1 and current_line['end'] - current_line['start'] >= min_length_ratio * width:
            lines_in_row.append(current_line)
        
        # 添加到行线列表
        if lines_in_row:
            horizontal_lines.append({'row': i, 'lines': lines_in_row})
    
    # 查找等距的水平线组
    h_line_groups = []
    for i in range(len(horizontal_lines)):
        for j in range(i + 1, len(horizontal_lines)):
            gap = horizontal_lines[j]['row'] - horizontal_lines[i]['row']
            if gap <= max_gap:
                # 检查是否有更多等距线
                group = [horizontal_lines[i], horizontal_lines[j]]
                next_row = horizontal_lines[j]['row'] + gap
                
                for k in range(j + 1, len(horizontal_lines)):
                    if horizontal_lines[k]['row'] == next_row:
                        group.append(horizontal_lines[k])
                        next_row += gap
                
                if len(group) >= min_lines:
                    h_line_groups.append({
                        'lines': group,
                        'gap': gap,
                        'orientation': 'horizontal'
                    })
    
    # 类似地检测垂直线...
    
    # 将线组转换为模式
    for group in h_line_groups:  # 加上v_line_groups
        pattern = {
            'pattern_type': 'grid_line_pattern',
            'orientation': group['orientation'],
            'gap': group['gap'],
            'lines': group['lines'],
            'confidence': len(group['lines']) / (height if group['orientation'] == 'horizontal' else width)
        }
        
        patterns.append(pattern)
    
    return patterns

# 其他函数实现... apply_grid_line_pattern, calculate_grid_line_similarity, 
# detect_corner_pattern, apply_corner_pattern, calculate_corner_similarity