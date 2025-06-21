% 定义目标关系
head_pred(extends_to_grid,1).
head_pred(yellow_fills_vertical,1).
head_pred(green_at_intersections,1).

% 背景知识谓词
body_pred(grid_size,3).
body_pred(color_value,2).
body_pred(h_line,1).
body_pred(v_line,1).
body_pred(line_y_pos,2).
body_pred(line_x_pos,2).
body_pred(yellow_object,1).
body_pred(x_min,2).
body_pred(y_min,2).
body_pred(width,2).
body_pred(height,2).
body_pred(color,2).
body_pred(grid_cell,7).
body_pred(column,2).
body_pred(on_grid_line,2).
body_pred(grid_intersection,2).
body_pred(has_adjacent_yellow,2).
body_pred(should_be_green,2).
body_pred(fills_column,2).
body_pred(adjacent,2).
body_pred(adjacent_pos,4).

% 搜索约束
max_vars(6).
max_body(6).
max_clauses(4).
timeout(30).  % 添加30秒超时
