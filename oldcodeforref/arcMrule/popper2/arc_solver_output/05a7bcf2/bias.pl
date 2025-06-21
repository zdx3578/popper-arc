% 定义目标关系
head(extends_to_grid/1).
head(yellow_fills_vertical/1).
head(green_at_intersections/1).
head(extends_to_grid/1).
head(green_at_intersections/1).
head(extends_to_grid/1).
head(green_at_intersections/1).

% 背景知识谓词
body(grid_size/3).
body(color_value/2).
body(h_line/1).
body(v_line/1).
body(line_y_pos/2).
body(line_x_pos/2).
body(yellow_object/1).
body(x_min/2).
body(y_min/2).
body(color/2).
body(on_grid_line/2).
body(grid_intersection/2).
body(has_adjacent_yellow/2).

% 搜索约束
max_vars(6).
max_body(8).
max_clauses(4).