% 05a7bcf2任务的预定义规则
% 规则1: 扩展网格 - 确保水平和垂直线形成完整网格
extends_to_grid(ID) :- grid_size(ID, W, H), create_grid_lines(ID, W, H).

% 规则2: 垂直填充黄色 - 在包含黄色对象的列中填充黄色
yellow_fills_vertical(ID) :- grid_size(ID, _, _), grid_cell(ID, R, C, L, _, _, _),
    column(C, X), yellow_object(Obj), x_min(Obj, X), fills_column(C, X).

% 规则3: 交叉点变绿 - 在有黄色对象附近的交叉点填充绿色
green_at_intersections(ID) :- grid_size(ID, _, _), grid_intersection(X, Y),
    has_adjacent_yellow(X, Y), should_be_green(X, Y).