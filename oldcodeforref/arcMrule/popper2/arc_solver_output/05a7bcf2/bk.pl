% 颜色定义
color_value(0, background).
color_value(1, red).
color_value(2, green).
color_value(4, yellow).
color_value(6, blue).

% 网格结构定义
grid_size(0, 10, 10).  % pair_id, width, height

% 输入网格中的线条
h_line(in_0_0).
line_y_pos(in_0_0, 3).
color(in_0_0, 6).  % 蓝色
v_line(in_0_1).
line_x_pos(in_0_1, 2).
color(in_0_1, 6).  % 蓝色
v_line(in_0_2).
line_x_pos(in_0_2, 7).
color(in_0_2, 6).  % 蓝色

% 黄色对象
yellow_object(in_0_3).
x_min(in_0_3, 4).
y_min(in_0_3, 2).
color(in_0_3, 4).  % 黄色
yellow_object(in_0_4).
x_min(in_0_4, 8).
y_min(in_0_4, 6).
color(in_0_4, 4).  % 黄色

% 辅助谓词
adjacent(X, Y) :- X is Y + 1.
adjacent(X, Y) :- X is Y - 1.

adjacent_pos(X1, Y1, X2, Y2) :- X1 = X2, adjacent(Y1, Y2).
adjacent_pos(X1, Y1, X2, Y2) :- Y1 = Y2, adjacent(X1, X2).

on_grid_line(X, Y) :- h_line(L), line_y_pos(L, Y).
on_grid_line(X, Y) :- v_line(L), line_x_pos(L, X).

grid_intersection(X, Y) :- 
    h_line(HL), line_y_pos(HL, Y),
    v_line(VL), line_x_pos(VL, X).

has_adjacent_yellow(X, Y) :-
    adjacent_pos(X, Y, NX, NY),
    yellow_object(Obj),
    x_min(Obj, NX),
    y_min(Obj, NY).