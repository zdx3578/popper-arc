% 综合修复的05a7bcf2任务背景知识
% 生成时间: 2025-06-13 09:35:19

% 全面的不连续谓词声明
:- discontiguous yellow_object/1.
:- discontiguous x_min/2.
:- discontiguous y_min/2.
:- discontiguous x_pos/2.
:- discontiguous y_pos/2.
:- discontiguous width/2.
:- discontiguous height/2.
:- discontiguous color/2.
:- discontiguous h_line/1.
:- discontiguous v_line/1.
:- discontiguous line_y_pos/2.
:- discontiguous line_x_pos/2.
:- discontiguous on_grid_line/2.
:- discontiguous grid_intersection/2.
:- discontiguous adjacent/2.
:- discontiguous adjacent_pos/4.
:- discontiguous fills_column/2.
:- discontiguous column/2.
:- discontiguous grid_cell/7.
:- discontiguous green_point/1.

% 基础事实 - 网格大小
grid_size(0, 10, 10).  % pair_id, width, height

% 颜色定义
color_value(0, background).
color_value(1, red).
color_value(2, green).
color_value(4, yellow).
color_value(6, blue).

% 水平线定义

% 垂直线定义

% 黄色对象定义
yellow_object(in_0_0).
x_min(in_0_0, 2).
y_min(in_0_0, 4).
color(in_0_0, 4).  % 黄色
yellow_object(in_0_1).
x_min(in_0_1, 2).
y_min(in_0_1, 11).
color(in_0_1, 4).  % 黄色
yellow_object(in_0_2).
x_min(in_0_2, 4).
y_min(in_0_2, 18).
color(in_0_2, 4).  % 黄色
yellow_object(in_0_3).
x_min(in_0_3, 3).
y_min(in_0_3, 25).
color(in_0_3, 4).  % 黄色

% 输出水平线定义

% 输出垂直线定义

% 绿色点定义
green_point(out_0_0).
x_pos(out_0_0, 20).
y_pos(out_0_0, 0).
color(out_0_0, 2).  % 绿色
green_point(out_0_1).
x_pos(out_0_1, 19).
y_pos(out_0_1, 1).
color(out_0_1, 2).  % 绿色
green_point(out_0_2).
x_pos(out_0_2, 20).
y_pos(out_0_2, 1).
color(out_0_2, 2).  % 绿色
green_point(out_0_3).
x_pos(out_0_3, 18).
y_pos(out_0_3, 2).
color(out_0_3, 2).  % 绿色
green_point(out_0_4).
x_pos(out_0_4, 19).
y_pos(out_0_4, 2).
color(out_0_4, 2).  % 绿色
green_point(out_0_5).
x_pos(out_0_5, 20).
y_pos(out_0_5, 2).
color(out_0_5, 2).  % 绿色
green_point(out_0_6).
x_pos(out_0_6, 19).
y_pos(out_0_6, 3).
color(out_0_6, 2).  % 绿色
green_point(out_0_7).
x_pos(out_0_7, 20).
y_pos(out_0_7, 3).
color(out_0_7, 2).  % 绿色
green_point(out_0_8).
x_pos(out_0_8, 29).
y_pos(out_0_8, 4).
color(out_0_8, 2).  % 绿色
green_point(out_0_9).
x_pos(out_0_9, 29).
y_pos(out_0_9, 5).
color(out_0_9, 2).  % 绿色
green_point(out_0_10).
x_pos(out_0_10, 18).
y_pos(out_0_10, 6).
color(out_0_10, 2).  % 绿色
green_point(out_0_11).
x_pos(out_0_11, 19).
y_pos(out_0_11, 6).
color(out_0_11, 2).  % 绿色
green_point(out_0_12).
x_pos(out_0_12, 20).
y_pos(out_0_12, 6).
color(out_0_12, 2).  % 绿色
green_point(out_0_13).
x_pos(out_0_13, 19).
y_pos(out_0_13, 7).
color(out_0_13, 2).  % 绿色
green_point(out_0_14).
x_pos(out_0_14, 20).
y_pos(out_0_14, 7).
color(out_0_14, 2).  % 绿色
green_point(out_0_15).
x_pos(out_0_15, 17).
y_pos(out_0_15, 8).
color(out_0_15, 2).  % 绿色
green_point(out_0_16).
x_pos(out_0_16, 18).
y_pos(out_0_16, 8).
color(out_0_16, 2).  % 绿色
green_point(out_0_17).
x_pos(out_0_17, 19).
y_pos(out_0_17, 8).
color(out_0_17, 2).  % 绿色
green_point(out_0_18).
x_pos(out_0_18, 20).
y_pos(out_0_18, 8).
color(out_0_18, 2).  % 绿色
green_point(out_0_19).
x_pos(out_0_19, 18).
y_pos(out_0_19, 9).
color(out_0_19, 2).  % 绿色
green_point(out_0_20).
x_pos(out_0_20, 19).
y_pos(out_0_20, 9).
color(out_0_20, 2).  % 绿色
green_point(out_0_21).
x_pos(out_0_21, 20).
y_pos(out_0_21, 9).
color(out_0_21, 2).  % 绿色
green_point(out_0_22).
x_pos(out_0_22, 20).
y_pos(out_0_22, 10).
color(out_0_22, 2).  % 绿色
green_point(out_0_23).
x_pos(out_0_23, 28).
y_pos(out_0_23, 11).
color(out_0_23, 2).  % 绿色
green_point(out_0_24).
x_pos(out_0_24, 29).
y_pos(out_0_24, 11).
color(out_0_24, 2).  % 绿色
green_point(out_0_25).
x_pos(out_0_25, 27).
y_pos(out_0_25, 12).
color(out_0_25, 2).  % 绿色
green_point(out_0_26).
x_pos(out_0_26, 28).
y_pos(out_0_26, 12).
color(out_0_26, 2).  % 绿色
green_point(out_0_27).
x_pos(out_0_27, 29).
y_pos(out_0_27, 12).
color(out_0_27, 2).  % 绿色
green_point(out_0_28).
x_pos(out_0_28, 19).
y_pos(out_0_28, 13).
color(out_0_28, 2).  % 绿色
green_point(out_0_29).
x_pos(out_0_29, 20).
y_pos(out_0_29, 13).
color(out_0_29, 2).  % 绿色
green_point(out_0_30).
x_pos(out_0_30, 20).
y_pos(out_0_30, 14).
color(out_0_30, 2).  % 绿色
green_point(out_0_31).
x_pos(out_0_31, 19).
y_pos(out_0_31, 15).
color(out_0_31, 2).  % 绿色
green_point(out_0_32).
x_pos(out_0_32, 20).
y_pos(out_0_32, 15).
color(out_0_32, 2).  % 绿色
green_point(out_0_33).
x_pos(out_0_33, 20).
y_pos(out_0_33, 16).
color(out_0_33, 2).  % 绿色
green_point(out_0_34).
x_pos(out_0_34, 20).
y_pos(out_0_34, 17).
color(out_0_34, 2).  % 绿色
green_point(out_0_35).
x_pos(out_0_35, 28).
y_pos(out_0_35, 18).
color(out_0_35, 2).  % 绿色
green_point(out_0_36).
x_pos(out_0_36, 29).
y_pos(out_0_36, 18).
color(out_0_36, 2).  % 绿色
green_point(out_0_37).
x_pos(out_0_37, 27).
y_pos(out_0_37, 19).
color(out_0_37, 2).  % 绿色
green_point(out_0_38).
x_pos(out_0_38, 28).
y_pos(out_0_38, 19).
color(out_0_38, 2).  % 绿色
green_point(out_0_39).
x_pos(out_0_39, 29).
y_pos(out_0_39, 19).
color(out_0_39, 2).  % 绿色
green_point(out_0_40).
x_pos(out_0_40, 27).
y_pos(out_0_40, 20).
color(out_0_40, 2).  % 绿色
green_point(out_0_41).
x_pos(out_0_41, 28).
y_pos(out_0_41, 20).
color(out_0_41, 2).  % 绿色
green_point(out_0_42).
x_pos(out_0_42, 29).
y_pos(out_0_42, 20).
color(out_0_42, 2).  % 绿色
green_point(out_0_43).
x_pos(out_0_43, 20).
y_pos(out_0_43, 21).
color(out_0_43, 2).  % 绿色
green_point(out_0_44).
x_pos(out_0_44, 20).
y_pos(out_0_44, 22).
color(out_0_44, 2).  % 绿色
green_point(out_0_45).
x_pos(out_0_45, 19).
y_pos(out_0_45, 23).
color(out_0_45, 2).  % 绿色
green_point(out_0_46).
x_pos(out_0_46, 20).
y_pos(out_0_46, 23).
color(out_0_46, 2).  % 绿色
green_point(out_0_47).
x_pos(out_0_47, 20).
y_pos(out_0_47, 24).
color(out_0_47, 2).  % 绿色
green_point(out_0_48).
x_pos(out_0_48, 28).
y_pos(out_0_48, 25).
color(out_0_48, 2).  % 绿色
green_point(out_0_49).
x_pos(out_0_49, 29).
y_pos(out_0_49, 25).
color(out_0_49, 2).  % 绿色
green_point(out_0_50).
x_pos(out_0_50, 20).
y_pos(out_0_50, 26).
color(out_0_50, 2).  % 绿色
green_point(out_0_51).
x_pos(out_0_51, 19).
y_pos(out_0_51, 27).
color(out_0_51, 2).  % 绿色
green_point(out_0_52).
x_pos(out_0_52, 20).
y_pos(out_0_52, 27).
color(out_0_52, 2).  % 绿色
green_point(out_0_53).
x_pos(out_0_53, 19).
y_pos(out_0_53, 28).
color(out_0_53, 2).  % 绿色
green_point(out_0_54).
x_pos(out_0_54, 20).
y_pos(out_0_54, 28).
color(out_0_54, 2).  % 绿色
green_point(out_0_55).
x_pos(out_0_55, 20).
y_pos(out_0_55, 29).
color(out_0_55, 2).  % 绿色

% 网格单元格定义
grid_cell(0, 0, 0, 0, 0, 2, 2).  % pair_id, cell_row, cell_col, left, top, right, bottom
grid_cell(0, 0, 1, 3, 0, 6, 2).
grid_cell(0, 0, 2, 8, 0, 9, 2).
grid_cell(0, 1, 0, 0, 3, 2, 6).
grid_cell(0, 1, 1, 3, 3, 6, 6).
grid_cell(0, 1, 2, 8, 3, 9, 6).
grid_cell(0, 2, 0, 0, 7, 2, 9).
grid_cell(0, 2, 1, 3, 7, 6, 9).
grid_cell(0, 2, 2, 8, 7, 9, 9).

% 辅助谓词定义
% 列定义谓词 - bias中使用但未定义的谓词
column(C, X) :- grid_cell(_, _, C, X, _, _, _).
% 填充列谓词 - bias中使用但未定义的谓词
fills_column(Col, X) :-
    number(Col), number(X),
    column(Col, X),
    yellow_object(YObj),
    x_min(YObj, X).

% 安全的on_grid_line谓词定义
on_grid_line(X, Y) :- 
    number(X), number(Y),
    h_line(L), 
    line_y_pos(L, Y).

on_grid_line(X, Y) :- 
    number(X), number(Y),
    v_line(L), 
    line_x_pos(L, X).

% 安全的adjacent谓词
adjacent(X, Y) :- number(X), number(Y), Y is X + 1.
adjacent(X, Y) :- number(X), number(Y), Y is X - 1.
adjacent(X, Y) :- number(Y), X is Y + 1.
adjacent(X, Y) :- number(Y), X is Y - 1.

% 安全的adjacent_pos谓词
adjacent_pos(X1, Y1, X2, Y2) :-
    number(X1), number(Y1), number(X2), number(Y2),
    (
        % 垂直相邻 - 严格分情况讨论
        (X1 = X2, Y2 is Y1 + 1, Y2 >= 0, Y2 < 10);
        (X1 = X2, Y2 is Y1 - 1, Y2 >= 0, Y2 < 10);
        % 水平相邻 - 严格分情况讨论
        (Y1 = Y2, X2 is X1 + 1, X2 >= 0, X2 < 10);
        (Y1 = Y2, X2 is X1 - 1, X2 >= 0, X2 < 10)
    ),
    !.

adjacent_pos(X1, Y1, X2, Y2) :- 
    number(X1), number(Y1), number(X2), number(Y2),
    Y1 = Y2, adjacent(X1, X2).

% 安全的网格交点谓词
grid_intersection(X, Y) :-
    number(X), number(Y),
    X >= 0, Y >= 0, X < 10, Y < 10,
    v_line(V), line_x_pos(V, X),
    h_line(H), line_y_pos(H, Y),
    !.

% 检查周围是否有黄色对象
has_adjacent_yellow(X, Y) :-
    number(X), number(Y),
    X >= 0, X < 10, Y >= 0, Y < 10,
    adjacent_pos(X, Y, NX, NY),
    yellow_object(Obj),
    x_min(Obj, NX),
    y_min(Obj, NY),
    !.

% 应该为绿色的点
should_be_green(X, Y) :-
    number(X), number(Y),
    grid_intersection(X, Y),
    has_adjacent_yellow(X, Y).

% 方向声明
direction(grid_size, (in, out, out)).
direction(h_line, (in)).
direction(v_line, (in)).
direction(line_y_pos, (in, out)).
direction(line_x_pos, (in, out)).
direction(yellow_object, (in)).
direction(green_point, (in)).
direction(x_min, (in, out)).
direction(y_min, (in, out)).
direction(x_pos, (in, out)).
direction(y_pos, (in, out)).
direction(color, (in, out)).
direction(on_grid_line, (in, in)).
direction(grid_intersection, (in, in)).
direction(adjacent, (in, in)).
direction(adjacent_pos, (in, in, in, in)).
direction(has_adjacent_yellow, (in, in)).
direction(should_be_green, (in, in)).
direction(fills_column, (in, in)).
direction(column, (in, in)).
direction(grid_cell, (in, in, in, out, out, out, out)).