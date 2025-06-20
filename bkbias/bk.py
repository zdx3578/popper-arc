########
#
'''
bk 首先生成task 多数据对的 所有像素 事实 -去背景
2 对象级别的事实

'''
#
# 线条  方向  关系垂直  or  并行

'''
% 目标
head_pred(target/5).            % target(Grid,X,Y,ColorIn,ColorOut)

% 输入事实
body_pred(cell/4).
body_pred(cyan_bar/5).
body_pred(orientation/5).
body_pred(perp_dir/2).
body_pred(extend_until/7).      % extend_until(G,X,Y,Dir,StopColor,FillColor,X2Y2)

% 工具
body_pred(succ/2). body_pred(leq/2).
max_body(6). max_vars(7).
'''

'''
1
'''