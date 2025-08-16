from textwrap import dedent

def output_bias_diagline():
    return dedent(r'''

% ==== 类型（用 newcoord 限制 outpix 坐标只能来自扩展算子）====
type(pair).  type(color).
type(xcoord). type(ycoord).    % 线段端点用
type(newcoord).                % 只能由 seg_extend_point 产生
type(int). type(dircomp).

% ==== 目标 ====
head_pred(outpix,4).
type(outpix,(pair,newcoord,newcoord,color,)).
direction(outpix,(in,out,out,out,)).

% ==== 仅暴露两个“对象级”原子 ====
% 1) seg/6: 提供每个颜色在图中的 “最大 45° 连续线段”（含单像素退化段）
body_pred(seg,6).
type(seg,(pair,color,xcoord,ycoord,xcoord,ycoord,)).
direction(seg,(in,out,out,out,out,out,)).

% 2) seg_extend_point/6: 从线段（词典序较小端）往其反方向，枚举 1..(Lpix+1) 的新像素
body_pred(seg_extend_point,6).
type(seg_extend_point,(xcoord,ycoord,xcoord,ycoord,newcoord,newcoord,)).
direction(seg_extend_point,(in,in,in,in,out,out,)).

% ==== 上限（唯一 1 条 2-体规则）====
max_clauses(1).
max_body(2).
max_vars(10).


    ''')

# 用法：
# lines = []

# with open('bias.pl','w', encoding='utf-8', newline='\n') as f:
#     f.write('\n'.join(lines))
