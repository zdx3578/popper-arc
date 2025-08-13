body_pred(int_1,1).
type(int_1,(int,)).
direction(int_1,(out,)).
body_pred(int_2,1).
type(int_2,(int,)).
direction(int_2,(out,)).
body_pred(int_0,1).
type(int_0,(int,)).
direction(int_0,(out,)).

max_vars(9).
max_body(3).
max_clauses(3).
type(pair). type(color). type(coord). type(epcoord).
type(stride).  % 用于 step/1
% type(obj).
type(int).

head_pred(outpix,4).
type(outpix,(pair,coord,coord,color,)).
direction(outpix,(in,out,out,out,)).

body_pred(inpix,4).
type(inpix,(pair,coord,coord,color,)).
direction(inpix,(in,out,out,out,)).

% body_pred(diag_pair_near,6).
% type(diag_pair_near,(pair,color,epcoord,epcoord,epcoord,epcoord,)).
% direction(diag_pair_near,(in,out,out,out,out,out,)).

body_pred(diag_pair_far,6).
type(diag_pair_far,(pair,color,epcoord,epcoord,epcoord,epcoord,)).
direction(diag_pair_far,(in,out,out,out,out,out,)).

% 步长候选（显式常量，小域即可）
% body_pred(step,1).
% type(step,(stride,)).
% direction(step,(out,)).

% 含步长的对角线生成（含端点，步进为 S）
body_pred(on_diag_between_k,7).
type(on_diag_between_k,(coord,coord,epcoord,epcoord,epcoord,epcoord,int,)).
direction(on_diag_between_k,(out,out,in,in,in,in,in,)).



body_pred(int_1,1).
type(int_1,(int,)).
direction(int_1,(out,)).
body_pred(int_2,1).
type(int_2,(int,)).
direction(int_2,(out,)).
% body_pred(int_3,1).
% type(int_3,(int,)).
% direction(int_3,(out,)).
% body_pred(int_4,1).
% type(int_4,(int,)).
% direction(int_4,(out,)).
% body_pred(int_5,1).
% type(int_5,(int,)).
% direction(int_5,(out,)).
% body_pred(int_6,1).
% type(int_6,(int,)).
% direction(int_6,(out,)).
% body_pred(int_7,1).
% type(int_7,(int,)).
% direction(int_7,(out,)).
% body_pred(int_8,1).
% type(int_8,(int,)).
% direction(int_8,(out,)).
% body_pred(int_9,1).
% type(int_9,(int,)).
% direction(int_9,(out,)).
body_pred(int_0,1).
type(int_0,(int,)).
direction(int_0,(out,)).

