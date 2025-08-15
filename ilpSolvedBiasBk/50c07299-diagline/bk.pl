:- style_check(-discontiguous).
% === col_0 ===
col_0(0).
% === col_1 ===
col_1(1).
% === col_2 ===
col_2(2).
% === col_3 ===
col_3(3).
% === col_4 ===
col_4(4).
% === col_5 ===
col_5(5).
% === col_6 ===
col_6(6).
% === col_7 ===
col_7(7).
% === col_8 ===
col_8(8).
% === col_9 ===
col_9(9).
% === constant ===
constant(p0,pair).
constant(p1,pair).
constant(p2,pair).
constant(0,coord).
constant(1,coord).
constant(2,coord).
constant(3,coord).
constant(4,coord).
constant(5,coord).
constant(6,coord).
constant(7,coord).
constant(8,coord).
constant(9,coord).
constant(10,coord).
constant(11,coord).
constant(12,coord).
constant(13,coord).
constant(14,coord).
constant(15,coord).
constant(0,color).
constant(1,color).
constant(2,color).
constant(3,color).
constant(4,color).
constant(5,color).
constant(6,color).
constant(7,color).
constant(8,color).
constant(9,color).
constant(0,int).
constant(1,int).
constant(2,int).
constant(3,int).
constant(4,int).
constant(5,int).
constant(6,int).
constant(7,int).
constant(8,int).
constant(9,int).
constant(pairid0in8982473621897940504,obj).
constant(pairid1in8982473621897940504,obj).
constant(pairid2in8982473621897940504,obj).
% === inbelongs ===
inbelongs(p0,pairid0in8982473621897940504,13,2).
inbelongs(p0,pairid0in8982473621897940504,14,1).
inbelongs(p1,pairid1in8982473621897940504,15,0).
inbelongs(p2,pairid2in8982473621897940504,10,5).
inbelongs(p2,pairid2in8982473621897940504,11,4).
inbelongs(p2,pairid2in8982473621897940504,12,3).
% === inpix ===
inpix(p0,13,2,2).
inpix(p0,14,1,2).
inpix(p1,15,0,2).
inpix(p2,10,5,2).
inpix(p2,11,4,2).
inpix(p2,12,3,2).
% === int_0 ===
int_0(0).
% === int_1 ===
int_1(1).
% === int_2 ===
int_2(2).
% === int_3 ===
int_3(3).
% === int_4 ===
int_4(4).
% === int_5 ===
int_5(5).
% === int_6 ===
int_6(6).
% === int_7 ===
int_7(7).
% === int_8 ===
int_8(8).
% === int_9 ===
int_9(9).
% === is_color_obj ===
is_color_obj(p0,pairid0in8982473621897940504,2).
is_color_obj(p0,pairid0in8982473621897940504,2).
is_color_obj(p1,pairid1in8982473621897940504,2).
is_color_obj(p2,pairid2in8982473621897940504,2).
is_color_obj(p2,pairid2in8982473621897940504,2).
is_color_obj(p2,pairid2in8982473621897940504,2).

        %%%% ========= 基础工具（最小集） =========
sub(A,B,C):- integer(A),integer(B), C is A-B.
abs_val(A,B):- integer(A), B is abs(A).
lex_leq(X1,Y1,X2,Y2) :- X1 < X2 ; (X1 =:= X2, Y1 =< Y2).

signum(N,S) :- N > 0, !, S = 1.
signum(N,S) :- N < 0, !, S = -1.
signum(_,0).

% 四个对角基础方向（零长度回退用）
diag_basis(-1, 1).  % NE：行-1，列+1
diag_basis( 1,-1).  % SW：行+1，列-1
diag_basis( 1, 1).  % SE：行+1，列+1
diag_basis(-1,-1).  % NW：行-1，列-1

%%%% ========= 方向（宽松版） =========
% 非零长度且 45°：由差分符号唯一确定
% 零长度（单像素）：枚举四个对角方向（任意）
diag_dir(X1,Y1,X2,Y2,Sx,Sy) :-
  sub(X2,X1,DX), sub(Y2,Y1,DY),
  abs_val(DX,ADx), abs_val(DY,ADy),
  ( ADx =:= ADy, ADx > 0 ->
      signum(DX,Sx), signum(DY,Sy)
  ; X1 =:= X2, Y1 =:= Y2 ->
      diag_basis(Sx,Sy)
  ).

%%%% ========= 最远对（有线段优先；否则单像素兜底） =========
% 严格最远（只接受非零长度）
diag_pair_far_strict(P,C,X1,Y1,X2,Y2) :-
  inpix(P,X1,Y1,C), inpix(P,X2,Y2,C),
  lex_leq(X1,Y1,X2,Y2),
  sub(X2,X1,DX), sub(Y2,Y1,DY),
  abs_val(DX,ADx), abs_val(DY,ADy),
  ADx =:= ADy, ADx > 0,
  \+ ( inpix(P,U1,V1,C), inpix(P,U2,V2,C),
       lex_leq(U1,V1,U2,V2),
       sub(U2,U1,DX2), sub(V2,V1,DY2),
       abs_val(DX2,A2), abs_val(DY2,B2),
       A2=:=B2, max(A2,B2) > ADx ).

% 单像素（恰有一个像素）
single_pix(P,C,X,Y) :-
  inpix(P,X,Y,C),
  \+ (inpix(P,U,V,C), (U \= X ; V \= Y)).

% 有线段就取“最远对”；没有线段才退化为“单像素对”
diag_pair_far_or_single(P,C,X1,Y1,X2,Y2) :-
  diag_pair_far_strict(P,C,X1,Y1,X2,Y2).
diag_pair_far_or_single(P,C,X,Y,X,Y) :-
  single_pix(P,C,X,Y),
  \+ diag_pair_far_strict(P,C,_,_,_,_).

%%%% ========= 核心：延长“自身 + 1” =========
% 关键点：总是以字典序较小的一端 (X1,Y1) 为“尾端”，
% 沿着反向 (-Sx,-Sy) 推进 K1=1..(Lpix+1)（Lpix=|X2-X1|+1）
extend_diag_out(P,C,Xo,Yo) :-
  diag_pair_far_or_single(P,C,X1,Y1,X2,Y2),
  diag_dir(X1,Y1,X2,Y2,Sx,Sy),
  sub(X2,X1,DX), abs_val(DX,ADx),
  Lplus is ADx + 2,           % Lpix+1
  ExSx is -Sx,  ExSy is -Sy,  % 从 (X1,Y1) 反向延
  between(1, Lplus, K1),
  Xo is X1 + K1*ExSx,
  Yo is Y1 + K1*ExSy.

