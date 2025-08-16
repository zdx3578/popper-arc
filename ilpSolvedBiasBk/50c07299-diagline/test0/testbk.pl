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
% === inbelongs ===
inbelongs(p0,pairid0in8982473621897940504,6,9).
inbelongs(p0,pairid0in8982473621897940504,7,8).
inbelongs(p0,pairid0in8982473621897940504,8,7).
inbelongs(p0,pairid0in8982473621897940504,9,6).
% === inpix ===
inpix(p0,6,9,2).
inpix(p0,7,8,2).
inpix(p0,8,7,2).
inpix(p0,9,6,2).
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
is_color_obj(p0,pairid0in8982473621897940504,2).
is_color_obj(p0,pairid0in8982473621897940504,2).


















%%%% ====== 基础 ======
sub(A, B, C):-
	 integer(A),
		integer(B),
	C is A - B.
abs_val(A, B):-
	 integer(A),
	B is abs(A).
lex_leq(X1, Y1, X2, Y2) :-
	X1 < X2;
(X1 =:= X2, Y1 =< Y2).

signum(N, S) :-
	N > 0,
	!,
	S = 1.
signum(N, S) :-
	N < 0,
	!,
	S =  - 1.
signum(_, 0).

%%%% ====== 45° 连续性判定 ======
% 连续：在 X1..X2 间，每一步都有像素（步长 Sx,Sy ∈ {-1,1}）
contiguous_diag(P, C, X1, Y1, X2, Y2) :-
	sub(X2, X1, DX),
	sub(Y2, Y1, DY),
	abs_val(DX, ADx),
	abs_val(DY, ADy),
	ADx =:= ADy,
	% 45°
	signum(DX, Sx),
	signum(DY, Sy),
	forall(between(0, ADx, K),
		(X is X1 + K*Sx,
			Y is Y1 + K*Sy,
			 inpix(P, X, Y, C))).

% 无法向两端再延一步（极大性）
not_extendable_head(P, C, X1, Y1, X2, Y2) :-
	signum(X2 - X1, Sx),
	signum(Y2 - Y1, Sy),
	X0 is X1 - Sx,
	Y0 is Y1 - Sy,
	 \+  inpix(P, X0, Y0, C).

not_extendable_tail(P, C, X1, Y1, X2, Y2) :-
	signum(X2 - X1, Sx),
	signum(Y2 - Y1, Sy),
	X3 is X2 + Sx,
	Y3 is Y2 + Sy,
	 \+  inpix(P, X3, Y3, C).

%%%% ====== 线段对象：最大连续 45° 段（含单像素退化） ======
% 非零长度且最大
% ===== 非零长度最大连续段 =====
seg(P, C, X1, Y1, X2, Y2) :-
	 inpix(P, X1, Y1, C),
	 inpix(P, X2, Y2, C),
	lex_leq(X1, Y1, X2, Y2),
	% 现在变量已绑定，可以安全用 </2, =</2
	sub(X2, X1, DX),
	sub(Y2, Y1, DY),
	abs_val(DX, ADx),
	abs_val(DY, ADy),
	ADx =:= ADy,
	ADx > 0,
	% 45° 且非零长度
	signum(DX, Sx),
	signum(DY, Sy),
	% 连续性：两端之间每一步都在 inpix 里
	forall(between(0, ADx, K),
		(X is X1 + K*Sx,
			Y is Y1 + K*Sy,
			 inpix(P, X, Y, C))),
	% 极大：两端再走一步就不是该色
	X0 is X1 - Sx,
	Y0 is Y1 - Sy,
	 \+  inpix(P, X0, Y0, C),
	X3 is X2 + Sx,
	Y3 is Y2 + Sy,
	 \+  inpix(P, X3, Y3, C),
	!.

% ===== 单像素退化段（该色仅此一个像素）=====
seg(P, C, X, Y, X, Y) :-
	 inpix(P, X, Y, C),
	 \+ (inpix(P, U, V, C),
		(U \= X;
	V \= Y)),
	!.


%%%% ====== 从“词典序较小端”反向延长（生成 Lpix+1 个点） ======
seg_extend_point(X1, Y1, X2, Y2, Xo, Yo) :-
	sub(X2, X1, DX),
	abs_val(DX, ADx),
	% ADx = |X2-X1|
	(ADx > 0 ->
	signum(DX, Sx),
		sub(Y2, Y1, DY),
		signum(DY, Sy);
	% 单像素：四个对角方向任取其一（若要“观测优先”，可在此改为从图中抽方向）
	Sx =  - 1,
		Sy = 1% 默认 NE，简洁稳定
		),
	Kmax is ADx + 2,
	% Lpix=ADx+1 → Lpix+1=ADx+2
	between(1, Kmax, K),
	% 统一从“词典序较小端”(X1,Y1) 反向推进
	Xo is X1 - K*Sx,
	Yo is Y1 - K*Sy.
