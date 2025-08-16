from textwrap import dedent

def output_bk_diagline():
    content = dedent(r'''



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



    ''')
    return content

# # 用法示例
# lines = []
# lines.append(output_bk_diagline())

# # 写入文件
# with open('rules.pl', 'w', encoding='utf-8', newline='\n') as f:
#     f.write('\n'.join(lines))
