from textwrap import dedent

def output_bk_diagline():
    content = dedent(r'''
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

    ''')
    return content

# # 用法示例
# lines = []
# lines.append(output_bk_diagline())

# # 写入文件
# with open('rules.pl', 'w', encoding='utf-8', newline='\n') as f:
#     f.write('\n'.join(lines))
