:- style_check(-discontiguous).

int_0(0). int_1(1).

% 安全算术/比较
sub(A,B,C):- integer(A),integer(B), C is A-B.
abs_val(A,B):- integer(A), B is abs(A).
geq(A,B):- integer(A),integer(B), A>=B.
leq(A,B):- integer(A),integer(B), A=<B.
eq(A,B):- integer(A),integer(B), A=:=B.

same_sign(A,B):- int_0(Z), geq(A,Z), geq(B,Z).
same_sign(A,B):- int_0(Z), leq(A,Z), leq(B,Z).

lex_leq(X1,Y1,X2,Y2):- X1<X2 ; (X1=:=X2, Y1=<Y2).


safe_mod(A,B,R):- integer(A), integer(B), B>0, R is A mod B.  % SWI 文档见 mod/2 :contentReference[oaicite:3]{index=3}
%  step(1). step(2). step(3). step(4). step(5).


on_diag_between_k(X,Y,X1,Y1,X2,Y2,S):-
  % 端点必须成 45°
  sub(X2,X1,DXe), sub(Y2,Y1,DYe),
  abs_val(DXe,ADx), abs_val(DYe,ADy), eq(ADx,ADy),
  % S >= 1
  geq(S,1),
  ( ADx =:= 0 ->
      % 零长度：只回端点自身（偏移=0，0 可被任意 S 整除）
      X = X1, Y = Y1
  ; % 一般情况：沿端点方向，且 |偏移| ∈ [0, ADx] 且 |偏移| mod S = 0
    sub(X,X1,DX1), sub(Y,Y1,DY1),
    abs_val(DX1,AD1), abs_val(DY1,AD1),     % |DX1|=|DY1|
    same_sign(DX1,DXe), same_sign(DY1,DYe), % 朝端点方向
    int_0(Z0), geq(AD1,Z0), leq(AD1,ADx),
    safe_mod(AD1,S,R), eq(R,Z0)
  ).

% 端点配对（像素级，等长=45°），Chebyshev 距离 d∞ = max(|dx|,|dy|)
diag_pair_near(P,C,X1,Y1,X2,Y2) :-             % 最近
  inpix(P,X1,Y1,C), inpix(P,X2,Y2,C), lex_leq(X1,Y1,X2,Y2),
  sub(X2,X1,DX), sub(Y2,Y1,DY), abs_val(DX,ADx), abs_val(DY,ADy), ADx=:=ADy,
  % 不存在更近的一对
  \+ (inpix(P,U1,V1,C), inpix(P,U2,V2,C), lex_leq(U1,V1,U2,V2),
      sub(U2,U1,DX2), sub(V2,V1,DY2), abs_val(DX2,A2), abs_val(DY2,B2),
      A2=:=B2, max(A2,B2) < ADx).

diag_pair_far(P,C,X1,Y1,X2,Y2) :-              % 最远
  inpix(P,X1,Y1,C), inpix(P,X2,Y2,C), lex_leq(X1,Y1,X2,Y2),
  sub(X2,X1,DX), sub(Y2,Y1,DY), abs_val(DX,ADx), abs_val(DY,ADy), ADx=:=ADy,
  % 不存在更远的一对
  \+ (inpix(P,U1,V1,C), inpix(P,U2,V2,C), lex_leq(U1,V1,U2,V2),
      sub(U2,U1,DX2), sub(V2,V1,DY2), abs_val(DX2,A2), abs_val(DY2,B2),
      A2=:=B2, max(A2,B2) > ADx).




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
constant(16,coord).
constant(17,coord).
constant(18,coord).
constant(19,coord).
constant(20,coord).
constant(21,coord).
constant(22,coord).
constant(23,coord).
constant(24,coord).
constant(25,coord).
constant(26,coord).
constant(27,coord).
constant(28,coord).
constant(29,coord).
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
constant(pairid0in3722106093910301873,obj).
constant(pairid0in6999552149255831699,obj).
constant(pairid1in3722106093910301873,obj).
constant(pairid1in6999552149255831699,obj).
constant(pairid2in3722106093910301873,obj).
constant(pairid2in6999552149255831699,obj).
% === inbelongs ===

% === inpix ===
inpix(p0,0,4,9).
inpix(p0,0,5,9).
inpix(p0,1,4,9).
inpix(p0,1,5,9).
inpix(p0,1,13,3).
inpix(p0,1,14,3).
inpix(p0,1,15,3).
inpix(p0,2,13,3).
inpix(p0,2,14,3).
inpix(p0,2,15,3).
inpix(p0,3,13,3).
inpix(p0,3,14,3).
inpix(p0,3,15,3).
inpix(p0,4,8,9).
inpix(p0,4,9,9).
inpix(p0,5,8,9).
inpix(p0,5,9,9).
inpix(p0,6,2,1).
inpix(p0,6,3,1).
inpix(p0,6,4,1).
inpix(p0,6,14,4).
inpix(p0,6,15,4).
inpix(p0,7,2,1).
inpix(p0,7,3,1).
inpix(p0,7,4,1).
inpix(p0,7,7,3).
inpix(p0,7,8,3).
inpix(p0,7,9,3).
inpix(p0,7,14,4).
inpix(p0,7,15,4).
inpix(p0,8,2,1).
inpix(p0,8,3,1).
inpix(p0,8,4,1).
inpix(p0,8,7,3).
inpix(p0,8,8,3).
inpix(p0,8,9,3).
inpix(p0,9,7,3).
inpix(p0,9,8,3).
inpix(p0,9,9,3).
inpix(p0,13,9,1).
inpix(p0,13,10,1).
inpix(p0,13,11,1).
inpix(p0,14,6,4).
inpix(p0,14,7,4).
inpix(p0,14,9,1).
inpix(p0,14,10,1).
inpix(p0,14,11,1).
inpix(p0,15,6,4).
inpix(p0,15,7,4).
inpix(p0,15,9,1).
inpix(p0,15,10,1).
inpix(p0,15,11,1).
inpix(p1,2,2,8).
inpix(p1,2,3,8).
inpix(p1,2,4,8).
inpix(p1,2,12,4).
inpix(p1,2,13,4).
inpix(p1,3,2,8).
inpix(p1,3,3,8).
inpix(p1,3,4,8).
inpix(p1,3,12,4).
inpix(p1,3,13,4).
inpix(p1,4,2,8).
inpix(p1,4,3,8).
inpix(p1,4,4,8).
inpix(p1,9,1,9).
inpix(p1,9,2,9).
inpix(p1,9,5,4).
inpix(p1,9,6,4).
inpix(p1,10,1,9).
inpix(p1,10,2,9).
inpix(p1,10,5,4).
inpix(p1,10,6,4).
inpix(p1,11,11,8).
inpix(p1,11,12,8).
inpix(p1,11,13,8).
inpix(p1,12,11,8).
inpix(p1,12,12,8).
inpix(p1,12,13,8).
inpix(p1,13,11,8).
inpix(p1,13,12,8).
inpix(p1,13,13,8).
inpix(p1,14,6,9).
inpix(p1,14,7,9).
inpix(p1,15,6,9).
inpix(p1,15,7,9).
inpix(p2,0,0,0).
inpix(p2,0,1,0).
inpix(p2,1,0,0).
inpix(p2,1,1,0).
inpix(p2,1,8,9).
inpix(p2,1,9,9).
inpix(p2,1,10,9).
inpix(p2,2,8,9).
inpix(p2,2,9,9).
inpix(p2,2,10,9).
inpix(p2,3,8,9).
inpix(p2,3,9,9).
inpix(p2,3,10,9).
inpix(p2,4,4,0).
inpix(p2,4,5,0).
inpix(p2,5,4,0).
inpix(p2,5,5,0).
inpix(p2,6,14,5).
inpix(p2,6,15,5).
inpix(p2,7,2,9).
inpix(p2,7,3,9).
inpix(p2,7,4,9).
inpix(p2,7,7,3).
inpix(p2,7,8,3).
inpix(p2,7,14,5).
inpix(p2,7,15,5).
inpix(p2,8,2,9).
inpix(p2,8,3,9).
inpix(p2,8,4,9).
inpix(p2,8,7,3).
inpix(p2,8,8,3).
inpix(p2,9,2,9).
inpix(p2,9,3,9).
inpix(p2,9,4,9).
inpix(p2,12,12,3).
inpix(p2,12,13,3).
inpix(p2,13,7,5).
inpix(p2,13,8,5).
inpix(p2,13,12,3).
inpix(p2,13,13,3).
inpix(p2,14,7,5).
inpix(p2,14,8,5).
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