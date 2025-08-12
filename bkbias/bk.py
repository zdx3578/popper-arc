from textwrap import dedent

def output_bk_diagline():
    content = dedent(r'''
        % 安全算术/比较
        sub(A,B,C):- integer(A),integer(B), C is A-B.
        abs_val(A,B):- integer(A), B is abs(A).
        geq(A,B):- integer(A),integer(B), A>=B.
        leq(A,B):- integer(A),integer(B), A=<B.
        eq(A,B):- integer(A),integer(B), A=:=B.

        same_sign(A,B):- int_0(Z), geq(A,Z), geq(B,Z).
        same_sign(A,B):- int_0(Z), leq(A,Z), leq(B,Z).

        lex_leq(X1,Y1,X2,Y2):- X1<X2 ; (X1=:=X2, Y1=<Y2).

        safe_mod(A,B,R):- integer(A), integer(B), B>0, R is A mod B.  % SWI 文档见 mod/2

        % step(1). step(2). step(3). step(4). step(5).

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
    ''')
    return content

# # 用法示例
# lines = []
# lines.append(output_bk_diagline())

# # 写入文件
# with open('rules.pl', 'w', encoding='utf-8', newline='\n') as f:
#     f.write('\n'.join(lines))
