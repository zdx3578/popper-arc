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

        signum(N,S) :- N > 0, !, S = 1.
        signum(N,S) :- N < 0, !, S = -1.
        signum(_,0).

        % 含步长 S 的 45° 连线（含端点）
        % 只依赖 SWI 的算术与 between/3；不依赖 same_sign/safe_mod 等
        on_diag_between_k(X,Y,X1,Y1,X2,Y2,S):-
          integer(S), S >= 1,
          DXe is X2 - X1,
          DYe is Y2 - Y1,
          ADx is abs(DXe),
          ADy is abs(DYe),
          ADx =:= ADy,                % 必须 45°
          signum(DXe,Sx),
          signum(DYe,Sy),
          between(0, ADx, K),         % K=0..ADx（含端点）
          0 is K mod S,               % 步长整除
          X is X1 + K*Sx,
          Y is Y1 + K*Sy.

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
