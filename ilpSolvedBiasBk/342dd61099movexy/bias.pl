%%%%%%%%%%%%  Head / Body signatures  %%%%%%%%%%%%

head_pred(outpix,4).                 % 目标：outpix(Pair,ROut,COut,Color)

body_pred(inpix,4).                  % 已知输入像素
body_pred(add,6).                    % 平移 (dx,dy)

type(outpix,(pair,coord,coord,color,)).
type(inpix,(pair,coord,coord,color,)).
type(add,(int,int,coord,coord,coord,coord,)).  % (DX,DY,Rin,Cin,Rout,Cout)

direction(outpix,(in,out,out,in,)).     % Pair & Color 已知 → 求 ROut,COut
direction(inpix,(in,out,out,in,)).      % 已知
direction(add,(in,in,in,in,out,out,)).  % Rout,COut 是输出



body_pred(col_1,1).
type(col_1,(color,)).
direction(col_1,(out,)).
body_pred(col_2,1).
type(col_2,(color,)).
direction(col_2,(out,)).
body_pred(col_3,1).
type(col_3,(color,)).
direction(col_3,(out,)).
body_pred(col_4,1).
type(col_4,(color,)).
direction(col_4,(out,)).
body_pred(col_5,1).
type(col_5,(color,)).
direction(col_5,(out,)).
body_pred(col_6,1).
type(col_6,(color,)).
direction(col_6,(out,)).
body_pred(col_7,1).
type(col_7,(color,)).
direction(col_7,(out,)).
body_pred(col_8,1).
type(col_8,(color,)).
direction(col_8,(out,)).
body_pred(col_9,1).
type(col_9,(color,)).
direction(col_9,(out,)).
body_pred(col_0,1).
type(col_0,(color,)).
direction(col_0,(out,)).


% 负整数常量 –9 到 –1
body_pred(int_n9,1).
type(int_n9,(int,)).
direction(int_n9,(out,)).

body_pred(int_n8,1).
type(int_n8,(int,)).
direction(int_n8,(out,)).

body_pred(int_n7,1).
type(int_n7,(int,)).
direction(int_n7,(out,)).

body_pred(int_n6,1).
type(int_n6,(int,)).
direction(int_n6,(out,)).

body_pred(int_n5,1).
type(int_n5,(int,)).
direction(int_n5,(out,)).

body_pred(int_n4,1).
type(int_n4,(int,)).
direction(int_n4,(out,)).

body_pred(int_n3,1).
type(int_n3,(int,)).
direction(int_n3,(out,)).

body_pred(int_n2,1).
type(int_n2,(int,)).
direction(int_n2,(out,)).

body_pred(int_n1,1).
type(int_n1,(int,)).
direction(int_n1,(out,)).



body_pred(int_1,1).
type(int_1,(int,)).
direction(int_1,(out,)).
body_pred(int_2,1).
type(int_2,(int,)).
direction(int_2,(out,)).
body_pred(int_3,1).
type(int_3,(int,)).
direction(int_3,(out,)).
body_pred(int_4,1).
type(int_4,(int,)).
direction(int_4,(out,)).
body_pred(int_5,1).
type(int_5,(int,)).
direction(int_5,(out,)).
body_pred(int_6,1).
type(int_6,(int,)).
direction(int_6,(out,)).
body_pred(int_7,1).
type(int_7,(int,)).
direction(int_7,(out,)).
body_pred(int_8,1).
type(int_8,(int,)).
direction(int_8,(out,)).
body_pred(int_9,1).
type(int_9,(int,)).
direction(int_9,(out,)).
body_pred(int_0,1).
type(int_0,(int,)).
direction(int_0,(out,)).



%%%%%%%%%%%%  Search limits (可依需要调)  %%%%%%%%

max_vars(8).
max_body(6).
max_clauses(6).

:- clause(C), #count{V : body_literal(C,inpix,4,V)} > 1.