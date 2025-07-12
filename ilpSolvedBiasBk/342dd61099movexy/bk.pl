inpix(tr0,4,4,7).
inpix(tr0,8,6,7).
inpix(tr1,7,5,2).
inpix(tr1,7,9,2).
inpix(tr2,2,4,9).
inpix(tr3,7,5,1).
inpix(tr3,7,9,1).

%%%%%%%%%%%%%%%%%%%%  颜色常量  %%%%%%%%%%%%%%%%%%%%

col_0(0).  col_1(1).  col_2(2).  col_3(3).
col_4(4).  col_5(5).  col_6(6).  col_7(7).
col_8(8).  col_9(9).


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



% === int_n1 ===
int_n1(-1).
% === int_n2 ===
int_n2(-2).
% === int_n3 ===
int_n3(-3).
% === int_n4 ===
int_n4(-4).
% === int_n5 ===
int_n5(-5).
% === int_n6 ===
int_n6(-6).
% === int_n7 ===
int_n7(-7).
% === int_n8 ===
int_n8(-8).
% === int_n9 ===
int_n9(-9).



%%%%%%%%%%%%%%%%%%%%  add/6: (DX,DY,Rin,Cin,Rout,Cout)  %%%%%%%%%

add(DX,DY,Rin,Cin,Rout,Cout) :-
    int(DX), int(DY), coord(Rin), coord(Cin),
    Rout is Rin+DY,
    Cout is Cin+DX.

%%%%%%%%%%%%%%%%%%%%  类型补充  %%%%%%%%%%%%%%%%%%%%%

int(_).          % “无类型”桩——Popper 只用来过类型检查
coord(_).        % 同上
color(_).        % 同上
pair(_).         % 同上
