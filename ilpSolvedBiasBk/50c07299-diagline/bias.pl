body_pred(int_1,1).
type(int_1,(int,)).
direction(int_1,(out,)).
body_pred(int_2,1).
type(int_2,(int,)).
direction(int_2,(out,)).
body_pred(int_0,1).
type(int_0,(int,)).
direction(int_0,(out,)).


type(pair). type(color). type(coord). type(epcoord).
type(stride).  % 用于 step/1
% type(obj).
type(int).
type(xcoord). type(ycoord).

head_pred(outpix,4).
type(outpix,(pair,coord,coord,color,)).
direction(outpix,(in,out,out,out,)).

body_pred(extend_diag_out,4).
type(extend_diag_out,(pair,color,coord,coord,)).
direction(extend_diag_out,(in,out,out,out,)).

max_clauses(1).
max_body(1).
max_vars(6).


