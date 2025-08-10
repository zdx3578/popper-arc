max_vars(10).
max_body(2).
max_clauses(2).

type(pair). type(color). type(coord). type(epcoord).

head_pred(outpix,4).
type(outpix,(pair,coord,coord,color,)).
direction(outpix,(in,out,out,out,)).

body_pred(diag_pair_near,6).
type(diag_pair_near,(pair,color,epcoord,epcoord,epcoord,epcoord,)).
direction(diag_pair_near,(in,out,out,out,out,out,)).

body_pred(diag_pair_far,6).
type(diag_pair_far,(pair,color,epcoord,epcoord,epcoord,epcoord,)).
direction(diag_pair_far,(in,out,out,out,out,out,)).

body_pred(on_diag_between,6).
type(on_diag_between,(coord,coord,epcoord,epcoord,epcoord,epcoord,)).
direction(on_diag_between,(out,out,in,in,in,in,)).



body_pred(inpix,4).
type(inpix,(pair,coord,coord,color,)).
direction(inpix,(in,out,out,out,)).