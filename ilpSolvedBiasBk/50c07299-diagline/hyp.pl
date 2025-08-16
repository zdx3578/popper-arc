outpix(V0, V1, V2, V3):-
	seg(V0, V3, V6, V4, V5, V7),
	seg_extend_point(V6, V4, V5, V7, V1, V2).