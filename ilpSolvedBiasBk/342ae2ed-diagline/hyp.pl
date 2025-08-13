outpix(V0,V1,V2,V3):- inpix(V0,V1,V2,V3).
outpix(V0,V1,V2,V3):- diag_pair_far(V0,V3,V8,V7,V5,V6),int_1(V4),on_diag_between_k(V1,V2,V8,V7,V5,V6,V4).
