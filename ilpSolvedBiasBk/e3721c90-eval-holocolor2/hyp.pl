outpix(V0,V1,V2,V3):- is_color_obj(V0,V4,V3),inbelongs(V0,V4,V1,V2).
outpix(V0,V1,V2,V3):- same_hole_but_diff_obj(V0,V7,V6,V5,V3),is_color_obj(V0,V7,V4),inbelongs(V0,V6,V1,V2).
