outpix(V0,V1,V2,V3):- col_1(V3),int_1(V4),inbelongs(V0,V5,V1,V2),objholes(V0,V5,V4).
outpix(V0,V1,V2,V3):- int_2(V4),objholes(V0,V5,V4),col_2(V3),inbelongs(V0,V5,V1,V2).
outpix(V0,V1,V2,V3):- int_4(V4),objholes(V0,V5,V4),col_7(V3),inbelongs(V0,V5,V1,V2).
outpix(V0,V1,V2,V3):- objholes(V0,V5,V4),col_3(V3),inbelongs(V0,V5,V1,V2),int_3(V4).
