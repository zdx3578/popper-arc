from textwrap import dedent

def output_bias_diagline():
    return dedent(r'''

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

    ''')

# 用法：
# lines = []

# with open('bias.pl','w', encoding='utf-8', newline='\n') as f:
#     f.write('\n'.join(lines))
