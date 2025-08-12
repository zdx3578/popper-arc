from textwrap import dedent

def output_bias_diagline():
    return dedent(r'''
        type(pair). type(color). type(coord). type(epcoord).
        type(stride).  % 用于 step/1
        % type(obj).
        type(int).

        head_pred(outpix,4).
        type(outpix,(pair,coord,coord,color,)).
        direction(outpix,(in,out,out,out,)).

        body_pred(inpix,4).
        type(inpix,(pair,coord,coord,color,)).
        direction(inpix,(in,out,out,out,)).

        body_pred(diag_pair_near,6).
        type(diag_pair_near,(pair,color,epcoord,epcoord,epcoord,epcoord,)).
        direction(diag_pair_near,(in,out,out,out,out,out,)).

        body_pred(diag_pair_far,6).
        type(diag_pair_far,(pair,color,epcoord,epcoord,epcoord,epcoord,)).
        direction(diag_pair_far,(in,out,out,out,out,out,)).

        % 步长候选（显式常量，小域即可）
        body_pred(step,1).
        type(step,(stride,)).
        direction(step,(out,)).

        % 含步长的对角线生成（含端点，步进为 S）
        body_pred(on_diag_between_k,7).
        type(on_diag_between_k,(coord,coord,epcoord,epcoord,epcoord,epcoord,int,)).
        direction(on_diag_between_k,(out,out,in,in,in,in,in,)).
    ''')

# 用法：
# lines = []

# with open('bias.pl','w', encoding='utf-8', newline='\n') as f:
#     f.write('\n'.join(lines))
