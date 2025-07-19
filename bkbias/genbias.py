# -*- coding: utf-8 -*-
"""Bias generation utilities extracted from objattr."""
from __future__ import annotations
from typing import List
import logging

def int_atom(v: int) -> str:
    return f"int_{v}" if v >= 0 else f"int_n{abs(v)}"

def color_atom(v: int) -> str:
    return f"col_{v}" if v >= 0 else f"col_n{abs(v)}"

logger = logging.getLogger(__name__)


def group_bias_lines(lines: List[str]) -> List[str]:
    """Group bias  Sorting  lines by predicate category."""
    cats = {
        'head_pred': [],
        'body_pred': [],
        'type': [],
        'direction': [],
        'other': [],
    }
    for ln in lines:
        s = ln.strip()
        if s.startswith('head_pred'):
            cats['head_pred'].append(ln)
        elif s.startswith('body_pred'):
            cats['body_pred'].append(ln)
        elif s.startswith('type(') or s.startswith('type '):
            cats['type'].append(ln)
        elif s.startswith('direction'):
            cats['direction'].append(ln)
        else:
            cats['other'].append(ln)
    grouped: List[str] = []
    for key in ['head_pred', 'body_pred', 'type', 'direction', 'other']:
        if cats[key]:
            grouped.append(f"% === {key} ===")
            grouped.extend(sorted(cats[key]))
    return grouped


def generate_bias(
    enable_pi: bool = True,
    *,
    max_clauses: int = 4,
    max_vars: int = 6,
    max_body: int = 4,
) -> str:
    """Return Popper bias string for predicting output pixels."""
    logger.debug(
        "Generating bias enable_pi=%s max_clauses=%s max_vars=%s max_body=%s",
        enable_pi,
        max_clauses,
        max_vars,
        max_body,
    )
    bias_lines: List[str] = []
    if enable_pi:
        bias_lines.extend(
            [
                "enable_pi.",
                # "max_inv_preds(1).",
                # "max_inv_arity(3).",
                # "max_inv_body(3).",
                # "max_inv_clauses(4).",
                f"max_clauses({max_clauses}).",
                f"max_vars({max_vars}).",
                f"max_body({max_body}).",
            ]
        )
    else:
        bias_lines.extend(
            [
                "body_pred(hole2color,2).",
                "type(hole2color,(int,color)).",
                "direction(hole2color,(in,out)).",
                f"max_clauses({max_clauses}).",
                f"max_vars({max_vars}).",
                f"max_body({max_body}).",
            ]
        )
    bias_lines.extend(
        [
            "head_pred(outpix,4).",
            "body_pred(inbelongs,4).",
            "body_pred(objholes,3).",
            "body_pred(grid_size,3).",
            "body_pred(color_value,1).",
            "body_pred(object,1).",
            "body_pred(belongs,2).",
            # "body_pred(x_min,2).",
            # "body_pred(y_min,2).",
            # "body_pred(width,2).",
            # "body_pred(height,2).",
            # "body_pred(color,2).",
            # "body_pred(size,2).",
            "body_pred(holes,2).",
            # "body_pred(add,3).",
            # "body_pred(sub,3).",
            "type(pair). type(obj).",
            "type(coord). type(color). type(int).",
            "type(outpix,(pair,coord,coord,color)).",
            "type(inbelongs,(pair,obj,coord,coord)).",
            "type(objholes,(pair,obj,int)).",
            "type(grid_size,(pair,int,int)).",
            "type(color_value,(color,)).",
            "type(object,(obj,)).",
            "type(belongs,(obj,pair)).",
            # "type(x_min,(obj,int)).",
            # "type(y_min,(obj,int)).",
            # "type(width,(obj,int)).",
            # "type(height,(obj,int)).",
            # "type(color,(obj,color)).",
            # "type(size,(obj,int)).",
            "type(holes,(obj,int)).",
            # "type(sub,(coord,coord,int)).",
            # "type(add,(coord,int,coord)).",
            "direction(outpix,(in,in,in,out)).",
            "direction(inbelongs,(in,out,in,in)).",
            "direction(objholes,(in,in,out)).",
            "direction(grid_size,(in,out,out)).",
            "direction(color_value,(out,)).",
            "direction(object,(out,)).",
            "direction(belongs,(in,in)).",
            # "direction(x_min,(in,out)).",
            # "direction(y_min,(in,out)).",
            # "direction(width,(in,out)).",
            # "direction(height,(in,out)).",
            # "direction(color,(in,out)).",
            # "direction(size,(in,out)).",
            "direction(holes,(in,out)).",
            # "direction(add,(in,in,out)).",
            # "direction(sub,(in,in,out)).",
        ]
    )
    if False:
        bias_lines.extend(
            [
                "head_pred(outpix,4).",
                "body_pred(inbelongs,4).",
                "body_pred(objholes,3).",
                "body_pred(grid_size,3).",
                "body_pred(color_value,1).",
                "body_pred(object,1).",
                "body_pred(belongs,2).",
                "body_pred(x_min,2).",
                "body_pred(y_min,2).",
                "body_pred(width,2).",
                "body_pred(height,2).",
                "body_pred(color,2).",
                "body_pred(size,2).",
                "body_pred(holes,2).",
                "body_pred(add,3).",
                "body_pred(sub,3).",
                "type(pair). type(obj).",
                "type(coord). type(color). type(int).",
                "type(outpix,(pair,coord,coord,color)).",
                "type(inbelongs,(pair,obj,coord,coord)).",
                "type(objholes,(pair,obj,int)).",
                "type(grid_size,(pair,int,int)).",
                "type(color_value,(color,)).",
                "type(object,(obj,)).",
                "type(belongs,(obj,pair)).",
                "type(x_min,(obj,int)).",
                "type(y_min,(obj,int)).",
                "type(width,(obj,int)).",
                "type(height,(obj,int)).",
                "type(color,(obj,color)).",
                "type(size,(obj,int)).",
                "type(holes,(obj,int)).",
                "type(sub,(coord,coord,int)).",
                "type(add,(coord,int,coord)).",
                "direction(outpix,(in,in,in,out)).",
                "direction(inbelongs,(in,out,in,in)).",
                "direction(objholes,(in,in,out)).",
                "direction(grid_size,(in,out,out)).",
                "direction(color_value,(out,)).",
                "direction(object,(out,)).",
                "direction(belongs,(in,in)).",
                "direction(x_min,(in,out)).",
                "direction(y_min,(in,out)).",
                "direction(width,(in,out)).",
                "direction(height,(in,out)).",
                "direction(color,(in,out)).",
                "direction(size,(in,out)).",
                "direction(holes,(in,out)).",
                "direction(add,(in,in,out)).",
                "direction(sub,(in,in,out)).",
            ]
        )

    const_ints = list(range(-9, 10))
    const_colors = list(range(0, 9))
    for k in const_ints:
        pred = int_atom(k)
        bias_lines.append(f"body_pred({pred},1).")
        bias_lines.append(f"type({pred},(int,)).")
        bias_lines.append(f"direction({pred},(out,)).")
    for k in const_colors:
        pred = color_atom(k)
        bias_lines.append(f"body_pred({pred},1).")
        bias_lines.append(f"type({pred},(color,)).")
        bias_lines.append(f"direction({pred},(out,)).")
    grouped = group_bias_lines(bias_lines)
    logger.debug("Generated bias with %d predicates", len(grouped))
    return "\n".join(grouped) + "\n"
