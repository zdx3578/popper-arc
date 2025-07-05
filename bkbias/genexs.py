# -*- coding: utf-8 -*-
"""Example generation utilities extracted from objattr."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import random
import logging
def determine_background_color(*args, **kwargs):
    from .objattr import determine_background_color as _dbc
    return _dbc(*args, **kwargs)

def prolog_atom(*args, **kwargs):
    from .objattr import prolog_atom as _pa
    return _pa(*args, **kwargs)

logger = logging.getLogger(__name__)


def nonbg_pixels(grid: List[List[int]], bg_color: int | None) -> List[Tuple[Tuple[int, int], int]]:
    pixels = []
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if bg_color is None or color != bg_color:
                pixels.append(((r, c), color))
    return pixels


def add_negatives(
    pair_id: int,
    out_grid: List[List[int]],
    bg_color: int | None,
    exs: List[str],
    k_factor: int = 2,
) -> None:
    pos_pixels = nonbg_pixels(out_grid, bg_color)
    colors = list(range(10))
    for (x, y), true_c in pos_pixels:
        wrong_colors = [c for c in colors if c != true_c and c != bg_color]
        random.shuffle(wrong_colors)
        for wrong_c in wrong_colors[:k_factor]:
            exs.append(f"neg(outpix(p{pair_id},{x},{y},{wrong_c})).")


def outpix_examples(task_data: Dict[str, Any], background_color: int | None = None, neg_factor: int = 2) -> List[str]:
    if background_color is None:
        background_color = determine_background_color(task_data)
    lines: List[str] = []
    lines.append(f":- discontiguous neg/1.")
    lines.append(f":- discontiguous pos/1.")
    for pair_id, pair in enumerate(task_data.get("train", [])):
        grid = pair["output"]
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                if background_color is not None and color == background_color:
                    continue
                lines.append(f"pos(outpix(p{pair_id},{r},{c},{color})).")
        add_negatives(pair_id, grid, background_color, lines, neg_factor)
    return lines


def objects_to_exs_lines(all_objects: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]]) -> List[str]:
    lines: List[str] = []
    inputs = {pid: objs for pid, objs in all_objects.get("input", [])}
    outputs = {pid: objs for pid, objs in all_objects.get("output", [])}
    for pair_id in sorted(inputs):
        in_names = [prolog_atom(info["obj_id"]) for info in inputs.get(pair_id, [])]
        out_names = [prolog_atom(info["obj_id"]) for info in outputs.get(pair_id, [])]
        in_list = "[" + ",".join(in_names) + "]"
        out_list = "[" + ",".join(out_names) + "]"
        lines.append(f"pos(target({in_list},{out_list})).")
    return lines


def grids_to_pix_lines(task_data: Dict[str, Any], background_color: int | None = None) -> List[str]:
    if background_color is None:
        background_color = determine_background_color(task_data)
    pix_lines: List[str] = []
    for pair_id, pair in enumerate(task_data.get("train", [])):
        for kind in ("input", "output"):
            grid = pair[kind]
            label = f"pairid{pair_id}_{'in' if kind=='input' else 'out'}"
            for i, row in enumerate(grid):
                for j, color in enumerate(row):
                    if background_color is not None and color == background_color:
                        continue
                    pix_lines.append(f"pix({label},{i},{j},{color}).")
    return pix_lines
