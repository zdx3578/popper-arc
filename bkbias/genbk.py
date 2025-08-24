# -*- coding: utf-8 -*-
"""BK generation utilities extracted from objattr."""
from __future__ import annotations
from typing import Dict, List, Tuple
import logging
from importlib import import_module

logger = logging.getLogger(__name__)

objattr = import_module('bkbias.objattr')

def int_atom(v: int) -> str:
    return f"int_{v}" if v >= 0 else f"int_n{abs(v)}"

def color_atom(v: int) -> str:
    return f"col_{v}" if v >= 0 else f"col_n{abs(v)}"


def inpix_bk_lines(task_data: Dict[str, any], background_color: int | None = None) -> List[str]:
    if background_color is None:
        background_color = objattr.determine_background_color(task_data)
    lines: List[str] = []
    for pair_id, pair in enumerate(task_data.get("train", [])):
        grid = pair["input"]
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                if background_color is not None and color == background_color:
                    continue
                lines.append(f"inpix(p{pair_id},{r},{c},{color}).")
    return lines


def add_sub_bk_lines(limit: int = 10) -> List[str]:
    """Return ``coord_const`` facts and all addition triples within ``[-limit,limit]``."""
    lines: List[str] = []
    rng = range(-limit, limit + 1)
    lim2 = 4
    for n in rng:
        lines.append(f"coord_const({n}).")
    for dx in rng:
        for x in rng:
            x2 = x + dx
            if -lim2 <= x2 <= lim2:
                lines.append(f"add({dx},{x},{x2}).")
    lines.append("sub(X,X2,DX):-add(DX,X,X2).")
    return lines


def group_bk_lines(lines: List[str]) -> List[str]:
    from collections import defaultdict
    import re
    bucket: Dict[str, List[str]] = defaultdict(list)
    directives: List[str] = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith('%'):
            continue
        if stripped.startswith(':-'):
            directives.append(ln)
            continue
        m = re.match(r'([a-zA-Z_][A-Za-z0-9_]*)\s*\(', stripped)
        if m:
            bucket[m.group(1)].append(ln)
        else:
            bucket[''].append(ln)
    grouped: List[str] = directives.copy()
    for pred in sorted(bucket):
        if pred:
            grouped.append(f"% === {pred} ===")
        grouped.extend(bucket[pred])
    return grouped


def save_bk(lines: List[str], path: str) -> None:
    grouped = group_bk_lines(lines)
    logger.debug("Saving BK to %s", path)
    with open(path, "w") as f:
        # Relax discontiguous warnings and declare dynamic preds so missing facts don't error
        f.write(':- style_check(-discontiguous).\n')
        f.write(':- dynamic inpix/4, inbelongs/4, objholes/3, grid_size/3, ' \
                'color_value/1, object/1, belongs/2, color/2, size/2, holes/2, ' \
                'add/3, sub/3, coord_const/1.\n')
        f.write("\n".join(grouped))


def objects_to_bk_lines(
    task_data: Dict[str, any],
    all_objects: Dict[str, List[Tuple[int, List[Dict[str, any]]]]],
    include_pixels: bool = True,
    *,
    enable_pi: bool = True,
    background_color: int | None = None,
    pixel_threshold_pct: int = 40,
) -> List[str]:
    lines: List[str] = []
    hole_color_map = objattr.compute_hole_color_mapping(
        task_data,
        background_color=background_color,
        pixel_threshold_pct=pixel_threshold_pct,
        debug=False,
    )
    pair_total = len(task_data.get("train", []))
    max_dim = 0
    colors = set()
    for pair in task_data.get("train", []):
        for grid in (pair["input"], pair["output"]):
            if not grid:
                continue
            max_dim = max(max_dim, len(grid), len(grid[0]))
            colors.update({c for row in grid for c in row})
            
    const_ints = list(range(-9, 10))
    const_colors = list(range(0, 9))
    for k in const_ints:
        lines.append(f"{int_atom(k)}({k}).")
    for k in const_colors:
        lines.append(f"{color_atom(k)}({k}).")
    for pid in range(pair_total):
        lines.append(f"constant(p{pid},pair).")
    for n in range(max_dim):
        lines.append(f"constant({n},coord).")
    obj_ids: set[str] = set()
    for pair_id, objs in all_objects.get("input", []):
        for info in objs:
            obj_ids.add(info["obj_id"])
    for n in range(10):
        lines.append(f"constant({n},int).")
    for oid in sorted(obj_ids):
        lines.append(f"constant({oid},obj).")
    seen_pairs: Dict[Tuple[str, int], Tuple[int, int]] = {}
    color_set = set()
    for cat in ("input", "output"):
        for pair_id, objs in all_objects.get(cat, []):
            if objs:
                seen_pairs[(cat, pair_id)] = objs[0]["grid_hw"]
            for info in objs:
                color_set.add(info["main_color"])
    for (cat, pair_id), (h, w) in seen_pairs.items():
        lines.append(f"grid_size({cat}_{pair_id},{h},{w}).")
    for color in sorted(color_set):
        lines.append(f"color_value({color}).")
    if not enable_pi:
        for h, c in hole_color_map.items():
            lines.append(f"hole2color({h},{c}).")
            lines.append(f"constant(int, {h}).")
    for pair_id, objs in all_objects.get("input", []):
        for info in objs:
            oid = info["obj_id"]
            if include_pixels:
                for _, (r, c) in info["obj"]:
                    lines.append(f"inbelongs(p{pair_id},{oid},{r},{c}).")
            if info['holes'] > 0:
                lines.append(f"objholes(p{pair_id},{oid},{info['holes']}).")
            logger.debug("Added pixel facts for object %s", oid)
    if include_pixels:
        lines.extend(inpix_bk_lines(task_data, background_color))
    for cat in ("input", "output"):
        for pair_id, objs in all_objects.get(cat, []):
            for info in objs:
                obj_name = objattr.prolog_atom(info["obj_id"])
                lines.append(f"object({obj_name}).")
                lines.append(f"belongs({obj_name},{cat}{pair_id}).")
                lines.append(f"color({obj_name},{info['main_color']}).")
                lines.append(f"size({obj_name},{info['size']}).")
                if 'obj_sort_ID' in info:
                    lines.append(f"objsortid({obj_name},{info['obj_sort_ID']}).")
                lines.append(f"holes({obj_name},{info['holes']}).")
    lines.extend(add_sub_bk_lines())
    logger.debug("Generated %d BK lines", len(lines))
    return lines
