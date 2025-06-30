import os
import json
import random
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, FrozenSet
import traceback
import subprocess
import pathlib
import shutil
import tempfile
import textwrap
import sys
import ast
import logging
from bkbias.extendplugin.objholecount import count_object_holes

# Mapping from ARC color numbers to emoji for debugging displays
COLOR_MAP = {
    0: "⬛",   # Black
    1: "🟦",   # Blue
    2: "🟥",   # Red
    3: "🟩",   # Green
    4: "🟨",   # Yellow
    5: "🟫",   # Brown / Gray
    6: "🟪",   # Purple
    7: "🟠",   # Orange
    8: "🔹",   # Light Blue
    9: "🔴",   # Dark Red
}

logger = logging.getLogger(__name__)


def grid_to_str(
    grid: List[List[int]], background_color: int | None = None
) -> str:
    """Return ``grid`` rendered using :data:`COLOR_MAP` emojis.

    If ``background_color`` is given and not ``0`` the display will swap the
    background and black colors so the background is rendered in black. Any
    original black cells will appear using the background color instead.
    """

    def disp(val: int) -> str:
        if background_color is not None and background_color != 0:
            if val == background_color:
                val = 0
            elif val == 0:
                val = background_color
        return COLOR_MAP.get(val, str(val))

    lines = ["".join(disp(v) for v in row) for row in grid]
    return "\n".join(lines)


def print_grid(
    grid: List[List[int]],
    title: str | None = None,
    *,
    background_color: int | None = None,
) -> None:
    """Print ``grid`` with optional ``title`` using emoji colors."""
    if title:
        print(f"\n{title}")
    print(grid_to_str(grid, background_color))


class IdManager:
    """Simple ID manager for assigning incremental IDs per category."""

    def __init__(self) -> None:
        self.tables: Dict[str, Dict[Any, int]] = {}
        self.next_id: Dict[str, int] = {}

    def get_id(self, category: str, value: Any) -> int:
        if isinstance(value, set):
            value = frozenset(value)
        if category not in self.tables:
            self.tables[category] = {}
            self.next_id[category] = 1
        table = self.tables[category]
        if value not in table:
            table[value] = self.next_id[category]
            self.next_id[category] += 1
        return table[value]

    def reset(self) -> None:
        self.tables = {}
        self.next_id = {}


managerid = IdManager()

# ---------------- Utility functions ----------------

def grid2grid_fromgriddiff(grid1, grid2):
    """Return difference grids between two equally sized grids."""
    if not grid1 or not grid2:
        return None, None
    rows, cols = len(grid1), len(grid1[0])
    diff1 = [[None for _ in range(cols)] for _ in range(rows)]
    diff2 = [[None for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if grid1[i][j] != grid2[i][j]:
                diff1[i][j] = grid1[i][j]
                diff2[i][j] = grid2[i][j]
    return diff1, diff2

# basic geometry helpers

def uppermost(patch):
    return min(i for _, (i, _) in patch)

def lowermost(patch):
    return max(i for _, (i, _) in patch)

def leftmost(patch):
    return min(j for _, (_, j) in patch)

def rightmost(patch):
    return max(j for _, (_, j) in patch)

def palette(element):
    if isinstance(element, (list, tuple)) and element and isinstance(element[0], (list, tuple)):
        return frozenset(v for row in element for v in row if v is not None)
    return frozenset(v for v, _ in element)

# simple transformations used by extend_obj

def rot90(grid):
    return tuple(row for row in zip(*grid[::-1]))

def rot180(grid):
    return tuple(tuple(row[::-1]) for row in grid[::-1])

def rot270(grid):
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

def hmirror(piece):
    return tuple(piece[::-1]) if isinstance(piece, tuple) else frozenset((v, (max(i for _, (i, _) in piece) - i, j)) for v, (i, j) in piece)

def vmirror(piece):
    return tuple(row[::-1] for row in piece) if isinstance(piece, tuple) else frozenset((v, (i, max(j for _, (_, j) in piece) - j)) for v, (i, j) in piece)

def dmirror(piece):
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a = leftmost(piece)
    b = uppermost(piece)
    return frozenset((v, (j - a + b, i - b + a)) for v, (i, j) in piece)

def cmirror(piece):
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

def extend_obj(obj):
    transformations = [
        ("vmirror", vmirror),
        ("cmirror", cmirror),
        ("hmirror", hmirror),
        ("dmirror", dmirror),
        ("rot90", lambda o: rot90(object_to_grid(o))),
        ("rot180", lambda o: rot180(object_to_grid(o))),
        ("rot270", lambda o: rot270(object_to_grid(o))),
    ]
    results = []
    for name, func in transformations:
        try:
            if name.startswith("rot"):
                res = grid_to_object(func(obj))
            else:
                res = func(obj)
            results.append((name, frozenset(res)))
        except Exception:
            pass
    return tuple(results)

def object_to_grid(obj):
    if not obj:
        return tuple()
    rows = 1 + max(r for _, (r, _) in obj)
    cols = 1 + max(c for _, (_, c) in obj)
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for v, (r, c) in obj:
        grid[r][c] = v
    return tuple(tuple(row) for row in grid)

def grid_to_object(grid):
    result = []
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v is not None:
                result.append((v, (r, c)))
    return frozenset(result)

# shift object to origin

def shift_to_origin(obj, preserve_colors=True):
    if not obj:
        return frozenset()
    min_row = min(r for _, (r, _) in obj)
    min_col = min(c for _, (_, c) in obj)
    if preserve_colors:
        return frozenset((color, (r - min_row, c - min_col)) for color, (r, c) in obj)
    else:
        return frozenset((0, (r - min_row, c - min_col)) for color, (r, c) in obj)

# simple object extraction from grid

def extract_objects(grid, background=None):
    """Return list of objects as sets of (color,(r,c))."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False]*cols for _ in range(rows)]
    objects = []
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    for i in range(rows):
        for j in range(cols):
            color = grid[i][j]
            if (background is not None and color == background) or visited[i][j]:
                continue
            # BFS
            q = deque([(i,j)])
            visited[i][j]=True
            obj = []
            while q:
                r,c = q.popleft()
                obj.append((color,(r,c)))
                for dr,dc in directions:
                    nr,nc = r+dr, c+dc
                    if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==color:
                        visited[nr][nc]=True
                        q.append((nr,nc))
            objects.append(frozenset(obj))
    return objects




def determine_background_color(
    task_data: Dict[str, Any],
    pixel_threshold_pct: int = 40,
    debug: bool = True,
) -> int | None:
    """Analyze all training grids and return the dominant background color."""
    if debug:
        print(
            f"Determining background color with threshold: {pixel_threshold_pct}%"
        )

    all_grids: List[List[List[int]]] = []
    for example in task_data.get("train", []):
        all_grids.append(example.get("input"))
        all_grids.append(example.get("output"))

    color_total_percentages: Dict[int, float] = defaultdict(float)
    color_appearance_count: Dict[int, int] = defaultdict(int)

    for grid in all_grids:
        if not grid:
            continue
        total_pixels = len(grid) * len(grid[0])
        if total_pixels == 0:
            continue

        color_counts: Dict[int, int] = defaultdict(int)
        for row in grid:
            for cell in row:
                color_counts[cell] += 1

        for color, count in color_counts.items():
            percentage = count / total_pixels * 100
            color_total_percentages[color] += percentage
            color_appearance_count[color] += 1

    color_avg_percentages: Dict[int, float] = {
        color: color_total_percentages[color] / color_appearance_count[color]
        for color in color_total_percentages
    }

    sorted_colors = sorted(
        color_avg_percentages.items(), key=lambda x: x[1], reverse=True
    )

    if debug:
        print("Color distribution across training data:")
        for col, pct in sorted_colors:
            print(f"  color {col}: {pct:.2f}%")

    if sorted_colors:
        max_color, max_percentage = sorted_colors[0]
        if max_percentage >= pixel_threshold_pct:
            if debug:
                print(
                    f"确定全局背景色: {max_color} (占比: {max_percentage:.2f}%)"
                )
            return max_color
    return None


def create_weighted_obj_info(pair_id: int, in_or_out: str, obj: FrozenSet[Tuple[int, Tuple[int, int]]],
                             grid_hw: Tuple[int, int], background: int | None = None,
                             obj_index: int | None = None) -> Dict[str, Any]:
    """Compute object attributes and return a dictionary."""
    obj_00 = shift_to_origin(obj)
    obj_000 = shift_to_origin(obj, preserve_colors=False)
    bounding_box = (
        uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)
    )
    size = len(obj)
    height = bounding_box[2] - bounding_box[0] + 1
    width = bounding_box[3] - bounding_box[1] + 1
    color_counts = defaultdict(int)
    for v, _ in obj:
        color_counts[v] += 1
    main_color = max(color_counts.items(), key=lambda x: x[1])[0]
    holes = count_object_holes(obj)
    # holemapping = {}
    hashid = hash(obj_000)
    # print(hashid)
    abshashid = abs(hashid)
    # print(abshashid)


    if obj_index is None:
        obj_id = f"pairid{pair_id}{in_or_out}{abshashid}"
    else:
        # obj_id = f"pairid{pair_id}{in_or_out}{obj_index}"
        obj_id = f"pairid{pair_id}{in_or_out}{abshashid}"
    obj_shape_ID = managerid.get_id("OBJshape", obj_000)
    info = {
        "pair_id": pair_id,
        "in_or_out": in_or_out,
        "obj": obj,
        "obj_params": (True, True, False),
        "grid_hw": grid_hw,
        "background": background,
        "obj_00": obj_00,
        "obj_000": obj_000,
        "bounding_box": bounding_box,
        "top": bounding_box[0],
        "left": bounding_box[1],
        "color_ranking": palette(obj),
        "obj000_ops": extend_obj(obj_000),
        "obj_ops": extend_obj(obj),
        "size": size,
        "height": height,
        "width": width,
        "main_color": main_color,
        "holes": holes,
        "obj_id": obj_id,
        "obj_shape_ID": obj_shape_ID,
        "rotated_variants": [("rot_0", obj_00)] + [
            (f"rot_{d}", shift_to_origin(grid_to_object({90: rot90, 180: rot180, 270: rot270}[d](object_to_grid(obj_00)))))
            for d in (90, 180, 270)
        ],
        "mirrored_variants": extend_obj(obj_000),
        "obj_weight": 0,
        "is_part_of": [],
        "has_parts": [],
        # "mapping"
    }
    return info


def extract_object_infos_from_grid(pair_id: int, in_or_out: str, grid: List[List[int]],
                                   background_color: int | None) -> List[Dict[str, Any]]:
    """Extract objects and attributes from a single grid."""
    objects = extract_objects(grid, background_color)
    h, w = len(grid), len(grid[0])
    infos: List[Dict[str, Any]] = []
    for idx, obj in enumerate(objects):
        infos.append(create_weighted_obj_info(pair_id, in_or_out, obj, (h, w), background_color, idx))
    return infos


def extract_objects_from_task(task_data: Dict[str, Any], background_color: int = None,
                              param: Tuple[bool, bool, bool] = (True, True, False)) -> Dict[str, List[Tuple[int, List[Dict[str, Any]]]]]:
    """Given a task dictionary, return extracted object infos for each pair."""
    managerid.reset()
    if background_color is None:
        background_color = determine_background_color(task_data)
    all_objects = {"input": [], "output": []}
    for pair_id, pair in enumerate(task_data.get("train", [])):
        input_grid = pair["input"]
        output_grid = pair["output"]
        input_infos = extract_object_infos_from_grid(pair_id, "in", input_grid, background_color)
        output_infos = extract_object_infos_from_grid(pair_id, "out", output_grid, background_color)
        all_objects["input"].append((pair_id, input_infos))
        all_objects["output"].append((pair_id, output_infos))
    return all_objects


def load_task(task_path: str) -> Dict[str, Any]:
    with open(task_path, "r") as f:
        return json.load(f)


def run_extraction(task_path: str) -> Dict[str, List[Tuple[int, List[Dict[str, Any]]]]]:
    task_data = load_task(task_path)
    return extract_objects_from_task(task_data)


def prolog_atom(text: str) -> str:
    """Convert arbitrary text to a safe Prolog atom."""
    return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in text)


def display_object(obj: FrozenSet[Tuple[int, Tuple[int, int]]]) -> None:
    """Print the object's shape using ``COLOR_MAP`` characters."""
    obj_origin = shift_to_origin(obj)
    if not obj_origin:
        print("(empty object)")
        return

    max_r = max(r for _, (r, _) in obj_origin)
    max_c = max(c for _, (_, c) in obj_origin)
    grid = [[' ' for _ in range(max_c + 1)] for _ in range(max_r + 1)]
    for color, (r, c) in obj_origin:
        grid[r][c] = COLOR_MAP.get(color, str(color))

    border = "+" + "-" * (max_c + 1) + "+"
    print(border)
    for row in grid:
        print("|" + ''.join(row) + "|")
    print(border)


def _find_matching_output_color(in_obj: FrozenSet[Tuple[int, Tuple[int, int]]],
                                out_objs: List[FrozenSet[Tuple[int, Tuple[int, int]]]]) -> int | None:
    """Return the main color of the output object at the same coordinates as ``in_obj``."""
    in_coords = {(r, c) for _, (r, c) in in_obj}
    for out_obj in out_objs:
        out_coords = {(r, c) for _, (r, c) in out_obj}
        if out_coords == in_coords:
            color_counts: Dict[int, int] = defaultdict(int)
            for v, _ in out_obj:
                color_counts[v] += 1
            return max(color_counts.items(), key=lambda x: x[1])[0]
    return None


def compute_hole_color_mapping(
    task_data: Dict[str, Any],
    background_color: int | None = None,
    pixel_threshold_pct: int = 40,
    debug: bool = False,
) -> Dict[int, int]:
    """Return mapping from hole count to output color based on identical objects."""
    mapping: Dict[int, int] = {}
    if background_color is None:
        background_color = determine_background_color(
            task_data, pixel_threshold_pct=pixel_threshold_pct, debug=debug
        )
    for pair_id, pair in enumerate(task_data.get("train", [])):
        in_grid = pair["input"]
        out_grid = pair["output"]
        in_objs = extract_objects(in_grid, background_color)
        out_objs = extract_objects(out_grid, background_color)
        for in_obj in in_objs:
            col = _find_matching_output_color(in_obj, out_objs)
            if col is None:
                continue
            holes = count_object_holes(in_obj)
            if holes not in mapping:
                mapping[holes] = col
                if debug:
                    print(f"pair {pair_id}: holes={holes} -> color {col}")
                    display_object(in_obj)
    if debug:
        print("hole-color mapping:", mapping)
    return mapping


def objects_to_bk_lines(
    task_data: Dict[str, Any],
    all_objects: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]],
    include_pixels: bool = True,
    *,
    enable_pi: bool = True,
    background_color: int | None = None,
    pixel_threshold_pct: int = 40,
) -> List[str]:
    """Convert extracted object infos into Popper background facts.

    Parameters
    ----------
    task_data : Dict[str, Any]
        The loaded ARC task.
    all_objects : Dict[str, List[Tuple[int, List[Dict[str, Any]]]]]
        Objects extracted from the task.
    include_pixels : bool, optional
        If ``True`` (default) include ``inbelongs/4`` pixel facts mapping
        objects to their coordinates.  When ``False`` these facts are
        omitted, producing a purely object level BK.
    """
    lines: List[str] = []
    hole_color_map = compute_hole_color_mapping(
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

    # max_dim = min(max_dim, 30)
    max_dim = 30

        # bk_lines = []
    const_ints  = [1,2,3,4,5,6,7,8,9,0]
    const_colors= [1,2,3,4,5,6,7,8,9,0]
    for k in const_ints:
        lines.append(f"int_{k}({k}).")
    for k in const_colors:
        lines.append(f"col_{k}({k}).")

    # constants
    for pid in range(pair_total):
        lines.append(f"constant(p{pid},pair).")
    for n in range(max_dim):
        lines.append(f"constant({n},coord).")
    for c in range(10):
        lines.append(f"constant({c},color).")
        # lines.append(f"constant(color, {c}).")

    obj_ids: set[str] = set()

    for pair_id, objs in all_objects.get("input", []):
        for info in objs:
            obj_ids.add(info["obj_id"])
    for n in range(10):
        lines.append(f"constant({n},int).")
        # lines.append(f"constant(int, {n}).")
    for oid in sorted(obj_ids):
        lines.append(f"constant({oid},obj).")

    # Optional invented predicate facts
    if not enable_pi:
        for h, c in hole_color_map.items():
            lines.append(f"hole2color({h},{c}).")
            lines.append(f"constant(int, {h}).")
            # lines.append(f"constant(color, {c}).")

    # in-grid pixel-object relations and hole counts
    for pair_id, objs in all_objects.get("input", []):
        for info in objs:
            oid = info["obj_id"]
            if include_pixels:
                for _, (r, c) in info["obj"]:
                    lines.append(f"inbelongs(p{pair_id},{oid},{r},{c}).")

            if info['holes'] > 0:
                lines.append(f"objholes(p{pair_id},{oid},{info['holes']}).")

    return lines


def group_bk_lines(lines: List[str]) -> List[str]:
    """Group BK facts by predicate name for better readability."""
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
    with open(path, "w") as f:
        f.write(':- style_check(-discontiguous).\n')
        f.write("\n".join(grouped))


def generate_bias(
    enable_pi: bool = True,
    *,
    max_clauses: int = 4,
    max_vars: int = 6,
    max_body: int = 4,
) -> str:
    """Return Popper bias string for predicting output pixels.

    Parameters
    ----------
    enable_pi : bool, optional
        Whether to enable predicate invention.
    max_clauses : int, optional
        Value for ``max_clauses`` in the bias (default ``4``).
    max_vars : int, optional
        Value for ``max_vars`` in the bias (default ``6``).
    max_body : int, optional
        Value for ``max_body`` in the bias (default ``4``).
    """

    bias_lines: List[str] = []



    if enable_pi:
        bias_lines.extend(
            [
                "enable_pi.",
                "max_inv_preds(1).",
                "max_inv_arity(3).",
                "max_inv_body(3).",
                "max_inv_clauses(4).",


                # "body_pred(hole2color,2).",
                # "invent_pred(hole2color,(int,color)).",
                # "type(hole2color,(int,color)).",
                # "direction(hole2color,(in,out)).",

                f"max_clauses({max_clauses}).",
                f"max_vars({max_vars}).",
                f"max_body({max_body}).",
                # "max_rules(4).  ",
            ]
        )
    else:
        bias_lines.extend([
            "body_pred(hole2color,2).",
            "type(hole2color,(int,color)).",
            "direction(hole2color,(in,out)).",

            f"max_clauses({max_clauses}).",
            f"max_vars({max_vars}).",
            f"max_body({max_body}).",
        ])

    bias_lines.extend(
        [
            "head_pred(outpix,4).",
            "body_pred(inbelongs,4).",
            "body_pred(objholes,3).",

            "type(pair). type(obj).",
            "type(coord). type(color). type(int).",
            "type(outpix,(pair,coord,coord,color)).",
            "type(inbelongs,(pair,obj,coord,coord)).",
            "type(objholes,(pair,obj,int)).",
            # "max_vars(6).",
            # "max_clauses(1).",
            # "max_body(3).",
            "direction(outpix,(in,in,in,out)).",
            "direction(inbelongs,(in,out,in,in)).",
            "direction(objholes,(in,in,out))."
        ]
    )
    const_ints  = [1,2,3,4,5,6,7,8,9,0]
    const_colors= [1,2,3,4,5,6,7,8,9,0]

    for k in const_ints:
        bias_lines.append(f"body_pred(int_{k},1).")
        bias_lines.append(f"type(int_{k},(int,)).")
        bias_lines.append(f"direction(int_{k},(out,)).")
    for k in const_colors:
        bias_lines.append(f"body_pred(col_{k},1).")
        bias_lines.append(f"type(col_{k},(color,)).")
        bias_lines.append(f"direction(col_{k},(out,)).")



    return "\n".join(bias_lines)+ "\n"



def grids_to_pix_lines(task_data: Dict[str, Any],
                      background_color: int | None = None) -> List[str]:
    """Convert all grids in the task to pixel facts lines."""
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


def objects_to_exs_lines(all_objects: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]]) -> List[str]:
    """Return positive examples describing input/output object mappings."""
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


def nonbg_pixels(grid: List[List[int]], bg_color: int | None) -> List[Tuple[Tuple[int, int], int]]:
    """Return coordinates and colors of pixels not equal to ``bg_color``."""
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
    """Append negative outpix examples to ``exs``.

    The negatives are generated using coordinates of existing positive pixels
    but with a randomly chosen wrong color.
    """

    pos_pixels = nonbg_pixels(out_grid, bg_color)
    colors = list(range(10))

    for (x, y), true_c in pos_pixels:
        wrong_colors = [c for c in colors if c != true_c and c != bg_color]
        random.shuffle(wrong_colors)
        for wrong_c in wrong_colors[:k_factor]:
            exs.append(f"neg(outpix(p{pair_id},{x},{y},{wrong_c})).")


def outpix_examples(task_data: Dict[str, Any], background_color: int | None = None, neg_factor: int = 2) -> List[str]:
    """Generate positive and negative examples for output pixels."""
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


def save_lines(lines: List[str], path: str) -> None:
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")


def generate_test_bk(
    in_grid: List[List[int]],
    out_grid: List[List[int]],
    output_dir: str,
    *,
    enable_pi: bool = True,
    background_color: int | None = None,
    pixel_threshold_pct: int = 40,
) -> str:
    """Generate BK for a single test pair and save to ``testbk.pl``.

    Returns the path to the created BK file."""
    os.makedirs(output_dir, exist_ok=True)
    task = {"train": [{"input": in_grid, "output": out_grid}]}
    if background_color is None:
        background_color = determine_background_color(
            task, pixel_threshold_pct=pixel_threshold_pct, debug=False
        )
    objs = extract_objects_from_task(task, background_color)
    bk_lines = objects_to_bk_lines(
        task,
        objs,
        include_pixels=True,
        enable_pi=enable_pi,
        background_color=background_color,
        pixel_threshold_pct=pixel_threshold_pct,
    )
    bk_path = os.path.join(output_dir, "testbk.pl")
    save_bk(bk_lines, bk_path)
    meta = {
        "input_size": [len(in_grid), len(in_grid[0]) if in_grid else 0],
        "output_size": [len(out_grid), len(out_grid[0]) if out_grid else 0],
    }
    with open(os.path.join(output_dir, "grid_meta.json"), "w") as f:
        json.dump(meta, f)
    return bk_path


def predict_from_prolog(
    hyp_path: str,
    bk_path: str,
    meta_path: str,
    pair_id: str = "p0",
) -> List[List[int]]:
    """Return predicted output grid using ``hyp_path`` and ``bk_path``.

    This implementation invokes ``swipl`` in a subprocess instead of using
    ``pyswip`` so that it can run in an isolated Prolog environment.
    """

    meta = json.load(open(meta_path))
    rows, cols = meta.get("output_size", meta.get("size", [0, 0]))

    cmd = [
        "swipl",
        "-q",
        "-s",
        hyp_path,
        "-s",
        bk_path,
        "-g",
        f"findall([X,Y,C],outpix({pair_id},X,Y,C),Ls),writeln(Ls),halt.",
    ]

    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(run.stderr)

    output_line = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else "[]"
    try:
        data = ast.literal_eval(output_line)
    except Exception:
        data = []

    import numpy as np

    grid = np.zeros((rows, cols), dtype=int)
    for row, col, color in data:
        if 0 <= row < rows and 0 <= col < cols:
            grid[row][col] = color

    return grid.tolist()


def evaluate_prediction(pred: List[List[int]], gold: List[List[int]]) -> Tuple[bool, float]:
    """Return (exact_match, pixel_accuracy) comparing ``pred`` with ``gold``."""

    import numpy as np

    pred_arr = np.array(pred)
    gold_arr = np.array(gold)
    if pred_arr.shape != gold_arr.shape:
        raise ValueError("Prediction and gold grid sizes differ")

    exact = bool((pred_arr == gold_arr).all())
    pix_acc = float((pred_arr == gold_arr).sum() / gold_arr.size)
    return exact, pix_acc


def save_grid_txt(grid: List[List[int]], path: str) -> None:
    """Save ``grid`` to a text file for quick visualisation."""
    import numpy as np

    np.savetxt(path, np.array(grid, dtype=int), fmt="%d")


def generate_files_from_task(
    task: str | Dict[str, Any],
    output_dir: str,
    *,
    use_pixels: bool = True,
    bk_use_pixels: bool | None = None,
    exs_use_pixels: bool | None = None,
    enable_pi: bool = True,
    pixel_threshold_pct: int = 40,
    background_color: int | None = None,
    max_clauses: int = 4,
    max_vars: int = 6,
    max_body: int = 4,
) -> Tuple[str, str, str]:
    """Generate BK, bias and exs files from a task JSON.

    Parameters
    ----------
    task : str | Dict[str, Any]
        ARC task JSON path or loaded data.
    output_dir : str
        Directory where ``bk.pl``, ``bias.pl`` and ``exs.pl`` will be written.
    use_pixels : bool, optional
        Global representation flag.  ``True`` (default) means pixel based
        representation, ``False`` means object based.
    bk_use_pixels : bool, optional
        Override for the BK representation.  If ``None`` the global flag is
        used.
    exs_use_pixels : bool, optional
        Override for the EXS representation.  If ``None`` the global flag is
        used.
    enable_pi : bool, optional
        If ``True`` include facts and bias declarations for the invented
        ``hole2color/2`` predicate.
    max_clauses : int, optional
        ``max_clauses`` value for the generated bias.
    max_vars : int, optional
        ``max_vars`` value for the generated bias.
    max_body : int, optional
        ``max_body`` value for the generated bias.
    """
    os.makedirs(output_dir, exist_ok=True)
    task_data = load_task(task) if isinstance(task, str) else task
    if background_color is None:
        background_color = determine_background_color(
            task_data, pixel_threshold_pct=pixel_threshold_pct, debug=False
        )
    objs = extract_objects_from_task(task_data, background_color)

    if bk_use_pixels is None:
        bk_use_pixels = use_pixels
    if exs_use_pixels is None:
        exs_use_pixels = use_pixels

    bk_lines = objects_to_bk_lines(
        task_data,
        objs,
        include_pixels=bk_use_pixels,
        enable_pi=enable_pi,
        background_color=background_color,
        pixel_threshold_pct=pixel_threshold_pct,
    )
    bias_content = generate_bias(
        enable_pi=enable_pi,
        max_clauses=max_clauses,
        max_vars=max_vars,
        max_body=max_body,
    )
    if exs_use_pixels:
        exs_lines = outpix_examples(task_data, background_color, neg_factor=3)
    else:
        exs_lines = objects_to_exs_lines(objs)

    bk_path = os.path.join(output_dir, "bk.pl")
    bias_path = os.path.join(output_dir, "bias.pl")
    exs_path = os.path.join(output_dir, "exs.pl")

    save_bk(bk_lines, bk_path)
    with open(bias_path, "w") as f:
        f.write(bias_content)
    save_lines(exs_lines, exs_path)

    return bk_path, bias_path, exs_path


def run_popper_from_dir(
    kb_dir: str,
    *,
    popper_root: str = "./popper",
    timeout: int = 600,
    debug: bool = False,
    keep_tmp: bool = False,
    show_output: bool = False,
) -> Tuple[str | None, Dict[str, Any]]:
    """Run Popper via its CLI in a dedicated subprocess.

    Parameters
    ----------
    kb_dir : str
        Directory containing ``bk.pl``, ``bias.pl`` and ``exs.pl``.
    popper_root : str, optional
        Path to the Popper source checkout (default ``"./popper"``).
    timeout : int, optional
        Maximum time allowed for Popper in seconds (default ``600``).
    debug : bool, optional
        If ``True`` pass ``--debug`` to Popper.
    keep_tmp : bool, optional
        Unused, kept for API compatibility.

    Returns
    -------
    Tuple[str | None, Dict[str, Any]]
        Path to the learned hypothesis ``program.pl`` if any and a dictionary
        with statistics such as ``score``. Popper's output is streamed to
        ``stdout.txt`` under ``kb_dir`` for real-time inspection.
    """

    kb_path = pathlib.Path(kb_dir).resolve()
    assert kb_path.is_dir(), f"{kb_path} not found"

    popper_py = pathlib.Path(popper_root, "popper.py").resolve()
    if not popper_py.exists():
        raise FileNotFoundError(popper_py)

    cmd = [sys.executable, str(popper_py), str(kb_path), "--timeout", str(timeout)]
    if debug:
        cmd.append("--debug")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    logger.debug("Running Popper command: %s", " ".join(cmd))

    out_dir = kb_path / "popper_run"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "stdout.txt"

    stdout_lines: List[str] = []
    with open(log_path, "w", encoding="utf8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                stdout_lines.append(line)
                if show_output:
                    print(line, end="")
            proc.wait(timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stdout_lines.append("[timeout]\n")
            log_file.write("[timeout]\n")
        finally:
            if proc.stdout:
                proc.stdout.close()

        rc = proc.returncode

    run_stdout = "".join(stdout_lines)

    (out_dir / "stderr.txt").write_text("", "utf8")
    if show_output:
        print(f"--- Popper stdout for {kb_path.name} ---")
        print(run_stdout)
        print(f"--- end stdout for {kb_path.name} ---")

    if rc != 0:
        logger.debug("Popper exited with code %s", rc)
        return None, {"rc": rc, "reason": "popper error"}

    prog_path = kb_path / "program.pl"
    if not prog_path.exists():
        logger.debug("program.pl not found in %s", kb_path)
        # Fallback: parse solution from stdout
        lines = []
        in_sol = False
        for ln in run_stdout.splitlines():
            if "SOLUTION" in ln and "**********" in ln:
                in_sol = not in_sol
                continue
            if in_sol:
                ln = ln.strip()
                if not ln or ln.startswith("Precision"):
                    continue
                lines.append(ln)
        if lines:
            prog_path.write_text("\n".join(lines) + "\n")
        else:
            return None, {"rc": 0, "reason": "no_solution"}

    score = None
    for line in run_stdout.splitlines():
        if line.startswith("Score:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except Exception:
                score = None
            break

    logger.debug("Popper succeeded with score %s", score)
    return str(prog_path), {"rc": 0, "score": score}


def run_popper_subprocess(kb_dir: str):
    """Run Popper in a separate subprocess and return the learned program."""
    import subprocess
    import tempfile
    import json

    cmd = ["popper", kb_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    lines = []
    in_sol = False
    for ln in result.stdout.splitlines():
        if "SOLUTION" in ln:
            in_sol = True
            continue
        if in_sol:
            if not ln.strip():
                break
            lines.append(ln.strip())
    return lines


def run_popper_from_files(bk_path: str, bias_path: str, exs_path: str):
    """Create a temporary kb directory and run Popper with the given files."""
    import shutil
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        shutil.copy(bk_path, os.path.join(tmpdir, "bk.pl"))
        shutil.copy(bias_path, os.path.join(tmpdir, "bias.pl"))
        shutil.copy(exs_path, os.path.join(tmpdir, "exs.pl"))
        return run_popper_subprocess(tmpdir)


def run_popper_for_task(
    task: str | Dict[str, Any],
    output_dir: str,
    *,
    use_pixels: bool = True,
    bk_use_pixels: bool | None = None,
    exs_use_pixels: bool | None = None,
    enable_pi: bool = True,
    pixel_threshold_pct: int = 40,
    ):
    """Generate Popper input files for ``task`` and run Popper."""
    bk, bias, exs = generate_files_from_task(
        task,
        output_dir,
        use_pixels=use_pixels,
        bk_use_pixels=bk_use_pixels,
        exs_use_pixels=exs_use_pixels,
        enable_pi=enable_pi,
        pixel_threshold_pct=pixel_threshold_pct,
    )
    return run_popper_subprocess(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Popper files and run the solver on an ARC task")
    parser.add_argument("task", help="Path to the ARC task JSON file")
    parser.add_argument("--out", default="popper_kb", help="Directory to store generated files")
    parser.add_argument(
        "--repr",
        choices=["pixels", "objects"],
        default="pixels",
        help="Global representation (pixels or objects).",
    )
    parser.add_argument(
        "--bk-repr",
        choices=["pixels", "objects"],
        default=None,
        help="Override BK representation",
    )
    parser.add_argument(
        "--exs-repr",
        choices=["pixels", "objects"],
        default=None,
        help="Override EXS representation",
    )
    parser.add_argument(
        "--enable-pi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable predicate invention (hole2color).",
    )
    parser.add_argument(
        "--bg-threshold",
        type=int,
        default=40,
        help="Background color detection threshold percentage",
    )
    args = parser.parse_args()




    if not os.path.exists(args.task):
        raise SystemExit(f"Task file {args.task} not found")

    use_pixels = args.repr == "pixels"
    bk_repr = None if args.bk_repr is None else args.bk_repr == "pixels"
    exs_repr = None if args.exs_repr is None else args.exs_repr == "pixels"

    bk_path, bias_path, exs_path = generate_files_from_task(
        args.task,
        args.out,
        use_pixels=use_pixels,
        bk_use_pixels=bk_repr,
        exs_use_pixels=exs_repr,
        enable_pi=args.enable_pi,
        pixel_threshold_pct=args.bg_threshold,
    )
    print(f"BK, bias and EXS files saved to {args.out}")

    try:
        prog_path, info = run_popper_from_dir(args.out)
        if prog_path is not None:
            print("Learned hypothesis saved to", prog_path)
            with open(prog_path) as f:
                print(f.read())
        else:
            print("Popper finished without finding a solution")
    except Exception as e:
        traceback.print_exc()
        print(f"Popper run failed: {e}")
        traceback.print_exc()
