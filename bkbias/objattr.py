import os
import json
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, FrozenSet

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




def determine_background_color(task_data: Dict[str, Any], debug: bool = False) -> int | None:
    """Analyze all training grids and return the dominant background color."""
    # Threshold percentage for determining a background color
    pixel_threshold_pct = 60

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
                             grid_hw: Tuple[int, int], background: int | None = None) -> Dict[str, Any]:
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
    obj_id = f"{pair_id}_{in_or_out}_{hash(obj_000)}_{bounding_box[0]}_{bounding_box[1]}"
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
        "obj_id": obj_id,
        "rotated_variants": [("rot_0", obj_00)] + [
            (f"rot_{d}", shift_to_origin(grid_to_object({90: rot90, 180: rot180, 270: rot270}[d](object_to_grid(obj_00)))))
            for d in (90, 180, 270)
        ],
        "mirrored_variants": extend_obj(obj_000),
        "obj_weight": 0,
        "is_part_of": [],
        "has_parts": [],
    }
    return info


def extract_object_infos_from_grid(pair_id: int, in_or_out: str, grid: List[List[int]],
                                   background_color: int | None) -> List[Dict[str, Any]]:
    """Extract objects and attributes from a single grid."""
    objects = extract_objects(grid, background_color)
    h, w = len(grid), len(grid[0])
    return [create_weighted_obj_info(pair_id, in_or_out, obj, (h, w), background_color) for obj in objects]


def extract_objects_from_task(task_data: Dict[str, Any], background_color: int = None,
                              param: Tuple[bool, bool, bool] = (True, True, False)) -> Dict[str, List[Tuple[int, List[Dict[str, Any]]]]]:
    """Given a task dictionary, return extracted object infos for each pair."""
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


if __name__ == "__main__":
    import sys
    task_path = sys.argv[1] if len(sys.argv) > 1 else "05a7bcf2.json"
    if os.path.exists(task_path):
        objs = run_extraction(task_path)
        print(json.dumps(objs, indent=2, default=str))
    else:
        print(f"Task file {task_path} not found")
