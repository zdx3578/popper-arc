import os
import json
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, FrozenSet


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
    obj_num_ID = "objID : " + str(managerid.get_id("OBJshape", obj_000))
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
        "obj_num_ID": obj_num_ID,
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


def objects_to_bk_lines(all_objects: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]]) -> List[str]:
    """Convert extracted object infos into simple Popper background facts."""
    lines: List[str] = []

    # gather grid sizes
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

    for cat in ("input", "output"):
        for pair_id, objs in all_objects.get(cat, []):
            for idx, info in enumerate(objs):
                obj_name = prolog_atom(info["obj_num_ID"]) or f"obj_{pair_id}_{idx}"
                lines.append(f"object({obj_name}).")
                lines.append(f"belongs({obj_name},{cat}_{pair_id}).")
                lines.append(f"x_min({obj_name},{info['left']}).")
                lines.append(f"y_min({obj_name},{info['top']}).")
                lines.append(f"width({obj_name},{info['width']}).")
                lines.append(f"height({obj_name},{info['height']}).")
                lines.append(f"color({obj_name},{info['main_color']}).")
                lines.append(f"size({obj_name},{info['size']}).")

    return lines


def save_bk(lines: List[str], path: str) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines))


def generate_bias() -> str:
    """Return a minimal bias string for the generated BK."""
    bias_lines = [
        "head_pred(target,1).",
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
        "body_pred(pix,4).",
        "max_vars(6).",
        "max_body(6).",
        "max_clauses(3).",
    ]
    return "\n".join(bias_lines)


def grids_to_exs_lines(task_data: Dict[str, Any],
                       background_color: int | None = None) -> List[str]:
    """Convert all grids in the task to pixel facts lines and add pos examples."""
    if background_color is None:
        background_color = determine_background_color(task_data)

    lines: List[str] = []

    # add a positive example for each pair id
    for pair_id, _ in enumerate(task_data.get("train", [])):
        lines.append(f"pos(target({pair_id})).")

    for pair_id, pair in enumerate(task_data.get("train", [])):
        for kind in ("input", "output"):
            grid = pair[kind]
            label = f"{pair_id}_{'in' if kind=='input' else 'out'}"
            for i, row in enumerate(grid):
                for j, color in enumerate(row):
                    if background_color is not None and color == background_color:
                        continue
                    lines.append(f"pix({label},{i},{j},{color}).")
    return lines


def save_lines(lines: List[str], path: str) -> None:
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")


def generate_files_from_task(task_path: str, output_dir: str) -> Tuple[str, str, str]:
    """Generate BK, bias and exs files from a task JSON."""
    os.makedirs(output_dir, exist_ok=True)
    task_data = load_task(task_path)
    background = determine_background_color(task_data)
    objs = extract_objects_from_task(task_data, background)
    bk_lines = objects_to_bk_lines(objs)
    bias_content = generate_bias()
    exs_lines = grids_to_exs_lines(task_data, background)

    bk_path = os.path.join(output_dir, "bk.pl")
    bias_path = os.path.join(output_dir, "bias.pl")
    exs_path = os.path.join(output_dir, "exs.pl")

    save_lines(bk_lines, bk_path)
    with open(bias_path, "w") as f:
        f.write(bias_content)
    save_lines(exs_lines, exs_path)

    return bk_path, bias_path, exs_path


def run_popper_from_dir(kb_dir: str):
    """Run Popper on the given directory containing bk.pl, bias.pl and exs.pl."""
    import importlib.util  # Ensure importlib.util exists before importing popper
    from popper.util import Settings
    from popper.loop import learn_solution

    settings = Settings(kbpath=kb_dir)
    return learn_solution(settings)


def run_popper_from_files(bk_path: str, bias_path: str, exs_path: str):
    """Create a temporary kb directory and run Popper with the given files."""
    import shutil
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        shutil.copy(bk_path, os.path.join(tmpdir, "bk.pl"))
        shutil.copy(bias_path, os.path.join(tmpdir, "bias.pl"))
        shutil.copy(exs_path, os.path.join(tmpdir, "exs.pl"))
        return run_popper_from_dir(tmpdir)


def run_popper_for_task(task_path: str, output_dir: str):
    """Generate Popper input files for ``task_path`` and run Popper."""
    bk, bias, exs = generate_files_from_task(task_path, output_dir)
    return run_popper_from_dir(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Popper files and run the solver on an ARC task")
    parser.add_argument("task", help="Path to the ARC task JSON file")
    parser.add_argument("--out", default="popper_kb", help="Directory to store generated files")
    args = parser.parse_args()

    if not os.path.exists(args.task):
        raise SystemExit(f"Task file {args.task} not found")

    bk_path, bias_path, exs_path = generate_files_from_task(args.task, args.out)
    print(f"BK, bias and EXS files saved to {args.out}")

    try:
        prog, score, stats = run_popper_from_dir(args.out)
        if prog is not None:
            print("Learned hypothesis:")
            print(prog)
        else:
            print("Popper finished without finding a solution")
    except Exception as e:
        print(f"Popper run failed: {e}")
