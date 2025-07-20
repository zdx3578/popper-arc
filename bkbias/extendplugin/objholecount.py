from collections import deque
from typing import List, Tuple, Iterable



Grid = List[List[int]]
Position = Tuple[int, int]


def flood(grid: Grid, start: Position, target: Iterable[int]) -> List[Position]:
    """Breadth-first search collecting cells with values in ``target``."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    q = deque([start])
    comp: List[Position] = []
    seen = {start}
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] in target:
                seen.add((nr, nc))
                q.append((nr, nc))
    return comp


def bbox(comp: Iterable[Position]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def count_holes000(grid: Grid, comp: Iterable[Position]) -> int:
    """Count enclosed zero regions in ``grid`` within ``comp`` bounding box."""
    r0, c0, r1, c1 = bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    obj = {(r - r0, c - c0) for r, c in comp}
    seen = [[False] * w for _ in range(h)]
    holes = 0
    for r in range(h):
        for c in range(w):
            if (r, c) in obj or seen[r][c]:
                continue
            blob = flood([[0 if (rr, cc) not in obj else 1 for cc in range(w)] for rr in range(h)], (r, c), {0})
            edge = False
            for rr, cc in blob:
                if rr in (0, h - 1) or cc in (0, w - 1):
                    edge = True
                seen[rr][cc] = True
            if not edge:
                holes += 1
    print(f"holes, {holes} in {len(comp)} cells"    )
    return holes

def count_holes(grid: Grid, comp: Iterable[Position]) -> int:
    """Count enclosed zero regions in ``grid`` within ``comp`` bounding box,
    but exclude thin lines (including polylines) which cannot have holes."""
    # 1. 先计算组件的边界框
    r0, c0, r1, c1 = bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    # 2. 如果宽度或高度都小于 3，视作线条，无洞
    if h < 3 or w < 3:
        # print(f"skipping line-like comp (h={h}, w={w}), holes=0")
        return None

    # 3. 正常计算洞数
    obj = {(r - r0, c - c0) for r, c in comp}
    seen = [[False] * w for _ in range(h)]
    holes = 0
    for r in range(h):
        for c in range(w):
            if (r, c) in obj or seen[r][c]:
                continue
            # flood out the background blob
            blob = flood(
                [[0 if (rr, cc) not in obj else 1 for cc in range(w)]
                 for rr in range(h)],
                (r, c), {0}
            )
            edge = False
            for rr, cc in blob:
                if rr in (0, h - 1) or cc in (0, w - 1):
                    edge = True
                seen[rr][cc] = True
            if not edge:
                holes += 1

    # print(f"holes: {holes} in {len(comp)} cells (h={h}, w={w})")
    return holes



def count_object_holes(obj: Iterable[Tuple[int, Position]]) -> int:
    """Return the number of holes inside an object."""
    from bkbias.objattr import shift_to_origin

    obj_origin = shift_to_origin(obj)
    coords = [(r, c) for _, (r, c) in obj_origin]
    if not coords:
        return 0

    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    bin_grid: Grid = [[0] * (max_c + 1) for _ in range(max_r + 1)]
    for r, c in coords:
        bin_grid[r][c] = 1

    return count_holes(bin_grid, coords)
