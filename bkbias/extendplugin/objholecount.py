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


def count_holes(grid: Grid, comp: Iterable[Position]) -> int:
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
    return holes


def count_object_holes(obj: Iterable[Tuple[int, Position]]) -> int:
    """Return the number of holes inside an object."""
    from objattr import shift_to_origin, object_to_grid

    obj_origin = shift_to_origin(obj)
    grid = object_to_grid(obj_origin)
    bin_grid: Grid = [[1 if v != 0 else 0 for v in row] for row in grid]
    comp = [(r, c) for r, row in enumerate(bin_grid) for c, v in enumerate(row) if v]
    return count_holes(bin_grid, comp)
