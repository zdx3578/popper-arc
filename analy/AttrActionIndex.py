from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, FrozenSet, Any
import numpy as np

from bkbias.objattr import (
    extract_object_infos_from_grid,
    determine_background_color,
)


@dataclass
class AttrIndex:
    """Indexes objects by various attributes for quick lookup."""

    by_color: Dict[int, List[int]] = field(default_factory=dict)
    by_holes: Dict[int, List[int]] = field(default_factory=dict)
    by_size: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    by_shape_sig: Dict[FrozenSet[Tuple[int, Tuple[int, int]]], List[int]] = field(
        default_factory=dict
    )


@dataclass
class RelIndex:
    """Stores relations between objects in a scene."""

    touching: Set[Tuple[int, int]] = field(default_factory=set)
    aligned_row: Dict[int, List[int]] = field(default_factory=dict)
    aligned_col: Dict[int, List[int]] = field(default_factory=dict)
    same_shape: Set[Tuple[int, int]] = field(default_factory=set)


@dataclass
class Scene:
    """Represents a grid scene with objects and their indexes."""

    grid: np.ndarray
    bg_color: Optional[int]
    objs: List[Dict[str, Any]]
    attr: AttrIndex
    rel: RelIndex
    H: int
    W: int


@dataclass
class PairState:
    """Stores information about one training pair."""

    pair_id: int
    in_scene: Scene
    out_scene: Scene
    delta_pixels: np.ndarray
    delta_hist: Dict[str, Dict[Any, int]]


def _build_attr_index(objs: List[Dict[str, Any]]) -> AttrIndex:
    idx = AttrIndex()
    for i, obj in enumerate(objs):
        color = obj.get("main_color")
        idx.by_color.setdefault(color, []).append(i)

        holes = obj.get("holes", 0)
        idx.by_holes.setdefault(holes, []).append(i)

        size = (obj.get("width", 0), obj.get("height", 0))
        idx.by_size.setdefault(size, []).append(i)

        shape_sig = obj.get("obj_000")
        if shape_sig is not None:
            idx.by_shape_sig.setdefault(frozenset(shape_sig), []).append(i)
    return idx


def _objects_touch(obj_a: FrozenSet[Tuple[int, Tuple[int, int]]], obj_b: FrozenSet[Tuple[int, Tuple[int, int]]]) -> bool:
    pixels_a = {coord for _, coord in obj_a}
    pixels_b = {coord for _, coord in obj_b}
    for r, c in pixels_a:
        if (r + 1, c) in pixels_b or (r - 1, c) in pixels_b or (r, c + 1) in pixels_b or (r, c - 1) in pixels_b:
            return True
    return False


def _build_rel_index(objs: List[Dict[str, Any]]) -> RelIndex:
    rel = RelIndex()
    for i, obj in enumerate(objs):
        top = obj.get("top")
        left = obj.get("left")
        rel.aligned_row.setdefault(top, []).append(i)
        rel.aligned_col.setdefault(left, []).append(i)

    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            obj_i = objs[i]
            obj_j = objs[j]
            if _objects_touch(obj_i["obj"], obj_j["obj"]):
                rel.touching.add((i, j))
            if obj_i.get("obj_000") == obj_j.get("obj_000"):
                rel.same_shape.add((i, j))
    return rel


def build_scene(grid: np.ndarray, objs: List[Dict[str, Any]], bg_color: Optional[int]) -> Scene:
    H, W = grid.shape
    attr = _build_attr_index(objs)
    rel = _build_rel_index(objs)
    return Scene(grid=grid, bg_color=bg_color, objs=objs, attr=attr, rel=rel, H=H, W=W)


def pair_states_from_task(task_data: Dict[str, Any], background_color: Optional[int] = None) -> List[PairState]:
    """Construct :class:`PairState` objects for all training pairs in ``task_data``."""
    if background_color is None:
        background_color = determine_background_color(task_data)

    pair_states: List[PairState] = []
    for pair_id, pair in enumerate(task_data.get("train", [])):
        in_grid = np.array(pair["input"], dtype=int)
        out_grid = np.array(pair["output"], dtype=int)

        in_objs = extract_object_infos_from_grid(pair_id, "in", pair["input"], background_color)
        out_objs = extract_object_infos_from_grid(pair_id, "out", pair["output"], background_color)

        in_scene = build_scene(in_grid, in_objs, background_color)
        out_scene = build_scene(out_grid, out_objs, background_color)

        delta_pixels = np.argwhere(in_grid != out_grid)
        delta_hist: Dict[str, Dict[Any, int]] = {"color": {}}
        for r, c in delta_pixels:
            before = int(in_grid[r, c])
            after = int(out_grid[r, c])
            key = (before, after)
            delta_hist["color"][key] = delta_hist["color"].get(key, 0) + 1

        pair_states.append(
            PairState(
                pair_id=pair_id,
                in_scene=in_scene,
                out_scene=out_scene,
                delta_pixels=delta_pixels,
                delta_hist=delta_hist,
            )
        )
    return pair_states
