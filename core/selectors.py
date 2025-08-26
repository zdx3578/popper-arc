# core/selectors.py
# -*- coding: utf-8 -*-
"""
Selector：针对 ObjInfo（dict）的精确布尔筛选器。

设计要点（遵守你的约束）：
- 仅操作已有的 ObjInfo 字段（如: 'main_color', 'holes', 'width', 'height',
  'obj_id', 'obj_shape_ID', 'obj_sort_ID', 'pair_id', 'in_or_out'）。
- 不内置任何新的模式识别（如 is_box 等）；后续若需要，可通过 sel_custom 注入外部判定。
- 支持组合：AND (&)、OR (|)、NOT (~)。
- 只做筛选与判定；不做其它副作用。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

ObjInfo = Dict[str, Any]
Predicate = Callable[[ObjInfo], bool]


# ---- 基类 ----

@dataclass(frozen=True)
class Selector:
    """最小可用选择器：包装一个对 ObjInfo 的布尔谓词。"""
    name: str
    predicate: Predicate
    params: Dict[str, Any] = field(default_factory=dict)

    def match(self, oi: ObjInfo) -> bool:
        """单个对象是否匹配。"""
        try:
            return bool(self.predicate(oi))
        except Exception:
            # 任何异常都视为不匹配，避免中断分析流水线
            return False

    def filter(self, objs: Iterable[ObjInfo]) -> List[ObjInfo]:
        """对一组对象做筛选，返回匹配的子集（保持输入顺序）。"""
        return [oi for oi in objs if self.match(oi)]

    def match_ids(self, objs: Iterable[ObjInfo]) -> List[str]:
        """返回匹配对象的 obj_id 列表（若无 obj_id 字段则返回空字符串占位）。"""
        out: List[str] = []
        for oi in objs:
            if self.match(oi):
                out.append(str(oi.get("obj_id", "")))
        return out

    # 组合操作符
    def __and__(self, other: "Selector") -> "Selector":
        return AndSelector(self, other)

    def __or__(self, other: "Selector") -> "Selector":
        return OrSelector(self, other)

    def __invert__(self) -> "Selector":
        return NotSelector(self)

    # 简单可读性输出
    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


# ---- 组合选择器 ----

@dataclass(frozen=True)
class AndSelector(Selector):
    left: Selector = field(default=None)
    right: Selector = field(default=None)

    def __init__(self, left: Selector, right: Selector):
        object.__setattr__(self, "name", f"AND({left.name},{right.name})")
        object.__setattr__(self, "predicate", lambda oi: left.match(oi) and right.match(oi))
        object.__setattr__(self, "params", {"left": left.to_dict(), "right": right.to_dict()})
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True)
class OrSelector(Selector):
    left: Selector = field(default=None)
    right: Selector = field(default=None)

    def __init__(self, left: Selector, right: Selector):
        object.__setattr__(self, "name", f"OR({left.name},{right.name})")
        object.__setattr__(self, "predicate", lambda oi: left.match(oi) or right.match(oi))
        object.__setattr__(self, "params", {"left": left.to_dict(), "right": right.to_dict()})
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True)
class NotSelector(Selector):
    inner: Selector = field(default=None)

    def __init__(self, inner: Selector):
        object.__setattr__(self, "name", f"NOT({inner.name})")
        object.__setattr__(self, "predicate", lambda oi: not inner.match(oi))
        object.__setattr__(self, "params", {"inner": inner.to_dict()})
        object.__setattr__(self, "inner", inner)


# ---- 工厂函数（仅围绕已有字段；不引入新识别逻辑） ----

def sel_all() -> Selector:
    """恒真选择器。"""
    return Selector(name="all", predicate=lambda oi: True)

def sel_key_eq(key: str, value: Any) -> Selector:
    """通用字段等值选择器：oi[key] == value。缺失字段时不匹配。"""
    def _pred(oi: ObjInfo) -> bool:
        return key in oi and oi.get(key) == value
    return Selector(name=f"{key}=={value!r}", predicate=_pred, params={"key": key, "value": value})

def sel_color_eq(color: int) -> Selector:
    """主颜色等值：oi['main_color'] == color。"""
    return sel_key_eq("main_color", int(color))

def sel_holes_eq(h: int) -> Selector:
    """洞数等值：oi['holes'] == h。"""
    return sel_key_eq("holes", int(h))

def sel_size_eq(width: int, height: int) -> Selector:
    """尺寸等值：oi['width']==width 且 oi['height']==height。"""
    def _pred(oi: ObjInfo) -> bool:
        return oi.get("width") == int(width) and oi.get("height") == int(height)
    return Selector(
        name=f"size==({width},{height})",
        predicate=_pred,
        params={"width": int(width), "height": int(height)}
    )

def sel_shape_id_eq(shape_id: Any) -> Selector:
    """形状 ID 等值：优先 obj_sort_ID，其次 obj_shape_ID。"""
    def _pred(oi: ObjInfo) -> bool:
        sid = oi.get("obj_sort_ID", oi.get("obj_shape_ID"))
        return sid == shape_id
    return Selector(name=f"shape_id=={shape_id!r}", predicate=_pred, params={"shape_id": shape_id})

def sel_in_or_out(val: str) -> Selector:
    """来源等值：oi['in_or_out'] in {'in','out'}。"""
    val = str(val)
    return sel_key_eq("in_or_out", val)

def sel_pair_id_eq(pid: Any) -> Selector:
    """pair_id 等值。"""
    return sel_key_eq("pair_id", pid)

def sel_obj_id_in(ids: Iterable[str]) -> Selector:
    """obj_id ∈ 给定集合。"""
    idset = {str(x) for x in ids}
    def _pred(oi: ObjInfo) -> bool:
        return str(oi.get("obj_id", "")) in idset
    return Selector(name=f"obj_id in {len(idset)} ids", predicate=_pred, params={"count": len(idset)})

def sel_custom(name: str, predicate: Predicate, **params) -> Selector:
    """
    自定义选择器：可把你已有的严格判定（例如 is_line_strict(基于 ObjInfo 的版本)）注入进来。
    注意：这里不实现任何新判定逻辑，只是包装你传入的 predicate。
    """
    return Selector(name=name, predicate=predicate, params=params or {})
