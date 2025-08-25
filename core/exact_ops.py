# core/exact_ops.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Iterable
import numpy as np
from dataclasses import dataclass

# ==== 基础数据契约（假设 Obj 与像素为下述最小接口） ====

@dataclass
class ObjLite:
    """与仓库内 Obj 结构对齐的极简接口：只用到 pixels 与 color。
       如果你已有 analy/AttrActionIndex.py 的对象类，确保属性名兼容或写一个小适配器。
    """
    pixels: np.ndarray  # shape=(N,2), dtype=int; 每行为 (r,c)
    color: int

# ==== 形状签名 / D4 规范形 ====

def shape_sig_from_pixels(pix: np.ndarray) -> frozenset:
    """位移不变签名：把像素集合以其左上为原点，转成偏移集合"""
    rmin, cmin = pix.min(axis=0)
    return frozenset((int(r - rmin), int(c - cmin)) for r, c in pix)

def _norm_offsets(S: Iterable[Tuple[int,int]]) -> frozenset:
    """把偏移集合平移到左上(0,0)"""
    rs = [r for r, _ in S]; cs = [c for _, c in S]
    rmin, cmin = min(rs), min(cs)
    return frozenset((r - rmin, c - cmin) for r, c in S)

def _d4_apply_offset(p: Tuple[int,int], tag: str) -> Tuple[int,int]:
    """在偏移坐标系中应用 D4 变换（围绕原点），返回新偏移"""
    r, c = p
    if tag == "id":     return ( r,  c)
    if tag == "rot90":  return ( c, -r)
    if tag == "rot180": return (-r, -c)
    if tag == "rot270": return (-c,  r)
    if tag == "flipH":  return ( r, -c)   # 水平镜像: 左右翻转
    if tag == "flipV":  return (-r,  c)   # 垂直镜像: 上下翻转
    if tag == "diag":   return ( c,  r)   # 主对角翻转
    if tag == "anti":   return (-c, -r)   # 副对角翻转
    raise ValueError(f"unknown D4 tag: {tag}")

_D4_TAGS: Tuple[str,...] = ("id","rot90","rot180","rot270","flipH","flipV","diag","anti")

def canonize_sig(sig: frozenset) -> frozenset:
    """把一个位移不变签名映射到 D4 规范形（8 个变换下字典序最小的形）"""
    cands = []
    for t in _D4_TAGS:
        S = {_d4_apply_offset(p, t) for p in sig}
        cands.append(sorted(_norm_offsets(S)))
    best = min(cands)
    return frozenset(best)

def apply_d4_to_pixels(pix: np.ndarray, tag: str) -> np.ndarray:
    """对绝对像素坐标应用 D4 变换：
       实现：先转偏移集合 → 应用 D4 → 归一化到左上 → 再加回原始 rmin,cmin，使绝对位置可比。
       注意：绝对位置只是“临时容器”，后续我们会再与 out 对象的最小坐标对齐，求出 (dx,dy)。
    """
    rmin, cmin = pix.min(axis=0)
    offsets = [(int(r - rmin), int(c - cmin)) for r, c in pix]
    trans = [_d4_apply_offset(p, tag) for p in offsets]
    norm  = sorted(_norm_offsets(trans))
    arr   = np.asarray(norm, dtype=np.int32)
    arr[:, 0] += int(rmin); arr[:, 1] += int(cmin)
    return arr

# ==== 严格全局平移 / D4+平移 检测 ====

def _bucket_by_sig(objs: List[ObjLite]) -> Dict[frozenset, List[np.ndarray]]:
    """按位移不变签名分桶，加速精确匹配（每个桶中保存像素数组，后续用于集合等式对比）"""
    buckets: Dict[frozenset, List[np.ndarray]] = {}
    for o in objs:
        sig = shape_sig_from_pixels(o.pixels)
        buckets.setdefault(sig, []).append(o.pixels)
    return buckets

def _pixels_set(a: np.ndarray) -> frozenset:
    return frozenset((int(r), int(c)) for r, c in a)

def exact_global_translate(
    in_objs: List[ObjLite],
    out_objs: List[ObjLite]
) -> Optional[Tuple[int,int]]:
    """严格全局平移检测：
       ∃唯一 (dx,dy) 使得 ∀ in 对象 Oin，平移 (dx,dy) 后的像素集合 == 某个 out 对象的像素集合。
       若存在，返回 (dx,dy)；否则返回 None。
    """
    out_buckets = _bucket_by_sig(out_objs)
    deltas: set[Tuple[int,int]] = set()

    for ia in in_objs:
        sig = shape_sig_from_pixels(ia.pixels)
        cand_pix_list = out_buckets.get(sig, [])
        if not cand_pix_list:
            return None  # 没有同形对象可匹配

        r1a, c1a = ia.pixels.min(axis=0)
        hit = False
        for pb in cand_pix_list:
            r1b, c1b = pb.min(axis=0)
            dx, dy = int(c1b - c1a), int(r1b - r1a)
            moved = {(int(r + dy), int(c + dx)) for r, c in ia.pixels}
            if moved == _pixels_set(pb):
                deltas.add((dx, dy)); hit = True; break
        if not hit:
            return None

    if len(deltas) == 1:
        return next(iter(deltas))
    return None  # 各对象位移不一致 → 非全局平移

def exact_global_d4_translate(
    in_objs: List[ObjLite],
    out_objs: List[ObjLite]
) -> Optional[Tuple[str,int,int]]:
    """严格全局 D4+平移检测：
       ∃唯一 (tag,dx,dy) 使得 ∀ in 对象 Oin，先做 D4(tag)，再平移 (dx,dy)，像素集合 == 某个 out 对象。
       若存在，返回 (tag,dx,dy)；否则 None。
    """
    # 先按“位移不变”签名分桶；D4 作用在偏移空间上，作用后签名会变化。
    out_buckets = _bucket_by_sig(out_objs)

    for tag in _D4_TAGS:
        deltas: set[Tuple[int,int]] = set()
        ok = True
        for ia in in_objs:
            pix_d4 = apply_d4_to_pixels(ia.pixels, tag)
            sig = shape_sig_from_pixels(pix_d4)
            cand_pix_list = out_buckets.get(sig, [])
            if not cand_pix_list:
                ok = False; break

            r1a, c1a = pix_d4.min(axis=0)
            matched = False
            for pb in cand_pix_list:
                r1b, c1b = pb.min(axis=0)
                dx, dy = int(c1b - c1a), int(r1b - r1a)
                moved = {(int(r + dy), int(c + dx)) for r, c in pix_d4}
                if moved == _pixels_set(pb):
                    deltas.add((dx, dy)); matched = True; break
            if not matched:
                ok = False; break

        if ok and len(deltas) == 1:
            dx, dy = next(iter(deltas))
            return (tag, dx, dy)

    return None

# ==== 严格颜色映射（在对齐后） ====

def exact_color_map(
    in_objs: List[ObjLite],
    out_objs: List[ObjLite],
    dx: int,
    dy: int,
    d4: str = "id"
) -> Dict[int,int]:
    """严格颜色映射：
       在已知全局 (d4,dx,dy) 对齐下，逐 in 对象：
       - 对 in 对象应用 D4 与平移
       - 在 out 中找像素集合完全相等的对象
       - 建立 Cin -> Cout 映射；若某 Cin 映射冲突（多值），就丢弃该 Cin
       返回单值映射字典（可能为空）
    """
    # 为快速定位，用“位移不变”签名索引 out
    out_buckets: Dict[frozenset, List[Tuple[np.ndarray,int]]] = {}
    for ob in out_objs:
        sig = shape_sig_from_pixels(ob.pixels)
        out_buckets.setdefault(sig, []).append((ob.pixels, ob.color))

    cmap: Dict[int,int] = {}
    bad: set[int] = set()

    for ia in in_objs:
        pix = ia.pixels
        if d4 != "id":
            pix = apply_d4_to_pixels(pix, d4)
        # 平移
        moved = np.empty_like(pix)
        moved[:, 0] = pix[:, 0] + int(dy)
        moved[:, 1] = pix[:, 1] + int(dx)

        sig = shape_sig_from_pixels(moved)
        cand = out_buckets.get(sig, [])
        if not cand:
            continue

        # 找像素集合完全相等的 out 对象
        moved_set = _pixels_set(moved)
        target_color = None
        for pb, col in cand:
            if moved_set == _pixels_set(pb):
                target_color = int(col); break
        if target_color is None:
            continue

        cin = int(ia.color); cout = target_color
        if cin in bad:
            continue
        if (cin in cmap) and (cmap[cin] != cout):
            # 冲突：丢掉该 Cin
            cmap.pop(cin, None)
            bad.add(cin)
        else:
            cmap[cin] = cout

    return cmap
