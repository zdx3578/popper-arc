# bkbias/object_matching_exact.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from core.exact_ops import (
    ObjLite, exact_global_translate, exact_global_d4_translate, exact_color_map
)

def _to_objlite_list(objs: List[Any]) -> List[ObjLite]:
    """把你仓库里 extract_objects_from_task 返回的对象集合适配成 ObjLite 列表。
       要求每个元素有属性 .pixels (Nx2 np.ndarray) 与 .color (int)。
       如果你的对象是 dict({'pixels':..., 'color':...})，此处做对应适配即可。
    """
    res: List[ObjLite] = []
    for o in objs:
        pix = getattr(o, "pixels", None) or o.get("pixels")
        col = getattr(o, "color", None)  or o.get("color")
        res.append(ObjLite(pixels=np.asarray(pix, dtype=np.int32), color=int(col)))
    return res

def analyze_task_transformations(task_data: Dict[str,Any], objs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """精确匹配版的任务变换分析（兼容旧入口）
       输入:
         - task_data: 你的任务 JSON（此处只用来拿 train 对，并可扩展）
         - objs: 旧管道里 extract_objects_from_task 的输出，形如
                 {'train': [{'in_objs':[...], 'out_objs':[...], 'pair_id':...}, ...], 'test': [...]}
       输出:
         - transformations: 与 generate_files_from_task 预期一致的结构（示例返回结构见下）
    """
    trains = objs.get("train", [])
    if not trains:
        return {"type":"none", "details":{}}

    # 严格：多对 pair 必须得到一致参数，否则返回 none（或退化为子集规则：此处先最小实现）
    d4_dxdy_set = set()
    all_in_ol, all_out_ol = [], []
    for pair in trains:
        in_ol  = _to_objlite_list(pair["in_objs"])
        out_ol = _to_objlite_list(pair["out_objs"])
        all_in_ol.append(in_ol); all_out_ol.append(out_ol)

        r1 = exact_global_d4_translate(in_ol, out_ol)
        if r1 is not None:
            d4_dxdy_set.add(r1)
        else:
            r0 = exact_global_translate(in_ol, out_ol)
            if r0 is not None:
                d4_dxdy_set.add(("id", r0[0], r0[1]))
            else:
                # 全局规则不成立；这里先返回 none（后续可扩展子集规则/线/框族）
                return {"type":"none", "details":{}}

    if len(d4_dxdy_set) != 1:
        return {"type":"none", "details":{}}

    d4, dx, dy = next(iter(d4_dxdy_set))

    # 颜色映射需在一致对齐下计算；多对 pair 取“单值一致”的交集
    cmaps = []
    for in_ol, out_ol in zip(all_in_ol, all_out_ol):
        cmap = exact_color_map(in_ol, out_ol, dx=dx, dy=dy, d4=d4)
        cmaps.append(cmap)
    # 求交集（只保留各 pair 一致的 Cin→Cout）
    keys = set.intersection(*(set(m.keys()) for m in cmaps)) if cmaps else set()
    final_cmap: Dict[int,int] = {}
    for k in keys:
        vals = {m[k] for m in cmaps}
        if len(vals) == 1:
            final_cmap[k] = vals.pop()

    # 组织成旧 generate_files_from_task 能理解的“变换描述”（示例；按你的实际格式改名即可）
    trans: Dict[str,Any] = {
        "type": "d4_translate_recolor" if final_cmap else "d4_translate",
        "details": {
            "d4": d4, "dx": dx, "dy": dy,
            "color_map": final_cmap
        }
    }
    return trans
