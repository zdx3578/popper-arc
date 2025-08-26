






from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
from collections import Counter

# -*- coding: utf-8 -*-
from dataclasses import dataclass

from analy.AttrActionIndex import PairState, Scene
from core.exact_ops import (exact_global_translate, exact_global_d4_translate,
                            exact_color_map, is_line_strict, is_box_strict,)
                            # solve_delay_line_params, solve_shrink_rect_k)
from core.selectors import Selector  # Filter(color==k)/Filter(is_line)/...

@dataclass
class RuleHypothesis:
    name: str                    # "translate", "recolor", "delay_line", "shrink_rect", "rotate90+translate", ...
    params: Dict[str, int|str|Tuple]  # {"dx":1,"dy":-1} / {"dir":"ne","k":1} / {"cmap":{2:7,...}}
    selector: Optional[Selector] = None   # None 表示全局；否则仅对满足 Sel 的对象生效
    order: int = 0   # 执行顺序，越小越先执行

@dataclass
class AnalysisResult:
    rules: List[RuleHypothesis]
    # 便于调试：对每个 pair 的预测输出
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    notes: str = ""


Obj = ObjLite = Dict[str, Any]  ## objattr  def create_weighted_obj_info   info objinfo


def _to_objlite_list(objs: List[Obj]) -> List[ObjLite]:
    return [ObjLite(pixels=o.pixels.astype(np.int32, copy=False), color=int(o.color)) for o in objs]

def _render_objects(objs: List[Obj], H: int, W: int, bg: int) -> np.ndarray:
    grid = np.full((H,W), fill_value=bg, dtype=np.int16)
    for o in objs:
        grid[o.pixels[:,0], o.pixels[:,1]] = o.color
    return grid

def _apply_translate(objs: List[Obj], dx: int, dy: int) -> List[Obj]:
    out=[]
    for o in objs:
        pix = o.pixels.copy()
        pix[:,0] += dy; pix[:,1] += dx
        r1,c1 = pix.min(axis=0); r2,c2 = pix.max(axis=0)
        out.append(Obj(
            id=o.id, color=o.color, pixels=pix, bbox=(int(r1),int(c1),int(r2),int(c2)),
            area=int(pix.shape[0]), width=int(c2-c1+1), height=int(r2-r1+1),
            holes=o.holes, shape_sig=o.shape_sig, canon_sig=o.canon_sig
        ))
    return out

def _apply_recolor(objs: List[Obj], cmap: Dict[int,int]) -> List[Obj]:
    out=[]
    for o in objs:
        col = cmap.get(int(o.color), int(o.color))
        out.append(Obj(
            id=o.id, color=col, pixels=o.pixels, bbox=o.bbox, area=o.area,
            width=o.width, height=o.height, holes=o.holes,
            shape_sig=o.shape_sig, canon_sig=o.canon_sig
        ))
    return out

# D4 应用到对象像素（严格离散）
from core.exact_ops import apply_d4_to_pixels
def _apply_d4(objs: List[Obj], tag: str) -> List[Obj]:
    if tag == "id": return [o for o in objs]
    out=[]
    for o in objs:
        pix = apply_d4_to_pixels(o.pixels, tag)
        r1,c1 = pix.min(axis=0); r2,c2 = pix.max(axis=0)
        # shape_sig/canon_sig 在应用 D4 后会变化；此处仅用于仿真渲染可不更新
        out.append(Obj(
            id=o.id, color=o.color, pixels=pix, bbox=(int(r1),int(c1),int(r2),int(c2)),
            area=int(pix.shape[0]), width=int(c2-c1+1), height=int(r2-r1+1),
            holes=o.holes, shape_sig=None, canon_sig=None  # 仿真阶段可忽略
        ))
    return out

# ---------- 分类桶（跨全体 train 的严格布尔聚合） ----------
@dataclass
class PairSummary:
    # pair 内颜色与计数
    colors_in: Set[int] = field(default_factory=set)
    colors_out: Set[int] = field(default_factory=set)
    color_count_in: Dict[int, int] = field(default_factory=dict)
    color_count_out: Dict[int, int] = field(default_factory=dict)

    # 颜色映射候选：仅在“一对一且计数匹配可判定”时填充
    proposed_color_map: Dict[int, int] = field(default_factory=dict)

    # 网格尺寸 (width, height)
    dims_in: Tuple[int, int] = (0, 0)
    dims_out: Tuple[int, int] = (0, 0)

    # 几何变换候选：D4（二面体群 8 种对称：R0/R90/R180/R270/F0/F90/F180/F270）与平移 (tx, ty)
    d4: Optional[str] = None
    translation: Optional[Tuple[int, int]] = None



@dataclass
class ClassifyBuckets:
    # colors_in: Set[int] = field(default_factory=set)
    # colors_out: Set[int] = field(default_factory=set)
    colors_in: Set[int] = field(default_factory=set)
    colors_out: Set[int] = field(default_factory=set)

    # --- 任务级（跨 pairs）交集（None 表示尚未初始化；首个 pair 时用它初始化）---
    colors_in_intersection: Optional[Set[int]] = None
    colors_out_intersection: Optional[Set[int]] = None

    # --- 任务级颜色总计数（像素级计数汇总）---
    color_count_in_total: Counter = field(default_factory=Counter)
    color_count_out_total: Counter = field(default_factory=Counter)

    # --- 任务级尺寸集合（有的任务不同 pair 尺寸可能不同）---
    dims_in_set: Set[Tuple[int, int]] = field(default_factory=set)
    dims_out_set: Set[Tuple[int, int]] = field(default_factory=set)

    # --- 任务级候选：颜色映射 / 对称与平移聚合 ---
    color_maps: List[Dict[int, int]] = field(default_factory=list)
    d4_set: Set[Tuple[str, int, int]] = field(default_factory=set)     # (d4_name, tx, ty)
    translation_set: Set[Tuple[int, int]] = field(default_factory=set)

    # --- 跨 pair 一致映射（对所有 pair 都成立的子映射；可能是部分映射）---
    consistent_color_map: Optional[Dict[int, int]] = None

    # --- 保留 pair 级摘要，便于后续 ILP 输出或调试 ---
    per_pair: List[PairSummary] = field(default_factory=list)

    holes_in: Set[int] = field(default_factory=set)
    holes_out: Set[int] = field(default_factory=set)
    sizes_in: Set[Tuple[int,int]] = field(default_factory=set)
    sizes_out: Set[Tuple[int,int]] = field(default_factory=set)
    canon_clusters_in: Dict[frozenset,int] = field(default_factory=dict)   # 规范形 -> 对象总数
    canon_clusters_out: Dict[frozenset,int] = field(default_factory=dict)
    # 结构存在性（严格布尔）
    has_lines: bool = False
    has_boxes: bool = False
    line_ids_per_pair: Dict[str, List[int]] = field(default_factory=dict)
    box_ids_per_pair: Dict[str, List[int]]  = field(default_factory=dict)
    # 全局规则候选（严格）
    global_translate: Optional[Tuple[int,int]] = None
    global_d4_translate: Optional[Tuple[str,int,int]] = None
    # 改色候选（交集 / 并集）
    recolor_intersection: Dict[int,int] = field(default_factory=dict)
    recolor_union: Dict[int, Set[int]] = field(default_factory=dict)






# class EnhancedPatternMetaAnalyzerExact(Analyzer):
#     """严格布尔匹配版：按 AttrIndex/RelIndex 做精确模式分析"""



#     # ====== 实现细节 ======

#     def _classify(self, train_pairs):
#         """利用 AttrIndex 做分桶：by_color/by_holes/by_size/by_canon_sig；辅助决定开启哪些动作族"""
#         # 这里不做任何模糊，只统计布尔结构：是否存在大规模 is_line/is_box；颜色桶数；D4 轨道数等
#         ...


class EnhancedPatternMetaAnalyzerExact:
    """严格布尔匹配：分类→提取全局/条件规则→仿真验证→产出规则"""

#     def analyze(self, train_pairs: List[PairState], base_feats) -> AnalyzerResult:
#         # 1) 分类 + 信息抽取（借助 AttrIndex/RelIndex）
#         #    此步主要用于快速分桶与“应考虑哪些动作族”的启发（如线/框）
#         buckets = self._classify(train_pairs)

#         # 2) 全局规则尝试（translate / D4+translate / recolor）
#         global_rules = self._extract_global_rules(train_pairs)

#         # 3) 条件规则（Sel 子集）：线段/矩形/按颜色或洞数分组
#         conditional_rules = self._extract_conditional_rules(train_pairs, buckets, global_rules)

#         # 4) 整合 + 去冗 + 宏化
#         rules = self._integrate(global_rules, conditional_rules)

#         # 5) （可选）在训练对上执行确认必须“像素全等”
#         assert self._verify_on_train(train_pairs, rules), "rules not exact on training pairs"

#         # 6) 转为 PredSpec/常量域/依赖
#         preds, constants, deps, hints = self._emit_open_set(rules)
#         return AnalyzerResult(preds=preds, ruledeps=deps, constants=constants,
#                               hints=hints, notes=f"rules={rules}")

    def analyze(self, train_pairs: List[PairState], base_feats=None) -> AnalysisResult:
        buckets = self._classify(train_pairs)

        rules: List[RuleHypothesis] = []

        # 1) 全局 D4+平移 或 纯平移
        if buckets.global_d4_translate is not None:
            tag, dx, dy = buckets.global_d4_translate
            rules.append(RuleHypothesis("d4_translate", {"d4":tag, "dx":dx, "dy":dy}, order=5))
        elif buckets.global_translate is not None:
            dx, dy = buckets.global_translate
            rules.append(RuleHypothesis("translate", {"dx":dx, "dy":dy}, order=5))

        # 2) 全局改色（若有单值一致映射）
        if buckets.recolor_intersection:
            rules.append(RuleHypothesis("recolor", {"cmap": dict(buckets.recolor_intersection)}, order=10))

        # TODO：3) 条件规则（线段延迟 / 矩形缩进 / 子集平移等）
        # 若 buckets.has_lines： 可调用 solve_delay_line_params 严格反推 (dir,k)
        # 若 buckets.has_boxes： 可调用 solve_shrink_rect_k 严格反推 k

        # 仿真执行验证（严格像素全等）
        predictions: Dict[str,np.ndarray] = {}
        for p in train_pairs:
            grid = p.in_scene.grid.copy()
            objs = p.in_scene.objs
            # 顺序执行
            for r in sorted(rules, key=lambda x: x.order):
                if r.name == "d4_translate":
                    objs = _apply_d4(objs, r.params["d4"])
                    objs = _apply_translate(objs, r.params["dx"], r.params["dy"])
                elif r.name == "translate":
                    objs = _apply_translate(objs, r.params["dx"], r.params["dy"])
                elif r.name == "recolor":
                    objs = _apply_recolor(objs, r.params["cmap"])
                # 其它规则留作后续扩展
            pred = _render_objects(objs, p.in_scene.H, p.in_scene.W,
                                   p.in_scene.bg_color if p.in_scene.bg_color is not None else 0)
            predictions[p.pair_id] = pred
            # 严格校验 —— 不通过也先返回（方便你看 openset），后续可作为 early stop
            # assert np.array_equal(pred, p.out_scene.grid), f"{p.pair_id} mismatch"

        notes = f"rules={rules}"
        return AnalysisResult(rules=rules, predictions=predictions, notes=notes)

    def _safe_color_counts(by_color: Dict[int, object]) -> Dict[int, int]:
        """兼容 value 既可能是像素坐标列表，也可能已经是计数的场景。"""
        counts = {}
        for c, v in by_color.items():
            if isinstance(v, int):
                counts[c] = v
            elif hasattr(v, '__len__'):
                counts[c] = len(v)
            else:
                # 回退：无法判断时按 1 计（避免崩），也可改为 0 或抛错，视你工程需要
                counts[c] = 1
        return counts


    def _infer_color_map_from_counts(cin: Dict[int, int], cout: Dict[int, int]) -> Dict[int, int]:
        """
        基于像素计数的最简单一对一映射推断：
        - 若输入颜色 c 的像素数在输出里存在唯一匹配的颜色，并且不与其他输入产生冲突，则建立 c->c'。
        - 否则保持为空（表示无法仅凭计数确定）。
        """
        out_by_count: Dict[int, List[int]] = {}
        for c, n in cout.items():
            out_by_count.setdefault(n, []).append(c)

        mapping: Dict[int, int] = {}
        used_out: Set[int] = set()
        for c_in, n in cin.items():
            candidates = out_by_count.get(n, [])
            # 只有在唯一候选且未被其它输入占用时，才建立映射
            if len(candidates) == 1 and candidates[0] not in used_out:
                mapping[c_in] = candidates[0]
                used_out.add(candidates[0])

        # 要求是单射（不强制满射），否则返回空表示不确定
        if len(mapping) == 0:
            return {}
        return mapping


    def _merge_consistent_maps(maps: List[Dict[int, int]]) -> Optional[Dict[int, int]]:
        """
        取所有 pair 映射的“点对一致交集”（只保留在每个非空映射中都一致的键值对）。
        若所有映射都为空或没有共同部分，则返回 {}；如果列表本身为空，返回 None。
        """
        if not maps:
            return None
        # 只统计非空映射
        non_empty = [m for m in maps if m]
        if not non_empty:
            return {}
        # 从第一个非空开始做“相等交集”
        base = dict(non_empty[0])
        for m in non_empty[1:]:
            for k in list(base.keys()):
                if k not in m or m[k] != base[k]:
                    base.pop(k)
            if not base:
                break
        return base


    def _get_dims(scene) -> Tuple[int, int]:
        """
        兼容不同 Scene 表示，尽量取 (width, height)。
        - 优先：scene.w, scene.h
        - 次之：scene.grid.shape -> (h, w) 需要翻转
        - 否则：(0,0)
        """
        w = getattr(scene, 'w', None)
        h = getattr(scene, 'h', None)
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            return (w, h)
        grid = getattr(scene, 'grid', None)
        if grid is not None and hasattr(grid, 'shape') and len(grid.shape) == 2:
            hh, ww = grid.shape
            return (int(ww), int(hh))
        return (0, 0)


    # --------- 关键：跨 train 的严格布尔聚合 ----------
    def _classify(self, train_pairs: List[PairState]) -> ClassifyBuckets:
        B = ClassifyBuckets()
        # 汇总容器
        rec_intersection_ready = False
        d4_set: Set[Tuple[str,int,int]] = set()
        tr_set: Set[Tuple[int,int]] = set()
        cmap_list: List[Dict[int,int]] = []

        for p in train_pairs:
            # 1) 属性集合（严格离散）
            inA, outA = p.in_scene.attr, p.out_scene.attr

            B.colors_in  |= set(inA.by_color.keys())
            B.colors_out |= set(outA.by_color.keys())
            B.holes_in   |= set(inA.by_holes.keys())
            B.holes_out  |= set(outA.by_holes.keys())
            B.sizes_in   |= set(inA.by_size.keys())
            B.sizes_out  |= set(outA.by_size.keys())

            for sig, ids in p.in_scene.attr.by_canon_sig.items():
                B.canon_clusters_in[sig] = B.canon_clusters_in.get(sig, 0) + len(ids)
            for sig, ids in p.out_scene.attr.by_canon_sig.items():
                B.canon_clusters_out[sig] = B.canon_clusters_out.get(sig, 0) + len(ids)

            # 2) 结构存在性（严格）
            line_ids = [o.id for o in p.in_scene.objs if is_line_strict(o)]
            box_ids  = [o.id for o in p.in_scene.objs if is_box_strict(o)]
            B.line_ids_per_pair[p.pair_id] = line_ids
            B.box_ids_per_pair[p.pair_id]  = box_ids

            # 3) 全局平移 / D4+平移（严格）
            in_ol  = _to_objlite_list(p.in_scene.objs)
            out_ol = _to_objlite_list(p.out_scene.objs)

            d4 = exact_global_d4_translate(in_ol, out_ol)
            if d4 is not None:
                d4_set.add(d4)
            else:
                tr = exact_global_translate(in_ol, out_ol)
                if tr is not None:
                    tr_set.add(tr)
                else:
                    # 本 pair 不满足任何全局对齐；先记录空，后续可能走条件规则
                    pass

            # 4) 在“单一对齐”下的严格改色
            if d4 is not None:
                tag, dx, dy = d4
                cmap = exact_color_map(in_ol, out_ol, dx=dx, dy=dy, d4=tag)
                cmap_list.append(cmap)
            elif len(tr_set) == 1:
                dx, dy = next(iter(tr_set))
                cmap = exact_color_map(in_ol, out_ol, dx=dx, dy=dy, d4="id")
                cmap_list.append(cmap)
            # 若尚未确定唯一对齐，不计算 cmap

        # 5) 聚合布尔与唯一性
        # （a）是否“每个 pair 都存在至少一个线/框对象”
        B.has_lines = all(len(B.line_ids_per_pair.get(p.pair_id, [])) > 0 for p in train_pairs)
        B.has_boxes = all(len(B.box_ids_per_pair.get(p.pair_id, []))  > 0 for p in train_pairs)

        # （b）全局对齐唯一性
        if len(d4_set) == 1:
            B.global_d4_translate = next(iter(d4_set))
        elif len(tr_set) == 1 and len(d4_set) == 0:
            B.global_translate = next(iter(tr_set))
        # 否则：不认定全局规则，后续可走条件子集

        # （c）严格改色：多对 pair 的 Cin->Cout 交集（仅保留单值一致）
        if cmap_list:
            keys = set.intersection(*(set(m.keys()) for m in cmap_list))
            rec_intersection: Dict[int,int] = {}
            rec_union: Dict[int, Set[int]] = {}
            for k in keys:
                vals = {m[k] for m in cmap_list}
                if len(vals) == 1:
                    rec_intersection[k] = vals.pop()
            # 并集（便于你观察冲突）
            for m in cmap_list:
                for k,v in m.items():
                    rec_union.setdefault(k,set()).add(v)

            B.recolor_intersection = rec_intersection
            B.recolor_union = rec_union

        return B



    def _extract_global_rules(self, train_pairs) -> List[RuleHypothesis]:
        # 尝试顺序：translate → D4+translate → recolor（基于已对齐的位置）
        r: List[RuleHypothesis] = []
        ok_t = self._try_global_translate(train_pairs)
        if ok_t: r.append(ok_t)
        ok_d4t = self._try_global_d4_translate(train_pairs)
        if ok_d4t: r.append(ok_d4t)
        ok_rc = self._try_global_recolor(train_pairs, r)  # 以 translate 对齐后生成 cmap
        if ok_rc: r.append(ok_rc)
        return r

    def _try_global_translate(self, train_pairs)->Optional[RuleHypothesis]:
        deltas = set()
        for p in train_pairs:
            res = exact_global_translate(p.in_scene.objs, p.out_scene.objs)
            if res is None: return None
            deltas.add(res)
        if len(deltas)==1:
            dx,dy = next(iter(deltas))
            return RuleHypothesis("translate", {"dx":dx,"dy":dy}, None, order=10)
        return None

    def _try_global_d4_translate(self, train_pairs)->Optional[RuleHypothesis]:
        opts=set()
        for p in train_pairs:
            res = exact_global_d4_translate(p.in_scene.objs, p.out_scene.objs)
            if res is None: return None
            opts.add(res)  # (d4, dx, dy)
        if len(opts)==1:
            d4,dx,dy = next(iter(opts))
            return RuleHypothesis("d4_translate", {"d4":d4,"dx":dx,"dy":dy}, None, order=9)
        return None

    def _try_global_recolor(self, train_pairs, prior_rules)->Optional[RuleHypothesis]:
        # 若已经有全局位移/对齐，基于对齐后的严格同位匹配构建单值 cmap
        has_align = any(r.name in ("translate","d4_translate") for r in prior_rules)
        if not has_align: return None
        # 取唯一的 (d4,dx,dy)
        d4,dx,dy = ("id",0,0)
        for r in prior_rules:
            if r.name=="translate": dx,dy=r.params["dx"],r.params["dy"]
            if r.name=="d4_translate": d4,dx,dy=r.params["d4"],r.params["dx"],r.params["dy"]
        cmap_all=[]
        for p in train_pairs:
            cmap = exact_color_map(p.in_scene.objs, p.out_scene.objs, dx, dy, d4)
            cmap_all.append(cmap)
        # 交集（必须一致）：
        keys = set.intersection(*(set(m.keys()) for m in cmap_all)) if cmap_all else set()
        final = {}
        for k in keys:
            vals = {m[k] for m in cmap_all}
            if len(vals)==1: final[k] = vals.pop()
        if final:
            return RuleHypothesis("recolor", {"cmap":final}, None, order=11)
        return None

    def _extract_conditional_rules(self, train_pairs, buckets, global_rules)->List[RuleHypothesis]:
        rules: List[RuleHypothesis] = []
        # 示例1：线类严格延迟
        if buckets.get("has_lines", False):
            dir_k_set=set()
            for p in train_pairs:
                # 只对 in_scene 中严格 is_line 的对象求 (dir,k)，并验证 out 上存在像素等式
                res = solve_delay_line_params(p.in_scene, p.out_scene)  # 返回 (dir,k) 或 None
                if res is None: dir_k_set.add(("fail",None)); break
                dir_k_set.add(res)
            if len(dir_k_set)==1 and ("fail",None) not in dir_k_set:
                dir,k = next(iter(dir_k_set))
                rules.append(RuleHypothesis("delay_line", {"dir":dir,"k":k},
                                            Selector.is_line(), order=5))
        # 示例2：矩形缩进（k 一致）
        if buckets.get("has_boxes", False):
            kset=set()
            for p in train_pairs:
                k = solve_shrink_rect_k(p.in_scene, p.out_scene)
                if k is None: kset.add(("fail",None)); break
                kset.add(k)
            if len(kset)==1 and ("fail",None) not in kset:
                k = next(iter(kset))
                rules.append(RuleHypothesis("shrink_rect", {"k":k}, Selector.is_box(), order=6))

        # 示例3：按颜色子集平移（如果全局平移失败）
        # 例如：只移动 color==k 的对象，(dx,dy) 唯一且一致
        # 同理可按 holes/size/shape_sig 做子集规则
        ...
        return rules

    def _integrate(self, global_rules, conditional_rules) -> List[RuleHypothesis]:
        # 1) 先执行对齐类（d4_translate/translate），再执行子集动作（delay/shrink），最后 recolor
        rules = sorted(global_rules + conditional_rules, key=lambda r: r.order)
        # 2) 去冗：若某条件规则覆盖对象集为空/被更强规则完全覆盖，移除
        # 3) 宏化（可选）：若某两个相邻规则在所有覆盖对象上强共现，替换为宏
        return rules

    def _verify_on_train(self, train_pairs, rules)->bool:
        # 严格执行、像素全等
        for p in train_pairs:
            pred = exec_rules_on_scene(p.in_scene, rules)  # Python 仿真执行
            if not np.array_equal(pred, p.out_scene.grid):
                return False
        return True

    def _emit_open_set(self, rules):
        # 把规则序列映射为 PredSpec/常量域/依赖：
        preds, consts, deps = [], {}, set()
        for r in rules:
            if r.name=="translate":
                preds += [PredSpec(name="translate", arity=4, vars=[
                        VarSpec("obj_in","object","in"),
                        VarSpec("dx","int","in"), VarSpec("dy","int","in"),
                        VarSpec("obj_out","object","out"),
                    ])]
                consts.setdefault("dx", []).append(r.params["dx"])
                consts.setdefault("dy", []).append(r.params["dy"])
                deps.update(["geom.pl","objrel.pl"])
            elif r.name=="d4_translate":
                preds += [PredSpec(name="rotate90k", arity=3, vars=[...]),
                          PredSpec(name="mirror", arity=3, vars=[...]),
                          PredSpec(name="translate", arity=4, vars=[...])]
                # 显式常量：rot/mirror 枚举值、dx/dy
                ...
            elif r.name=="recolor":
                preds += [PredSpec(name="recolor", arity=3, vars=[...])]
                for cin, cout in r.params["cmap"].items():
                    consts.setdefault("cin", []).append(cin)
                    consts.setdefault("cout", []).append(cout)
                deps.update(["color.pl"])
            elif r.name=="delay_line":
                preds += [PredSpec(name="delay_line", arity=4, vars=[...])]
                consts.setdefault("dir", []).append(r.params["dir"])
                consts.setdefault("k", []).append(r.params["k"])
                deps.update(["geom.pl"])
            elif r.name=="shrink_rect":
                preds += [PredSpec(name="shrink_rect", arity=3, vars=[...])]
                consts.setdefault("k", []).append(r.params["k"])
                deps.update(["geom.pl"])
            # 其它规则同理……
        # 去重常量、形成 AnalyzerHints
        for k in list(consts.keys()):
            consts[k] = sorted(set(consts[k]))
        hints = AnalyzerHints(max_vars=6, max_body=min(len(rules),3), max_clauses=2)
        return preds, consts, RuleDependency(files=list(deps)), hints
