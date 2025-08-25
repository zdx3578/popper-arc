# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
# from core.analyzer_iface import Analyzer, AnalyzerResult, PredSpec, VarSpec, RuleDependency, AnalyzerHints
from AttrActionIndex import PairState, Scene, Obj
from core.exact_ops import (exact_global_translate, exact_global_d4_translate,
                            exact_color_map, is_line_strict, is_box_strict,
                            solve_delay_line_params, solve_shrink_rect_k)
from core.selectors import Selector  # Filter(color==k)/Filter(is_line)/...

@dataclass
class RuleHypothesis:
    name: str                    # "translate", "recolor", "delay_line", "shrink_rect", "rotate90+translate", ...
    params: Dict[str, int|str|Tuple]  # {"dx":1,"dy":-1} / {"dir":"ne","k":1} / {"cmap":{2:7,...}}
    selector: Optional[Selector] = None   # None 表示全局；否则仅对满足 Sel 的对象生效
    order: int = 0               # 执行顺序，越小越先执行

class EnhancedPatternMetaAnalyzerExact(Analyzer):
    """严格布尔匹配版：按 AttrIndex/RelIndex 做精确模式分析"""

    def analyze(self, train_pairs: List[PairState], base_feats) -> AnalyzerResult:
        # 1) 分类 + 信息抽取（借助 AttrIndex/RelIndex）
        #    此步主要用于快速分桶与“应考虑哪些动作族”的启发（如线/框）
        buckets = self._classify(train_pairs)

        # 2) 全局规则尝试（translate / D4+translate / recolor）
        global_rules = self._extract_global_rules(train_pairs)

        # 3) 条件规则（Sel 子集）：线段/矩形/按颜色或洞数分组
        conditional_rules = self._extract_conditional_rules(train_pairs, buckets, global_rules)

        # 4) 整合 + 去冗 + 宏化
        rules = self._integrate(global_rules, conditional_rules)

        # 5) （可选）在训练对上执行确认必须“像素全等”
        assert self._verify_on_train(train_pairs, rules), "rules not exact on training pairs"

        # 6) 转为 PredSpec/常量域/依赖
        preds, constants, deps, hints = self._emit_open_set(rules)
        return AnalyzerResult(preds=preds, ruledeps=deps, constants=constants,
                              hints=hints, notes=f"rules={rules}")

    # ====== 实现细节 ======

    def _classify(self, train_pairs):
        """利用 AttrIndex 做分桶：by_color/by_holes/by_size/by_canon_sig；辅助决定开启哪些动作族"""
        # 这里不做任何模糊，只统计布尔结构：是否存在大规模 is_line/is_box；颜色桶数；D4 轨道数等
        ...

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
