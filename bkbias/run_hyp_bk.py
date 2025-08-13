#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone tester: run hyp.pl + testbk.pl + meta.json and (optionally) expected.json.
- 隔离进程，避免 SWI 引擎重定义残留
- 集合语义渲染 + 冲突检测
- 五个探针便于定位问题
"""
import argparse, json, sys, multiprocessing as mp
from pathlib import Path

def _run_once(hyp_path, bk_path, meta_path, pair_id, expected_path=None, prefill_bg=None, limit_probe=10):
    from pyswip import Prolog
    import numpy as np

    def q(pl, s): return list(pl.query(s))

    meta = json.loads(Path(meta_path).read_text())
    rows, cols = meta.get("output_size", meta.get("size", [0, 0]))
    grid = np.zeros((rows, cols), dtype=int)
    if prefill_bg is not None:
        grid[:, :] = int(prefill_bg)

    pl = Prolog()
    # 载入顺序：BK -> HYP
    pl.consult(str(bk_path))
    pl.consult(str(hyp_path))

    # 探针
    pin = q(pl, f"aggregate_all(count, inpix({pair_id},_,_,_),N)")
    n_in = pin[0]["N"] if pin else 0
    has_i0 = bool(q(pl, "int_0(Z)"))
    has_i1 = bool(q(pl, "int_1(S)"))
    pairs = q(pl, f"diag_pair_far({pair_id},C,X1,Y1,X2,Y2)")
    sample_pairs = pairs[:limit_probe]
    sample_pts = []
    if sample_pairs and has_i1:
        p0 = sample_pairs[0]
        X1,Y1,X2,Y2 = p0["X1"],p0["Y1"],p0["X2"],p0["Y2"]
        sample_pts = q(pl, f"int_1(S), on_diag_between_k(X,Y,{X1},{Y1},{X2},{Y2},S)")
        sample_pts = sample_pts[:limit_probe]

    # outpix 集合语义 + 冲突检测
    sols = q(pl, f"outpix({pair_id},X,Y,C)")
    painted, conflicts = {}, []
    for s in sols:
        x,y,c = int(s["X"]), int(s["Y"]), int(s["C"])
        if 0 <= x < rows and 0 <= y < cols:
            key = (x,y)
            if key in painted and painted[key] != c:
                conflicts.append((x,y,painted[key],c))
            painted[key] = c
    for (x,y), c in painted.items():
        grid[x,y] = c

    diff = {"total_mismatch": 0, "cells": []}
    if expected_path:
        exp = np.array(json.loads(Path(expected_path).read_text()))
        if exp.shape != grid.shape:
            print(f"[ERROR] expected shape {exp.shape} != predicted {grid.shape}", file=sys.stderr)
        else:
            mism = (exp != grid)
            nmis = int(mism.sum())
            diff["total_mismatch"] = nmis
            if nmis > 0:
                xs, ys = np.where(mism)
                for i in range(min(200, nmis)):
                    x, y = int(xs[i]), int(ys[i])
                    diff["cells"].append({"x":x,"y":y,"exp":int(exp[x,y]),"got":int(grid[x,y])})

    diag = {
        "inpix_count": n_in,
        "has_int_0": has_i0,
        "has_int_1": has_i1,
        "diag_pair_far_count": len(pairs),
        "diag_pair_far_samples": sample_pairs,
        "on_diag_between_k_samples": sample_pts,
        "outpix_count": len(sols),
        "conflicts": conflicts[:50],
    }
    return {"grid": grid.tolist(), "diagnostics": diag, "diff": diff}

def _worker(q, *args, **kwargs):
    try:
        q.put(_run_once(*args, **kwargs))
    except Exception as e:
        q.put({"error": repr(e)})

def run_isolated(*args, **kwargs):
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(q,)+args, kwargs=kwargs, daemon=True)
    p.start(); p.join()
    return q.get() if not q.empty() else {"error":"subprocess failed"}

def main():
    ap = argparse.ArgumentParser("Run hyp.pl + testbk.pl quickly")
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--bk", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pair", default="p0")
    ap.add_argument("--expected")
    ap.add_argument("--save")
    ap.add_argument("--prefill-bg", type=int, help="若不透传输入，可用背景色预填网格")
    args = ap.parse_args()

    res = run_isolated(args.hyp, args.bk, args.meta, args.pair, args.expected, args.prefill_bg)
    if "error" in res:
        print("[FATAL]", res["error"], file=sys.stderr); sys.exit(2)

    d = res["diagnostics"]
    print("== PROBES ==")
    print(f"inpix_count={d['inpix_count']}, has_int_0={d['has_int_0']}, has_int_1={d['has_int_1']}")
    print(f"diag_pair_far_count={d['diag_pair_far_count']}, outpix_count={d['outpix_count']}")
    if d["conflicts"]:
        print(f"[WARN] conflicts: {d['conflicts']}")
    if not d["has_int_0"] or not d["has_int_1"]:
        print("[HINT] 缺少 int_0/1 会导致 on_diag_between_k 或 S=1 失败")

    if d["diag_pair_far_samples"]:
        s0 = d["diag_pair_far_samples"][0]
        print(f"sample diag_pair_far: C={s0['C']} ({s0['X1']},{s0['Y1']}) -> ({s0['X2']},{s0['Y2']})")
    if d["on_diag_between_k_samples"]:
        print(f"sample on_diag_between_k points: {d['on_diag_between_k_samples']}")

    if args.expected:
        print("== DIFF ==")
        print(f"total_mismatch={res['diff']['total_mismatch']}")
        for cell in res["diff"]["cells"]:
            print(cell)

    if args.save:
        # Path(args.save).write_text(json.dumps(res["grid"]))
        with open(args.save, "w") as f:
            f.write("# Human-readable grid\n")
            for row in res["grid"]:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("\n# JSON format\n")
            f.write(json.dumps(res["grid"], indent=2))
        print(f"[OK] saved to {args.save}")

if __name__ == "__main__":
    main()

