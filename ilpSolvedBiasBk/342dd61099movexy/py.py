#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Popper files (bias.pl, bk.pl, exs.pl) from ARC task JSON.
Now with 12‑class negative generation (neg_sampler.py).
Usage:
    python gen_arc_popper.py <task.json|dir> <out_root>
        [--invent]      # allow predicate invention
        [--hard-loop]   # CEGIS-style hard negative augmentation
        [--neg-per N]   # max negatives per class (default 30)
"""

import json, sys, pathlib, argparse, re, importlib
from copy import deepcopy
from neg_sampler import gen_negatives, MAX_NEG_PER_CLASS as DEF_MAX_NEG

PAIR_ID_RE = re.compile(r"([a-f0-9]{8})\.json$", re.I)

# ----- task utils -------------------------------------------------- #
def load_tasks(src: pathlib.Path):
    if src.is_file() and src.suffix == ".json":
        return [src]
    return sorted(p for p in src.rglob("*.json"))

def grids_in_task(task):
    for pair in task["train"]:
        yield pair["input"]; yield pair["output"]
    for pair in task["test"]:
        yield pair["input"]

def maxdim_task(task):
    return max(max(len(g), len(g[0])) for g in grids_in_task(task))

# ----- bias writer ------------------------------------------------- #
def write_bias(path: pathlib.Path, maxdim: int, invent: bool):
    parts = [
        "% -- generated bias.pl --\n",
        "#const use_neg_cache=1.\n",
        "max_clauses(4).\nmax_body(3).\nmax_vars(16).\nallow_singletons.\n",
        "head_pred(outpix,4).\n",
        "body_pred(inpix,4).\n",
        "type(outpix,(pair,coord,coord,color,)).\n",
        "type(inpix,(pair,coord,coord,color,)).\n",
        "direction(outpix,(in,in,out,out,)).\n",
        "direction(inpix,(in,out,out,out,)).\n",
        "allow_singletons.\n",
    ]
    if invent:
        parts += [
            "enable_pi.\nenable_recursion.\ninvented_pred(move,4).\n",
            "type(move,(coord,coord,coord,coord,)).\n",
            "direction(move,(in,out,out,out,)).\n",
        ]
    # row/col constants 0..maxdim (+1 for OOD)
    for i in range(maxdim + 1):
        parts += [
            f"body_pred(row{i},1).\n", f"type(row{i},(coord,)).\n",
            f"direction(row{i},(out,)).\n",
            # f"body_pred(col{i},1).\n", f"type(col{i},(coord,)).\n",
            # f"direction(col{i},(out,)).\n",
        ]
    # color constants 0..10
    for c in range(11):
        parts += [
            f"body_pred(colVal{c},1).\n", f"type(colVal{c},(color,)).\n",
            f"direction(colVal{c},(out,)).\n",
        ]
    path.write_text("".join(parts))

def emit_const_facts(maxdim: int):
    for i in range(maxdim + 1):
        yield f"row{i}({i}).\n"; #yield f"col{i}({i}).\n"
    for c in range(11):
        yield f"colVal{c}({c}).\n"

# ---------- pair helper class ---------- #
class PairObj:
    __slots__ = ("pid","in_grid","w","h","pos")
    def __init__(self, pid, in_grid, out_grid):
        self.pid = pid
        self.in_grid = in_grid
        self.h = len(in_grid)
        self.w = len(in_grid[0])
        # 正例集合 (过滤颜色 8)
        self.pos = {(x,y,c)
                    for y,row in enumerate(out_grid)
                    for x,c in enumerate(row) if c!=8}

# ---------- main task writer ---------- #
def write_task_files(json_path: pathlib.Path, out_root: pathlib.Path,
                     invent: bool, max_neg_cls: int, cegis: bool):
    task = json.loads(json_path.read_text())
    task_id = json_path.stem
    task_dir = out_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    maxdim = maxdim_task(task)
    write_bias(task_dir / "bias.pl", maxdim, invent)

    bk_lines, exs_lines = [], []
    pair_objs = []

    # helper: deterministic pid per pair
    def pid(prefix, idx): return f"T{task_id}_{prefix}{idx}"
    # def pid(prefix, idx): return f"T{idx}"

    # ---- read train pairs ----
    for idx, tr in enumerate(task["train"]):
        pid_tr = pid("tr", idx)
        in_g, out_g = tr["input"], tr["output"]
        # BK: inpix
        for y,row in enumerate(in_g):
            for x,val in enumerate(row):
                if val != 8:
                    bk_lines.append(f"inpix({pid_tr},{x},{y},{val}).\n")
        # Positives
        p = PairObj(pid_tr, in_g, out_g)
        pair_objs.append(p)
        for (x,y,c) in p.pos:
            exs_lines.append(f"pos(outpix({pid_tr},{x},{y},{c})).\n")

    # # ---- TEST inputs: 仅作 BK，可选 ----
    # for idx, te in enumerate(task["test"]):
    #     pid_te = pid("te", idx)
    #     for y,row in enumerate(te["input"]):
    #         for x,val in enumerate(row):
    #             if val != 8:
    #                 bk_lines.append(f"inpix({pid_te},{x},{y},{val}).\n")

    # ---- 常量 facts ----
    bk_lines.extend(emit_const_facts(maxdim))

    # ---- 负例采样 ----
    # 基本负例
    for p in pair_objs:
        negs = gen_negatives(p)
        for (x,y,c) in negs:
            exs_lines.append(f"neg(outpix({p.pid},{x},{y},{c})).\n")

    # 跨 pair 投射 (NEG_CROSS_PAIR)
    for i,p_a in enumerate(pair_objs):
        for j,p_b in enumerate(pair_objs):
            if i==j: continue
            for (x,y,c) in p_a.pos:
                # 只要坐标合法
                if 0<=x<p_b.w and 0<=y<p_b.h and (x,y,c) not in p_b.pos:
                    exs_lines.append(f"neg(outpix({p_b.pid},{x},{y},{c})).\n")

    # ---- 写文件 ----
    (task_dir / "bk.pl").write_text("".join(bk_lines))
    (task_dir / "exs.pl").write_text("".join(exs_lines))

    print(f"[{task_id}]  positives={sum(len(p.pos) for p in pair_objs):4d} "
          f"negatives={len(exs_lines) - sum(len(p.pos) for p in pair_objs):4d}")

# ---------- CLI ---------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="ARC task .json or directory")
    ap.add_argument("out_root", help="output directory")
    ap.add_argument("--invent", action="store_true",
                    help="enable predicate invention for move/4")
    ap.add_argument("--hard-loop", action="store_true",
                    help="enable CEGIS hard‑negative loop (not implemented in demo)")
    ap.add_argument("--neg-per", type=int, default=DEF_MAX_NEG,
                    help=f"max negatives per class (default {DEF_MAX_NEG})")
    args = ap.parse_args()

    # 动态调整 neg_sampler 超参
    import neg_sampler
    neg_sampler.MAX_NEG_PER_CLASS = args.neg_per

    src = pathlib.Path(args.src)
    out_root = pathlib.Path(args.out_root)
    tasks = load_tasks(src)
    for jp in tasks:
        write_task_files(jp, out_root, args.invent,
                         args.neg_per, args.hard_loop)
    print(f"Generated Popper files for {len(tasks)} task(s) under {args.out_root}")

if __name__ == "__main__":
    main()
