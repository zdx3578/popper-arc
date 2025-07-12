# -*- coding: utf-8 -*-
"""
12‑class negative‑example generator for ARC‑Popper pipeline.

依赖约定:
  pair.pid      : 唯一 ID (str)
  pair.pos      : set((x,y,c)) 正例 outpix 元组
  pair.in_grid  : 2‑D list  输入网格 (便于镜像/邻域)
  pair.w, pair.h: 网格宽高 (int)
"""

import random
from itertools import product

# 全局超参
MAX_NEG_PER_CLASS   = 30   # 每类最多采样
NEAR_MISS_DXY       = [(1,0),(-1,0),(0,1),(0,-1)]
COLORS              = list(range(11))  # 0‑10, 留出10作 OOD
SPARSE_NOISE_RATIO  = 0.01             # 1% 稀疏噪声

# ---------- 小工具 ---------- #
def in_bounds(x, y, w, h):
    return 0 <= x < w and 0 <= y < h

def mirror_xy(x, y, w, h):
    return w-1-x, y

def mirror_y(x, y, w, h):
    return x, h-1-y

def rotate_180(x, y, w, h):
    return w-1-x, h-1-y

# ---------- 主函数 ---------- #
def gen_negatives(pair, hypothesis=None):
    """返回 set((x,y,c))."""
    w, h = pair.w, pair.h
    pos  = pair.pos
    neg  = set()

    # ===== 2‑1 位置扰动 =====
    for (x, y, c) in pos:
        tries = random.sample(list(product(range(w), range(h))), k=min(8, w*h))
        for nx, ny in tries:
            if (nx, ny, c) not in pos:
                neg.add((nx, ny, c))

    # ===== 2‑2 颜色扰动 =====
    for (x, y, c) in pos:
        for c2 in COLORS:
            if c2 != c:
                neg.add((x, y, c2))

    # ===== 2‑3 行列互换 =====
    for (x, y, c) in pos:
        if in_bounds(y, x, w, h) and (y, x, c) not in pos:
            neg.add((y, x, c))

    # ===== 2‑4 镜像 / 旋转 =====
    for (x, y, c) in pos:
        for nx, ny in {mirror_xy(x,y,w,h), mirror_y(x,y,w,h), rotate_180(x,y,w,h)}:
            if (nx, ny, c) not in pos:
                neg.add((nx, ny, c))

    # ===== 2‑5 邻域漂移 =====
    for (x, y, c) in pos:
        for dx, dy in NEAR_MISS_DXY:
            nx, ny = x+dx, y+dy
            if in_bounds(nx, ny, w, h) and (nx, ny, c) not in pos:
                neg.add((nx, ny, c))

    # ===== 2‑6 计数冲突 (复制) =====
    sample_dup = random.sample(list(pos), min(len(pos), MAX_NEG_PER_CLASS))
    for (x, y, c) in sample_dup:
        nx, ny = (x+2) % w, (y+2) % h
        if (nx, ny, c) not in pos:
            neg.add((nx, ny, c))

    # ===== 2‑7 对象混叠 (交换颜色) =====
    if len(pos) >= 2:
        (x1,y1,c1), (x2,y2,c2) = random.sample(list(pos), 2)
        if c1 != c2:
            neg.update({(x1,y1,c2), (x2,y2,c1)})

    # ===== 2‑8 全局违例 (破坏对角) =====
    d = min(w,h)
    for i in range(d):
        if (i,i,0) not in pos:
            neg.add((i, i, 0))

    # ===== 2‑9 稀疏噪声 =====
    noise_ct = max(1, int(w*h*SPARSE_NOISE_RATIO))
    for _ in range(noise_ct):
        nx, ny = random.randrange(w), random.randrange(h)
        c  = random.choice(COLORS)
        if (nx, ny, c) not in pos:
            neg.add((nx, ny, c))

    # ===== 2‑10 CEGIS 硬负例 =====
    if hypothesis:
        miss_preds = hypothesis(pair) - pos
        neg.update(random.sample(list(miss_preds),
                                 min(len(miss_preds), MAX_NEG_PER_CLASS)))

    # ===== 2‑12 域外 (颜色10 / 边界) =====
    neg.add((w-1, h-1, 10))  # 颜色 10：ARC 域外
    # (行/列=w) 需在 bias 中声明 row{w} col{h} 以保证可接地
    neg.add((w,   0,   0))
    neg.add((0,   h,   0))

    # 限制总量
    if len(neg) > 12 * MAX_NEG_PER_CLASS:
        neg = set(random.sample(list(neg), 12 * MAX_NEG_PER_CLASS))
    return neg

