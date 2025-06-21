from arc_types import *
import numpy as np
from constants import *

from contextlib import contextmanager
import logging
import traceback









def identity(
    x: Any
) -> Any:
    """ identity function """
    return x


def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)


def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)


def multiply(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)


def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)


def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])


def even(
    n: Integer
) -> Boolean:
    """ evenness """
    return n % 2 == 0


def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)


def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)


def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b


def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b


def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container


def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))


def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b


def difference(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ set difference """
    return type(a)(e for e in a if e not in b)

from typing import Dict, Any, List, Tuple, Callable, Optional

def advanced_difference(
    a: FrozenSet,
    b: FrozenSet
) -> Dict[str, Any]:
    """高级集合差异分析，包含多层次比较"""
    # 基础差异分析
    common = a.intersection(b)
    a_unique = a - b
    b_unique = b - a

    # 排序差异
    sorted_a_uniq = sorted(a_unique, key=lambda x: (x[0], x[1]))
    sorted_b_uniq = sorted(b_unique, key=lambda x: (x[0], x[1]))

    # 二次差异比较
    diff_a_unique = sorted(set(sorted_a_uniq) - set(sorted_b_uniq))
    diff_b_unique = sorted(set(sorted_b_uniq) - set(sorted_a_uniq))

    return {
        "common": common,
        "first_level_diff": {
            "a_unique": a_unique,
            "b_unique": b_unique,
            "sorted_diffs": {
                "a": sorted_a_uniq,
                "b": sorted_b_uniq
            }
        },
        "second_level_diff": {
            "diff_a": diff_a_unique,
            "diff_b": diff_b_unique
        }
    }

def dedupe(
    tup: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(tup) if tup.index(e) == i)


def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))


def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))


def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b


def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)


def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)


def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)


def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)


def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))


def valmin(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ minimum by custom function """
    return compfunc(min(container, key=compfunc, default=0))


def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc)


def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc)


def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)


def leastcommon(
    container: Container
) -> Any:
    """ least common item """
    return min(set(container), key=container.count)


def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})


def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b


def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b


def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)


def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)


def crement(
    x: Numerical
) -> Numerical:
    """ incrementing positive and decrementing negative """
    if isinstance(x, int):
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)
    return (
        0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
        0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1)
    )


def sign(
    x: Numerical
) -> Numerical:
    """ sign """
    if isinstance(x, int):
        return 0 if x == 0 else (1 if x > 0 else -1)
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1)
    )


def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0


# def toivec(
#     i: Integer
# ) -> IntegerTuple:
#     """ vector pointing vertically """
#     return (i, 0)


# def tojvec(
#     j: Integer
# ) -> IntegerTuple:
#     """ vector pointing horizontally """
#     return (0, j)


def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))


def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))


def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))


def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)


def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))


def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]


def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))


def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)


def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))


def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))


def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)


def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)


def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))


def branch(
    condition: Boolean,
    a: Any,
    b: Any
) -> Any:
    """ if else branching """
    return a if condition else b


def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))


def chain(
    h: Callable,
    g: Callable,
    f: Callable,
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))


def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target


def rbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)


def lbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the leftmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)


def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))


def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))


def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)


def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)


def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))


def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))


def mpapply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors and merge """
    return merge(papply(function, a, b))


def prapply(
    function,
    a: Container,
    b: Container
) -> FrozenSet:
    """ apply function on cartesian product """
    return frozenset(function(i, j) for j in b for i in a)


def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    # if 0 in values:
    #     return 0
    # else:
    return max(set(values), key=values.count)

mostcolorcount = mostcolor

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)
leastcolorcount = leastcolor

def height(
    piece: Piece
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1



def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def hwratio(
    piece: Piece
) -> Integer:
    """ height to width ratio """
    return divide(height(piece), width(piece))

def hratio(
    piece: Piece, piece2: Piece
) -> Integer:
    """ height to width ratio """
    return divide(height(piece2), height(piece))

def wratio(
    piece: Piece, piece2: Piece
) -> Integer:
    """ height to width ratio """
    return divide(width(piece2), width(piece))

def hratioI(
    piece: Piece, piece2: Piece
) -> Integer:
    """ height to width ratio """
    return divide(height(piece), height(piece2))

def wratioI(
    piece: Piece, piece2: Piece
) -> Integer:
    """ height to width ratio """
    return divide(width(piece), width(piece2))

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))


def portrait(
    piece: Piece
) -> Boolean:
    """ whether height is greater than width """
    return height(piece) > width(piece)



def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)

samecolorElemCount = colorcount

from collections import Counter

#same as ?
def all_colorcount(
    element: Element
) -> dict:
    """计算所有颜色的出现次数"""
    # 提取所有颜色值
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    # 计算颜色出现次数
    return dict(Counter(values))

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)


def sizefilter(
    container: Container,
    n: Integer
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)

from typing import Set, Tuple, Union, FrozenSet
from arc_types import Patch, IntegerTuple, Indices

def asindices_patch(patch: Patch) -> Indices:
    """
    提取 Patch 中所有的坐标。

    Args:
        patch (Patch): 要提取坐标的 Patch。

    Returns:
        Indices: 提取出的坐标集合。
    """
    coords: Set[Tuple[int, int]] = set()
    for elem in patch:
        if isinstance(elem, tuple) and isinstance(elem[1], tuple):
            _, (i, j) = elem
            coords.add((i, j))
        else:
            i, j = elem
            coords.add((i, j))
    return frozenset(coords)

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    # print('asindices,asindices,asindices')
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))


def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

color_indices = ofcolor

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))


def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))


def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))


def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))


def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])


def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch


def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))


def shift(
    patch: Patch,
    directions: IntegerTuple
) -> Patch:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)


def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))


def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})


def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})


def neighbors(
    loc: IntegerTuple
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)


def objects(
    grid: Grid,
    univalued: Boolean,
    diagonal: Boolean,
    without_bg: Boolean
) -> Objects:
    """ Extract objects occurring on the grid """
    # 计算背景颜色
    # bg = mostcolor(grid) if without_bg else None
    # bg = 0 if without_bg else None
    element = grid
    # print(f"element: {element}, type: {type(element)}")
    # values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    # Corrected version
    if isinstance(element, np.ndarray):
        # 处理 numpy 数组 - 将其转换为嵌套列表格式
        values = element.flatten().tolist()
    elif isinstance(element, (tuple, list)) and len(element) > 0 and isinstance(element[0], (tuple, list, np.ndarray)):
        # Handle 2D grid case (nested tuples, lists, or numpy arrays)
        if isinstance(element, np.ndarray):
            values = element.flatten().tolist()
        else:
            values = [v for r in element for v in r]
    else:
        # Handle object case (list of (value, location) pairs)
        try:
            values = [v for v, _ in element]
        except ValueError:
            # 如果解包失败，尝试直接展平
            if hasattr(element, '__iter__'):
                values = list(element) if not isinstance(element, (str, bytes)) else []
            else:
                values = []

    if without_bg:
        if  0 in values:
            bg = 0
        else:
            bg = mostcolor(grid)
    else:
        bg=None

    objs = set()  # 存放所有对象
    occupied = set()  # 记录已经属于某个对象的单元格
    h, w = len(grid), len(grid[0])  # 网格的高度和宽度
    unvisited = asindices(grid)  # 获取所有单元格的坐标
    diagfun = neighbors if diagonal else dneighbors  # 确定邻居获取函数

    for loc in unvisited:
        if loc in occupied:
            continue  # 如果该单元格已经属于某个对象，则跳过
        val = grid[loc[0]][loc[1]]  # 获取当前单元格的值
        if val == bg:
            continue  # 如果该单元格是背景颜色，跳过
        obj = {(val, loc)}  # 创建一个新对象，包含当前单元格
        cands = {loc}  # 候选单元格集合，用于扩展对象
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]  # 获取候选单元格的值
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))  # 如果符合条件，将该单元格加入对象
                    occupied.add(cand)  # 标记该单元格为已占用
                    # 获取邻居单元格，并过滤掉越界的单元格
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied  # 更新候选单元格集合，去掉已占用的单元格
        objs.add(frozenset(obj))  # 将当前对象加入对象集合

    return frozenset(objs)  # 返回所有对象



def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )


def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )


def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))


def lowermost(
    patch: Patch
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))


def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))


def rightmost(
    patch: Patch
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))


def square(
    piece: Piece
) -> Boolean:
    """ whether the piece forms a square """
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else height(piece) * width(piece) == len(piece) and height(piece) == width(piece)

def is_square(
    piece: Piece
) -> Boolean:
    """ whether the piece forms a square """
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else height(piece) * width(piece) == len(piece) and height(piece) == width(piece)


def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1


def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1

def sorted_frozenset(fset: frozenset) -> list:
    """
    对 frozenset 中的元组进行排序。

    - 第一种形式：frozenset({(int, int), ...})
    - 第二种形式：frozenset({(int, Any), ...})

    该函数首先按元组的第一个元素排序，如果第一个元素相同，则按第二个元素排序。

    参数:
    - fset (frozenset): 要排序的 frozenset，其中每个元素都是长度为2的元组。

    返回:
    - list: 排序后的元组列表。
    """
    return sorted(fset, key=lambda x: (x[0], x[1]))


def hmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0


def vmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a column for which both patches have cells """
    return len(set(j for i, j in toindices(a)) & set(j for i, j in toindices(b))) > 0


def manhattan(
    a: Patch,
    b: Patch,
    diagonal: bool = True
) -> Integer:
    """
    计算两个patches之间的最小距离

    Args:
        a: 第一个patch
        b: 第二个patch
        diagonal: 是否考虑对角线距离
            False: 曼哈顿距离 |x1-x2| + |y1-y2|
            True: 切比雪夫距离 max(|x1-x2|, |y1-y2|)
    """
    if not diagonal:
        # 传统曼哈顿距离
        return min(abs(ai - bi) + abs(aj - bj)
                  for ai, aj in toindices(a)
                  for bi, bj in toindices(b))
    else:
        # 切比雪夫距离（允许对角线移动）
        return min(max(abs(ai - bi), abs(aj - bj))
                  for ai, aj in toindices(a)
                  for bi, bj in toindices(b))

def adjacent(
    a: Patch,
    b: Patch,
    diagonal: bool = True
) -> Boolean:
    """
    判断两个patches是否相邻

    Args:
        diagonal: True则对角线也算相邻
    """
    if diagonal:
        # 使用切比雪夫距离判断相邻
        return manhattan(a, b, diagonal=True) == 1
    else:
        # 传统四方向相邻
        return manhattan(a, b) == 1


def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1


def centerofmass(
    patch: Patch
) -> IntegerTuple:
    """ center of mass """
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindices(patch))))


def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})



def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))


def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

colorofobj = color

def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)


def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def object_to_grid(obj: Object) -> Grid:

    if not obj:
        return [[]]

    # 获取网格范围
    coords = [(i, j) for _, (i, j) in obj]
    max_i = max(i for i, _ in coords) + 1
    max_j = max(j for _, j in coords) + 1

    # # 创建空网格
    grid = [[None for _ in range(max_j)] for _ in range(max_i)]

    # 填充值
    for value, (i, j) in obj:
        grid[i][j] = value
    tpl = tuple(tuple(inner) for inner in grid)

    return tpl

def grid_to_object(grid: Grid) -> Object:

    return frozenset((grid[i][j], (i, j)) for i in range(len(grid)) for j in range(len(grid[0])))

# 使用示例:
"""
grid = [[1, 2], [3, 4]]
obj = asobject(grid)
restored_grid = object_to_grid(obj)
assert grid == restored_grid
"""

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))


def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])

def upper_third(grid: Grid) -> Grid:
    """ Upper third of grid """
    third = len(grid) // 3
    return grid[:third]


def middle_third(grid: Grid) -> Grid:
    """ Middle third of grid """
    third = len(grid) // 3
    return grid[third:2 * third]


def lower_third(grid: Grid) -> Grid:
    """ Lower third of grid """
    third = len(grid) // 3
    return grid[2 * third + (len(grid) % 3 != 0):]


def left_third(grid: Grid) -> Grid:
    """ Left third of grid """
    return rot270(upper_third(rot90(grid)))


def center_third(grid: Grid) -> Grid:
    """ Center third of grid """
    return rot270(middle_third(rot90(grid)))


def right_third(grid: Grid) -> Grid:
    """ Right third of grid """
    return rot270(lower_third(rot90(grid)))

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]


def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)


def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)


def dmirror(
    piece: Piece
) -> Piece:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)


def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))


def fill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices """
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)


def paint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid """
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)


def underfill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    g = list(list(r) for r in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return tuple(tuple(r) for r in g)


def underpaint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid where there is background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    g = list(list(r) for r in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return tuple(tuple(r) for r in g)


def hupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid horizontally """
    g = tuple()
    for row in grid:
        r = tuple()
        for value in row:
            r = r + tuple(value for num in range(factor))
        g = g + (r,)
    return g


def vupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid vertically """
    g = tuple()
    for row in grid:
        g = g + tuple(row for num in range(factor))
    return g


def upscale(
    element: Element,
    factor: Integer
) -> Element:
    """ upscale object or grid """
    if isinstance(element, tuple):
        g = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            g = g + tuple(upscaled_row for num in range(factor))
        return g
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        o = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    o.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(o), (di_inv, dj_inv))


def get_mode(
    values: List[Union[int, float]]
) -> Union[int, float]:
    """计算列表中的众数"""
    if not values:
        return 0
    counts = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]

def downscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """downscale grid with noise tolerance using mode value in each window"""
    if not grid or not grid[0]:
        return tuple()

    h, w = len(grid), len(grid[0])
    result = []

    # 按factor×factor的窗口扫描
    for i in range(0, h, factor):
        row = []
        for j in range(0, w, factor):
            # 收集窗口内的所有值
            window_values = []
            for di in range(factor):
                for dj in range(factor):
                    if i + di < h and j + dj < w:
                        window_values.append(grid[i + di][j + dj])
            # 使用众数作为该区域的值
            row.append(get_mode(window_values))
        if row:
            result.append(tuple(row))

    return tuple(result)


def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))


def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b


def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))


def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))


def vsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))


def cellwise(
    a: Grid,
    b: Grid,
    fallback: Integer
) -> Grid:
    """ cellwise match of two grids """
    h, w = len(a), len(a[0])
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            value = a_value if a_value == b[i][j] else fallback
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid


def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)


def switch(
    grid: Grid,
    a: Integer,
    b: Integer
) -> Grid:
    """ color switching """
    return tuple(tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid)


def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)


def position(
    a: Patch,
    b: Patch
) -> IntegerTuple:
    """ relative position between two patches """
    ia, ja = center(toindices(a))
    ib, jb = center(toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)


# def index(
def color_at_location(
    grid: Grid,
    loc: IntegerTuple
) -> Integer:
    """ color at location """
    if len(loc) != 2:
        return None
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]]


def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))


def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})


def connect(
    a: IntegerTuple,
    b: IntegerTuple
) -> Indices:
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()


def cover(
    grid: Grid,
    patch: Patch
) -> Grid:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))


def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])


def move(
    grid: Grid,
    obj: Object,
    offset: IntegerTuple
) -> Grid:
    """ move object on grid """
    return paint(cover(grid, obj), shift(obj, offset))


def tophalf(
    grid: Grid
) -> Grid:
    """ upper half of grid """
    return grid[:len(grid) // 2]


def bottomhalf(
    grid: Grid
) -> Grid:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]


def lefthalf(
    grid: Grid
) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))


def righthalf(
    grid: Grid
) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))


# def vfrontier(
#     location: IntegerTuple
# ) -> Indices:
#     """ vertical frontier """
#     return frozenset((i, location[1]) for i in range(30))


# def hfrontier(
#     location: IntegerTuple
# ) -> Indices:
#     """ horizontal frontier """
#     return frozenset((location[0], j) for j in range(30))


def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))


def delta(
    patch: Patch
) -> Indices:
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)


def gravitate(
    source: Patch,
    destination: Patch
) -> IntegerTuple:
    """ direction to move source until adjacent to destination """
    si, sj = center(source)
    di, dj = center(destination)
    i, j = 0, 0
    if vmatching(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacent(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shift(source, (i, j))
    return (gi - i, gj - j)


def inbox(
    patch: Patch
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)

# gpt
def inbox0(patch: Patch) -> Indices:
    """ 提取 patch 的内部一层边界 """
    # 提取最外层边界
    outer_box = box(patch)
    if not outer_box:
        return frozenset()

    # 提取内部区域
    inner_patch = patch - outer_box
    if not inner_patch:
        return frozenset()

    # 计算内部边界坐标
    ai, aj = uppermost(inner_patch) + 1, leftmost(inner_patch) + 1
    bi, bj = lowermost(inner_patch) - 1, rightmost(inner_patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}

    return frozenset(vlines | hlines)

def extract_all_boxes(patch: Patch) -> List[Indices]:
    all_boxes = []
    current_patch = set(patch)  # 创建副本以进行修改

    while current_patch:
        outer_box = box(current_patch)
        if not outer_box:
            break
        all_boxes.append(outer_box)
        # 从当前补丁中移除已提取的外层边界
        current_patch -= set(outer_box)
        # 提取内部边界
        internal_box = inbox(current_patch)
        if not internal_box:
            break
        all_boxes.append(internal_box)
        current_patch -= set(internal_box)

    return all_boxes


def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def box(
    patch: Patch
) -> Indices:
    """Outline of patch, ensuring it's at least 2x2 and forms a rectangle or square."""
    if len(patch) == 0:
        return frozenset()

    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)

    height_patch = abs(bi - ai) + 1
    width_patch = abs(bj - aj) + 1

    # 检查是否至少为2x2
    if height_patch < 2 or width_patch < 2:
        return frozenset()
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))


def occurrences(
    grid: Grid,
    obj: Object
) -> Indices:
    """ locations of occurrences of object in grid """
    occs = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    oh, ow = shape(obj)
    h2, w2 = h - oh + 1, w - ow + 1
    for i in range(h2):
        for j in range(w2):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if not (0 <= a < h and 0 <= b < w and grid[a][b] == v):
                    occurs = False
                    break
            if occurs:
                occs.add((i, j))
    return frozenset(occs)


# def frontiers(
#     grid: Grid
# ) -> Objects:
#     """ set of frontiers """
#     h, w = len(grid), len(grid[0])
#     row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
#     column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
#     hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
#     vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
#     return hfrontiers | vfrontiers

def frontiers(grid: Grid) -> Objects:
    """ set of frontiers """
    if not grid or not grid[0]:
        return frozenset()  # 空网格返回空集合

    try:
        h, w = len(grid), len(grid[0])
        # 检查行是否全相同
        row_indices = tuple(i for i, r in enumerate(grid)
                          if r and len(set(r)) == 1)
        # 检查列是否全相同
        column_indices = tuple(j for j, c in enumerate(dmirror(grid))
                             if c and len(set(c)) == 1)

        # 如果没有找到frontiers，返回空集合
        if not row_indices and not column_indices:
            return frozenset()

        # 构建frontiers
        hfrontiers = frozenset({
            frozenset({(grid[i][j], (i, j)) for j in range(w)})
            for i in row_indices
        }) if row_indices else frozenset()

        vfrontiers = frozenset({
            frozenset({(grid[i][j], (i, j)) for i in range(h)})
            for j in column_indices
        }) if column_indices else frozenset()

        return hfrontiers , vfrontiers

    except Exception as e:
        print(f"Error in frontiers: {e}")
        return frozenset()  # 发生异常时返回空集合

def split_by_frontiers(grid: Grid) -> List[Grid]:
    """根据frontiers分割网格"""
    h, w = len(grid), len(grid[0])
    hfrs, vfrs = frontiers(grid)

    # 如果没有分割线，返回原网格
    if not hfrs and not vfrs:
        return [grid]

    # 直接从frontiers获取分割位置
    row_splits = sorted({next(iter(fr))[1][0] for fr in hfrs}) if hfrs else []
    col_splits = sorted({next(iter(fr))[1][1] for fr in vfrs}) if vfrs else []

    # 添加边界
    row_splits = [0] + row_splits + [h]
    col_splits = [0] + col_splits + [w]

    # 分割网格
    return [
        crop(grid, (start_i, start_j),
            (end_i - start_i, end_j - start_j))
        for i, (start_i, end_i) in enumerate(zip(row_splits, row_splits[1:]))
        for j, (start_j, end_j) in enumerate(zip(col_splits, col_splits[1:]))
    ]


def compress(
    grid: Grid
) -> Grid:
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)


def hperiod(
    obj: Object
) -> Integer:
    """ horizontal periodicity """
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w


def vperiod(
    obj: Object
) -> Integer:
    """ vertical periodicity """
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h

# ```python  asobject(grid)
# filepath: /Users/zhangdexiang/github/VSAHDC/arc-dsl/dsl.py

def period(obj: Object) -> Tuple[int, int, bool]:
    """
    计算对象的水平及垂直周期，并判断是否存在周期性。
    返回 (horizontal_period, vertical_period, is_periodic)
    """
    # 先标准化对象
    normalized = normalize(obj)
    h = height(normalized)
    w = width(normalized)

    # 计算水平周期
    hp = w
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            hp = p
            break

    # 计算垂直周期
    vp = h
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            vp = p
            break

    # 是否存在周期性（需判断是否该周期小于原本尺寸）
    is_periodic = (hp < w) or (vp < h)
    print("水平周期:", hp, "垂直周期:", vp, "是否有周期:", is_periodic)

    return (hp, vp, is_periodic)