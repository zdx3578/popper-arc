from itertools import product
from typing import FrozenSet, Tuple, Union
from dsl import *
from dataclasses import dataclass

from dsl2 import *
from arc_types import *
import pandas as pd
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.width', 1000)         # 调整输出宽度
pd.set_option('display.max_colwidth', None)  # 显示完整列内容
from copy import deepcopy

from objutil2plus import *
import sys
# sys.path.append("bateson/")
# from bateson import bateson_algorithm



class IdManager:
    def __init__(self):
        # 初始化字段：tables 用于存储各 category 下的值与 ID 映射；next_id 用于记录下一个可用的 ID
        self.tables = {}    # 例如: {'shape': {'shape_1': 1, 'shape_2': 2, ...}}
        self.next_id = {}   # 例如: {'shape': 1}

    def get_id(self, category, value):
        """
        获取 category 分类下 value 对应的 ID，
        如果 value 不存在，则分配新的 ID。
        """
        if isinstance(value, set):
            value = frozenset(value)

        if category not in self.tables:
            self.tables[category] = {}
            self.next_id[category] = 1

        category_table = self.tables[category]

        if value not in category_table:
            category_table[value] = self.next_id[category]
            self.next_id[category] += 1

        return category_table[value]

    def print_all_ids(self):
        """
        打印所有类别下的所有对象及其对应的 ID。
        """
        for category, category_table in self.tables.items():
            print(f"\n\nCategory: {category}, length: {len(category_table)} \n")
            for value, id_val in category_table.items():
                print(f"ID : {id_val} -> Object content -> : \n                  {value}")

    def reset(self):
        """
        清空所有数据
        """
        self.tables = {}
        self.next_id = {}
        print("All data has been reset.")




# managerid = IdManager()


def process_single_data(task: List[Any]) -> bool:
    # analysys_in_out_pattern_000(task)
    # result = analysys_in_out_pattern(task)
    # if result:
    #     return True

    analysys_out_out_pattern(task)



def analysys_out_out_pattern(task) -> bool:
    train_data = task['train']
    test_data = task['test']
    successful_obj_pairs = []
    df = pd.DataFrame(columns=columns_outout)
    temp_pd_data = []

    for paramid, out_param in enumerate(param_combinations2):  # 遍历 param_combinations
        obj_id_sets=[]
        for pair_id, data_pair in enumerate(train_data):
            I = input_grid = data_pair['input']
            O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None
            height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
            height_o, width_o = height(O), width(O)

            out_obj_set = objects_info_from_one_params(the_pair_id=pair_id,in_or_out="out",grid=O, bools=out_param, hw=(height_o, width_o) )
            for out_obj in out_obj_set:  # 遍历 out_obj_set
                display_diff_color_ofa_matrices(out_obj.obj)
                display_matrices(out_obj.obj)
                bateson_algorithm(out_obj.obj)


            in_obj_set = objects_info_from_one_params(the_pair_id=pair_id,in_or_out="in",grid=I, bools=out_param, hw=(height_i, width_i) )
            len_out_obj_set = len(out_obj_set),
            len_in_obj_set = len(in_obj_set)
            print(paramid, "\nOutput parameters:", out_param,"pair id : ",  pair_id  , "  | Number of output objects:", len_out_obj_set)
            # for out_obj in out_obj_set:  # 遍历 out_obj_set
            #     display_matrices(out_obj.obj)
            obj_ids = count_obj_ids(out_obj_set,"obj_ID")
            print(f"\nout_obj_set 的 obj_ID 种类:\n {obj_ids}")
            obj_id_sets.append(obj_ids)
        ###!!! todo same id or same position?? same 00 or same 000
        all_same = all(obj_id_sets[0] == obj_id_set for obj_id_set in obj_id_sets) and (len(obj_id_sets[0]) > 1)
        if all_same:
            print("\n\n相同 > 1  所有 out_obj_set 的 obj_ID 种类相同！paramid : ", paramid, "Output parameters:", out_param)
            print("\n\n\n")
        else:
            print("\n\n不同 or = 1 存在不同的 obj_ID 种类！")
            print("\n\n\n")


def analysys_in_out_pattern(task) -> bool:
    # Initialize accumulators
    train_data = task['train']

    successful_params = []
    all_transformations = {}



    # Iterate through parameter combinations and accumulate successes
    for paramid, out_param in enumerate(param_combinations):

        # Check all training pairs with current parameter
        param_works_for_all_pairs = True
        success_pairs = []
        for pair_id, data_pair in enumerate(train_data):
            I = input_grid = data_pair['input']
            O = output_grid = data_pair.get('output')
            height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
            height_o, width_o = height(O), width(O)

            out_obj_set = objects_info_from_one_params(the_pair_id=pair_id,in_or_out="out",grid=O, bools=out_param, hw=(height_o, width_o) )
            # input_obj_set = objects_info_from_one_params(the_pair_id=pair_id,in_or_out="in",grid=I, bools=out_param, hw=(height_i, width_i) )

            input_obj_set = input_obj_setsparam = all_objects_from_grid_all_parma(the_pair_id=pair_id,in_or_out="in",grid=I, hw=(height_i, width_i))

            successful_obj = []
            # Check that all output objects are solvable with current parameter
            all_out_obj_satisfied = True
            for out_obj in out_obj_set:  # 遍历 out_obj_set
                # display_diff_color_ofa_matrices(out_obj.obj)
                # display_matrices(out_obj.obj)
                found_valid_in_outobj = False
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:  # 遍历 input_obj_set
                        if in_obj.obj == out_obj.obj:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same_same")
                            successful_obj.append(tempdata)
                            break  # 存在一个满足条件即可退出内层循环
                #! 先平移  00  再其他旋转等  same_same",match_op
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        if in_obj.obj_00 == out_obj.obj_00:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_same00",'move')
                            successful_obj.append(tempdata)
                            break  # 存在一个满足条件即可退出内层循环
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        match_op = next((name for name, res in out_obj.obj_ops if res == in_obj.obj), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same_same",match_op)
                            successful_obj.append(tempdata)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        match_op = next((name for name, res in out_obj.obj_ops if res == in_obj.obj_00), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_same",match_op)
                            successful_obj.append(tempdata)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        if in_obj.obj_000 == out_obj.obj_000:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_0",'move+color')
                            successful_obj.append(tempdata)
                            break  # 存在一个满足条件即可退出内循环
                        # elif in_obj.obj_000 in out_obj.obj000_ops: ##any(x in out_obj.obj000_ops for x in in_obj.obj000_ops):    # 至少存在一个共同元素
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        match_op = next((name for name, res in out_obj.obj000_ops if res == in_obj.obj_000), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_0",match_op)
                            successful_obj.append(tempdata)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        if any(x in out_obj.obj000_ops for x in in_obj.obj000_ops):    # 至少存在一个共同元素
                            # in_obj.obj_00 == out_obj.obj_00:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "extend222")
                            successful_obj.append(tempdata)
                            break
                if not found_valid_in_outobj:  # 如果没有找到满足条件的 in_obj
                    all_out_obj_satisfied = False
                    break  # 跳出中间层循环
            sorted_successful_obj = sorted(    successful_obj,    key=lambda x: int(x[0][3].split(' : ')[1])  )
            success_pairs.append((paramid, out_param, pair_id ,sorted_successful_obj))
            # success_pairs.append((successful_obj,))

            if not all_out_obj_satisfied:
                # Some output object was unsolvable - current parameter doesn't work for all pairs
                param_works_for_all_pairs = False
                break  # Move to the next parameter combination

        if param_works_for_all_pairs:
            # Current parameter worked for all training pairs - accumulate it
            # successful_params.append((paramid, out_param,success_pairs))
            successful_params.append((success_pairs))
            print(" - -  pretty_print")
            pretty_print(success_pairs)
            result = find_rule(success_pairs,task)
            if result:
                return True


    # print("\n\n all  all  pretty_print")
    # pretty_print(successful_params)


    # Construct the final result
    # success = len(successful_params) > 0
    return






def find_rule(df,task):
    test_data = task['test']
    op1, op2 = None, None
    found_mismatch = False
    for d in df:
        obj_pairs = d[3]
        if all(obj_pairs[0][2] == obj_pair[2] for obj_pair in obj_pairs) :
            # print("all obj one pair have the same -- obj:",obj_pairs[0][2])
            op1 = obj_pairs[0][2] if op1 is None else op1
            if op1 != obj_pairs[0][2]:
                found_mismatch = True
                op1, op2 = None, None
                break # 说明不是同一个 obj op


            if  ( len(obj_pairs[0]) > 3 and obj_pairs[0][3] ) :
                if all(obj_pairs[0][3] == obj_pair[3] for obj_pair in obj_pairs) :
                    # print("all obj one pair have the same -- operator:",obj_pairs[0][3])
                    op2 = obj_pairs[0][3] if op2 is None else op2
                    if op2 != obj_pairs[0][3]:
                        op1, op2 = None, None
                        found_mismatch = True
                        break # 说明不是同一个 obj op
        if found_mismatch:
            break  # 跳出最外层的 for 循环
    if op1 and op2 :
        print("op1: ",op1)
        print("op2: ",op2)
        # print("found rule")
        return apply_rule(op1,op2,task, df)
        # return op1, op2

#! 罗列所有的规则
def apply_rule(op1,op2,task,df):
    test_data = task['test']
    data_pair = test_data[0]
    I = input_grid = data_pair['input']
    O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None
    background  = None
    height_widht = None
    if all(df[0][3][0][0][6] == obj_pairs[3][0][0][6] for obj_pairs in df) :
        print(f"all obj one pair have the background: {df[0][3][0][0][6]}, test var {df[0][3][0][0][2]}, {df[0][3][0][0][3]}")
        background = df[0][3][0][0][6]
    else:
        background = 0
    if all(df[0][3][0][0][7] == obj_pairs[3][0][0][7] for obj_pairs in df) :
        print(f"all obj one pair have the height_widht: {df[0][3][0][0][7]}, test var {df[0][3][-1][0][4]}, {df[0][3][-1][0][3]}")
        height_widht = df[0][3][0][0][7]

    if op1 == "same00_same00" and op2 == "move":
        # print("apply_rule: same00_same00 and move")
        # 确认移动位置在多pair之间是相同的，相同id 相同的位置 ，假设 df[0][3] 是一个列表或可迭代对象，且每个元素需要匹配其内部的某个位置，例如 [0][0][3]
        if all(
            all(d[3][i][0][3] == df[0][3][i][0][3] for d in df) and all(d[3][i][0][3] == df[0][3][i][0][3] for d in df)
            for i in range(len(df[0][3]))
            ):
            # print(f"\n\n！相同对象在相同的位置 ,test var: {df[0][3][0][0][2]}")
            input_obj_set = objects_info_from_one_params(the_pair_id="test",in_or_out="in",grid=I, bools=df[0][3][0][0][2], hw=("test") )
            moved_objs = move_in_obj_based_on_out(df,input_obj_set)
            output = paint_objects(moved_objs, background,height_widht)
            # display_matrices(output)
            assert output == O
            print("                   !  ok  !                     apply_rule: same00_same00 and move ok ")
            if output == O:
                return True
        # return  ( assert output == O)
    if op1 == "same00_same" and op2 == "vmirror":         #7468f01a
        #! if one op just run one op
        if all( len(d[3]) == 1 for d in df) :
            print
            input_obj_set = objects_info_from_one_params(the_pair_id="",in_or_out="in",grid=I, bools=df[0][3][0][1][2], hw=("") )
            obj = input_obj_set[0].obj
            obj00 = shift_pure_obj_to_00(obj)
            op2fun = globals()[op2]
            out = op2fun(obj00)
            output = object_to_grid(out)
            assert output == O
            if output == O:
                    return True

        print


    elif op1 == "same_same" :
        # print(" - -  pretty_print")
        pretty_print(df)
        print(op2)
        # function = findfunction(op2    getattr(solvers_module, f'solve_{key}')
        if all( len(d[3]) == 1 for d in df) :
            op2fun = globals()[op2]
            out = op2fun(I)
            assert out == O
            if out == O:
                    return True




    elif op1 == "same00_0" and op2 == "move+color":
        print("apply_rule: same00_0 and move+color")






@dataclass
class ObjInf:
    pair_id: Integer
    in_or_out: str
    objparam: Tuple[bool, bool, bool]  # 3个bool
    obj: Objects       # 假设这是一个通用对象
    obj_00: Objects    # 假设这是一个通用对象
    obj_ID: int
    obj_000: Objects   # 假设这是一个通用对象
    grid_H_W: Tuple[Integer, Integer]    # 假设是一个 (height, width) 的元组
    bounding_box: Tuple[Integer, Integer, Integer, Integer]    # 列表 [minr, minc, maxr, maxc]
    color_ranking: Tuple[Tuple[int, int], ...]   # 从大到小的 多对( color count , color );
    background: int
    obj000_ops:list
    obj_ops:list





def lessforprintobj(obj):
    return (obj.pair_id,obj.in_or_out,obj.objparam,obj.obj_ID,obj.bounding_box, obj.color_ranking, obj.background, obj.grid_H_W)


def all_pureobjects_from_grid0(param_combinations, the_pair_id: int, in_or_out: str, grid: Grid, hw:list, weight = 0 ) -> FrozenSet[Object]:
    acc: FrozenSet[Object] = frozenset()  # 初始化空集合
    for params in param_combinations:
        acc = acc.union(objects_fromone_params(the_pair_id, in_or_out, grid, params,hw))
        # print()

    return acc

def all_pureobjects_from_grid(param_combinations, the_pair_id: int, in_or_out: str, grid: Grid, hw:list, weight=0, background_color=None) -> FrozenSet[Object]:
    """
    从网格中提取对象，可选择过滤掉完全由背景色组成的对象

    参数:
        param_combinations: 对象提取参数组合
        the_pair_id: 样本对ID
        in_or_out: 输入或输出标识
        grid: 输入网格
        hw: 高宽列表
        weight: 权重参数，默认为0
        background_color: 背景色，默认为None。如果指定（0-9），则过滤掉全部由该背景色组成的对象

    返回:
        过滤后的对象集合
    """
    #! fun  determine_background_color
    background_color = 0 if background_color is None else background_color

    acc: FrozenSet[Object] = frozenset()  # 初始化空集合
    # acc = acc.union(objects_fromone_params(the_pair_id, in_or_out, grid, param_combinations, hw))
    for params in param_combinations:
        acc = acc.union(objects_fromone_params(the_pair_id, in_or_out, grid, params,hw))

    # 如果指定了背景色，过滤掉全部由背景色组成的对象
    if background_color is not None:
        filtered_acc = frozenset(
            obj for obj in acc if not all(color == background_color for color, _ in obj)
        )
        print(f"ObjFun: Filtered objects based on background color {background_color} ,  delete: {len(acc) - len(filtered_acc)} objects")
        return filtered_acc
    else:
        print(f"ObjFun: No background color filter applied, total objects: {len(acc)}")

    return acc

def pureobjects_from_grid(param_combinations, the_pair_id: int, in_or_out: str, grid: Grid, hw:list, weight=0, background_color=None) -> FrozenSet[Object]:
    """
    从网格中提取对象，可选择过滤掉完全由背景色组成的对象

    参数:
        param_combinations: 对象提取参数组合
        the_pair_id: 样本对ID
        in_or_out: 输入或输出标识
        grid: 输入网格
        hw: 高宽列表
        weight: 权重参数，默认为0
        background_color: 背景色，默认为None。如果指定（0-9），则过滤掉全部由该背景色组成的对象

    返回:
        过滤后的对象集合
    """
    acc: FrozenSet[Object] = frozenset()  # 初始化空集合
    acc = acc.union(objects_fromone_params(the_pair_id, in_or_out, grid, param_combinations, hw))

    # 如果指定了背景色，过滤掉全部由背景色组成的对象
    if background_color is not None:
        filtered_acc = frozenset(
            obj for obj in acc if not all(color == background_color for color, _ in obj)
        )
        print(f"ObjFun: Filtered objects based on background color {background_color}; all obj:{len(acc)}, filtered obj:{len(filtered_acc)} ,  delete: {len(acc) - len(filtered_acc)} objects")
        return filtered_acc

    return acc


def objects_info_from_one_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
    b1, b2, b3 = bools  # 解包布尔值
    # return objects( grid, b1, b2, b3)
    result = []
    bg = mostcolor(grid)
    for obj in objects(grid, b1, b2, b3):
        # 对每个 obj，计算对应平移后的版本
        # 假设 obj 本身是一个表示对象的集合；如果不是，则请调整调用方式
        obj00 = shift_pure_obj_to_00(obj)
        obj000 = shift_pure_obj_to_0_0_0(obj)
        new_obj = ObjInf(
            pair_id='pair_id: '+str(the_pair_id),
            in_or_out=in_or_out,
            objparam=bools,  # 使用传入的布尔值
            obj=obj,         # 原始对象
            obj_00=obj00,
            obj_000=obj000,
            # obj_ID=managerid.get_id("OBJshape", obj000),
            obj_ID="objID : "+str(managerid.get_id("OBJshape", obj000)),
            grid_H_W=hw,            # 默认值，根据需要调整
            bounding_box=(uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)),   # 默认值，根据需要调整
            color_ranking=palette(obj)    ,   # 默认空 tuple
            background = bg,
            obj000_ops=extend_obj(obj000),
            obj_ops = extend_obj(obj)
        )
        result.append(new_obj)
    return result

def all_objects_from_grid_all_parma(the_pair_id: int, in_or_out: str, grid: Grid, hw: list):
    result = []  # 初始化空列表存放所有结果
    for params in param_combinations:
        objs = objects_info_from_one_params(the_pair_id, in_or_out, grid, params, hw)
        result.extend(objs)  # 将当前参数组合的结果加入 result
    return result

def objop(obj,op):
    return grid_to_object(op(object_to_grid(obj)))

def s_filtered(s) :
    return frozenset(e for e in s if e[0] is not None)


def extend_obj(obj):
    """
    对传入的 obj 分别进行不同的变换，并返回一个包含
    (操作名称, 变换结果) 的元组，每个操作结果经过 s_filtered 处理（如果需要）。
    """
    transformations = [
        ("vmirror", vmirror),
        ("cmirror", cmirror),
        ("hmirror", hmirror),
        ("dmirror", dmirror),
        ("rot90", lambda o: s_filtered(objop(o, rot90))),
        ("rot180", lambda o: s_filtered(objop(o, rot180))),
        ("rot270", lambda o: s_filtered(objop(o, rot270))),
    ]
    results = tuple((name, func(obj)) for name, func in transformations)
    return results



from dsl import *
from constants import *
import dsl2
import inspect
from arc_types import *
import logging



def is_arg1_is_arg2_subgrid(grid1: Grid, grid2: Grid) -> Union[Tuple[bool, str, Tuple[int, int], Tuple[int, int]], bool]:
    """
    检查 grid1 是否是 grid2 的子网格。

    参数:
    - grid1: Grid - 第一个矩形网格。
    - grid2: Grid - 第二个矩形网格。

    返回:
    - bool: 如果 grid1 是 grid2 的子网格，返回 True；否则返回 False。
    """
    h1, w1 = len(grid1), len(grid1[0])
    h2, w2 = len(grid2), len(grid2[0])

    # 检查 grid1 的尺寸是否小于或等于 grid2
    if h1 > h2 or w1 > w2:
        return False

    # 遍历 grid2，检查是否存在与 grid1 匹配的子网格
    for i in range(h2 - h1 + 1):
        for j in range(w2 - w1 + 1):
            match = True
            for x in range(h1):
                for y in range(w1):
                    if grid1[x][y] != grid2[i + x][j + y]:
                        match = False
                        break
                if not match:
                    break
            if match:
                # return (True, 'crop', (i, j),(h1, w1))
                return True

    return False



columns = [
    "pair_id", "out", "outparam", "output_id",
    "input_id",
    "outbounding_box", "outcolor",
    # "pair_id2",
    "in", "inparam",
    # "input_id",
    "inbounding_box", "incolor",
    "label"
]
columns_outout = [
    "pair_id", "out", "outparam", "output_id",
    # "input_id",
    "outbounding_box", "outcolor",
]
def parsed_pd_data(raw_data):
    data = {
        "pair_id": raw_data[0][0],
        "out": raw_data[0][1],
        "outparam": raw_data[0][2],
        "output_id": (raw_data[0][3].split()[-1]),
        "outbounding_box": raw_data[0][4],
        "outcolor": next(iter(raw_data[0][5])),
        # "pair_id_2": raw_data[1][0],
        # "pair_id_2": raw_data[0][0],
        "in": raw_data[1][1],
        "inparam": str(raw_data[1][2]),
        "input_id": (raw_data[1][3].split()[-1]),
        "inbounding_box": raw_data[1][4],
        "incolor": next(iter(raw_data[1][5])),
        "label": raw_data[2]
    }
    if len(raw_data) > 3 and raw_data[3]:
        data["operation"] = raw_data[3]
    return data
def parsed_pd_outout_data(raw_data):
    data = {
        "pair_id": raw_data[0][0],
        "out": raw_data[0][1],
        "outparam": raw_data[0][2],
        # "output_id": int(raw_data[0][3].split()[-1]),
        "output_id": (raw_data[0][3].split()[-1]),
        "outbounding_box": raw_data[0][4],
        "outcolor": next(iter(raw_data[0][5])),
    }
    if len(raw_data) > 3 and raw_data[3]:
        data["operation"] = raw_data[3]
    return data

# def count_obj_ids(obj_set,colum):
#     return set(obj.colum for obj in obj_set)
def count_obj_ids(obj_set, attr_name):
    return {getattr(obj, attr_name) for obj in obj_set}

def show_count_col(df,col_id):
            value_counts = df[col_id].value_counts()
            sorted_value_counts = value_counts.sort_values(ascending=True)
            print("\n","count_number  排序:  ",col_id)
            print(sorted_value_counts)
            print("\n")




def shift_pure_obj_to_0_0_0(obj):
    """
    对于纯对象集合（以 set 表示），将所有对象坐标平移到 (0,0) 并将颜色重设为 0.
    假设每个对象 e 格式为 (color, (r, c))，其中 color, r, c 为整数.
    """
    if not obj:
        return set()
    obj_list = list(obj)
    # 提取所有 (r, c)
    rc_list = [e[1] for e in obj_list]
    min_row = min(r for r, c in rc_list)
    min_col = min(c for r, c in rc_list)
    new_set = set()
    for e in obj_list:
        # 忽略原始 color，统一设置为 0
        _, (r, c) = e
        new_obj = (0, (r - min_row, c - min_col))
        new_set.add(new_obj)
    return new_set


def shift_pure_obj_to_00(obj):
    """
    对于纯对象集合，将所有对象坐标平移到 (0,0)，但保持原始颜色不变.
    假设每个对象 e 格式为 (color, (r, c)).
    """
    if not obj:
        return set()
    obj_list = list(obj)
    rc_list = [e[1] for e in obj_list]
    min_row = min(r for r, c in rc_list)
    min_col = min(c for r, c in rc_list)
    new_set = set()
    for e in obj_list:
        color, (r, c) = e
        new_obj = (color, (r - min_row, c - min_col))
        new_set.add(new_obj)
    return new_set




def pretty_print(dataall, indent=1):
    """
    格式化打印嵌套数据结构，每个字段一行，嵌套列表的每个子列表也分行打印。
    :param dataall: 要打印的数据（可以是元组、列表、集合、字典等）
    :param indent: 当前缩进级别（用于递归调用）
    """
    # 定义缩进字符串
    indent_str = " " * (indent * 3)
    print("\n\npretty_print function")

    if isinstance(dataall, (tuple, list)):
        # 如果是元组或列表
        print(indent_str + "[")
        for data in dataall:
            if isinstance(data, (tuple, list)) and len(data) == 4:
                # 如果是包含四个字段的元组 (paramid, out_param, len_out_obj_set, successful_obj)
                paramid, out_param, pair_id, successful_obj = data
                print(indent_str + " " * 4 + f"paramid: {paramid}")
                print(indent_str + " " * 4 + f"out_param: {out_param}")
                print(indent_str + " " * 4 + f"pair_id:: {pair_id}")
                print(indent_str + " " * 4 + "successful_obj:")
                # 打印 successful_obj 中的每个元素
                if isinstance(successful_obj, (tuple, list)):
                    print(indent_str + " " * 8 + "[")
                    for item in successful_obj:
                        # print(indent_str + str(item))
                        print(indent_str + " " * 8 + str(item))
                        # pretty_print(item, indent + 3)  # 递归打印 successful_obj 中的每个元素
                    print(indent_str + " " * 8 + "]")
                else:
                    print(indent_str + " " * 8 + str(successful_obj))
            else:
                # 对其他类型的元组或列表递归打印
                pretty_print(data, indent + 1)
        print(indent_str + "]")
    elif isinstance(dataall, frozenset):
        # 如果是 frozenset，转换为列表后递归打印
        print(indent_str + "frozenset({")
        pretty_print(sorted(dataall), indent + 1)  # 排序以便输出更整齐
        print(indent_str + "})")
    elif isinstance(dataall, dict):
        # 如果是字典，逐键值对打印
        for key, value in dataall.items():
            print(indent_str + f"{key}:")
            pretty_print(value, indent + 1)
    else:
        # 普通数据类型直接打印
        print(indent_str + str(dataall))



def foranalysisshow(df):
    value_counts = df['outparam'].value_counts()

    sorted_value_counts = value_counts.sort_values(ascending=True)
    print("\n按 outparam count  排序:")
    print(sorted_value_counts)
    print("\n")

    df['out_param_count'] = df['outparam'].map(value_counts)
    sorted_df = df.sort_values(by=["out_param_count","outparam", "pair_id", "output_id"], ascending=[True, True, True, True])

    column_widths = {col: max(sorted_df[col].astype(str).apply(len).max(), len(col)) + 3 for col in sorted_df.columns}

    # 输出时检测 outparam 的变化并插入空行
    previous_outparam = None
    output_lines = []
    header = "".join(f"{col:^{column_widths[col]}}" for col in sorted_df.columns)

    output_lines.append(header)
    for _, row in sorted_df.iterrows():
        current_outparam = row['outparam']
        # 如果 outparam 发生变化，插入一个空行
        if previous_outparam is not None and current_outparam != previous_outparam:
            output_lines.append("")  # 插入空行
            # print("\n".join(output_lines))
        # 添加当前行到输出

        formatted_row = []
        for col in sorted_df.columns:
            value = str(row[col])
            width222 = column_widths[col]
            formatted_row.append(f"{value:^{width222}}")
        output_lines.append("".join(formatted_row))
        # 更新 previous_outparam
        previous_outparam = current_outparam
    # 打印最终输出
    print("\n".join(output_lines))


# printlist = lambda x: print("\n".join(map(str, x)))
def printlist(x):
    # for l in list:
    #     print(l)
    print("\n\n".join(map(str, x)))
    # lambda x: print("\n".join(map(str, x)))

def forprintlist(xx):
    for x in xx:
        print("\n\n")
        print("\n\n".join(map(str, x)))

param_combinations3: List[Tuple[bool, bool, bool]] = [
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, True, False),
    (True, True, True),
    (True, False, False),
    (True, False, True)  ]

param_combinations2: List[Tuple[bool, bool, bool]] = [
    (True, True, False),
    (True, True, True),
    (True, False, False),
    (True, False, True),
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True) ]

param_combinations: List[Tuple[bool, bool]] = [
    (True, True, False),
    (True, False, False),
    (False, False, False),
    (False, True, False) ]

def objects_fromone_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
    b1, b2, b3 = bools  # 解包布尔值
    return objects( grid, b1, b2, b3)

# def objects_fromone_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
#     b1, b2, b3 = bools  # 解包布尔值
#     return objects( grid, b1, b2, b3)

# all_objects_from_grid 函数
def all_objects_from_grid(the_pair_id: int, in_or_out: str, grid: Grid, hw:list) -> FrozenSet[Object]:
    acc: FrozenSet[Object] = frozenset()  # 初始化空集合
    for params in param_combinations:
        acc = acc.union(objects_fromone_params(the_pair_id, in_or_out, grid, params,hw))
        # print()
    result = []
    bg = mostcolor(grid)
    for obj in acc:
        # 对每个 obj，计算对应平移后的版本
        # 假设 obj 本身是一个表示对象的集合；如果不是，则请调整调用方式
        obj00 = shift_pure_obj_to_00(obj)
        obj000 = shift_pure_obj_to_0_0_0(obj)
        new_obj = ObjInf(
            pair_id='pair_id: '+str(the_pair_id),
            in_or_out=in_or_out,
            objparam="all",  # 使用传入的布尔值
            obj=obj,         # 原始对象
            obj_00=obj00,
            obj_000=obj000,
            # obj_ID=managerid.get_id("OBJshape", obj000),
            obj_ID="obj-ID:"+str(managerid.get_id("OBJshape", obj000)),
            grid_H_W=hw,            # 默认值，根据需要调整
            bounding_box=(uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)),    # 默认值，根据需要调整
            color_ranking=palette(obj)    ,     # 默认空 tuple
            background = bg,
            obj000_ops=extend_obj(obj000),
            obj_ops = extend_obj(obj)
        )
        result.append(new_obj)
    return result

def paint_objects(obj_set, background,hw):
    if background is None:
        background = -1
    grid = canvas(background, hw)
    grid = [list(row) for row in grid]

    for obj in obj_set:
        for node in obj:
            color, (r, c) = node
            grid[r][c] = color
    tpl = tuple(tuple(inner) for inner in grid)
    return tpl


def move_in_obj_based_on_out(successful_obj,new_in_obj_list):
    pairs = successful_obj[0][3]
    moved_objs = []
    for pair in pairs:
        out_obj_info, in_obj_info, op_type, op = pair
        # if in_obj_info[3] == out_obj_info[3]:
        bbox = out_obj_info[4]  # bounding_box
        # print(f"匹配到 obj_ID {in_obj_info[3]}，使用 bounding_box {bbox} 来移动 in_obj.obj")

        in_obj = get_in_obj_by_id(in_obj_info[3],new_in_obj_list)
        if in_obj is not None:
            in_obj.obj = shift_object_by_bbox(in_obj.obj,in_obj.bounding_box, bbox)
            # print(f"in_obj.obj 已经移动到新位置: {in_obj.obj}")
            moved_objs.append(in_obj.obj)
    # pretty_print(moved_objs)
    return moved_objs

def shift_object_by_bbox(obj_set,bbox0, bbox):
    dr0, dc0 = bbox0[0], bbox0[1]
    dr, dc = bbox[0], bbox[1]
    new_obj_set = set()
    for color, (r, c) in obj_set:
        new_obj_set.add((color, (r -dr0 + dr, c -dc0 + dc)))
    return new_obj_set


def get_in_obj_by_id(obj_id, global_in_obj_list):
    for obj in global_in_obj_list:
        if obj.obj_ID == obj_id:
            return obj
    return None


