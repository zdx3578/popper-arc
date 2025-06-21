#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Visualizing all 1000 train tasks + 120 evaluation tasks
#
# ### ARC-AGI has come for the 3rd time! Let us immerse ourselves into the universe of [p][i][x][e][l][s]

# In[ ]:





# In[ ]:


# DATA_PATH = '/kaggle/input/arc-prize-2025'
# DATA_PATH = '/Users/zhangdexiang/github/ARC-AGI-2/arc-prize-2025'
FIGURES_PATH = 'task_figures'


import os

# 定义多个可能的路径
DATA_PATHS = [
    '/kaggle/input/arc-prize-2025',
    '/Users/zhangdexiang/github/ARC-AGI-2/arc-prize-2025',
    '/home/zdx/github/VSAHDC/'
    '/another/backup/path'
]

# 遍历路径列表，找到第一个存在的路径
for path in DATA_PATHS:
    if os.path.exists(path):
        DATA_PATH = path
        print(f"DATA_PATH is set to: {DATA_PATH}")
        break
else:
    # 如果所有路径都不存在，抛出异常
    raise FileNotFoundError("None of the specified paths exist!")


# In[ ]:





# In[ ]:


import numpy as np, pandas as pd, json, os
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import pprint
pp = pprint.PrettyPrinter(indent=1)
from matplotlib import colors
import copy # for creating full copy of JSON object
from tqdm.notebook import tqdm
from PIL import Image
import time


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

cmap = colors.ListedColormap(
   ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])
norm = colors.Normalize(vmin=0, vmax=10)

def plot_one(ax, i, task, train_or_test, input_or_output, is_solution=False, is_pred=False):
    if is_pred: input_matrix = task
    elif is_solution: input_matrix = task[i]
    else: input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    if is_pred:
        title = 'test prediction'
    else:
        title = train_or_test + ' ' + input_or_output
    ax.set_title(title)


def plot_task(task1, text, task_solution=None, save_file=None):
    num_train = len(task1['train'])
    num_test = len(task1['test'])
    #num_test  = len(task['test'])

    w = num_train

    if task_solution is not None:
        w += num_test

    fig, axs  = plt.subplots(2, w, figsize=(3*w ,3*2))
    plt.suptitle(f'{text}', fontsize=int(3*w*1.5), fontweight='bold', y=1)

    for j in range(num_train):
        plot_one(axs[0, j], j, task1, 'train', 'input')
        plot_one(axs[1, j], j, task1, 'train', 'output')

    if task_solution is not None:
        for k in range(num_test):
            plot_one(axs[0, j+k+1], k, task1, 'test', 'input')
            plot_one(axs[1, j+k+1], k, task_solution, 'test', 'output', is_solution=True)


    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')
#     plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')

    plt.show()


# In[ ]:
import sys
import os
# 定义可能的路径列表
possible_pypaths = [
    '/kaggle/input/3-28arcdsl'
    '/kaggle/input/3-28arcdsl/forpopper2',
    '/kaggle/input/3-28arcdsl/bateson',
    '/Users/zhangdexiang/github/VSAHDC/arcv2',
    '/Users/zhangdexiang/github/VSAHDC/arcv2/forpopper2',
    '/Users/zhangdexiang/github/VSAHDC/arcv2/bateson',
    '/home/zdx/github/VSAHDC/arcv2',
    '/home/zdx/github/VSAHDC/arcv2/forpopper2',
    '/home/zdx/github/VSAHDC/arcv2/bateson',

    '/another/path/to/check'
]

# 遍历路径列表，检查并按需加载
for path in possible_pypaths:
    if os.path.exists(path):
        print(f"Adding path to sys.path: {path}")
        sys.path.append(path)
    else:
        print(f"Path does not exist, skipping: {path}")

# 打印最终的 sys.path 以确认结果
print("Current sys.path:")
for p in sys.path:
    print(p)

from dsl import *
from dsl2 import *
from dataclasses import dataclass

print(sys.version)








from objutil import  *
# from objutil2plus import * # shift_pure_obj_to_0_0_0,shift_pure_obj_to_00,IdManager


managerid = IdManager()

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
    color_ranking: Tuple[Tuple[int, int], ...]
    background: int
    obj000_ops:list
    obj_ops:list
    obj_weight:int        #相同颜色，相同位置，相同行列
    obj_VSA:None

# def objects_fromone_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
#     b1, b2, b3 = bools  # 解包布尔值
#     return objects( grid, b1, b2, b3)

# param_combinations: List[Tuple[bool, bool]] = [
#     (True, True, False) ]



# def all_objects_from_grid(the_pair_id: int, in_or_out: str, grid: Grid, hw:list, weight = 0 ) -> FrozenSet[Object]:
#     acc: FrozenSet[Object] = frozenset()  # 初始化空集合
#     for params in param_combinations:
#         acc = acc.union(objects_fromone_params(the_pair_id, in_or_out, grid, params,hw))
#         # print()
#     result = []
#     bg = mostcolor(grid)
#     for obj in acc:
#         # 对每个 obj，计算对应平移后的版本
#         # 假设 obj 本身是一个表示对象的集合；如果不是，则请调整调用方式
#         obj00 = shift_pure_obj_to_00(obj)
#         obj000 = shift_pure_obj_to_0_0_0(obj)
#         new_obj = ObjInf(
#             pair_id='pair_id: '+str(the_pair_id),
#             in_or_out=in_or_out,
#             objparam="all",  # 使用传入的布尔值
#             obj=obj,         # 原始对象
#             obj_00=obj00,
#             obj_000=obj000,
#             # obj_ID=managerid.get_id("OBJshape", obj000),
#             obj_ID="obj-ID:"+str(managerid.get_id("OBJshape", obj000)),
#             grid_H_W=hw,            # 默认值，根据需要调整
#             bounding_box=(uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)),    # 默认值，根据需要调整
#             color_ranking=palette(obj)    ,     # 默认空 tuple
#             background = bg,
#             obj000_ops=extend_obj(obj000),
#             obj_ops = extend_obj(obj),
#             obj_VSA = None
#         )
#         result.append(new_obj)
#     return result

# def display_matrices(diff1: List[Tuple[int, Tuple[int, int]]],HW:list,
#                           diff2: Optional[List[Tuple[int, Tuple[int, int]]]] = None,
#                           diff3: Optional[List[Tuple[int, Tuple[int, int]]]] = None):
#     """
#     展示所有不同元素位置的二维矩阵，不按数值分组，所有内容一次性打印到一起。

#     参数:
#     - diff1: 必填，包含不同元素及其位置的集合。
#     - diff2, diff3: 可选，额外的不同元素及其位置集合。
#     """
#     # 合并所有不同元素的位置
#     combined = list(diff1) + (diff2 if diff2 else []) + (diff3 if diff3 else [])

#     if not combined:
#         print("无差异")
#         return

#     # 确定矩阵的大小（所有位置的最大行和最大列）
#     # max_row = max(pos[0] for _, pos in combined) + 1
#     # max_col = max(pos[1] for _, pos in combined) + 1
#     max_row = HW[0]
#     max_col = HW[1]

#     # 初始化空矩阵，初始内容为空格
#     matrix = [[' ' for _ in range(max_col)] for _ in range(max_row)]

#     # 填充矩阵：如果同一位置有多个值，则用逗号连接显示
#     for value, (row, col) in combined:
#         current = matrix[row][col]
#         text = str(value)
#         if current == ' ':
#             matrix[row][col] = text
#         else:
#             matrix[row][col] = current + ',' + text

#     # 打印带有边框的矩阵
#     border = "+" + "-" * (max_col * 2 - 1) + "+"
#     print(border)
#     for row in matrix:
#         print("|" + " ".join(row) + "|")
#     print(border)


# In[ ]:


if not os.path.exists(FIGURES_PATH):
    os.mkdir(FIGURES_PATH)


# In[ ]:





# In[ ]:


train_tasks   = load_json(f'{DATA_PATH}/arc-agi_training_challenges.json')
train_sols    = load_json(f'{DATA_PATH}/arc-agi_training_solutions.json')

eval_tasks = load_json(f'{DATA_PATH}/arc-agi_evaluation_challenges.json')
eval_sols  = load_json(f'{DATA_PATH}/arc-agi_evaluation_solutions.json')

test_tasks   = load_json(f'{DATA_PATH}/arc-agi_test_challenges.json')
import torch
# get_ipython().system('pip install torch-hd')
import os

def install_package(package_name):
    exit_code = os.system(f'pip install {package_name}')
    if exit_code == 0:
        print(f"Successfully installed {package_name}")
    else:
        print(f"Failed to install {package_name}")

# 调用函数安装 torch-hd
install_package('torch-hd')
install_package('seaborn')

DEFAULT_SIMILARITY_THRESHOLD = 0.7


# In[ ]:


import torchhd as hd

import numpy as np
from scipy.ndimage import label


import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['font.sans-serif'] = ['Noto Sans']#['Droid Sans Fallback']  # 使用 DroidSansFallback 字体
plt.rcParams['axes.unicode_minus'] = False                 # 解决负号显示问题

from matplotlib import font_manager

# 列出所有可用字体
# for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
#     print(font)

from datetime import datetime
import os

# 颜色常量
COLORS = list(range(10))

# 初始化HDC编码器
dimension = 10000  # 高维向量维度
vsa = hd.FHRRTensor(dimension)  # 选择FHRR作为VSA模型

# 创建颜色编码
# color_encodings = {color: vsa.random() for color in COLORS}
color_encodings = hd.FHRRTensor.random(10, dimension)

# 创建位置编码器
max_grid_size = 30  # 假设最大网格尺寸为30x30



class PositionEncoder:
    def __init__(self, dimension, n_modules=5, scale_factor=1.42, s_min=4):
        """
        初始化位置编码器。
        :param dimension: 每个模块的维度。
        :param n_modules: 模块数量。
        :param scale_factor: 尺度因子。
        :param s_min: 最小尺度。
        """
        self.dimension = dimension
        self.n_modules = n_modules
        self.scale_factor = scale_factor
        self.s_min = s_min

    def encode_position(self, r, c, max_r, max_c):
        """
        编码位置。
        :param r: 行坐标。
        :param c: 列坐标。
        :param max_r: 最大行值（用于归一化）。
        :param max_c: 最大列值（用于归一化）。
        :return: 编码后的高维向量。
        """
        # 归一化坐标
        r_norm = r / max_r
        c_norm = c / max_c

        # 初始化编码向量
        pos_enc = hd.FHRRTensor(torch.zeros(self.dimension))

        # 遍历每个模块
        for i in range(self.n_modules):
            scale = self.s_min * (self.scale_factor ** i)
            module_encoding = self._encode_module(r_norm, c_norm, scale)
            pos_enc = pos_enc.bundle(module_encoding)

        return pos_enc

    def _encode_module(self, r_norm, c_norm, scale):
        """
        编码单个模块。
        :param r_norm: 归一化的行坐标。
        :param c_norm: 归一化的列坐标。
        :param scale: 当前模块的尺度。
        :return: 单个模块的编码向量。
        """
        # 创建相位向量
        r_phases = torch.ones(self.dimension, dtype=torch.float32) * r_norm * 2 * torch.pi / scale
        c_phases = torch.ones(self.dimension, dtype=torch.float32) * c_norm * 2 * torch.pi / scale

        # 转换为复数向量
        r_enc = torch.complex(torch.cos(r_phases), torch.sin(r_phases))
        c_enc = torch.complex(torch.cos(c_phases), torch.sin(c_phases))

        # 应用分数幂操作
        r_enc_fhrr = self.fractional_power(hd.FHRRTensor(r_enc), r_norm)
        c_enc_fhrr = self.fractional_power(hd.FHRRTensor(c_enc), c_norm)

        # 绑定两个编码向量
        module_encoding = r_enc_fhrr.bind(c_enc_fhrr)

        return module_encoding

    @staticmethod
    def fractional_power(v, alpha):
        """
        分数幂操作。
        :param v: 输入向量。
        :param alpha: 幂指数。
        :return: 应用分数幂后的向量。
        """
        return torch.pow(v, alpha)


n_modules = 15     # 模块数量
encoder = PositionEncoder(dimension, n_modules)


def encode_pixel(pixel_data, max_r, max_c):
    """
    编码单个像素

    参数:
        pixel_data: 格式为 (color, (row, col)) 的像素数据
        max_r: 网格最大行数
        max_c: 网格最大列数
    """
    color, position = pixel_data
    r, c = position

    # 编码位置
    pos_enc = encoder.encode_position(r, c, max_r, max_c)

    # 编码颜色 (确保是FHRRTensor)
    color_enc = color_encodings[color]

    # 正确使用bind方法: pos_enc是FHRRTensor对象，调用其bind方法
    pixel_enc = pos_enc.bind(color_enc)

    return pixel_enc

def encode_object(obj, grid_size):
    """
    编码单个对象

    参数:
        obj: 对象的像素列表，每个像素格式为 (color, (row, col))
        grid_size: 网格尺寸 (height, width)
    """
    max_r, max_c = grid_size

    # 编码对象的所有像素
    pixel_encodings = []
    for pixel in obj:
        pixel_enc = encode_pixel(pixel, max_r, max_c)
        pixel_encodings.append(pixel_enc)

    # 合并所有像素编码
    if len(pixel_encodings) == 0:
        # 如果没有像素，返回一个随机编码
        return hd.FHRRTensor.random(1, dimension)[0]

    # 正确使用bundle方法
    # 从第一个编码开始
    obj_encoding = pixel_encodings[0]

    # 逐个添加其他编码
    for enc in pixel_encodings[1:]:
        obj_encoding = obj_encoding.bundle(enc)

    return obj_encoding


#!to be add diff grid
def encode_all_object_sets(object_sets, train_data_size):
    """
    为所有训练数据对的所有对象生成编码

    参数:
        object_sets: 对象集合字典，键的格式为 "in_obj_set_{pair_id}" 或 "out_obj_set_{pair_id}"
        train_data_size: 训练数据对的数量

    返回:
        all_encodings: 字典，包含所有对象的编码
        {
            "in_encodings_{pair_id}": [编码1, 编码2, ...],
            "out_encodings_{pair_id}": [编码1, 编码2, ...],
            ...
        }
    """
    all_encodings = {}

    # 遍历所有训练数据对
    for pair_id in range(train_data_size):
        # 处理输入对象集
        if f"in_obj_set_{pair_id}" in object_sets:
            in_encodings = []

            for obj in object_sets[f"in_obj_set_{pair_id}"]:
                # 获取网格尺寸
                grid_H_W = obj.grid_H_W

                # 编码对象
                obj_encoding = encode_object(obj.obj, grid_H_W)
                in_encodings.append(obj_encoding)

                # 将编码存储到对象中(如果对象有obj_VSA属性)
                if hasattr(obj, 'obj_VSA'):
                    obj.obj_VSA = obj_encoding

            # 将输入编码存储到结果字典
            all_encodings[f"in_encodings_{pair_id}"] = in_encodings

        # 处理输出对象集
        if f"out_obj_set_{pair_id}" in object_sets:
            out_encodings = []

            for obj in object_sets[f"out_obj_set_{pair_id}"]:
                # 获取网格尺寸
                grid_H_W = obj.grid_H_W

                # 编码对象
                obj_encoding = encode_object(obj.obj, grid_H_W)
                out_encodings.append(obj_encoding)

                # 将编码存储到对象中(如果对象有obj_VSA属性)
                if hasattr(obj, 'obj_VSA'):
                    obj.obj_VSA = obj_encoding

            # 将输出编码存储到结果字典
            all_encodings[f"out_encodings_{pair_id}"] = out_encodings

    return all_encodings


def compare_object_encodings(encoding1, encoding2):
    """
    比较两个对象编码的相似度

    参数:
        encoding1: 第一个对象编码 (FHRRTensor)
        encoding2: 第二个对象编码 (FHRRTensor)

    返回:
        相似度值，范围为[-1, 1]
    """
    # 使用余弦相似度
    similarity = encoding1.cosine_similarity(encoding2)
    return similarity.item()

KAGGLE_RESULTS_DIR = "/kaggle/working/vsa_analysis_results"
# os.makedirs(KAGGLE_RESULTS_DIR, exist_ok=True)


# 修改后的相似度矩阵计算函数
def compute_similarity_matrix(encodings, threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """
    计算编码集合中所有对象之间的相似度矩阵

    参数:
        encodings: 对象编码列表

    返回:
        similarity_matrix: n×n的相似度矩阵，其中n是编码数量
    """
    n = len(encodings)
    similarity_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i ==j:
                similarity_matrix[i, j] = 0
            else:
                simival = compare_object_encodings(encodings[i], encodings[j])
                if simival < threshold:
                    similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = simival

    return similarity_matrix



def plot_weight_grids(weight_grids, text, save_file=None):
    """绘制所有权重网格"""
    # 按类型分组
    grid_types = {
        'train_input': [],
        'train_output': [],
        'test_input': []
    }

    for grid_id in weight_grids:
        for grid_type in grid_types:
            if grid_id.startswith(grid_type):
                grid_types[grid_type].append(grid_id)

    # 确定布局
    num_pairs = max(len(grid_types['train_input']),
                   len(grid_types['train_output']),
                   len(grid_types['test_input']))

    # 创建图形
    fig, axs = plt.subplots(2, num_pairs, figsize=(3*num_pairs, 6))
    plt.suptitle(f'{text}', fontsize=int(3*num_pairs*1.5), fontweight='bold', y=1)

    # 绘制权重网格
    for i, grid_id in enumerate(grid_types['train_input']):
        _, grid_norm = normalize_weight_grid(weight_grids[grid_id])
        axs[0, i].imshow(grid_norm, cmap=cmap, norm=norm)
        axs[0, i].set_title(f"Input {i}")
        axs[0, i].grid(True)

    for i, grid_id in enumerate(grid_types['train_output']):
        _, grid_norm = normalize_weight_grid(weight_grids[grid_id])
        axs[1, i].imshow(grid_norm, cmap=cmap, norm=norm)
        axs[1, i].set_title(f"Output {i}")
        axs[1, i].grid(True)

    # 设置样式
    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')

    plt.show()



# ## Plot all 1000 train tasks

# In[ ]:


import warnings
from weightgird import *

# 屏蔽 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
from IPython.display import display, Image, Markdown, HTML


for jj, tid in enumerate(train_tasks):
    tid = '009d5c81'
    # tid = '045e512c'
    if tid in train_tasks.keys():
        train_or_eval = 'train'
        task = train_tasks[tid]
        task_solution = train_sols[tid]
    else:
        train_or_eval = 'eval'
        task = eval_tasks[tid]
        task_solution = eval_sols[tid]

    save_file = f"{FIGURES_PATH}/{tid}_train.png"
    print(f'Train task {jj}: {tid}')
    plot_task(task, f"({jj}) {tid}   {train_or_eval}",
              task_solution=task_solution,
              save_file=None)
    time.sleep(0.05)

    object_sets = {}
    print("\n\nlen object_sets ",len(object_sets))

    train_data = task['train']
    test_data = task['test']

    weight_grids = apply_object_weights_for_arc_task(task)
    # 正确的循环方式
    for grid_id, weight_grid in weight_grids.items():
        # grid_id 是如 'train_input_0', 'train_output_0', 'test_input_0' 这样的字符串
        display_weight_grid(weight_grid, title=f"{grid_id}")
        corrected_grid,cgridint = normalize_weight_grid(weight_grid)
        # display_weight_grid(corrected_grid, title=f"修正后的权重网格 - {grid_id}")
        display_weight_grid(cgridint, title=f"修正后的权重网格 - {grid_id}")
        print('\n\n\n\n\n\n\n\n\n\n')
        # display_matrices(weight_grid, is_grid_format=True)
        # corrected_grid = apply_weight_correction(weight_grid, scale_factor=10)
        # display_weight_grid(corrected_grid, title=f"修正后的权重网格 - {grid_id}")
    plot_weight_grids(weight_grids, f" ! Machine can see the gird weight ! - 任务 {tid}")

    # 按train_input, train_output, test_input分组显示
    # grid_types = ['train_input', 'train_output', 'test_input']
    # for grid_type in grid_types:
    #     for pair_id in range(len(train_data if 'train' in grid_type else test_data)):
    #         grid_id = f"{grid_type}_{pair_id}"
    #         if grid_id in weight_grids:
    #             weight_grid = weight_grids[grid_id]
    #             display_matrices(weight_grid, is_grid_format=True)
    #             corrected_grid,cgridint = normalize_weight_grid(weight_grid)
    #             display_weight_grid(corrected_grid, title=f"修正后的权重网格 - {grid_id}")
    #             display_weight_grid(cgridint, title=f"修正后的权重网格 - {grid_id}")
    #             print

    for pair_id, data_pair in enumerate(train_data):
        # print(data_pair)
        I  = data_pair['input']
        O  = data_pair['output']

        diff_I,diff_O = grid2grid_fromgriddiff(I, O)



        I = tuple(tuple(row) for row in I)
        O = tuple(tuple(row) for row in O)
        # print(I)
        # print(O)
        height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
        height_o, width_o = height(O), width(O)


        # weight_grid_in,weight_grid_out = process_grid_with_weights(I, O)
        # 显示权重热图
        # visualize_weights_as_heatmap(corrected_grid, title="权重热图")
        # weight_grid = apply_weight_correction(weight_grid, scale_factor=10)
        # display_matrices(weight_grid,(height_o, width_o))



        object_sets[f"out_obj_set_{pair_id}"]  = all_objects_from_grid(the_pair_id=pair_id,in_or_out="out",grid=O, hw=(height_o, width_o) )
        object_sets[f"in_obj_set_{pair_id}"]  = all_objects_from_grid(the_pair_id=pair_id,in_or_out="in",grid=I, hw=(height_i, width_i) )

        object_sets[f"out_obj_set_diff_{pair_id}"]  = all_objects_from_grid(the_pair_id=pair_id,in_or_out="out",grid=diff_O, hw=(height_o, width_o),weight = 10 )
        object_sets[f"in_obj_set_diff_{pair_id}"]  = all_objects_from_grid(the_pair_id=pair_id,in_or_out="in",grid=diff_I, hw=(height_i, width_i),weight = 10 )

        print("\n\nlen object_sets ",len(object_sets))
        # print(object_sets[f"out_obj_set_{pair_id}"])

        # object_encodings = encode_object_set(object_sets, pair_id)
        # analysis_result = analyze_encoded_object_set(object_encodings, object_sets["in_obj_set_0"])






        print("\n\n\n\n\n\n\n\n","in")
        print(f"Length of object_sets[f'in_obj_set_{pair_id}']: {len(object_sets[f'in_obj_set_{pair_id}'])}")
        for obj in object_sets[f"in_obj_set_{pair_id}"] :
            display_matrices(obj.obj,obj.grid_H_W)
            # obj_vsa_encoding = encode_object(obj.obj, obj.grid_H_W)
            # print(obj_vsa_encoding)

        print("\n\n\n\n\n\n\n\n","out")
        print(f"Length of object_sets[f'out_obj_set_{pair_id}']: {len(object_sets[f'out_obj_set_{pair_id}'])}")
        for obj in object_sets[f"out_obj_set_{pair_id}"] :
            display_matrices(obj.obj,obj.grid_H_W)

        # time.sleep(1.5)

    # object_encodings = encode_object_set(object_sets, pair_id)
    #     analysis_result = analyze_encoded_object_set(object_encodings, object_sets["in_obj_set_0"])

    print("\n\n\n\n\n\n\n\n","vsa log log")
    from datetime import datetime, timezone
    # 获取当前 UTC 时间
    current_utc_time = datetime.now(timezone.utc)
    # 格式化为 YYYY-MM-DD HH:MM:SS.sss
    formatted_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 截取前三位毫秒
    # 打印时间
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS.sss formatted): {formatted_time}")
    #! add diff grid  len *2  fix 后续
    all_encodings = encode_all_object_sets(object_sets, len(train_data)*2  )

    # 分析所有编码
    results_dir = "vsa_analysis_results"

    # analysis_results = analyze_all_object_sets(all_encodings, object_sets, results_dir, threshold=DEFAULT_SIMILARITY_THRESHOLD)

    # display_all_results()


    from datetime import datetime, timezone
    # 获取当前 UTC 时间
    current_utc_time = datetime.now(timezone.utc)
    # 格式化为 YYYY-MM-DD HH:MM:SS.sss
    formatted_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 截取前三位毫秒
    # 打印时间
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS.sss formatted): {formatted_time}")




    print("\n\nlen object_sets ",len(object_sets))
    print("\n\n\n\n")
    if jj == 0: break


# In[ ]:





# In[ ]:





# ## Plot all 120 evaluation tasks

# In[ ]:


for jj, tid in enumerate(eval_tasks):

    if tid in train_tasks.keys():
        train_or_eval = 'train'
        task = train_tasks[tid]
        task_solution = train_sols[tid]
    else:
        train_or_eval = 'eval'
        task = eval_tasks[tid]
        task_solution = eval_sols[tid]

    print(f'Eval task {jj}: {tid}')

    save_file = f"{FIGURES_PATH}/{tid}_eval.png"
    plot_task(task, f"({jj}) {tid}   {train_or_eval}",
              task_solution=task_solution,
              save_file=None)
    time.sleep(0.5)
    print()
    print()
    print()
    if jj == 0: break


# In[ ]:




