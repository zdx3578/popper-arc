#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Visualizing all 1000 train tasks  And  Machine see what gird weight info + 120 evaluation tasks
# 
# ### ARC-AGI has come for the 3rd time! Let us immerse ourselves into the universe of [p][i][x][e][l][s]
# 
# ### just plot same in out grid size date

# In[ ]:





# In[1]:


DATA_PATH = '/kaggle/input/arc-prize-2025'
FIGURES_PATH = 'task_figures'


# In[ ]:





# In[2]:


import numpy as np, pandas as pd, json, os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[3]:


from dataclasses import dataclass
import sys
print(sys.version)
sys.path.append('/kaggle/input/3-28arcdsl')
from dsl import *
from dsl2 import *
sys.path.append('/kaggle/input/3-28arcdsl/forpopper2')
sys.path.append('/kaggle/input/3-28arcdsl/bateson')

from objutil import  *
# from objutil2plus import * # shift_pure_obj_to_0_0_0,shift_pure_obj_to_00,IdManager


managerid = IdManager()



def objects_fromone_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
    b1, b2, b3 = bools  # 解包布尔值
    return objects( grid, b1, b2, b3)

# param_combinations: List[Tuple[bool, bool]] = [
#     (True, True, False) ]
    




# In[4]:


if not os.path.exists(FIGURES_PATH):
    os.mkdir(FIGURES_PATH)


# In[ ]:





# In[5]:


train_tasks   = load_json(f'{DATA_PATH}/arc-agi_training_challenges.json')
train_sols    = load_json(f'{DATA_PATH}/arc-agi_training_solutions.json')

eval_tasks = load_json(f'{DATA_PATH}/arc-agi_evaluation_challenges.json')
eval_sols  = load_json(f'{DATA_PATH}/arc-agi_evaluation_solutions.json')

test_tasks   = load_json(f'{DATA_PATH}/arc-agi_test_challenges.json')
import torch
get_ipython().system('pip install torch-hd')

DEFAULT_SIMILARITY_THRESHOLD = 0.7


# In[6]:


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


def plot_weight_grids2(weight_grids, text, save_file=None):
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
        # _, grid_norm = normalize_weight_grid(weight_grids[grid_id])
        axs[0, i].imshow(weight_grids[grid_id], cmap='viridis', norm=norm)
        axs[0, i].set_title(f"Input {i}")
        axs[0, i].grid(True)

    for i, grid_id in enumerate(grid_types['train_output']):
        # _, grid_norm = normalize_weight_grid(weight_grids[grid_id])
        axs[1, i].imshow(weight_grids[grid_id], cmap='viridis', norm=norm)
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

# In[8]:


import warnings
import sys
import os
# 定义可能的路径列表
possible_pypaths = [
    '/kaggle/input/3-28arcdsl'
    '/kaggle/input/3-28arcdsl/forpopper2',
    '/kaggle/input/3-28arcdsl/bateson',
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
from weightgird import *


# 屏蔽 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
from IPython.display import display, Image, Markdown, HTML


for jj, tid in enumerate(train_tasks):
    try:
        # tid = '009d5c81'
        if tid in train_tasks.keys():
            train_or_eval = 'train'
            task = train_tasks[tid]
            task_solution = train_sols[tid]
        else:
            train_or_eval = 'eval'
            task = eval_tasks[tid]
            task_solution = eval_sols[tid]
        # 检查所有训练样例的尺寸一致性
        skip_task = False
        train_data = task['train']
        for pair_id, data_pair in enumerate(train_data):
            I = data_pair['input']
            O = data_pair['output']
            
            # 获取输入和输出的尺寸
            height_i, width_i = len(I), len(I[0]) if I else 0
            height_o, width_o = len(O), len(O[0]) if O else 0
            
            # 检查尺寸是否一致
            if height_i != height_o or width_i != width_o:
                print(f"任务 {tid} 的样例 {pair_id} 尺寸不一致: 输入 {height_i}x{width_i}, 输出 {height_o}x{width_o}")
                skip_task = True
                break
        
        # 如果发现尺寸不一致，跳过当前任务
        if skip_task:
            print(f"跳过任务 {tid} 处理，继续下一个任务")
            continue
            
    
        save_file = f"{FIGURES_PATH}/{tid}_train.png"
        print(f'Train task {jj}: {tid}')
        plot_task(task, f"  origin ARC grid show : ({jj}) {tid}   {train_or_eval}", 
                  task_solution=task_solution, 
                  save_file=None)
        # time.sleep(0.5)
        try:
            weight_grids = apply_object_weights_for_arc_task(task)
            
            # plot_weight_grids(weight_grids, f"权重网格 - 任务 {tid}")
            plot_weight_grids(weight_grids, f" ! Machine can see the gird weight ! - ({jj}) {tid}")

            # plot_weight_grids2(weight_grids, f" ! Machine can see the gird weight ! - ({jj}) {tid}")
            
            for grid_id, weight_grid in weight_grids.items():
            # grid_id 是如 'train_input_0', 'train_output_0', 'test_input_0' 这样的字符串
                display_weight_grid(weight_grid, title=f"{grid_id}")

        except Exception as e:
            print(f"无法绘制权重网格: {e}")
    
        # print("\n\nlen object_sets ",len(object_sets))
        print("\n\n\n\n")

        
        # if jj == 4: break

    
    except Exception as e:
        print(f"\n处理任务 {tid} 时出错: {str(e)}")
        print("继续处理下一个任务...\n")
        continue


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


    # 检查所有训练样例的尺寸一致性
    skip_task = False
    train_data = task['train']
    for pair_id, data_pair in enumerate(train_data):
        I = data_pair['input']
        O = data_pair['output']
        
        # 获取输入和输出的尺寸
        height_i, width_i = len(I), len(I[0]) if I else 0
        height_o, width_o = len(O), len(O[0]) if O else 0
        
        # 检查尺寸是否一致
        if height_i != height_o or width_i != width_o:
            print(f"任务 {tid} 的样例 {pair_id} 尺寸不一致: 输入 {height_i}x{width_i}, 输出 {height_o}x{width_o}")
            skip_task = True
            break
    
    # 如果发现尺寸不一致，跳过当前任务
    if skip_task:
        print(f"跳过任务 {tid} 处理，继续下一个任务")
        continue
        

    save_file = f"{FIGURES_PATH}/{tid}_train.png"
    print(f'Train task {jj}: {tid}')
    plot_task(task, f" ! origin ARC grid show : ({jj}) {tid}   {train_or_eval}", 
              task_solution=task_solution, 
              save_file=None)
    # time.sleep(0.5)
    
    weight_grids = apply_object_weights_for_arc_task(task)
    # plot_weight_grids(weight_grids, f"权重网格 - 任务 {tid}")
    plot_weight_grids(weight_grids, f" ! Machine can see the gird weight ! - ({jj}) {tid}")

    

    
    # print("\n\nlen object_sets ",len(object_sets))
    print("\n\n\n\n")
    # if jj == 4: break



    
    
    # save_file = f"{FIGURES_PATH}/{tid}_eval.png"
    # plot_task(task, f"({jj}) {tid}   {train_or_eval}", 
    #           task_solution=task_solution, 
    #           save_file=None)
    # time.sleep(0.5)
    # print()
    # print()
    # print()
    # if jj == 0: break


# In[ ]:




