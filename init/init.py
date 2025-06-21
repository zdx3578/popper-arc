import os
import json
import sys
import traceback



def prepare_paths():
    possible_pypaths = [
        '/kaggle/input/3-28arcdsl'
        '/kaggle/input/3-28arcdsl/forpopper2',
        '/kaggle/input/3-28arcdsl/bateson',
        '/Users/zhangdexiang/github/VSAHDC/arcv2',
        '/Users/zhangdexiang/github/VSAHDC/arcv2/forpopper2',
        '/Users/zhangdexiang/github/VSAHDC',
        '/home/zdx/github/VSAHDC/arcv2',
        '/home/zdx/github/VSAHDC/arcv2/forpopper2',
        '/home/zdx/github/VSAHDC',
        '/home/zdx/github/VSAHDC/arcMrule',
        '/home/zdx/github/VSAHDC/arcMrule/diffstar',
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


def load_json(arcfile_path):
    with open(arcfile_path) as f:
        data = json.load(f)
    return data

# DATA_PATH = '/kaggle/input/3-28arcdsl/forpopper2'
# DATA_PATH = '/Users/zhangdexiang/github/ARC-AGI-2/arc-prize-2025'
DATA_PATH = './data'

def prepare_arc_data():
    train_tasks   = load_json(f'{DATA_PATH}/arc-agi_training_challenges.json')
    train_sols    = load_json(f'{DATA_PATH}/arc-agi_training_solutions.json')

    eval_tasks = load_json(f'{DATA_PATH}/arc-agi_evaluation_challenges.json')
    eval_sols  = load_json(f'{DATA_PATH}/arc-agi_evaluation_solutions.json')

    test_tasks   = load_json(f'{DATA_PATH}/arc-agi_test_challenges.json')


    # 确保数据目录存在
    return train_tasks, train_sols, eval_tasks, eval_sols, test_tasks

