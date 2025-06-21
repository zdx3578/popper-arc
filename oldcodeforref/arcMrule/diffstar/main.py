import argparse
import os


import sys
import os
# 定义可能的路径列表
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
    # '/home/zdx/github/VSAHDC/arcMrule',
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


# from arc_solver import ARCSolver
from diffstar_arc_solver_weighted import WeightedARCSolver








def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='ARC任务解决器')
    parser.add_argument('--task-id', help='要处理的单个任务ID')
    parser.add_argument('--data-dir', help='ARC数据集目录路径')
    parser.add_argument('--output', '-o', default='results.json', help='结果输出文件路径')
    parser.add_argument('--process-all', action='store_true', help='处理所有任务')
    parser.add_argument('--task-type', choices=['train', 'eval', 'test'], default='train',
                        help='处理的任务类型(用于--process-all)')
    parser.add_argument('--limit', type=int, help='处理的最大任务数量')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    # 初始化解决器
    solver = WeightedARCSolver(data_dir=args.data_dir, debug=args.debug)

    # 处理单个任务或所有任务
    if args.process_all:
        # 处理指定类型的所有任务
        results = solver.process_all_tasks(task_type=args.task_type, limit=args.limit)
    else:
        # 如果没有指定任务ID，则默认选择第一个任务
        task_id = args.task_id
        if not task_id:
            # 根据任务类型选择第一个任务
            if args.task_type == 'train' and solver.train_tasks:
                task_id = next(iter(solver.train_tasks.keys()))
            elif args.task_type == 'eval' and solver.eval_tasks:
                task_id = next(iter(solver.eval_tasks.keys()))
            elif args.task_type == 'test' and solver.test_tasks:
                task_id = next(iter(solver.test_tasks.keys()))
            else:
                print("错误：未指定任务ID且无法自动选择任务")
                return 1

        task_data = solver.load_task(task_id)
        if not task_data:
            print(f"无法加载任务: {task_id}")
            return 1

        # 处理任务
        results = solver.process_task(task_data)

    # 保存结果
    if not solver.save_results(results, args.output):
        print(f"无法保存结果到: {args.output}")
        return 1

    print(f"处理完成，结果已保存到: {args.output}")
    return 0

if __name__ == '__main__':
    exit(main())


# python main.py --task-id 009d5c81


# python main.py --process-all --task-type train --debug


# python main.py --process-all --task-type eval --limit 10

# python main.py --data-dir /path/to/data --process-all --task-type train

# python main.py --debug --task-id 009d5c81