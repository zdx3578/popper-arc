import os
import json
import sys
import traceback

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




import matplotlib.pyplot as plt
from arc_solver_modular_integration import ARCSolverModular
from grid_extension_plugin_integration import GridExtensionPriorKnowledge





def visualize_grid(grid, title=""):
    """可视化ARC网格"""
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='tab10')
    plt.title(title)
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)

    # 添加网格线
    height, width = len(grid), len(grid[0])
    for i in range(width + 1):
        plt.axvline(i - 0.5, color='gray', lw=0.5)
    for i in range(height + 1):
        plt.axhline(i - 0.5, color='gray', lw=0.5)

    return plt

def load_task(task_path):
    """加载ARC任务文件"""
    with open(task_path, 'r') as f:
        return json.load(f)

def run_solver(task_path, output_dir="./popper_files"):
    """运行ARC求解器与Popper集成"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载任务
    task_data = load_task(task_path)

    # 创建求解器实例
    solver = ARCSolverModular(debug=True)

    # 注册插件
    grid_plugin = GridExtensionPriorKnowledge()
    solver.register_plugin(grid_plugin)

    # 设置任务数据
    solver.train_pairs = [(pair['input'], pair['output']) for pair in task_data['train']]
    solver.test_pairs = [(pair['input'], pair['output']) for pair in task_data['test']]
    solver.task_data = task_data

    # 找到适用的插件
    solver.applicable_plugins = [p for p in solver.prior_knowledge_plugins
                                if p.is_applicable(task_data)]

    print(f"加载了 {len(solver.train_pairs)} 个训练对和 {len(solver.test_pairs)} 个测试对")
    print(f"找到 {len(solver.applicable_plugins)} 个适用的先验知识插件")

    # 保存Popper文件
    solver.save_popper_files(output_dir)
    print(f"Popper文件--将保存到 {output_dir} 目录")

    # 尝试学习规则
    try:
        print("开始使用Popper学习规则...")
        learned_rules = solver.learn_rules_with_popper(output_dir)
        print(f"学习到 {len(learned_rules)} 条规则")

        # 保存学习到的规则
        with open(os.path.join(output_dir, "learned_rules.txt"), "w") as f:
            for rule in learned_rules:
                f.write(f"{rule}\n")
    except Exception as e:
        print(f"学习规则时出错: {e}")
        print(traceback.format_exc())
        learned_rules = []

    # 应用规则到测试用例
    results = []
    for i, (input_grid, expected_output) in enumerate(solver.test_pairs):
        print(f"\n处理测试用例 {i+1}...")

        # 应用规则
        output_grid = solver.apply_learned_rules(input_grid, learned_rules)
        results.append((output_grid, expected_output))

        # 可视化结果
        try:
            plt_input = visualize_grid(input_grid, f"测试输入 {i+1}")
            plt_input.savefig(os.path.join(output_dir, f"test_{i+1}_input.png"))
            plt.close()

            plt_output = visualize_grid(output_grid, f"预测输出 {i+1}")
            plt_output.savefig(os.path.join(output_dir, f"test_{i+1}_output.png"))
            plt.close()

            plt_expected = visualize_grid(expected_output, f"期望输出 {i+1}")
            plt_expected.savefig(os.path.join(output_dir, f"test_{i+1}_expected.png"))
            plt.close()
        except Exception as e:
            print(f"可视化结果时出错: {e}")

    # 评估结果
    correct = 0
    for i, (output_grid, expected) in enumerate(results):
        matches = all(output_grid[y][x] == expected[y][x]
                    for y in range(len(output_grid))
                    for x in range(len(output_grid[0])))
        print(f"测试用例 {i+1}: {'正确' if matches else '不正确'}")
        if matches:
            correct += 1

    print(f"\n总结: {correct}/{len(results)} 个测试用例正确")
    return correct, len(results)

if __name__ == "__main__":
    task_path = "05a7bcf2.json"

    # import os

    print("Current working directory:", os.getcwd())

    if os.path.exists(task_path):
        run_solver(task_path)
    else:
        print(f"找不到任务文件: {task_path}")
        print("请确保05a7bcf2.json文件在当前目录")