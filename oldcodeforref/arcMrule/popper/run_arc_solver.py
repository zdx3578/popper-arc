import os
import json
import matplotlib.pyplot as plt
from arc_solver_modular import ARCSolverModular
from grid_extension_plugin import GridExtensionPriorKnowledge

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

def run_solver(task_path, output_dir="./popper_files"):
    """运行ARC求解器"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建求解器实例
    solver = ARCSolverModular(debug=True)
    
    # 注册插件
    grid_plugin = GridExtensionPriorKnowledge()
    solver.register_plugin(grid_plugin)
    
    # 加载任务
    solver.load_task(task_path)
    
    # 保存Popper文件
    solver.save_popper_files(output_dir)
    
    # 尝试学习规则
    try:
        learned_rules = solver.learn_rules_with_popper(output_dir)
        print(f"学习到 {len(learned_rules)} 条规则")
    except Exception as e:
        print(f"学习规则时出错: {e}")
        learned_rules = []
    
    # 处理测试案例
    solutions = []
    for i, (input_grid, expected_output) in enumerate(solver.test_pairs):
        output_grid = solver.apply_learned_rules(input_grid, learned_rules)
        solutions.append((output_grid, expected_output))
        
        # 可视化并保存结果
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
            
            # 保存JSON格式的输出
            with open(os.path.join(output_dir, f"test_{i+1}_output.json"), 'w') as f:
                json.dump(output_grid, f)
                
        except ImportError:
            print("注意: 未能导入matplotlib，跳过可视化")
    
    # 评估结果
    correct_count = 0
    for i, (output_grid, expected_output) in enumerate(solutions):
        if solver._compare_grids(output_grid, expected_output):
            correct_count += 1
            print(f"测试用例 {i+1}: 正确")
        else:
            print(f"测试用例 {i+1}: 不正确")
    
    print(f"共 {len(solutions)} 个测试用例，正确率: {correct_count/len(solutions)*100:.1f}%")
    
    return solutions

if __name__ == "__main__":
    task_path = "05a7bcf2.json"  # 替换为实际路径
    
    if not os.path.exists(task_path):
        print(f"找不到任务文件: {task_path}")
        print("请提供正确的任务JSON文件路径")
    else:
        run_solver(task_path)