"""
ARC任务求解器主运行文件
用于解决05a7bcf2.json任务并生成Popper规则文件
"""
import os
import sys
import time
import json
import numpy as np
import traceback
from datetime import datetime

# 导入ARC求解器
from arc_solver import ARCSolver

def visualize_grid(grid, title=""):
    """简单可视化网格（控制台输出）"""
    print(f"\n{title}")

    # 颜色映射字符
    color_map_emoji = {
        0: "⬛",  # 背景
        1: "🟥",  # 红色
        2: "🟩",  # 绿色
        3: "🟦",  # 蓝色
        4: "🟨",  # 黄色
        5: "🟪",  # 紫色
        6: "🟦",  # 蓝色
        7: "⬜",  # 白色
        8: "🟫",  # 棕色
        9: "🟧",  # 橙色
    }

    color_map = {
        0: "⬛",   # 黑色
        1: "🟦",   # 蓝色
        2: "🟥",   # 红色
        3: "🟩",   # 绿色
        4: "🟨",   # 黄色
        5: "🟫",  # 灰色（白色中等方块代替）
        6: "🟪",   # 粉紫
        7: "🟠",   # 橙色
        8: "🔹",   # 浅蓝（大蓝菱形）
        9: "🔴",   # 酒红（复用红色方块）
    }


    for row in grid:
        print("".join(color_map.get(c, "⬜") for c in row))

def main():
    """主函数"""
    print("=" * 50)
    print("ARC任务求解器 - 05a7bcf2任务")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 设置任务路径
    task_path = "05a7bcf2.json"
    if not os.path.exists(task_path):
        print(f"错误: 找不到任务文件 {task_path}")
        print("请确保05a7bcf2.json文件在当前目录下")
        return 1

    # 创建输出目录
    output_dir = "arc_solver_output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 开始计时
        start_time = time.time()

        # 创建求解器实例
        print("\n创建ARC求解器...")
        solver = ARCSolver(debug=True)

        # 解析并解决任务
        print(f"\n解决任务: {task_path}")
        solutions = solver.solve_task(task_path)

        # 计算用时
        elapsed_time = time.time() - start_time
        print(f"\n任务处理完成, 用时: {elapsed_time:.2f}秒")

        # 输出结果
        if solutions:
            print(f"\n成功解决 {len(solutions)} 个测试用例")

            # 可视化每个解决方案
            for i, solution in enumerate(solutions):
                visualize_grid(solution, f"测试用例 {i+1} 解决方案")

                # 保存解决方案
                solution_path = f"{output_dir}/solution_{i+1}.json"
                with open(solution_path, 'w') as f:
                    json.dump(solution.tolist(), f)
                print(f"  解决方案已保存到: {solution_path}")
        else:
            print("\n未找到解决方案")

        # 检查Popper规则文件
        popper_dir = f"{output_dir}/05a7bcf2"
        if os.path.exists(f"{popper_dir}/bk.pl"):
            print(f"\nPopper配置文件已生成在: {popper_dir}")
            print("  - bk.pl: 背景知识文件")
            print("  - bias.pl: 偏置文件")
            print("  - pos.pl: 正例文件")
            print("  - neg.pl: 负例文件")

        return 0

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())