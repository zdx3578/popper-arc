



def determine_background_color(self, task_data):
    """
    分析所有训练数据确定背景色，只返回占比最大的背景色值

    Args:
        task_data: 任务数据

    Returns:
        背景色值(单个整数)，如果没有满足条件的背景色则返回None
    """
    from collections import defaultdict

    all_grids = []

    # 收集所有输入和输出网格
    for example in task_data['train']:
        all_grids.append(example['input'])
        all_grids.append(example['output'])

    # 统计所有网格中各颜色的总占比
    color_total_percentages = defaultdict(float)
    color_appearance_count = defaultdict(int)

    for grid in all_grids:
        if not grid:  # 跳过空网格
            continue

        total_pixels = len(grid) * len(grid[0])
        if total_pixels == 0:  # 避免除零错误
            continue

        color_counts = defaultdict(int)

        for row in grid:
            for cell in row:
                color_counts[cell] += 1

        # 计算每种颜色的百分比并累加
        for color, count in color_counts.items():
            percentage = (count / total_pixels * 100)
            color_total_percentages[color] += percentage
            color_appearance_count[color] += 1

    # 计算每种颜色的平均占比
    color_avg_percentages = {}
    for color, total_pct in color_total_percentages.items():
        color_avg_percentages[color] = total_pct / color_appearance_count[color]

    # 按平均占比排序颜色
    sorted_colors = sorted(
        color_avg_percentages.items(),
        key=lambda x: x[1],
        reverse=True
    )

    background_color = None

    # 只获取占比最大的颜色作为背景色
    if sorted_colors:
        max_color, max_percentage = sorted_colors[0]
        # 确保占比超过阈值
        if max_percentage >= self.diff_analyzer.pixel_threshold_pct:
            background_color = max_color
            if self.debug:
                print(f"确定全局背景色: {max_color} (占比: {max_percentage:.2f}%)")

    return background_color