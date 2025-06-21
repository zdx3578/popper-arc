"""
工具函数模块

提供各种辅助函数，供其他模块调用。
"""

def get_obj_shape_hash(obj_info):
    """从对象信息中获取形状哈希值"""
    try:
        # 首先尝试从ID中提取哈希值
        if hasattr(obj_info, 'obj_id'):
            # 提取ID中的哈希部分
            id_parts = obj_info.obj_id.split('_')
            if len(id_parts) >= 3:
                return id_parts[2]  # 返回哈希部分

        # 如果从ID中获取失败，尝试传统方法
        if hasattr(obj_info, 'obj_000'):
            return hash(tuple(map(tuple, obj_info.obj_000)))
        return None
    except (TypeError, AttributeError):
        return None

def get_hashable_representation(obj_set):
    """
    将对象集合转换为可哈希的表示

    Args:
        obj_set: 对象集合（可以是frozenset或其他可迭代对象）

    Returns:
        可哈希的表示（元组）
    """
    sorted_elements = []
    for value, loc in obj_set:
        i, j = loc
        sorted_elements.append((value, i, j))

    return tuple(sorted(sorted_elements))


def create_position_object_map(obj_infos):
    """
    创建位置到对象的映射

    Args:
        obj_infos: 对象信息列表

    Returns:
        字典 {(row, col): [obj_info, ...]}
    """
    from collections import defaultdict
    pos_to_obj = defaultdict(list)

    for obj_info in obj_infos:
        for _, (i, j) in obj_info.original_obj:
            pos_to_obj[(i, j)].append(obj_info)

    return pos_to_obj


def are_shapes_similar(shape1, shape2, tolerance=0.8):
    """
    判断两个形状是否相似

    Args:
        shape1, shape2: 两个形状（像素集合）
        tolerance: 相似度阈值

    Returns:
        是否相似的布尔值
    """
    # 获取两个形状的像素位置
    pixels1 = {loc for _, loc in shape1}
    pixels2 = {loc for _, loc in shape2}

    # 计算交集大小
    intersection = pixels1.intersection(pixels2)

    # 计算相似度（交集占较小集合的比例）
    similarity = len(intersection) / min(len(pixels1), len(pixels2))

    return similarity >= tolerance


def transform_grid_to_dict(grid):
    """
    将网格转换为字典表示

    Args:
        grid: 输入网格

    Returns:
        {(i, j): value} 的字典表示
    """
    result = {}
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            value = grid[i][j]
            if value != 0:  # 忽略背景值
                result[(i, j)] = value

    return result


def transform_dict_to_grid(grid_dict, height, width, background=0):
    """
    将字典表示转换回网格

    Args:
        grid_dict: {(i, j): value} 的字典表示
        height, width: 网格尺寸
        background: 背景值

    Returns:
        二维网格
    """
    grid = [[background for _ in range(width)] for _ in range(height)]

    for (i, j), value in grid_dict.items():
        if 0 <= i < height and 0 <= j < width:
            grid[i][j] = value

    return grid


def apply_offset_to_grid(grid, dr, dc, height, width, background=0):
    """
    对网格应用位移

    Args:
        grid: 输入网格
        dr, dc: 行和列的位移
        height, width: 目标网格尺寸
        background: 背景值

    Returns:
        位移后的网格
    """
    # 转换为字典表示
    grid_dict = transform_grid_to_dict(grid)

    # 应用位移
    new_dict = {(i+dr, j+dc): value for (i, j), value in grid_dict.items()}

    # 转换回网格
    return transform_dict_to_grid(new_dict, height, width, background)