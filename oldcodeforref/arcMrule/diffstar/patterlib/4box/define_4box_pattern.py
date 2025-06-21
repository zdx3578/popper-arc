# 在IntegratedARCPatternSystem类中添加该模式定义
def _get_built_in_spatial_patterns(self):
    """获取内置的空间关系模式"""
    patterns = [
        # 其他已有模式...
        
        # 添加4-Box模式
        {
            "id": "four_box_pattern",
            "name": "4-Box围绕模式",
            "description": "中心像素/对象被另一种颜色在上下左右四个方向完全包围",
            "detector": self._detect_four_box_pattern,
            "parameters": {
                "require_exact_match": True,  # 是否要求严格的四向包围
                "allow_diagonal": False,      # 不考虑对角线方向
                "boundary_counts": False      # 网格边界不算作包围
            }
        }
    ]
    return patterns