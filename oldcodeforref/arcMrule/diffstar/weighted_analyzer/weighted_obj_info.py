"""
加权对象信息类

扩展基础ObjInfo类，添加权重相关功能。
"""

from arcMrule.diffstar.arc_diff_analyzer import ObjInfo


class WeightedObjInfo(ObjInfo):
    """增强型对象信息类，存储对象的所有相关信息、变换和权重"""

    def __init__(self, pair_id, in_or_out, obj, obj_params=None, grid_hw=None, background=0):
        """
        初始化对象信息

        Args:
            pair_id: 训练对ID
            in_or_out: 'in', 'out', 'diff_in', 'diff_out'等
            obj: 原始对象 (frozenset形式)
            obj_params: 对象参数 (univalued, diagonal, without_bg)
            grid_hw: 网格尺寸 [height, width]
            background: 背景值
        """
        # 调用父类初始化
        super().__init__(pair_id, in_or_out, obj, obj_params, grid_hw, background)

        # 新增：对象权重属性
        self.obj_weight = 0  # 初始权重为0，后续根据各种规则增加权重

    def to_dict(self):
        """转换为可序列化的字典表示"""
        result = super().to_dict()
        result["obj_weight"] = self.obj_weight  # 添加权重到字典表示
        return result

    def increase_weight(self, amount):
        """增加对象权重"""
        self.obj_weight += amount
        return self.obj_weight

    def set_weight(self, value):
        """设置对象权重"""
        self.obj_weight = value
        return self.obj_weight