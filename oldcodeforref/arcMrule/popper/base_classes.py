"""
ARC求解器的基础类定义
包含所有共享的类，避免循环导入
"""
import abc
from typing import Dict, List, Tuple, Any

class PriorKnowledgePlugin(abc.ABC):
    """先验知识插件接口"""

    @abc.abstractmethod
    def get_plugin_name(self) -> str:
        """获取插件名称"""
        pass

    @abc.abstractmethod
    def is_applicable(self, task_data: Dict) -> bool:
        """判断此先验知识是否适用于当前任务"""
        pass

    @abc.abstractmethod
    def generate_facts(self, pair_id: int, input_objects: List, output_objects: List) -> List[str]:
        """生成特定于任务的Popper事实"""
        pass

    @abc.abstractmethod
    def generate_positive_examples(self, pair_id: int) -> List[str]:
        """生成特定于任务的正例"""
        pass

    @abc.abstractmethod
    def generate_negative_examples(self, pair_id: int) -> List[str]:
        """生成特定于任务的负例"""
        pass

    def generate_bias(self) -> str:
        """生成特定于任务的Popper偏置"""
        return ""

    def apply_solution(self, input_grid, learned_rules=None):
        """应用插件特定的解决方案"""
        # 默认实现返回输入网格的副本
        return [row[:] for row in input_grid]