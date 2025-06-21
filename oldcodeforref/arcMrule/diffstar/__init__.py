"""
diffstar子包 - 基于差异分析的ARC求解工具

此子包提供了使用差异分析方法解决ARC任务的各种工具。
"""

# from .arc_diff_analyzer import ARCDiffAnalyzer
# from .diffstar_arc_solver_weighted import WeightedARCDiffSolver
# from .diffstar_weighted_arc_analyzer import WeightedARCDiffAnalyzer

# 导入原始分析器
# from arcMrule.diffstar import ARCDiffAnalyzer

# 导入加权分析器 (原始版本)
# from arcMrule.diffstar import WeightedARCDiffAnalyzer

# 导入重构后的加权分析器
from arcMrule.diffstar.weighted_analyzer import WeightedARCDiffAnalyzer


__all__ = ['WeightedARCDiffAnalyzer']
# __all__ = ['ARCDiffAnalyzer', 'WeightedARCDiffSolver', 'WeightedARCDiffAnalyzer']