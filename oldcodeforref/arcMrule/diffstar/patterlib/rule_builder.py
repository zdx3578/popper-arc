"""
规则构建器 - 用于创建标准化的规则对象，确保所有规则具有一致的结构
"""

class RuleBuilder:
    """
    规则构建器，用于创建带有标准化字段的规则对象
    """

    def __init__(self, total_train_pairs=None):
        """
        初始化规则构建器

        Args:
            total_train_pairs: 训练数据对的总数量，用于判断规则是否覆盖所有训练数据
        """
        self.total_train_pairs = total_train_pairs

    def create_rule(self, rule_type, **fields):
        """
        创建通用规则，适用于所有规则类型

        Args:
            rule_type: 规则类型（如 'color_operation_rule', 'shape_operation_rule', 'conditional_rule', 等）
            **fields: 规则包含的所有字段，支持任意键值对

        特殊处理的字段:
            supporting_pairs: 支持此规则的数据对列表
            confidence: 规则的置信度
            description: 规则的描述

        返回:
            包含所有必要字段的规则字典
        """
        # 创建基础规则结构
        rule = {
            'rule_type': rule_type,
        }

        # 确保所有字段都被包含，即使没有提供也设置默认值
        rule['confidence'] = fields.pop('confidence', 0.0)
        rule['description'] = fields.pop('description', f"{rule_type} 规则")

        # 添加supporting_pairs信息
        supporting_pairs = fields.pop('supporting_pairs', [])
        rule['supporting_pairs'] = supporting_pairs
        rule['supporting_pairs_count'] = len(supporting_pairs)

        # 判断是否覆盖所有训练数据对
        if self.total_train_pairs is not None:
            rule['supporting_pairs_ifisallpair'] = (len(supporting_pairs) == self.total_train_pairs)
        else:
            rule['supporting_pairs_ifisallpair'] = False

        # 添加剩余所有字段，确保兼容性
        rule.update(fields)

        return rule

    # 以下是针对特定规则类型的快捷方法

    def create_global_rule(self, operation_type, target, supporting_pairs, **fields):
        """通用全局规则构建方法，适用于颜色和形状操作"""
        result = self.create_rule(
            f"global_{operation_type}_operation",
            operation_type=operation_type,
            target=target,
            operation=fields.pop('operation', ''),
            supporting_pairs=supporting_pairs,
            coverage=len(supporting_pairs),
            **fields
        )

        # 为了兼容性，同时设置特定的键
        if operation_type == 'color':
            result['color'] = target
        elif operation_type == 'shape':
            result['shape_hash'] = target

        return result

    def create_color_rule(self, color, operation, supporting_pairs, **fields):
        """颜色规则快捷方法 - 内部调用create_rule"""
        return self.create_global_rule(
            'color',
            color,
            supporting_pairs,
            operation=operation,
            **fields
        )

    def create_shape_rule(self, shape_hash, operation, supporting_pairs, **fields):
        """形状规则快捷方法 - 内部调用create_rule"""
        return self.create_global_rule(
            'shape',
            shape_hash,
            supporting_pairs,
            operation=operation,
            **fields
        )

    def create_conditional_rule(self, condition, effect, supporting_pairs, **fields):
        """条件规则快捷方法 - 内部调用create_rule"""
        return self.create_rule(
            'conditional_rule',
            condition=condition,
            effect=effect,
            supporting_pairs=supporting_pairs,
            **fields
        )

    def create_composite_rule(self, base_rule, conditional_rules, supporting_pairs, **fields):
        """复合规则快捷方法 - 内部调用create_rule"""
        return self.create_rule(
            'composite_rule',
            base_rule=base_rule,
            conditional_rules=conditional_rules,
            supporting_pairs=supporting_pairs,
            **fields
        )