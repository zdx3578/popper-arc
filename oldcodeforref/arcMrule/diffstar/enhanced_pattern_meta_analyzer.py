"""
增强版模式元分析器

专注于整合全局规则和条件规则，并匹配测试数据中的形状和颜色。
"""

import itertools
from collections import defaultdict
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional

# from arcMrule.diffstar.patterlib.rule_builder import RuleBuilder
from arcMrule.diffstar.patterlib.rule_builder import RuleBuilder
from arcMrule.diffstar.patterlib.pattern_analysis_mixin import PatternAnalysisMixin

# class EnhancedPatternMetaAnalyzer:
class EnhancedPatternMetaAnalyzer(PatternAnalysisMixin):
    """
    增强版模式元分析器：整合全局和条件模式，匹配测试形状
    """

    def __init__(self, debug=False, debug_print=None, task=None):
        """初始化增强版模式元分析器"""
        self.debug = debug
        self.debug_print = debug_print
        self.task = task
        total_train_pairs = len(task['train'])
        self.rule_builder = RuleBuilder(total_train_pairs)

        # 存储基础模式
        self.base_patterns = []
        self.patterns_by_type = defaultdict(list)

        # 形状和颜色的映射库
        self.shape_hash_to_info = {}  # 形状哈希 -> 详细信息
        self.color_to_info = {}       # 颜色 -> 详细信息

        # 通用操作规则
        self.global_operation_rules = []

        # 条件规则
        self.conditional_rules = []

        # 复合规则
        self.composite_rules = []

        # 测试数据匹配规则
        self.test_matched_rules = []

    def process_patterns(self, patterns, test_instance=None):
        """
        处理模式并生成规则，可选择提供测试实例进行匹配

        Args:
            patterns: 从关系库中提取的基础模式
            test_instance: 可选的测试实例数据
        """
        self.base_patterns = patterns

        if self.debug:
            self.debug_print(f"开始处理 {len(patterns)} 个基础模式...")

        # 1. 按类型对模式进行分类
        self._classify_patterns()

        # 2. 提取形状和颜色信息
        self._extract_shape_color_info()

        # 3. 提取全局操作规则
        self._extract_global_operation_rules()

        # 4. 提取条件规则
        self._extract_conditional_rules()

        # 5. 整合全局和条件规则
        self._integrate_rules()

        # 6. 如果提供了测试实例，匹配测试数据中的形状和颜色
        if test_instance:
            self._match_rules_to_test_instance(test_instance)

        if self.debug:
            self.debug_print(f"分析完成，生成了:")
            self.debug_print(f"  - {len(self.global_operation_rules)} 条全局操作规则")
            self.debug_print(f"  - {len(self.conditional_rules)} 条条件规则")
            self.debug_print(f"  - {len(self.composite_rules)} 条复合规则")
            if test_instance:
                self.debug_print(f"  - {len(self.test_matched_rules)} 条与测试数据匹配的规则")

        return self._get_final_results()

    def _classify_patterns(self):
        """对模式按类型和子类型分类"""
        for pattern in self.base_patterns:
            pattern_type = pattern.get('type', 'unknown')
            self.patterns_by_type[pattern_type].append(pattern)

        if self.debug:
            self.debug_print("模式分类结果:")
            for ptype, patterns in self.patterns_by_type.items():
                self.debug_print(f"  - {ptype}: {len(patterns)} 个模式")

    def _extract_shape_color_info(self):
        """提取形状和颜色的详细信息"""
        # 处理形状操作模式
        for pattern in self.patterns_by_type.get('shape_operation_pattern', []):
            shape_hash = pattern.get('shape_hash')
            if shape_hash:
                if shape_hash not in self.shape_hash_to_info:
                    self.shape_hash_to_info[shape_hash] = {
                        'operations': defaultdict(list),
                        'supporting_pairs': set(pattern.get('supporting_pairs', [])),
                        'confidence': pattern.get('confidence', 0)
                    }

                operation = pattern.get('operation')
                if operation:
                    self.shape_hash_to_info[shape_hash]['operations'][operation].append(pattern)

        # 处理颜色操作模式
        for pattern in self.patterns_by_type.get('color_operation_pattern', []):
            color = pattern.get('color')
            if color is not None:  # 颜色可能是0
                if color not in self.color_to_info:
                    self.color_to_info[color] = {
                        'operations': defaultdict(list),
                        'supporting_pairs': set(pattern.get('supporting_pairs', [])),
                        'confidence': pattern.get('confidence', 0)
                    }

                operation = pattern.get('operation')
                if operation:
                    self.color_to_info[color]['operations'][operation].append(pattern)

    def _extract_global_operation_rules0(self):
        """提取全局操作规则，如'移除所有颜色为X的对象'"""
        # 处理颜色操作模式
        for color, info in self.color_to_info.items():
            for operation, patterns in info['operations'].items():
                if patterns:  # 确保有模式支持此操作
                    # 计算操作覆盖率
                    all_supporting_pairs = set()
                    for pattern in patterns:
                        all_supporting_pairs.update(pattern.get('supporting_pairs', []))

                    # 创建全局规则
                    rule = {
                        'rule_type': 'global_color_operation',
                        'color': color,
                        'operation': operation,
                        'supporting_pairs': list(all_supporting_pairs),
                        'coverage': len(all_supporting_pairs),
                        'patterns': patterns,
                        'confidence': max(p.get('confidence', 0) for p in patterns)
                    }

                    self.global_operation_rules.append(rule)

        # 按覆盖率排序
        self.global_operation_rules.sort(key=lambda x: (x['coverage'], x['confidence']), reverse=True)

    def _extract_global_operation_rules00(self):
        """提取全局操作规则，如'移除所有颜色为X的对象'"""
        # 处理颜色操作模式
        for color, info in self.color_to_info.items():
            for operation, patterns in info['operations'].items():
                if patterns:  # 确保有模式支持此操作
                    # 计算操作覆盖率
                    all_supporting_pairs = set()
                    for pattern in patterns:
                        all_supporting_pairs.update(pattern.get('supporting_pairs', []))

                    # 创建全局规则
                    rule = self.rule_builder.create_color_rule(
                        color=color,
                        operation=operation,
                        supporting_pairs=list(all_supporting_pairs),
                        patterns=patterns,
                        confidence=max(p.get('confidence', 0) for p in patterns),
                        description=f"对颜色为{color}的所有对象 被执行{operation}操作"
                    )

                    self.global_operation_rules.append(rule)

        # 按覆盖率排序
        self.global_operation_rules.sort(key=lambda x: (x['coverage'], x['confidence']), reverse=True)

    def _extract_global_operation_rules(self):
        """提取全局操作规则，如'移除所有颜色为X的对象'"""
        # 处理颜色操作模式
        for color, info in self.color_to_info.items():
            for operation, patterns in info['operations'].items():
                if patterns:  # 确保有模式支持此操作
                    # 计算操作覆盖率
                    all_supporting_pairs = set()
                    for pattern in patterns:
                        all_supporting_pairs.update(pattern.get('supporting_pairs', []))

                    # 创建全局规则
                    rule = self.rule_builder.create_color_rule(
                        color=color,
                        operation=operation,
                        supporting_pairs=list(all_supporting_pairs),
                        patterns=patterns,
                        confidence=max(p.get('confidence', 0) for p in patterns),
                        description=f"对颜色为{color}的所有对象 被执行{operation}操作"
                    )

                    # 如果是添加操作并且规则适用于所有训练样例，分析背后的模式
                    if operation == 'added' and rule.get('supporting_pairs_ifisallpair', False):
                        # 分析添加操作背后可能的模式
                        underlying_pattern = self._analyze_underlying_pattern_for_addition(color, rule)
                        if underlying_pattern:
                            rule['underlying_pattern'] = underlying_pattern
                            rule['description'] = f"{rule['description']} (基于{underlying_pattern['pattern_type']}模式)"
                            if self.debug:
                                self.debug_print(f"发现颜色{color}的添加操作可能基于{underlying_pattern['pattern_type']}模式")

                    self.global_operation_rules.append(rule)

        # 按覆盖率排序
        self.global_operation_rules.sort(key=lambda x: (x['coverage'], x['confidence']), reverse=True)




    def _extract_conditional_rules(self):
        """提取条件规则，如'当移除形状X时，将颜色Y变为Z'"""
        # 处理条件模式
        for pattern in self.patterns_by_type.get('conditional_pattern', []):
            subtype = pattern.get('subtype')
            if subtype == 'removal_color_change':
                condition = pattern.get('condition', {})
                effect = pattern.get('effect', {})

                if 'shape_hash' in condition and 'color_change' in effect:
                    shape_hash = condition.get('shape_hash')
                    from_color = effect['color_change'].get('from_color')
                    to_color = effect['color_change'].get('to_color')

                    if shape_hash and from_color is not None and to_color is not None:
                        rule = self.rule_builder.create_conditional_rule(
                            condition={
                                'shape_hash': shape_hash,
                                'operation': 'remove'
                            },
                            effect={
                                'from_color': from_color,
                                'to_color': to_color
                            },
                            supporting_pairs=pattern.get('supporting_pairs', []),
                            confidence=pattern.get('confidence', 0),
                            description=pattern.get('description', f"当移除形状{shape_hash}时，将颜色{from_color}变为{to_color}")
                        )

                        self.conditional_rules.append(rule)

        # 按支持对集合的大小和置信度排序
        self.conditional_rules.sort(
            key=lambda x: (len(x.get('supporting_pairs', [])), x.get('confidence', 0)),
            reverse=True
        )

    def _extract_conditional_rules0(self):
        """提取条件规则，如'当移除形状X时，将颜色Y变为Z'"""
        # 处理条件模式
        for pattern in self.patterns_by_type.get('conditional_pattern', []):
            subtype = pattern.get('subtype')
            if subtype == 'removal_color_change':
                condition = pattern.get('condition', {})
                effect = pattern.get('effect', {})

                if 'shape_hash' in condition and 'color_change' in effect:
                    shape_hash = condition.get('shape_hash')
                    from_color = effect['color_change'].get('from_color')
                    to_color = effect['color_change'].get('to_color')

                    if shape_hash and from_color is not None and to_color is not None:
                        rule = {
                            'rule_type': 'conditional_color_change',
                            'condition': {
                                'shape_hash': shape_hash,
                                'operation': 'remove'
                            },
                            'effect': {
                                'from_color': from_color,
                                'to_color': to_color
                            },
                            'supporting_pairs': pattern.get('supporting_pairs', []),
                            'confidence': pattern.get('confidence', 0),
                            'description': pattern.get('description', '')
                        }

                        self.conditional_rules.append(rule)

        # 按支持对集合的大小和置信度排序
        self.conditional_rules.sort(
            key=lambda x: (len(x.get('supporting_pairs', [])), x.get('confidence', 0)),
            reverse=True
        )

    def _integrate_rules(self):
        """整合全局操作规则和条件规则，创建复合规则"""
        # 按颜色分组条件规则
        color_to_conditional_rules = defaultdict(list)
        for rule in self.conditional_rules:
            from_color = rule['effect']['from_color']
            color_to_conditional_rules[from_color].append(rule)

        # 查找可以组合的规则
        for global_rule in self.global_operation_rules:
            color = global_rule.get('color')
            operation = global_rule.get('operation')

            # 如果是移除操作，查找相关的条件规则
            if operation == 'removed' and color in color_to_conditional_rules:
                related_conditional_rules = color_to_conditional_rules[color]

                # 创建组合规则
                all_supporting_pairs = set(global_rule.get('supporting_pairs', []))
                for cond_rule in related_conditional_rules:
                    all_supporting_pairs.update(cond_rule.get('supporting_pairs', []))

                composite_rule = self.rule_builder.create_composite_rule(
                    base_rule={
                        'type': 'global_color_operation',
                        'color': color,
                        'operation': operation
                    },
                    conditional_rules=related_conditional_rules,
                    supporting_pairs=list(all_supporting_pairs),
                    confidence=global_rule.get('confidence', 0),
                    description=f"处理颜色为{color}的对象的综合规则",
                    coverage_score=len(all_supporting_pairs)  # 添加覆盖率分数
                )

                # 计算综合置信度
                if related_conditional_rules:
                    avg_cond_confidence = sum(r.get('confidence', 0) for r in related_conditional_rules) / len(related_conditional_rules)
                    composite_rule['confidence'] = (global_rule.get('confidence', 0) + avg_cond_confidence) / 2

                # 添加到复合规则列表
                self.composite_rules.append(composite_rule)

        # 按覆盖率和置信度排序
        self.composite_rules.sort(
            key=lambda x: (x.get('coverage_score', 0), x.get('confidence', 0)),
            reverse=True
        )

    def _integrate_rules0(self):
        """
        整合全局操作规则和条件规则，创建复合规则

        这一步骤特别重要，它解决了您提到的问题：整合通用的移除规则和基于形状的条件变化
        """
        # 按颜色分组条件规则
        color_to_conditional_rules = defaultdict(list)
        for rule in self.conditional_rules:
            from_color = rule['effect']['from_color']
            color_to_conditional_rules[from_color].append(rule)

        # 查找可以组合的规则
        for global_rule in self.global_operation_rules:
            color = global_rule.get('color')
            operation = global_rule.get('operation')

            # 如果是移除操作，查找相关的条件规则
            if operation == 'removed' and color in color_to_conditional_rules:
                related_conditional_rules = color_to_conditional_rules[color]

                # 创建组合规则
                composite_rule = {
                    'rule_type': 'composite_rule',
                    'base_rule': {
                        'type': 'global_color_operation',
                        'color': color,
                        'operation': operation
                    },
                    'conditional_rules': [],
                    'all_supporting_pairs': set(global_rule.get('supporting_pairs', [])),
                    'confidence': global_rule.get('confidence', 0),
                    'description': f"处理颜色为{color}的对象的综合规则"
                }

                # 添加相关的条件规则
                for cond_rule in related_conditional_rules:
                    composite_rule['conditional_rules'].append(cond_rule)
                    composite_rule['all_supporting_pairs'].update(cond_rule.get('supporting_pairs', []))

                # 计算综合置信度
                if composite_rule['conditional_rules']:
                    avg_cond_confidence = sum(r.get('confidence', 0) for r in composite_rule['conditional_rules']) / len(composite_rule['conditional_rules'])
                    composite_rule['confidence'] = (global_rule.get('confidence', 0) + avg_cond_confidence) / 2

                # 转换supporting_pairs为列表，以便序列化
                composite_rule['all_supporting_pairs'] = list(composite_rule['all_supporting_pairs'])

                # 计算覆盖率分数
                composite_rule['coverage_score'] = len(composite_rule['all_supporting_pairs'])

                # 添加到复合规则列表
                self.composite_rules.append(composite_rule)

        # 按覆盖率和置信度排序
        self.composite_rules.sort(
            key=lambda x: (x.get('coverage_score', 0), x.get('confidence', 0)),
            reverse=True
        )

    def _match_rules_to_test_instance(self, test_instance):
        """
        将规则与测试实例进行匹配

        Args:
            test_instance: 测试数据实例
        """
        test_shapes = self._extract_shapes_from_test(test_instance)
        test_colors = self._extract_colors_from_test(test_instance)

        if self.debug:
            self.debug_print(f"测试实例中发现 {len(test_shapes)} 个形状和 {len(test_colors)} 个颜色")

        # 匹配全局操作规则
        matched_global_rules = []
        for rule in self.global_operation_rules:
            if rule['color'] in test_colors:
                matched_global_rules.append({
                    'rule': rule,
                    'match_type': 'color_exact',
                    'match_score': 1.0,
                    'matched_elements': [rule['color']]
                })

        # 匹配条件规则
        matched_conditional_rules = []
        for rule in self.conditional_rules:
            shape_hash = rule['condition']['shape_hash']
            from_color = rule['effect']['from_color']

            # 检查形状和颜色是否都匹配
            if shape_hash in test_shapes and from_color in test_colors:
                matched_conditional_rules.append({
                    'rule': rule,
                    'match_type': 'shape_and_color_exact',
                    'match_score': 1.0,
                    'matched_elements': {
                        'shape': shape_hash,
                        'color': from_color
                    }
                })
            # 只有形状匹配
            elif shape_hash in test_shapes:
                matched_conditional_rules.append({
                    'rule': rule,
                    'match_type': 'shape_exact',
                    'match_score': 0.7,
                    'matched_elements': {
                        'shape': shape_hash
                    }
                })
            # 只有颜色匹配
            elif from_color in test_colors:
                matched_conditional_rules.append({
                    'rule': rule,
                    'match_type': 'color_exact',
                    'match_score': 0.5,
                    'matched_elements': {
                        'color': from_color
                    }
                })

        # 匹配复合规则
        matched_composite_rules = []
        for rule in self.composite_rules:
            base_color = rule['base_rule']['color']
            matched_conditions = []

            # 检查基础颜色是否匹配
            if base_color in test_colors:
                # 检查条件规则
                for cond_rule in rule['conditional_rules']:
                    shape_hash = cond_rule['condition']['shape_hash']
                    if shape_hash in test_shapes:
                        matched_conditions.append(cond_rule)

                # 如果基础颜色匹配且至少有一个条件匹配
                if matched_conditions:
                    matched_composite_rules.append({
                        'rule': rule,
                        'match_type': 'composite_exact',
                        'match_score': 1.0,
                        'matched_elements': {
                            'base_color': base_color,
                            'matched_conditions': len(matched_conditions),
                            'total_conditions': len(rule['conditional_rules'])
                        }
                    })
                else:
                    # 只有基础颜色匹配
                    matched_composite_rules.append({
                        'rule': rule,
                        'match_type': 'base_color_only',
                        'match_score': 0.6,
                        'matched_elements': {
                            'base_color': base_color
                        }
                    })

        # 合并所有匹配并排序
        all_matches = []
        all_matches.extend(matched_global_rules)
        all_matches.extend(matched_conditional_rules)
        all_matches.extend(matched_composite_rules)

        # 按匹配分数排序
        all_matches.sort(key=lambda x: x['match_score'], reverse=True)

        self.test_matched_rules = all_matches

    def _extract_shapes_from_test(self, test_instance):
        """
        从测试实例中提取形状

        Args:
            test_instance: 测试数据实例

        Returns:
            提取的形状哈希集合
        """
        # 这里是一个简化实现，实际应用中需根据测试实例的具体结构提取
        # 假设test_instance包含一个shapes字段
        shapes = []
        if isinstance(test_instance, dict) and 'shapes' in test_instance:
            shapes = test_instance['shapes']
        elif hasattr(test_instance, 'shapes'):
            shapes = test_instance.shapes

        # 如果没有明确的形状信息，使用所有已知形状作为回退
        if not shapes:
            shapes = list(self.shape_hash_to_info.keys())

        return set(shapes)

    def _extract_colors_from_test(self, test_instance):
        """
        从测试实例中提取颜色

        Args:
            test_instance: 测试数据实例

        Returns:
            提取的颜色集合
        """
        # 这里是一个简化实现，实际应用中需根据测试实例的具体结构提取
        # 假设test_instance包含一个colors字段
        colors = []
        if isinstance(test_instance, dict) and 'colors' in test_instance:
            colors = test_instance['colors']
        elif hasattr(test_instance, 'colors'):
            colors = test_instance.colors

        # 如果没有明确的颜色信息，使用所有已知颜色作为回退
        if not colors:
            colors = list(self.color_to_info.keys())

        return set(colors)

    def _get_final_results(self):
        """
        生成最终结果

        Returns:
            包含所有规则和匹配的结果字典
        """
        results = {
            'global_rules': self.global_operation_rules,
            'conditional_rules': self.conditional_rules,
            'composite_rules': self.composite_rules
        }
        #! plan for test
        if self.test_matched_rules:
            results['test_matched_rules'] = self.test_matched_rules

            # 添加推荐的执行计划
            execution_plan = self._generate_execution_plan()
            if execution_plan:
                results['recommended_execution_plan'] = execution_plan

        return results

    def _generate_execution_plan(self):
        """
        生成推荐的规则执行计划

        Returns:
            执行计划列表
        """
        # 如果没有与测试匹配的规则，无法生成执行计划
        if not self.test_matched_rules:
            return []

        execution_plan = []

        # 首先添加匹配度最高的复合规则
        composite_matches = [m for m in self.test_matched_rules
                           if m['rule']['rule_type'] == 'composite_rule'
                           and m['match_score'] > 0.7]

        for match in composite_matches:
            rule = match['rule']
            execution_plan.append({
                'step_type': 'apply_composite_rule',
                'rule': rule,
                'match_info': match,
                'priority': match['match_score'] * 100
            })

        # 然后添加未被复合规则覆盖的条件规则
        covered_conditionals = set()
        for match in composite_matches:
            for cond_rule in match['rule'].get('conditional_rules', []):
                covered_conditionals.add(id(cond_rule))

        conditional_matches = [m for m in self.test_matched_rules
                              if 'rule' in m and
                              m['rule'].get('rule_type') == 'conditional_rule' and
                              id(m['rule']) not in covered_conditionals and
                              m['match_score'] > 0.5]

        for match in conditional_matches:
            execution_plan.append({
                'step_type': 'apply_conditional_rule',
                'rule': match['rule'],
                'match_info': match,
                'priority': match['match_score'] * 80
            })

        # 最后添加未被其他规则覆盖的全局操作规则
        covered_globals = set()
        for match in composite_matches:
            base_rule = match['rule'].get('base_rule', {})
            if base_rule.get('type') == 'global_color_operation':
                color = base_rule.get('color')
                operation = base_rule.get('operation')
                covered_globals.add((color, operation))

        global_matches = [m for m in self.test_matched_rules
                         if m['rule'].get('rule_type') == 'global_color_operation' and
                         (m['rule'].get('color'), m['rule'].get('operation')) not in covered_globals and
                         m['match_score'] > 0.5]

        for match in global_matches:
            execution_plan.append({
                'step_type': 'apply_global_rule',
                'rule': match['rule'],
                'match_info': match,
                'priority': match['match_score'] * 60
            })

        # 按优先级排序
        execution_plan.sort(key=lambda x: x['priority'], reverse=True)

        return execution_plan