"""
整合版ARC关系与模式系统

将ARCRelationshipLibraries与EnhancedPatternLibrary整合为一个统一系统，
实现从基础关系库到高级模式提取的完整功能链。
"""

from collections import defaultdict
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional, Union
import json
import os

class IntegratedARCPatternSystem:
    """
    整合版ARC关系与模式系统
    
    结合ARCRelationshipLibraries的关系库构建能力和
    EnhancedPatternLibrary的模式检测与管理功能
    """
    
    def __init__(self, debug=False, debug_print=None):
        """初始化整合系统"""
        self.debug = debug
        self.debug_print = debug_print or (lambda x: print(x) if debug else None)
        
        # ============= 关系库组件（原ARCRelationshipLibraries） =============
        # 对象属性库
        self.object_attributes = defaultdict(dict)  # obj_id -> 属性
        
        # 操作库
        self.operations = defaultdict(list)  # 操作类型 -> 操作列表
        self.operation_by_object = defaultdict(list)  # obj_id -> 操作列表
        
        # 属性-操作映射
        self.attribute_operation_maps = {}  # (属性类型,属性值) -> 操作列表
        
        # 对象索引库
        self.objects_by_pair = defaultdict(lambda: defaultdict(list))  # pair_id -> io_type -> obj_id列表
        self.objects_by_attribute = defaultdict(lambda: defaultdict(list))  # 属性类型 -> 属性值 -> obj_id列表
        
        # 相似对象分组
        self.similar_objects = defaultdict(list)  # 相似度哈希 -> obj_id列表
        
        # 属性相关性
        self.attribute_correlations = {}  # (属性1,属性2) -> 相关性分数
        
        # 统计数据
        self.statistics = {}
        self.total_objects = 0
        self.total_pairs = 0
        
        # ============= 模式库组件（原PatternLibrary） =============
        # 内置模式库
        self.built_in_patterns = {
            "spatial": self._get_built_in_spatial_patterns(),
            "color": self._get_built_in_color_patterns(),
            "shape": self._get_built_in_shape_patterns(),
            "transformation": self._get_built_in_transformation_patterns()
        }
        
        # 用户定义的模式库
        self.user_patterns = defaultdict(list)
        
        # 提取的模式
        self.extracted_patterns = defaultdict(list)
        
        # 复合规则
        self.composite_rules = []
        
        # 模式检测统计
        self.pattern_detection_stats = {}
    
    # ============= 关系库构建方法 =============
    
    def build_libraries_from_data(self, mapping_rules, all_objects):
        """
        从分析结果构建所有关系库
        
        Args:
            mapping_rules: 映射规则列表
            all_objects: 所有对象信息
        """
        if self.debug:
            self.debug_print("开始构建多维度关系库...")

        self.total_pairs = len(mapping_rules)

        # 1. 首先处理所有对象，建立基础属性库
        self._build_object_attribute_libraries(all_objects)

        # 2. 处理所有映射规则，建立操作库
        self._build_operation_libraries(mapping_rules)

        # 3. 建立属性-操作关系库
        self._build_attribute_operation_maps()

        # 4. 建立相似对象分组
        self._group_similar_objects()

        # 5. 分析属性相关性
        self._analyze_attribute_correlations()

        # 6. 计算统计数据
        self._calculate_statistics()

        if self.debug:
            self._log_library_statistics()
    
    def _build_object_attribute_libraries(self, all_objects):
        """构建对象属性库"""
        valid_io_types = ['input', 'output', 'diff_in', 'diff_out']
        for io_type in valid_io_types:
            if io_type in all_objects:
                for pair_id, obj_infos in all_objects.get(io_type, []):
                    if self.debug and len(obj_infos) > 0:
                        self.debug_print(f"处理 {io_type} 类型的对象，pair_id={pair_id}，对象数量={len(obj_infos)}")
                    for obj_info in obj_infos:
                        self._process_single_object(pair_id, io_type, obj_info)

    def _build_operation_libraries(self, mapping_rules):
        """构建操作库"""
        # 实现构建操作库的逻辑
        pass
        
    def _build_attribute_operation_maps(self):
        """构建属性-操作关系库"""
        # 实现构建属性-操作关系库的逻辑
        pass
        
    def _process_single_object(self, pair_id, io_type, obj_info):
        """
        处理单个对象，更新相关库
        
        Args:
            pair_id: 数据对ID
            io_type: 输入或输出类型
            obj_info: 对象信息
        """
        # 获取对象ID和基本属性
        obj_id = obj_info.obj_id
        self.total_objects += 1

        if io_type not in self.objects_by_pair[pair_id]:
            if self.debug:
                self.debug_print(f"警告: 创建对象索引库中未预期的io_type: {io_type}")
            self.objects_by_pair[pair_id][io_type] = []

        # 添加到对象索引库
        self.objects_by_pair[pair_id][io_type].append(obj_id)
        
        # 提取并存储对象属性
        self._extract_and_store_object_attributes(obj_info, obj_id)
    
    def _extract_and_store_object_attributes(self, obj_info, obj_id):
        """提取并存储对象属性"""
        # 实现提取和存储对象属性的逻辑
        pass
        
    # ... 其他原ARCRelationshipLibraries方法 ...
    
    # ============= 模式库方法 =============
    
    def _get_built_in_spatial_patterns(self):
        """获取内置的空间关系模式"""
        return [
            {
                "id": "surrounded_pattern",
                "name": "包围模式",
                "description": "一种颜色的对象被另一种颜色的对象包围",
                "detector": self._detect_surrounded_pattern,
                "parameters": {"min_surrounding_ratio": 0.7}
            },
            # ... 其他空间模式 ...
        ]
    
    def _get_built_in_color_patterns(self):
        """获取内置的颜色关系模式"""
        return [
            {
                "id": "color_pair_pattern",
                "name": "颜色对模式",
                "description": "两种特定颜色总是成对出现",
                "detector": self._detect_color_pair_pattern,
                "parameters": {"min_occurrence": 2}
            },
            # ... 其他颜色模式 ...
        ]
    
    def _get_built_in_shape_patterns(self):
        """获取内置的形状关系模式"""
        return [
            {
                "id": "shape_container_pattern",
                "name": "形状容器模式",
                "description": "一个形状包含另一个形状",
                "detector": self._detect_shape_container_pattern,
                "parameters": {"container_threshold": 0.8}
            },
            # ... 其他形状模式 ...
        ]
    
    def _get_built_in_transformation_patterns(self):
        """获取内置的变换模式"""
        return [
            {
                "id": "growth_pattern",
                "name": "增长模式",
                "description": "对象按特定方向扩展",
                "detector": self._detect_growth_pattern,
                "parameters": {"min_growth_factor": 1.5}
            },
            {
                "id": "box_formation_pattern",
                "name": "围盒形成模式",
                "description": "对象周围形成围盒",
                "detector": self._detect_box_formation_pattern,
                "parameters": {"box_completion_threshold": 0.8}
            },
            # ... 其他变换模式 ...
        ]
    
    # 模式检测器方法
    def _detect_surrounded_pattern(self, data, params):
        """检测包围模式"""
        # 现在可以直接使用关系库数据进行检测
        # 无需额外传递数据
        patterns = []
        # ... 实现检测逻辑 ...
        return patterns
    
    # ... 其他模式检测方法 ...
    
    def add_user_pattern(self, category, pattern):
        """添加用户定义的模式"""
        if "id" not in pattern or "detector" not in pattern:
            raise ValueError("模式必须包含id和detector")
        
        self.user_patterns[category].append(pattern)
        return True
    
    # ============= 模式提取与规则生成 =============
    
    def extract_patterns_and_rules(self):
        """
        从关系库提取模式并生成规则
        
        Returns:
            提取的模式和生成的规则
        """
        if self.debug:
            self.debug_print("开始从关系库提取模式...")
        
        # 清空之前的结果
        self.extracted_patterns = defaultdict(list)
        self.composite_rules = []
        
        # 1. 提取模式
        self._extract_all_patterns()
        
        # 2. 生成规则
        self._integrate_patterns_into_rules()
        
        # 3. 排序规则
        self._sort_composite_rules()
        
        return {
            "extracted_patterns": self.extracted_patterns,
            "composite_rules": self.composite_rules
        }
    
    def _extract_all_patterns(self):
        """提取所有类型的模式"""
        # 1. 提取空间关系模式
        self._extract_spatial_patterns()
        
        # 2. 提取颜色关系模式
        self._extract_color_patterns()
        
        # 3. 提取形状关系模式
        self._extract_shape_patterns()
        
        # 4. 提取变换模式
        self._extract_transformation_patterns()
        
        # 5. 提取跨对模式（原find_patterns_across_pairs）
        self._extract_cross_pair_patterns()
    
    def _extract_spatial_patterns(self):
        """提取空间关系模式"""
        spatial_patterns = self.built_in_patterns["spatial"] + self.user_patterns.get("spatial", [])
        
        for pattern_def in spatial_patterns:
            try:
                detector = pattern_def.get("detector")
                params = pattern_def.get("parameters", {})
                
                if detector:
                    # 由于已经有了关系库，可以直接调用检测器，无需额外参数
                    detected_patterns = detector(params=params)
                    
                    # 处理检测到的模式
                    if detected_patterns:
                        for pattern in detected_patterns:
                            pattern["pattern_type"] = pattern_def["id"]
                            pattern["pattern_name"] = pattern_def["name"]
                            pattern["pattern_category"] = "spatial"
                        
                        self.extracted_patterns["spatial"].extend(detected_patterns)
                        
                        if self.debug:
                            self.debug_print(f"发现 {len(detected_patterns)} 个 {pattern_def['name']} 模式")
            
            except Exception as e:
                if self.debug:
                    self.debug_print(f"提取 {pattern_def.get('name')} 模式时出错: {e}")
    
    # 类似地实现其他模式提取方法...
    
    def _extract_cross_pair_patterns(self):
        """提取跨对模式（原find_patterns_across_pairs的功能）"""
        # 实现原find_patterns_across_pairs的逻辑
        cross_pair_patterns = []
        # ... 实现跨对模式提取逻辑 ...
        
        self.extracted_patterns["cross_pair"] = cross_pair_patterns
    
    def _integrate_patterns_into_rules(self):
        """整合提取的模式，生成复合规则"""
        # 1. 整合基本规则
        self._integrate_basic_rules()
        
        # 2. 整合高级模式规则
        categories = ["spatial", "color", "shape", "transformation", "cross_pair"]
        for category in categories:
            for pattern in self.extracted_patterns.get(category, []):
                self._create_rule_from_pattern(pattern, category)
    
    def _create_rule_from_pattern(self, pattern, category):
        """根据模式创建规则"""
        pattern_type = pattern.get("pattern_type")
        
        # 调用相应的规则创建方法
        method_name = f"_create_{pattern_type}_rule"
        if hasattr(self, method_name):
            getattr(self, method_name)(pattern)
        else:
            # 创建通用规则
            self._create_generic_rule(pattern, category)
    
    def _create_generic_rule(self, pattern, category):
        """创建通用规则"""
        rule = {
            'rule_type': f'{category}_pattern_rule',
            'pattern_type': pattern.get('pattern_type'),
            'pattern_name': pattern.get('pattern_name'),
            'conditions': pattern.get('conditions', {}),
            'effects': pattern.get('effects', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.7),
            'description': f"基于{pattern.get('pattern_name')}的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    # ============= 模式应用与规则执行 =============
    
    def apply_rules_to_grid(self, input_grid, test_features=None):
        """
        应用提取的规则到输入网格
        
        Args:
            input_grid: 输入网格
            test_features: 测试输入的特征信息
            
        Returns:
            转换后的输出网格
        """
        # 如果没有预先提取模式和规则，则执行提取
        if not self.composite_rules:
            self.extract_patterns_and_rules()
        
        # 创建输出网格副本
        output_grid = [list(row) for row in input_grid]
        
        # 如果有测试特征，则匹配规则
        if test_features:
            matched_rules = self._match_rules_to_test(test_features)
        else:
            # 使用所有规则，按置信度排序
            matched_rules = sorted(
                self.composite_rules, 
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )
        
        # 应用匹配的规则
        for rule in matched_rules:
            self._apply_single_rule(rule, output_grid)
        
        return output_grid
    
    def _match_rules_to_test(self, test_features):
        """将规则与测试特征匹配"""
        matched_rules = []
        
        # ... 实现规则匹配逻辑 ...
        
        return matched_rules
    
    def _apply_single_rule(self, rule, grid):
        """应用单条规则到网格"""
        rule_type = rule.get('rule_type')
        
        # 根据规则类型调用相应的应用方法
        if rule_type == 'spatial_pattern_rule':
            self._apply_spatial_rule(rule, grid)
        elif rule_type == 'color_pattern_rule':
            self._apply_color_rule(rule, grid)
        elif rule_type == 'shape_pattern_rule':
            self._apply_shape_rule(rule, grid)
        elif rule_type == 'transformation_rule':
            self._apply_transformation_rule(rule, grid)
        elif rule_type == 'composite_rule':
            self._apply_composite_rule(rule, grid)
        
        return grid
    
    # 各类规则应用方法...
    
    # ============= 实用工具方法 =============
    
    def _log_library_statistics(self):
        """打印库统计信息"""
        self.debug_print("关系库统计信息:")
        self.debug_print(f"  - 总对象数: {self.total_objects}")
        self.debug_print(f"  - 总数据对数: {self.total_pairs}")
        self.debug_print(f"  - 属性类型数: {len(self.objects_by_attribute)}")
        
        self.debug_print("\n模式库统计信息:")
        for category, patterns in self.extracted_patterns.items():
            self.debug_print(f"  - {category} 类别: {len(patterns)} 个模式")
        
        self.debug_print(f"\n总复合规则数: {len(self.composite_rules)}")
    
    def export_to_json(self, file_path):
        """导出库状态到JSON文件"""
        # 准备可序列化的数据
        export_data = {
            "statistics": self.statistics,
            "total_objects": self.total_objects,
            "total_pairs": self.total_pairs,
            "extracted_patterns": {},
            "composite_rules": []
        }
        
        # 处理提取的模式
        for category, patterns in self.extracted_patterns.items():
            export_data["extracted_patterns"][category] = []
            for pattern in patterns:
                # 移除不可序列化的检测器函数
                serializable_pattern = {k: v for k, v in pattern.items() if k != 'detector'}
                export_data["extracted_patterns"][category].append(serializable_pattern)
        
        # 处理复合规则
        for rule in self.composite_rules:
            export_data["composite_rules"].append(rule)
        
        # 保存到文件
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_from_json(self, file_path):
        """从JSON文件导入库状态"""
        with open(file_path, 'r') as f:
            import_data = json.load(f)
        
        # 更新统计信息
        self.statistics = import_data.get("statistics", {})
        self.total_objects = import_data.get("total_objects", 0)
        self.total_pairs = import_data.get("total_pairs", 0)
        
        # 更新提取的模式
        self.extracted_patterns = defaultdict(list)
        for category, patterns in import_data.get("extracted_patterns", {}).items():
            self.extracted_patterns[category] = patterns
        
        # 更新复合规则
        self.composite_rules = import_data.get("composite_rules", [])