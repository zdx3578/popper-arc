"""
增强型模式提取与集成系统

能够识别多种类型的常见模式并维护动态模式库
"""

from collections import defaultdict
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional, Union
import json
import os

class PatternLibrary:
    """模式库：存储和管理已识别的模式"""
    
    def __init__(self, library_path=None):
        # 内置模式库
        self.built_in_patterns = {
            "spatial": self._get_built_in_spatial_patterns(),
            "color": self._get_built_in_color_patterns(),
            "shape": self._get_built_in_shape_patterns(),
            "transformation": self._get_built_in_transformation_patterns()
        }
        
        # 用户定义的模式库
        self.user_patterns = defaultdict(list)
        
        # 从文件加载用户模式库
        if library_path and os.path.exists(library_path):
            self.load_from_file(library_path)
    
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
            {
                "id": "alignment_pattern",
                "name": "对齐模式",
                "description": "对象沿水平或垂直方向对齐",
                "detector": self._detect_alignment_pattern,
                "parameters": {"alignment_threshold": 2}
            },
            {
                "id": "proximity_pattern", 
                "name": "邻近模式",
                "description": "特定颜色的对象总是出现在另一种颜色的对象附近",
                "detector": self._detect_proximity_pattern,
                "parameters": {"proximity_threshold": 3}
            }
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
            {
                "id": "color_group_pattern",
                "name": "颜色组模式",
                "description": "多种颜色总是在一起出现",
                "detector": self._detect_color_group_pattern,
                "parameters": {"min_colors": 3, "min_occurrence": 2}
            }
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
            {
                "id": "shape_symmetry_pattern",
                "name": "形状对称模式",
                "description": "形状具有水平或垂直对称性",
                "detector": self._detect_shape_symmetry_pattern,
                "parameters": {"symmetry_threshold": 0.9}
            }
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
                "id": "replacement_pattern",
                "name": "替换模式",
                "description": "特定颜色或形状的对象被另一种替换",
                "detector": self._detect_replacement_pattern,
                "parameters": {"replacement_threshold": 0.7}
            },
            {
                "id": "box_formation_pattern",
                "name": "围盒形成模式",
                "description": "对象周围形成围盒",
                "detector": self._detect_box_formation_pattern,
                "parameters": {"box_completion_threshold": 0.8}
            }
        ]

    # 模式检测器方法 - 这些将是实际的模式检测算法
    def _detect_surrounded_pattern(self, objects_data, params):
        """检测包围模式"""
        # 具体实现略
        return []
    
    def _detect_alignment_pattern(self, objects_data, params):
        """检测对齐模式"""
        # 具体实现略
        return []
    
    def _detect_proximity_pattern(self, objects_data, params):
        """检测邻近模式"""
        # 具体实现略
        return []
    
    def _detect_color_pair_pattern(self, objects_data, params):
        """检测颜色对模式"""
        # 具体实现略
        return []
        
    def _detect_color_group_pattern(self, objects_data, params):
        """检测颜色组模式"""
        # 具体实现略
        return []
    
    def _detect_shape_container_pattern(self, objects_data, params):
        """检测形状容器模式"""
        # 具体实现略
        return []
        
    def _detect_shape_symmetry_pattern(self, objects_data, params):
        """检测形状对称模式"""
        # 具体实现略
        return []
        
    def _detect_growth_pattern(self, objects_data, params):
        """检测增长模式"""
        # 具体实现略
        return []
        
    def _detect_replacement_pattern(self, objects_data, params):
        """检测替换模式"""
        # 具体实现略
        return []
        
    def _detect_box_formation_pattern(self, objects_data, params):
        """检测围盒形成模式"""
        # 具体实现略
        return []
    
    def add_user_pattern(self, category, pattern):
        """添加用户定义的模式"""
        if "id" not in pattern or "detector" not in pattern:
            raise ValueError("模式必须包含id和detector")
        
        self.user_patterns[category].append(pattern)
        return True
    
    def get_all_patterns(self):
        """获取所有可用的模式"""
        all_patterns = {}
        
        # 合并内置模式和用户模式
        for category in set(list(self.built_in_patterns.keys()) + list(self.user_patterns.keys())):
            all_patterns[category] = []
            
            if category in self.built_in_patterns:
                all_patterns[category].extend(self.built_in_patterns[category])
            
            if category in self.user_patterns:
                all_patterns[category].extend(self.user_patterns[category])
        
        return all_patterns
    
    def get_patterns_by_category(self, category):
        """获取特定类别的所有模式"""
        patterns = []
        
        if category in self.built_in_patterns:
            patterns.extend(self.built_in_patterns[category])
        
        if category in self.user_patterns:
            patterns.extend(self.user_patterns[category])
        
        return patterns
    
    def save_to_file(self, file_path):
        """将用户模式保存到文件"""
        # 将方法对象转换为名称（无法直接序列化函数对象）
        serializable_patterns = {}
        
        for category, patterns in self.user_patterns.items():
            serializable_patterns[category] = []
            for pattern in patterns:
                serializable_pattern = pattern.copy()
                if "detector" in serializable_pattern:
                    serializable_pattern["detector"] = serializable_pattern["detector"].__name__
                serializable_patterns[category].append(serializable_pattern)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_patterns, f, indent=2)
    
    def load_from_file(self, file_path):
        """从文件加载用户模式"""
        with open(file_path, 'r') as f:
            loaded_patterns = json.load(f)
        
        # 需要将检测器名称转换回方法引用
        for category, patterns in loaded_patterns.items():
            for pattern in patterns:
                if "detector" in pattern and isinstance(pattern["detector"], str):
                    detector_name = pattern["detector"]
                    if hasattr(self, detector_name):
                        pattern["detector"] = getattr(self, detector_name)
                    else:
                        # 如果找不到检测器方法，使用空检测器
                        pattern["detector"] = lambda x, y: []
            
            self.user_patterns[category] = patterns


class EnhancedPatternExtractor:
    """增强型模式提取器：识别和集成各种类型的模式"""
    
    def __init__(self, pattern_library=None, debug=False, debug_print=None):
        """初始化增强型模式提取器"""
        self.debug = debug
        self.debug_print = debug_print or (lambda x: print(x) if debug else None)
        
        # 使用提供的模式库或创建新的库
        self.pattern_library = pattern_library or PatternLibrary()
        
        # 存储提取的模式
        self.extracted_patterns = defaultdict(list)
        
        # 复合规则
        self.composite_rules = []
        
        # 初始化数据结构
        self.objects_by_color = defaultdict(list)
        self.objects_by_shape = defaultdict(list)
        self.objects_by_position = defaultdict(list)
        self.shape_to_colors = defaultdict(set)
        self.color_to_shapes = defaultdict(set)
    
    def process_objects_data(self, objects_data):
        """
        处理对象数据，提取各种模式
        
        Args:
            objects_data: 包含所有对象信息的数据结构
        """
        # 预处理对象数据
        self._preprocess_objects(objects_data)
        
        # 提取各类模式
        self._extract_spatial_patterns()
        self._extract_color_patterns()
        self._extract_shape_patterns()
        self._extract_transformation_patterns()
        
        # 整合提取的模式，生成复合规则
        self._integrate_patterns_into_rules()
        
        return {
            "extracted_patterns": self.extracted_patterns,
            "composite_rules": self.composite_rules
        }
    
    def _preprocess_objects(self, objects_data):
        """预处理对象数据，建立索引"""
        # 假设objects_data是一个字典，包含输入和输出对象
        # 例如：{'input': {...}, 'output': {...}}
        
        for io_type in ['input', 'output']:
            if io_type not in objects_data:
                continue
                
            for pair_id, pair_objects in objects_data[io_type].items():
                for obj_id, obj_info in pair_objects.items():
                    # 按颜色索引
                    color = obj_info.get('color')
                    if color is not None:
                        self.objects_by_color[color].append((pair_id, io_type, obj_id))
                    
                    # 按形状索引
                    shape_hash = obj_info.get('shape_hash')
                    if shape_hash:
                        self.objects_by_shape[shape_hash].append((pair_id, io_type, obj_id))
                        
                        # 建立形状-颜色映射
                        if color is not None:
                            self.shape_to_colors[shape_hash].add(color)
                            self.color_to_shapes[color].add(shape_hash)
                    
                    # 按位置索引（网格划分，例如3x3网格）
                    if 'position' in obj_info:
                        x, y = obj_info['position']
                        # 假设网格大小是10x10，分为3x3区域
                        grid_x = min(2, x // 4)
                        grid_y = min(2, y // 4)
                        position_key = f"{grid_x}_{grid_y}"
                        self.objects_by_position[position_key].append((pair_id, io_type, obj_id))
    
    def _extract_spatial_patterns(self):
        """提取空间关系模式"""
        spatial_patterns = self.pattern_library.get_patterns_by_category("spatial")
        
        for pattern_def in spatial_patterns:
            try:
                detector = pattern_def.get("detector")
                params = pattern_def.get("parameters", {})
                
                if detector:
                    # 调用检测器函数
                    detected_patterns = detector(
                        {
                            "objects_by_position": self.objects_by_position,
                            "objects_by_color": self.objects_by_color,
                            "objects_by_shape": self.objects_by_shape
                        }, 
                        params
                    )
                    
                    # 添加到提取的模式列表
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
    
    def _extract_color_patterns(self):
        """提取颜色关系模式"""
        color_patterns = self.pattern_library.get_patterns_by_category("color")
        
        # 类似于_extract_spatial_patterns的实现
        for pattern_def in color_patterns:
            try:
                detector = pattern_def.get("detector")
                params = pattern_def.get("parameters", {})
                
                if detector:
                    # 检测颜色模式
                    detected_patterns = detector(
                        {
                            "objects_by_color": self.objects_by_color,
                            "color_to_shapes": self.color_to_shapes
                        }, 
                        params
                    )
                    
                    # 添加到提取的模式列表
                    if detected_patterns:
                        for pattern in detected_patterns:
                            pattern["pattern_type"] = pattern_def["id"]
                            pattern["pattern_name"] = pattern_def["name"]
                            pattern["pattern_category"] = "color"
                        
                        self.extracted_patterns["color"].extend(detected_patterns)
                        
                        if self.debug:
                            self.debug_print(f"发现 {len(detected_patterns)} 个 {pattern_def['name']} 模式")
            
            except Exception as e:
                if self.debug:
                    self.debug_print(f"提取 {pattern_def.get('name')} 模式时出错: {e}")
    
    def _extract_shape_patterns(self):
        """提取形状关系模式"""
        shape_patterns = self.pattern_library.get_patterns_by_category("shape")
        
        # 实现类似于之前的方法
        for pattern_def in shape_patterns:
            try:
                detector = pattern_def.get("detector")
                params = pattern_def.get("parameters", {})
                
                if detector:
                    # 检测形状模式
                    detected_patterns = detector(
                        {
                            "objects_by_shape": self.objects_by_shape,
                            "shape_to_colors": self.shape_to_colors
                        }, 
                        params
                    )
                    
                    # 添加到提取的模式列表
                    if detected_patterns:
                        for pattern in detected_patterns:
                            pattern["pattern_type"] = pattern_def["id"]
                            pattern["pattern_name"] = pattern_def["name"]
                            pattern["pattern_category"] = "shape"
                        
                        self.extracted_patterns["shape"].extend(detected_patterns)
                        
                        if self.debug:
                            self.debug_print(f"发现 {len(detected_patterns)} 个 {pattern_def['name']} 模式")
            
            except Exception as e:
                if self.debug:
                    self.debug_print(f"提取 {pattern_def.get('name')} 模式时出错: {e}")
    
    def _extract_transformation_patterns(self):
        """提取变换模式"""
        transformation_patterns = self.pattern_library.get_patterns_by_category("transformation")
        
        for pattern_def in transformation_patterns:
            try:
                detector = pattern_def.get("detector")
                params = pattern_def.get("parameters", {})
                
                if detector:
                    # 检测变换模式
                    detected_patterns = detector(
                        {
                            "objects_by_shape": self.objects_by_shape,
                            "objects_by_color": self.objects_by_color,
                            "shape_to_colors": self.shape_to_colors,
                            "color_to_shapes": self.color_to_shapes
                        }, 
                        params
                    )
                    
                    # 添加到提取的模式列表
                    if detected_patterns:
                        for pattern in detected_patterns:
                            pattern["pattern_type"] = pattern_def["id"]
                            pattern["pattern_name"] = pattern_def["name"]
                            pattern["pattern_category"] = "transformation"
                        
                        self.extracted_patterns["transformation"].extend(detected_patterns)
                        
                        if self.debug:
                            self.debug_print(f"发现 {len(detected_patterns)} 个 {pattern_def['name']} 模式")
            
            except Exception as e:
                if self.debug:
                    self.debug_print(f"提取 {pattern_def.get('name')} 模式时出错: {e}")
    
    def _integrate_patterns_into_rules(self):
        """集成提取的模式，生成复合规则"""
        # 1. 首先处理全局操作规则和条件规则（保留原有功能）
        self._integrate_basic_rules()
        
        # 2. 处理特殊的空间关系模式
        self._integrate_spatial_patterns()
        
        # 3. 处理颜色关系模式
        self._integrate_color_patterns()
        
        # 4. 处理形状关系模式
        self._integrate_shape_patterns()
        
        # 5. 处理变换模式
        self._integrate_transformation_patterns()
        
        # 6. 排序规则
        self._sort_composite_rules()
    
    def _integrate_basic_rules(self):
        """整合基本的全局操作规则和条件规则"""
        # 这里保留原来的功能，对全局规则和条件规则进行整合
        # ... 原有的整合代码 ...
        
    def _integrate_spatial_patterns(self):
        """整合空间关系模式"""
        for pattern in self.extracted_patterns.get("spatial", []):
            # 根据模式类型创建不同的规则
            pattern_type = pattern.get("pattern_type")
            
            if pattern_type == "surrounded_pattern":
                # 创建包围模式规则
                self._create_surrounded_pattern_rule(pattern)
            
            elif pattern_type == "alignment_pattern":
                # 创建对齐模式规则
                self._create_alignment_pattern_rule(pattern)
            
            elif pattern_type == "proximity_pattern":
                # 创建邻近模式规则
                self._create_proximity_pattern_rule(pattern)
    
    def _create_surrounded_pattern_rule(self, pattern):
        """创建基于包围模式的规则"""
        # 这里实现具体的规则创建逻辑
        # 例如：如果检测到物体A被物体B包围，则在输出中移除物体A
        
        rule = {
            'rule_type': 'spatial_pattern_rule',
            'pattern_type': 'surrounded_pattern',
            'conditions': pattern.get('conditions', {}),
            'effects': pattern.get('effects', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于包围模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _create_alignment_pattern_rule(self, pattern):
        """创建基于对齐模式的规则"""
        # 实现规则创建逻辑
        
        rule = {
            'rule_type': 'spatial_pattern_rule',
            'pattern_type': 'alignment_pattern',
            'conditions': pattern.get('conditions', {}),
            'effects': pattern.get('effects', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于对齐模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _create_proximity_pattern_rule(self, pattern):
        """创建基于邻近模式的规则"""
        # 实现规则创建逻辑
        
        rule = {
            'rule_type': 'spatial_pattern_rule',
            'pattern_type': 'proximity_pattern',
            'conditions': pattern.get('conditions', {}),
            'effects': pattern.get('effects', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于邻近模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _integrate_color_patterns(self):
        """整合颜色关系模式"""
        for pattern in self.extracted_patterns.get("color", []):
            pattern_type = pattern.get("pattern_type")
            
            if pattern_type == "color_pair_pattern":
                self._create_color_pair_rule(pattern)
            
            elif pattern_type == "color_group_pattern":
                self._create_color_group_rule(pattern)
    
    def _create_color_pair_rule(self, pattern):
        """创建基于颜色对的规则"""
        # 实现规则创建逻辑
        
        rule = {
            'rule_type': 'color_pattern_rule',
            'pattern_type': 'color_pair_pattern',
            'colors': pattern.get('colors', []),
            'effects': pattern.get('effects', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于颜色对模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _create_color_group_rule(self, pattern):
        """创建基于颜色组的规则"""
        # 实现规则创建逻辑
        
        rule = {
            'rule_type': 'color_pattern_rule',
            'pattern_type': 'color_group_pattern',
            'colors': pattern.get('colors', []),
            'effects': pattern.get('effects', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于颜色组模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _integrate_shape_patterns(self):
        """整合形状关系模式"""
        for pattern in self.extracted_patterns.get("shape", []):
            pattern_type = pattern.get("pattern_type")
            
            if pattern_type == "shape_container_pattern":
                self._create_shape_container_rule(pattern)
            
            elif pattern_type == "shape_symmetry_pattern":
                self._create_shape_symmetry_rule(pattern)
    
    def _create_shape_container_rule(self, pattern):
        """创建基于形状容器的规则"""
        # 实现规则创建逻辑
        
        rule = {
            'rule_type': 'shape_pattern_rule',
            'pattern_type': 'shape_container_pattern',
            'shapes': pattern.get('shapes', []),
            'effects': pattern.get('effects', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于形状容器模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _create_shape_symmetry_rule(self, pattern):
        """创建基于形状对称的规则"""
        # 实现规则创建逻辑
        
        rule = {
            'rule_type': 'shape_pattern_rule',
            'pattern_type': 'shape_symmetry_pattern',
            'shapes': pattern.get('shapes', []),
            'effects': pattern.get('effects', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于形状对称模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _integrate_transformation_patterns(self):
        """整合变换模式"""
        for pattern in self.extracted_patterns.get("transformation", []):
            pattern_type = pattern.get("pattern_type")
            
            if pattern_type == "growth_pattern":
                self._create_growth_rule(pattern)
            
            elif pattern_type == "replacement_pattern":
                self._create_replacement_rule(pattern)
            
            elif pattern_type == "box_formation_pattern":
                self._create_box_formation_rule(pattern)
    
    def _create_growth_rule(self, pattern):
        """创建基于增长模式的规则"""
        # 实现规则创建逻辑
        
        rule = {
            'rule_type': 'transformation_rule',
            'pattern_type': 'growth_pattern',
            'target': pattern.get('target', {}),
            'growth_direction': pattern.get('growth_direction', {}),
            'growth_factor': pattern.get('growth_factor', 1.0),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于增长模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _create_replacement_rule(self, pattern):
        """创建基于替换模式的规则"""
        # 实现规则创建逻辑
        
        rule = {
            'rule_type': 'transformation_rule',
            'pattern_type': 'replacement_pattern',
            'source': pattern.get('source', {}),
            'target': pattern.get('target', {}),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于替换模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _create_box_formation_rule(self, pattern):
        """创建基于围盒形成的规则"""
        # 实现规则创建逻辑 - 这对应于您特别提到的"添加的颜色周围的被围起来的box模式"
        
        rule = {
            'rule_type': 'transformation_rule',
            'pattern_type': 'box_formation_pattern',
            'inner_color': pattern.get('inner_color'),
            'box_color': pattern.get('box_color'),
            'box_thickness': pattern.get('box_thickness', 1),
            'supporting_pairs': pattern.get('supporting_pairs', []),
            'confidence': pattern.get('confidence', 0.8),
            'description': f"基于围盒形成模式的规则: {pattern.get('description', '')}"
        }
        
        self.composite_rules.append(rule)
    
    def _sort_composite_rules(self):
        """按照置信度和支持对数量排序复合规则"""
        self.composite_rules.sort(
            key=lambda x: (
                len(x.get('supporting_pairs', [])), 
                x.get('confidence', 0)
            ),
            reverse=True
        )
