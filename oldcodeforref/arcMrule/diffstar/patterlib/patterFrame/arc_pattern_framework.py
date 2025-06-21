"""
ARC模式分析框架：多维空间关系模式库
提供统一的模式定义、检测、匹配和应用机制
"""

from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Callable, Union, Optional
import json
import copy

class ARCPatternFramework:
    """
    ARC模式分析框架 - 核心组件
    用于定义、检测、匹配和应用各种模式
    """
    
    def __init__(self):
        """初始化模式框架"""
        # 模式定义库
        self.pattern_definitions = {}
        
        # 模式层次结构
        self.pattern_hierarchy = {}
        
        # 模式检测器映射
        self.pattern_detectors = {}
        
        # 模式应用器映射
        self.pattern_appliers = {}
        
        # 模式相似度函数
        self.pattern_similarity_funcs = {}
        
        # 模式优先级
        self.pattern_priorities = {}
        
        # 初始化标准模式库
        self._initialize_standard_patterns()
    
    def _initialize_standard_patterns(self):
        """初始化标准的模式库"""
        # 1. 空间关系模式
        self._register_spatial_patterns()
        
        # 2. 颜色关系模式
        self._register_color_patterns()
        
        # 3. 形状关系模式
        self._register_shape_patterns()
        
        # 4. 变换关系模式
        self._register_transformation_patterns()
    
    def _register_spatial_patterns(self):
        """注册空间关系模式"""
        # 1. 四向包围模式(4-Box)
        self.register_pattern(
            pattern_id='four_box_pattern',
            category='spatial',
            name='4-Box围绕模式',
            description='中心像素/对象被另一种颜色在上下左右四个方向完全包围',
            detector=self._detect_four_box_pattern,
            applier=self._apply_four_box_pattern,
            similarity_func=self._calculate_four_box_similarity,
            parameters={
                'require_exact_match': True,
                'allow_diagonal': False, 
                'boundary_counts': False
            },
            priority=80
        )
        
        # 2. 八向包围模式(8-Box)
        self.register_pattern(
            pattern_id='eight_box_pattern',
            category='spatial',
            name='8-Box围绕模式',
            description='中心像素/对象被另一种颜色在所有八个方向(包括对角线)完全包围',
            detector=self._detect_eight_box_pattern,
            applier=self._apply_eight_box_pattern,
            similarity_func=self._calculate_eight_box_similarity,
            parameters={
                'require_exact_match': True,
                'minimum_surrounding_ratio': 0.7,
                'boundary_counts': False
            },
            priority=75
        )
        
        # 3. 边界模式(Border)
        self.register_pattern(
            pattern_id='border_pattern',
            category='spatial',
            name='边界模式',
            description='一种颜色形成另一种颜色的边界/轮廓',
            detector=self._detect_border_pattern,
            applier=self._apply_border_pattern,
            similarity_func=self._calculate_border_similarity,
            parameters={
                'min_border_coverage': 0.8,
                'diagonal_counts': True
            },
            priority=70
        )
        
        # 4. 连接模式(Connection)
        self.register_pattern(
            pattern_id='connection_pattern',
            category='spatial',
            name='连接模式',
            description='两个或多个相同颜色的对象通过另一种颜色的路径连接',
            detector=self._detect_connection_pattern,
            applier=self._apply_connection_pattern,
            similarity_func=self._calculate_connection_similarity,
            parameters={
                'min_connection_length': 1,
                'max_connection_length': 10,
                'diagonal_allowed': False
            },
            priority=65
        )
        
        # 5. 分隔模式(Separator)
        self.register_pattern(
            pattern_id='separator_pattern',
            category='spatial',
            name='分隔模式',
            description='一种颜色的线/对象将网格分隔成独立的区域',
            detector=self._detect_separator_pattern,
            applier=self._apply_separator_pattern,
            similarity_func=self._calculate_separator_similarity,
            parameters={
                'continuous_separator': True,
                'count_isolated_regions': True
            },
            priority=60
        )
        
        # 6. 包含模式(Containment)
        self.register_pattern(
            pattern_id='containment_pattern',
            category='spatial',
            name='包含模式',
            description='一个对象完全包含在另一个对象内部',
            detector=self._detect_containment_pattern,
            applier=self._apply_containment_pattern,
            similarity_func=self._calculate_containment_similarity,
            parameters={
                'min_containment_ratio': 1.0,
                'max_contained_objects': 5
            },
            priority=85
        )
        
        # 7. 相邻模式(Adjacency)
        self.register_pattern(
            pattern_id='adjacency_pattern',
            category='spatial',
            name='相邻模式',
            description='特定颜色的对象总是与另一种颜色的对象相邻',
            detector=self._detect_adjacency_pattern,
            applier=self._apply_adjacency_pattern,
            similarity_func=self._calculate_adjacency_similarity,
            parameters={
                'min_adjacency_ratio': 0.7,
                'count_diagonal': False
            },
            priority=50
        )
    
    def _register_color_patterns(self):
        """注册颜色关系模式"""
        # 1. 颜色对模式(Color Pair)
        self.register_pattern(
            pattern_id='color_pair_pattern',
            category='color',
            name='颜色对模式',
            description='两种特定颜色总是成对出现',
            detector=self._detect_color_pair_pattern,
            applier=self._apply_color_pair_pattern,
            similarity_func=self._calculate_color_pair_similarity,
            parameters={
                'min_occurrence': 2,
                'proximity_required': False
            },
            priority=70
        )
        
        # 2. 颜色组模式(Color Group)
        self.register_pattern(
            pattern_id='color_group_pattern',
            category='color',
            name='颜色组模式',
            description='多种颜色总是一起出现',
            detector=self._detect_color_group_pattern,
            applier=self._apply_color_group_pattern,
            similarity_func=self._calculate_color_group_similarity,
            parameters={
                'min_group_size': 3,
                'min_occurrence': 2
            },
            priority=65
        )
        
        # 3. 颜色排列模式(Color Sequence)
        self.register_pattern(
            pattern_id='color_sequence_pattern',
            category='color',
            name='颜色排列模式',
            description='多种颜色按特定顺序排列',
            detector=self._detect_color_sequence_pattern,
            applier=self._apply_color_sequence_pattern,
            similarity_func=self._calculate_color_sequence_similarity,
            parameters={
                'min_sequence_length': 3,
                'direction_sensitive': True,
                'allow_gaps': False
            },
            priority=75
        )
        
        # 4. 颜色频率模式(Color Frequency)
        self.register_pattern(
            pattern_id='color_frequency_pattern',
            category='color',
            name='颜色频率模式',
            description='不同颜色按特定的数量比例出现',
            detector=self._detect_color_frequency_pattern,
            applier=self._apply_color_frequency_pattern,
            similarity_func=self._calculate_color_frequency_similarity,
            parameters={
                'min_colors': 2,
                'frequency_tolerance': 0.1
            },
            priority=55
        )
        
        # 5. 颜色替换模式(Color Replacement)
        self.register_pattern(
            pattern_id='color_replacement_pattern',
            category='color',
            name='颜色替换模式',
            description='一种颜色在输出中被另一种颜色替换',
            detector=self._detect_color_replacement_pattern,
            applier=self._apply_color_replacement_pattern,
            similarity_func=self._calculate_color_replacement_similarity,
            parameters={
                'replacement_ratio_threshold': 0.8,
                'consider_position': False
            },
            priority=90
        )
    
    def _register_shape_patterns(self):
        """注册形状关系模式"""
        # 1. 形状相似度模式(Shape Similarity)
        self.register_pattern(
            pattern_id='shape_similarity_pattern',
            category='shape',
            name='形状相似度模式',
            description='对象在形状上相似，但可能颜色、大小或朝向不同',
            detector=self._detect_shape_similarity_pattern,
            applier=self._apply_shape_similarity_pattern,
            similarity_func=self._calculate_shape_similarity,
            parameters={
                'min_similarity_threshold': 0.8,
                'consider_rotation': True,
                'consider_reflection': True
            },
            priority=65
        )
        
        # 2. 形状包含模式(Shape Containment)
        self.register_pattern(
            pattern_id='shape_containment_pattern',
            category='shape',
            name='形状包含模式',
            description='一个形状包含在另一个形状内部',
            detector=self._detect_shape_containment_pattern,
            applier=self._apply_shape_containment_pattern,
            similarity_func=self._calculate_shape_containment_similarity,
            parameters={
                'containment_threshold': 0.9,
                'min_size_ratio': 1.5
            },
            priority=70
        )
        
        # 3. 形状增长模式(Shape Growth)
        self.register_pattern(
            pattern_id='shape_growth_pattern',
            category='shape',
            name='形状增长模式',
            description='形状按照特定规则增长或扩展',
            detector=self._detect_shape_growth_pattern,
            applier=self._apply_shape_growth_pattern,
            similarity_func=self._calculate_shape_growth_similarity,
            parameters={
                'min_growth_factor': 1.2,
                'growth_direction_sensitive': True
            },
            priority=75
        )
        
        # 4. 形状对称模式(Shape Symmetry)
        self.register_pattern(
            pattern_id='shape_symmetry_pattern',
            category='shape',
            name='形状对称模式',
            description='对象具有轴对称或点对称特性',
            detector=self._detect_shape_symmetry_pattern,
            applier=self._apply_shape_symmetry_pattern,
            similarity_func=self._calculate_shape_symmetry_similarity,
            parameters={
                'symmetry_threshold': 0.9,
                'check_horizontal': True,
                'check_vertical': True,
                'check_diagonal': False,
                'check_rotational': False
            },
            priority=60
        )
        
        # 5. 形状分割模式(Shape Division)
        self.register_pattern(
            pattern_id='shape_division_pattern',
            category='shape',
            name='形状分割模式',
            description='一个形状被分割成多个相似的子形状',
            detector=self._detect_shape_division_pattern,
            applier=self._apply_shape_division_pattern,
            similarity_func=self._calculate_shape_division_similarity,
            parameters={
                'min_divisions': 2,
                'division_similarity_threshold': 0.7
            },
            priority=65
        )
    
    def _register_transformation_patterns(self):
        """注册变换关系模式"""
        # 1. 平移模式(Translation)
        self.register_pattern(
            pattern_id='translation_pattern',
            category='transformation',
            name='平移模式',
            description='对象按特定方向和距离移动',
            detector=self._detect_translation_pattern,
            applier=self._apply_translation_pattern,
            similarity_func=self._calculate_translation_similarity,
            parameters={
                'max_distance': 10,
                'direction_sensitive': True
            },
            priority=80
        )
        
        # 2. 旋转模式(Rotation)
        self.register_pattern(
            pattern_id='rotation_pattern',
            category='transformation',
            name='旋转模式',
            description='对象围绕中心点或特定点旋转',
            detector=self._detect_rotation_pattern,
            applier=self._apply_rotation_pattern,
            similarity_func=self._calculate_rotation_similarity,
            parameters={
                'rotation_angles': [90, 180, 270],
                'rotation_center': 'object_center'  # or 'grid_center'
            },
            priority=75
        )
        
        # 3. 镜像模式(Reflection)
        self.register_pattern(
            pattern_id='reflection_pattern',
            category='transformation',
            name='镜像模式',
            description='对象沿特定轴线反射',
            detector=self._detect_reflection_pattern,
            applier=self._apply_reflection_pattern,
            similarity_func=self._calculate_reflection_similarity,
            parameters={
                'reflection_axes': ['horizontal', 'vertical', 'diagonal'],
                'reflection_line': 'center'  # or 'custom'
            },
            priority=70
        )
        
        # 4. 缩放模式(Scaling)
        self.register_pattern(
            pattern_id='scaling_pattern',
            category='transformation',
            name='缩放模式',
            description='对象按比例放大或缩小',
            detector=self._detect_scaling_pattern,
            applier=self._apply_scaling_pattern,
            similarity_func=self._calculate_scaling_similarity,
            parameters={
                'min_scale_factor': 0.5,
                'max_scale_factor': 3.0,
                'uniform_scaling': True
            },
            priority=65
        )
        
        # 5. 合并模式(Merging)
        self.register_pattern(
            pattern_id='merging_pattern',
            category='transformation',
            name='合并模式',
            description='多个对象合并成一个更大的对象',
            detector=self._detect_merging_pattern,
            applier=self._apply_merging_pattern,
            similarity_func=self._calculate_merging_similarity,
            parameters={
                'min_objects_to_merge': 2,
                'merge_rule': 'union'  # or 'intersection', 'majority'
            },
            priority=85
        )
        
        # 6. 分裂模式(Splitting)
        self.register_pattern(
            pattern_id='splitting_pattern',
            category='transformation',
            name='分裂模式',
            description='一个对象分裂成多个更小的对象',
            detector=self._detect_splitting_pattern,
            applier=self._apply_splitting_pattern,
            similarity_func=self._calculate_splitting_similarity,
            parameters={
                'min_split_parts': 2,
                'split_rule': 'equal'  # or 'custom'
            },
            priority=80
        )
    
    def register_pattern(self, pattern_id, category, name, description, 
                        detector, applier, similarity_func, parameters, priority):
        """
        注册一个模式到框架中
        
        Args:
            pattern_id: 模式唯一标识符
            category: 模式类别(spatial, color, shape, transformation等)
            name: 模式名称
            description: 模式描述
            detector: 模式检测函数
            applier: 模式应用函数
            similarity_func: 模式相似度计算函数
            parameters: 模式参数字典
            priority: 模式优先级(0-100)
        """
        # 创建模式定义
        pattern_def = {
            'id': pattern_id,
            'category': category,
            'name': name,
            'description': description,
            'parameters': parameters,
            'default_priority': priority
        }
        
        # 注册模式定义
        self.pattern_definitions[pattern_id] = pattern_def
        
        # 注册检测器、应用器和相似度函数
        self.pattern_detectors[pattern_id] = detector
        self.pattern_appliers[pattern_id] = applier
        self.pattern_similarity_funcs[pattern_id] = similarity_func
        self.pattern_priorities[pattern_id] = priority
        
        # 更新层次结构
        if category not in self.pattern_hierarchy:
            self.pattern_hierarchy[category] = []
        self.pattern_hierarchy[category].append(pattern_id)
    
    def detect_pattern(self, pattern_id, input_data, params=None):
        """
        使用指定的检测器检测模式
        
        Args:
            pattern_id: 要检测的模式ID
            input_data: 输入数据
            params: 可选的参数覆盖
            
        Returns:
            检测到的模式实例列表
        """
        if pattern_id not in self.pattern_detectors:
            raise ValueError(f"未知的模式ID: {pattern_id}")
        
        detector = self.pattern_detectors[pattern_id]
        
        # 合并参数
        merged_params = self.pattern_definitions[pattern_id]['parameters'].copy()
        if params:
            merged_params.update(params)
        
        # 执行检测
        return detector(input_data, merged_params)
    
    def apply_pattern(self, pattern_id, pattern_instance, input_data, params=None):
        """
        应用模式
        
        Args:
            pattern_id: 要应用的模式ID
            pattern_instance: 检测到的模式实例
            input_data: 要应用模式的数据
            params: 可选的参数覆盖
            
        Returns:
            应用模式后的数据
        """
        if pattern_id not in self.pattern_appliers:
            raise ValueError(f"未知的模式ID: {pattern_id}")
        
        applier = self.pattern_appliers[pattern_id]
        
        # 合并参数
        merged_params = self.pattern_definitions[pattern_id]['parameters'].copy()
        if params:
            merged_params.update(params)
        
        # 执行应用
        return applier(pattern_instance, input_data, merged_params)
    
    def calculate_pattern_similarity(self, pattern_id, instance1, instance2, params=None):
        """
        计算两个模式实例的相似度
        
        Args:
            pattern_id: 模式ID
            instance1: 第一个模式实例
            instance2: 第二个模式实例
            params: 可选的参数覆盖
            
        Returns:
            相似度分数(0-1)
        """
        if pattern_id not in self.pattern_similarity_funcs:
            raise ValueError(f"未知的模式ID: {pattern_id}")
        
        similarity_func = self.pattern_similarity_funcs[pattern_id]
        
        # 合并参数
        merged_params = self.pattern_definitions[pattern_id]['parameters'].copy()
        if params:
            merged_params.update(params)
        
        # 计算相似度
        return similarity_func(instance1, instance2, merged_params)
    
    def detect_all_patterns(self, input_data, categories=None, min_confidence=0.5):
        """
        在输入数据中检测所有模式
        
        Args:
            input_data: 输入数据
            categories: 可选的类别过滤
            min_confidence: 最小置信度阈值
            
        Returns:
            检测到的所有模式实例
        """
        all_patterns = []
        
        # 筛选要检测的模式类别
        if categories:
            target_categories = categories
        else:
            target_categories = list(self.pattern_hierarchy.keys())
        
        # 对每个类别
        for category in target_categories:
            if category not in self.pattern_hierarchy:
                continue
                
            # 对每个模式
            for pattern_id in self.pattern_hierarchy[category]:
                try:
                    # 检测模式
                    detected_patterns = self.detect_pattern(pattern_id, input_data)
                    
                    # 过滤低置信度的模式
                    filtered_patterns = [
                        p for p in detected_patterns 
                        if p.get('confidence', 1.0) >= min_confidence
                    ]
                    
                    # 添加模式元数据
                    for pattern in filtered_patterns:
                        pattern['pattern_id'] = pattern_id
                        pattern['category'] = category
                        pattern['name'] = self.pattern_definitions[pattern_id]['name']
                        pattern['priority'] = self.pattern_priorities[pattern_id]
                    
                    all_patterns.extend(filtered_patterns)
                except Exception as e:
                    print(f"检测模式 {pattern_id} 时出错: {e}")
        
        # 按优先级排序
        all_patterns.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return all_patterns
    
    def save_to_json(self, file_path):
        """将模式库保存到JSON文件"""
        # 准备可序列化的数据
        serializable_data = {
            'pattern_definitions': self.pattern_definitions,
            'pattern_hierarchy': self.pattern_hierarchy,
            'pattern_priorities': self.pattern_priorities
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
    
    def load_from_json(self, file_path):
        """从JSON文件加载模式库定义"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 加载模式定义
        if 'pattern_definitions' in data:
            self.pattern_definitions.update(data['pattern_definitions'])
        
        # 加载模式层次结构
        if 'pattern_hierarchy' in data:
            for category, patterns in data['pattern_hierarchy'].items():
                if category not in self.pattern_hierarchy:
                    self.pattern_hierarchy[category] = []
                self.pattern_hierarchy[category].extend(patterns)
        
        # 加载模式优先级
        if 'pattern_priorities' in data:
            self.pattern_priorities.update(data['pattern_priorities'])

    # 各种检测器和应用器方法的具体实现
    # 这里只给出几个关键方法的实现示例
    
    def _detect_four_box_pattern(self, input_data, params):
        """
        检测4-Box围绕模式
        
        Args:
            input_data: 输入数据，包含网格和对象信息
            params: 参数字典
            
        Returns:
            检测到的4-Box模式列表
        """
        grid = input_data.get('grid')
        if grid is None:
            return []
            
        require_exact_match = params.get('require_exact_match', True)
        allow_diagonal = params.get('allow_diagonal', False)
        boundary_counts = params.get('boundary_counts', False)
        
        patterns = []
        height, width = len(grid), len(grid[0])
        
        for i in range(height):
            for j in range(width):
                center_color = grid[i][j]
                if center_color == 0:  # 跳过背景
                    continue
                
                # 检查四个方向
                directions = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                
                # 统计周围颜色
                surrounding_colors = {}
                valid_directions = 0
                
                for ni, nj in directions:
                    # 检查边界
                    if 0 <= ni < height and 0 <= nj < width:
                        valid_directions += 1
                        neighbor_color = grid[ni][nj]
                        
                        if neighbor_color not in surrounding_colors:
                            surrounding_colors[neighbor_color] = 0
                        surrounding_colors[neighbor_color] += 1
                    elif boundary_counts:
                        # 如果边界也算作包围
                        boundary_color = -1  # 特殊值表示边界
                        if boundary_color not in surrounding_colors:
                            surrounding_colors[boundary_color] = 0
                        surrounding_colors[boundary_color] += 1
                        valid_directions += 1
                
                # 找出最主要的包围颜色
                if surrounding_colors:
                    main_surrounding = max(surrounding_colors.items(), key=lambda x: x[1])
                    main_color, main_count = main_surrounding
                    
                    # 严格模式：必须完全被同一种颜色包围
                    if require_exact_match and main_count < valid_directions:
                        continue
                    
                    # 不能被自己的颜色包围
                    if main_color == center_color:
                        continue
                        
                    # 边界不算作有效的包围色
                    if main_color == -1:
                        continue
                    
                    # 计算包围比例
                    surround_ratio = main_count / valid_directions
                    
                    # 创建模式对象
                    pattern = {
                        'center_position': (i, j),
                        'center_color': center_color,
                        'surrounding_color': main_color,
                        'surrounding_ratio': surround_ratio,
                        'confidence': surround_ratio
                    }
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _apply_four_box_pattern(self, pattern_instance, input_data, params):
        """
        应用4-Box模式
        
        Args:
            pattern_instance: 检测到的模式实例
            input_data: 输入数据
            params: 参数
            
        Returns:
            应用模式后的数据
        """
        grid = copy.deepcopy(input_data.get('grid', []))
        if not grid:
            return {'grid': grid}
            
        transformation = pattern_instance.get('transformation', {})
        transform_type = transformation.get('type')
        center_color = pattern_instance.get('center_color')
        surrounding_color = pattern_instance.get('surrounding_color')
        
        # 找出所有匹配的模式实例
        matching_positions = []
        height, width = len(grid), len(grid[0])
        
        for i in range(height):
            for j in range(width):
                if grid[i][j] == center_color:
                    # 检查四个方向
                    directions = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                    surrounded = True
                    
                    for ni, nj in directions:
                        if not (0 <= ni < height and 0 <= nj < width and grid[ni][nj] == surrounding_color):
                            surrounded = False
                            break
                    
                    if surrounded:
                        matching_positions.append((i, j))
        
        # 应用变换
        if transform_type == 'color_change' and 'new_color' in transformation:
            new_color = transformation['new_color']
            for i, j in matching_positions:
                grid[i][j] = new_color
        
        elif transform_type == 'remove':
            for i, j in matching_positions:
                grid[i][j] = 0  # 假设0是背景色
        
        elif transform_type == 'expand':
            # 扩展中心颜色到周围
            for i, j in matching_positions:
                # 四个方向
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                    if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] == surrounding_color:
                        grid[ni][nj] = center_color
        
        return {'grid': grid}
    
    def _calculate_four_box_similarity(self, instance1, instance2, params):
        """
        计算两个4-Box模式实例的相似度
        
        Args:
            instance1: 第一个模式实例
            instance2: 第二个模式实例
            params: 参数
            
        Returns:
            相似度分数(0-1)
        """
        # 颜色匹配是最重要的
        center_color_match = instance1.get('center_color') == instance2.get('center_color')
        surrounding_color_match = instance1.get('surrounding_color') == instance2.get('surrounding_color')
        
        # 比例相似度
        ratio1 = instance1.get('surrounding_ratio', 0)
        ratio2 = instance2.get('surrounding_ratio', 0)
        ratio_similarity = 1.0 - abs(ratio1 - ratio2)
        
        # 计算综合相似度
        if center_color_match and surrounding_color_match:
            return 0.8 + 0.2 * ratio_similarity
        elif center_color_match or surrounding_color_match:
            return 0.4 + 0.2 * ratio_similarity
        else:
            return 0.2 * ratio_similarity