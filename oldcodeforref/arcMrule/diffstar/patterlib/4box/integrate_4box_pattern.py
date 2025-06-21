# 在IntegratedARCPatternSystem类中更新以下方法

def _apply_spatial_rule(self, rule, grid):
    """应用空间规则的实现"""
    rule_pattern_type = rule.get('pattern_type')
    
    if rule_pattern_type == 'surrounded_pattern':
        # 原有代码...
        pass
    elif rule_pattern_type == 'four_box_pattern':
        # 调用4-Box规则应用方法
        return self._apply_4box_pattern_rule(rule, grid)
    
    return grid

def _extract_spatial_patterns(self):
    """提取空间关系模式"""
    spatial_patterns = self.built_in_patterns["spatial"] + self.user_patterns.get("spatial", [])
    
    for pattern_def in spatial_patterns:
        try:
            detector = pattern_def.get("detector")
            params = pattern_def.get("parameters", {})
            pattern_id = pattern_def.get("id")
            
            if detector:
                # 调用检测器
                detected_patterns = detector(params)
                
                # 处理检测到的模式
                if detected_patterns:
                    for pattern in detected_patterns:
                        pattern["pattern_type"] = pattern_id
                        pattern["pattern_name"] = pattern_def.get("name", pattern_id)
                        pattern["pattern_category"] = "spatial"
                    
                    self.extracted_patterns["spatial"].extend(detected_patterns)
                    
                    if self.debug:
                        self.debug_print(f"发现 {len(detected_patterns)} 个 {pattern_def['name']} 模式")
        
        except Exception as e:
            if self.debug:
                self.debug_print(f"提取 {pattern_def.get('name')} 模式时出错: {e}")