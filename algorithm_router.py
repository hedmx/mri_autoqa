#!/usr/bin/env python3
"""
算法路由器 - 基于A/S/F/M约束空间选择算法参数
修复版本：正确解析metadata.json字段
"""

import json
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np

from config import ROI_SETTINGS, SEQUENCE_SNR_STANDARDS, G_FACTOR_TABLE


class AlgorithmRouter:
    """
    基于约束空间的路由器
    输入: metadata.json
    输出: 算法参数配置
    """
    
    def __init__(self):
        self.supported_anatomies = list(ROI_SETTINGS.keys())
        self.supported_sequences = list(SEQUENCE_SNR_STANDARDS.keys())
        
    def route(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        主路由函数
        
        Args:
            metadata: 从metadata.json加载的字典
            
        Returns:
            算法参数配置字典
        """
        # 提取约束空间
        constraint = metadata.get('constraint_space', {})
        
        # 解析A/S/F/M
        anatomy = self._extract_anatomy(constraint, metadata)
        sequence = self._extract_sequence(constraint, metadata)
        field_strength = self._extract_field_strength(constraint, metadata)
        acquisition_mode = self._extract_acquisition_mode(constraint, metadata)
        
        # Parallel Imaging Information - 修复字段名
        # 首先尝试小写下划线版本（metadata.json的实际字段名）
        parallel_info = metadata.get('parallel_imaging', {})
        if not parallel_info:
            # 尝试大写驼峰版本（向后兼容）
            parallel_info = metadata.get('ParallelImaging', {})
        
        # 修复键名：metadata.json中使用的是"used"，不是"is_parallel"
        is_parallel = parallel_info.get('used', False)
        acceleration_factor = parallel_info.get('acceleration_factor', 1.0)
        
        # 构建路由结果
        route_result = {
            'algorithm': 'single_image_v3_corrected',
            'constraint_space': {
                'A': anatomy,
                'S': sequence,
                'F': field_strength,
                'M': acquisition_mode
            },
            'adjustments': {
                'noise_correction_required': is_parallel,
                'acceleration_factor': acceleration_factor,
                'anatomy_specific_roi': self._get_roi_template(anatomy),
                'sequence_standards': self._get_sequence_standards(sequence),
                'field_strength': field_strength,
                'acquisition_mode': acquisition_mode,
                'g_factor': self._get_g_factor(acceleration_factor) if is_parallel else 1.0
            },
            'validation': {
                'anatomy_supported': anatomy in self.supported_anatomies,
                'sequence_supported': sequence in self.supported_sequences,
                'has_parallel_info': 'parallel_imaging' in metadata or 'ParallelImaging' in metadata
            }
        }
        
        return route_result
    
    def _extract_anatomy(self, constraint: Dict, metadata: Dict) -> str:
        """提取Anatomical Region"""
        # 优先使用constraint_space
        anatomy = constraint.get('A', '').lower()
        if anatomy:
            return anatomy
        
        # ===========================================
        # 修复：使用metadata.json中实际的字段名
        # ===========================================
        
        # 首先尝试小写下划线版本（metadata.json的实际字段名）
        anatomical_region = metadata.get('anatomical_region', '')
        
        if anatomical_region:
            # 如果是字典，提取standardized字段
            if isinstance(anatomical_region, dict):
                std_anatomy = anatomical_region.get('standardized', '').lower()
                if std_anatomy:
                    return self._map_anatomy(std_anatomy)
            # 如果是字符串，直接使用
            elif isinstance(anatomical_region, str):
                return self._map_anatomy(anatomical_region.lower())
        
        # 尝试大写驼峰版本（向后兼容）
        anatomical_region = metadata.get('AnatomicalRegion', {})
        if isinstance(anatomical_region, dict):
            std_anatomy = anatomical_region.get('standardized', '').lower()
            if std_anatomy:
                return self._map_anatomy(std_anatomy)
        
        # 从StudyDescription推断
        study_desc = metadata.get('study_info', {}).get('study_description', '').lower()
        if study_desc:
            if 'lumbar' in study_desc or 'lspine' in study_desc:
                return 'lumbar'
            elif 'cervical' in study_desc or 'cspine' in study_desc:
                return 'cervical'
            elif 'thoracic' in study_desc or 'tspine' in study_desc:
                return 'thoracic'
            elif 'brain' in study_desc or 'head' in study_desc:
                return 'brain'
        
        # 从SeriesDescription推断
        series_desc = metadata.get('series_info', {}).get('series_description', '').lower()
        if series_desc:
            if 'lumbar' in series_desc or 'lspine' in series_desc:
                return 'lumbar'
            elif 'cervical' in series_desc or 'cspine' in series_desc:
                return 'cervical'
            elif 'thoracic' in series_desc or 'tspine' in series_desc:
                return 'thoracic'
            elif 'brain' in series_desc or 'head' in series_desc:
                return 'brain'
            elif 'spine' in series_desc:
                # 如果是脊柱但未指定部位，根据序列描述进一步推断
                if 'c' in series_desc or 'cerv' in series_desc:
                    return 'cervical'
                elif 't' in series_desc or 'thor' in series_desc:
                    return 'thoracic'
                else:
                    return 'lumbar'  # 默认腰椎
        
        return 'default'  # 修改默认值
    
    def _map_anatomy(self, anatomy: str) -> str:
        """映射解剖区域到标准名称"""
        anatomy_lower = anatomy.lower()
        
        # 映射通用名称到具体部位
        if anatomy_lower == 'spine':
            return 'lumbar'  # 默认映射为腰椎
        elif anatomy_lower == 'spinal':
            return 'lumbar'
        elif 'lumbar' in anatomy_lower or 'lspine' in anatomy_lower:
            return 'lumbar'
        elif 'cervical' in anatomy_lower or 'cspine' in anatomy_lower:
            return 'cervical'
        elif 'thoracic' in anatomy_lower or 'tspine' in anatomy_lower:
            return 'thoracic'
        elif 'brain' in anatomy_lower or 'head' in anatomy_lower:
            return 'brain'
        elif 'abdomen' in anatomy_lower or 'abdominal' in anatomy_lower:
            return 'abdomen'
        elif 'pelvis' in anatomy_lower or 'pelvic' in anatomy_lower:
            return 'pelvis'
        
        # 检查是否在支持的解剖区域列表中
        for supported_anatomy in self.supported_anatomies:
            if supported_anatomy.lower() in anatomy_lower:
                return supported_anatomy
        
        # 如果都不匹配，尝试直接返回
        return anatomy_lower
    
    def _extract_sequence(self, constraint: Dict, metadata: Dict) -> str:
        """提取Sequence Type"""
        # 优先使用constraint_space
        sequence = constraint.get('S', '').lower()
        if sequence:
            # 清理Sequence标识符（移除'w', 'fat_sat'等后缀）
            if 't1' in sequence:
                return 't1'
            elif 't2' in sequence:
                return 't2'
            elif 'pd' in sequence:
                return 'pd'
            elif 'flair' in sequence:
                return 'flair'
        
        # 从metadata.json的sequence_info字段提取
        sequence_info = metadata.get('sequence_info', {})
        sequence_type = sequence_info.get('sequence_type', '').lower()
        if sequence_type:
            # 清理序列类型
            if 't1' in sequence_type:
                return 't1'
            elif 't2' in sequence_type:
                return 't2'
            elif 'pd' in sequence_type:
                return 'pd'
            elif 'flair' in sequence_type:
                return 'flair'
        
        # 从sequence_subtype提取
        sequence_subtype = sequence_info.get('sequence_subtype', '').lower()
        if sequence_subtype:
            if 't1' in sequence_subtype or 'spin' in sequence_subtype:
                return 't1'
            elif 't2' in sequence_subtype or 'fse' in sequence_subtype or 'tse' in sequence_subtype:
                return 't2'
            elif 'pd' in sequence_subtype:
                return 'pd'
            elif 'flair' in sequence_subtype:
                return 'flair'
        
        # 从SeriesDescription推断
        series_desc = metadata.get('series_info', {}).get('series_description', '').lower()
        if series_desc:
            if 't1' in series_desc:
                return 't1'
            elif 't2' in series_desc:
                return 't2'
            elif 'pd' in series_desc or 'proton' in series_desc:
                return 'pd'
            elif 'flair' in series_desc:
                return 'flair'
        
        return 't1'  # Default值
    
    def _extract_field_strength(self, constraint: Dict, metadata: Dict) -> str:
        """提取Field Strength"""
        # 优先使用constraint_space
        field_strength = constraint.get('F', '').lower()
        if field_strength:
            return field_strength
        
        # 从acquisition_params字段提取
        acquisition_params = metadata.get('acquisition_params', {})
        field_strength_raw = acquisition_params.get('field_strength_t', '')
        
        if field_strength_raw:
            # 清理场强字符串
            field_str = str(field_strength_raw).lower()
            if '3' in field_str or '3.0' in field_str:
                return '3.0t'
            elif '1.5' in field_str:
                return '1.5t'
            elif '1.0' in field_str:
                return '1.0t'
        
        # 从MagneticFieldStrength字段推断（如果存在）
        mag_strength = metadata.get('MagneticFieldStrength', 1.5)
        if isinstance(mag_strength, (int, float)):
            if mag_strength >= 2.5:
                return '3.0t'
            else:
                return '1.5t'
        
        return '1.5t'  # 默认场强
    
    def _extract_acquisition_mode(self, constraint: Dict, metadata: Dict) -> str:
        """提取采集模式"""
        # 优先使用constraint_space
        mode = constraint.get('M', '').lower()
        if mode:
            return mode
        
        # 从parallel_imaging字段推断
        parallel_info = metadata.get('parallel_imaging', {})
        if not parallel_info:
            # 尝试大写驼峰版本
            parallel_info = metadata.get('ParallelImaging', {})
        
        # 修复：使用正确的键名
        if parallel_info.get('used', False):
            return 'parallel_imaging'
        else:
            return 'conventional'
    
    def _get_roi_template(self, anatomy: str) -> Dict[str, float]:
        """获取Anatomical Region特定的ROI模板"""
        # 首先尝试直接匹配
        template = ROI_SETTINGS.get(anatomy.lower(), None)
        if template:
            return template
        
        # 尝试模糊匹配
        for key in ROI_SETTINGS.keys():
            if key.lower() in anatomy.lower() or anatomy.lower() in key.lower():
                return ROI_SETTINGS[key]
        
        # 返回默认模板
        return ROI_SETTINGS['default']
    
    def _get_sequence_standards(self, sequence: str) -> Dict[str, float]:
        """获取Sequence特定的SNR标准"""
        # 清理序列名称
        clean_seq = sequence.lower()
        if 't1' in clean_seq:
            return SEQUENCE_SNR_STANDARDS.get('t1', SEQUENCE_SNR_STANDARDS['default'])
        elif 't2' in clean_seq:
            return SEQUENCE_SNR_STANDARDS.get('t2', SEQUENCE_SNR_STANDARDS['default'])
        elif 'pd' in clean_seq:
            return SEQUENCE_SNR_STANDARDS.get('pd', SEQUENCE_SNR_STANDARDS['default'])
        elif 'flair' in clean_seq:
            return SEQUENCE_SNR_STANDARDS.get('flair', SEQUENCE_SNR_STANDARDS['default'])
        
        # 默认使用t1标准
        return SEQUENCE_SNR_STANDARDS.get(clean_seq, SEQUENCE_SNR_STANDARDS['t1'])
    
    def _get_g_factor(self, acceleration_factor: float) -> float:
        """获取g-factor值"""
        # 查找最近的R值
        available_R = sorted(G_FACTOR_TABLE.keys())
        
        if acceleration_factor in G_FACTOR_TABLE:
            nearest_R = acceleration_factor
        else:
            nearest_R = min(available_R, key=lambda x: abs(x - acceleration_factor))
        
        g_mean, _, _, _ = G_FACTOR_TABLE[nearest_R]
        
        # 插值Processing
        if acceleration_factor != nearest_R:
            lower_R = max([r for r in available_R if r <= acceleration_factor], default=nearest_R)
            upper_R = min([r for r in available_R if r >= acceleration_factor], default=nearest_R)
            
            if lower_R != upper_R:
                lower_g = G_FACTOR_TABLE[lower_R][0]
                upper_g = G_FACTOR_TABLE[upper_R][0]
                ratio = (acceleration_factor - lower_R) / (upper_R - lower_R)
                g_mean = lower_g + ratio * (upper_g - lower_g)
        
        return float(g_mean)


# 便捷函数
def route_from_metadata_file(metadata_path: Path) -> Dict[str, Any]:
    """从metadata.json文件直接路由"""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        router = AlgorithmRouter()
        return router.route(metadata)
        
    except Exception as e:
        print(f"路由Failed {metadata_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


# 测试函数
def test_router():
    """测试路由器"""
    # 模拟您的metadata.json结构
    test_metadata = {
        "anatomical_region": "spine",
        "parallel_imaging": {
            "used": True,
            "acceleration_factor": 2.0,
            "method": "GRAPPA"
        },
        "sequence_info": {
            "sequence_type": "t1",
            "sequence_subtype": "fse_tse"
        },
        "acquisition_params": {
            "field_strength_t": "1.5t"
        },
        "study_info": {
            "study_description": "HZSDEYY^spine"
        },
        "series_info": {
            "series_description": "T1_tse_sag_320"
        }
    }
    
    router = AlgorithmRouter()
    result = router.route(test_metadata)
    
    print("测试路由结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n关键字段验证:")
    print(f"解剖区域: {result['constraint_space']['A']} (应该为: lumbar)")
    print(f"序列类型: {result['constraint_space']['S']} (应该为: t1)")
    print(f"场强: {result['constraint_space']['F']} (应该为: 1.5t)")
    print(f"并行扫描: {result['adjustments']['noise_correction_required']} (应该为: True)")
    print(f"加速因子: {result['adjustments']['acceleration_factor']} (应该为: 2.0)")


if __name__ == "__main__":
    # 运行测试
    print("运行AlgorithmRouter测试...")
    test_router()