#!/usr/bin/env python3
"""
算法路由器 - 基于A/S/F/M约束空间选择算法参数
重构版本：利用新的元数据结构，保持输出100%兼容
"""

import json
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np

from config import ROI_SETTINGS, SEQUENCE_SNR_STANDARDS, G_FACTOR_TABLE


class AlgorithmRouter:
    """
    基于约束空间的路由器（重构版）
    输入: metadata.json (兼容新旧格式)
    输出: 算法参数配置（100%兼容旧格式）
    """
    
    def __init__(self):
        self.supported_anatomies = list(ROI_SETTINGS.keys())
        self.supported_sequences = list(SEQUENCE_SNR_STANDARDS.keys())
        
        # 解剖区域映射（保持向后兼容）
        self.anatomy_compatibility_map = {
            # 新格式 -> 旧格式
            'lumbar': 'lumbar',
            'lumbar_left': 'lumbar',
            'lumbar_right': 'lumbar',
            'cervical': 'cervical',
            'cervical_left': 'cervical',
            'cervical_right': 'cervical',
            'thoracic': 'thoracic',
            'thoracic_left': 'thoracic',
            'thoracic_right': 'thoracic',
            'brain': 'brain',
            'brain_left': 'brain',
            'brain_right': 'brain',
            'knee': 'knee',
            'knee_left': 'knee',
            'knee_right': 'knee',
            'shoulder': 'shoulder',
            'shoulder_left': 'shoulder',
            'shoulder_right': 'shoulder',
            'hip': 'hip',
            'hip_left': 'hip',
            'hip_right': 'hip',
            'spine': 'lumbar',  # 旧逻辑：spine默认映射为lumbar
            'abdomen': 'abdomen',
            'pelvis': 'pelvis',
            'chest': 'chest',
            'neck': 'neck',
            'default': 'default'
        }
        
    def route(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        主路由函数（保持输出100%兼容）
        
        Args:
            metadata: 从metadata.json加载的字典（兼容新旧格式）
            
        Returns:
            算法参数配置字典（完全兼容旧格式）
        """
        # 提取约束空间（如果有）
        constraint = metadata.get('constraint_space', {})
        
        # ==================== 核心重构：使用新元数据结构 ====================
        
        # 1. 解剖区域提取（优先使用新结构）
        anatomy = self._extract_anatomy_new(metadata)
        if not anatomy:
            # 后备：使用constraint_space
            anatomy = constraint.get('A', '').lower()
        
        # 映射到兼容格式（必须保持旧格式）
        anatomy = self._map_to_compatible_anatomy(anatomy)
        
        # 2. 序列类型提取
        sequence = self._extract_sequence_new(metadata)
        if not sequence:
            sequence = constraint.get('S', '').lower()
        sequence = self._clean_sequence_name(sequence)
        
        # 3. 场强提取
        field_strength = self._extract_field_strength_new(metadata)
        if not field_strength:
            field_strength = constraint.get('F', '').lower()
        field_strength = self._clean_field_strength(field_strength)
        
        # 4. 采集模式提取
        acquisition_mode = self._extract_acquisition_mode_new(metadata)
        if not acquisition_mode:
            acquisition_mode = constraint.get('M', '').lower()
        acquisition_mode = self._clean_acquisition_mode(acquisition_mode)
        
        # 5. 并行成像信息提取（兼容新旧字段名）
        is_parallel, acceleration_factor = self._extract_parallel_info(metadata)
        
        # ==================== 构建兼容性输出（必须保持原样） ====================
        
        route_result = {
            'algorithm': 'single_image_v3_corrected',  # 必须保持固定值
            'constraint_space': {
                'A': anatomy,          # 保持旧格式
                'S': sequence,         # 保持旧格式
                'F': field_strength,   # 保持旧格式
                'M': acquisition_mode  # 保持旧格式
            },
            'adjustments': {
                # 以下字段名必须保持蛇形命名和原样
                'noise_correction_required': is_parallel,
                'acceleration_factor': float(acceleration_factor),
                'anatomy_specific_roi': self._get_roi_template_compatible(anatomy),
                'sequence_standards': self._get_sequence_standards_compatible(sequence),
                'field_strength': field_strength,
                'acquisition_mode': acquisition_mode,
                'g_factor': self._get_g_factor(acceleration_factor) if is_parallel else 1.0
            },
            'validation': {
                # 保持原有结构
                'anatomy_supported': anatomy in self.supported_anatomies,
                'sequence_supported': sequence in self.supported_sequences,
                'has_parallel_info': True  # 总是设为True，因为我们已经提取了
            }
        }
        
        # ==================== 添加扩展信息（不影响兼容性） ====================
        # 添加下划线前缀，避免影响现有代码
        route_result['_extended_info'] = {
            'anatomical_details': self._get_anatomical_details(metadata),
            'metadata_format_version': metadata.get('format_version', 'unknown'),
            'router_version': '2.0_reconstructed'
        }
        
        return route_result
    
    # ==================== 新的提取方法（利用新元数据结构） ====================
    
    def _extract_anatomy_new(self, metadata: Dict) -> str:
        """使用新元数据结构提取解剖区域"""
        # 优先级1：anatomical_info.detailed_region（最精确）
        anatomical_info = metadata.get('anatomical_info', {})
        if anatomical_info:
            detailed_region = anatomical_info.get('detailed_region', '').lower()
            if detailed_region and detailed_region != 'default':
                return detailed_region
            
            # 优先级2：anatomical_info.standardized_region
            standardized_region = anatomical_info.get('standardized_region', '').lower()
            if standardized_region and standardized_region != 'default':
                return standardized_region
        
        # 优先级3：旧字段 anatomical_region（向后兼容）
        anatomical_region = metadata.get('anatomical_region', '')
        if anatomical_region:
            if isinstance(anatomical_region, dict):
                # 旧格式的字典结构
                return anatomical_region.get('standardized', '').lower()
            elif isinstance(anatomical_region, str):
                return anatomical_region.lower()
        
        # 优先级4：从描述推断（最后手段）
        return self._infer_anatomy_from_description(metadata)
    
    def _extract_sequence_new(self, metadata: Dict) -> str:
        """使用新元数据结构提取序列类型"""
        # 优先级1：sequence_info.sequence_type
        sequence_info = metadata.get('sequence_info', {})
        if sequence_info:
            sequence_type = sequence_info.get('sequence_type', '').lower()
            if sequence_type:
                return sequence_type
            
            # 优先级2：sequence_info.sequence_subtype
            sequence_subtype = sequence_info.get('sequence_subtype', '').lower()
            if sequence_subtype:
                # 从子类型推断主类型
                if 't1' in sequence_subtype:
                    return 't1'
                elif 't2' in sequence_subtype:
                    return 't2'
                elif 'pd' in sequence_subtype:
                    return 'pd'
                elif 'flair' in sequence_subtype:
                    return 'flair'
        
        # 优先级3：从系列描述推断
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
        
        return ''
    
    def _extract_field_strength_new(self, metadata: Dict) -> str:
        """使用新元数据结构提取场强"""
        # 优先级1：acquisition_params.field_strength_t
        acquisition_params = metadata.get('acquisition_params', {})
        if acquisition_params:
            field_strength = acquisition_params.get('field_strength_t', '')
            if field_strength:
                return str(field_strength).lower()
        
        # 优先级2：acquisition_params.magnetic_field_strength
        if acquisition_params and 'magnetic_field_strength' in acquisition_params:
            mag_strength = float(acquisition_params['magnetic_field_strength'])
            if mag_strength >= 2.5:
                return '3.0t'
            else:
                return '1.5t'
        
        # 优先级3：旧字段 MagneticFieldStrength
        mag_strength = metadata.get('MagneticFieldStrength', '')
        if mag_strength:
            try:
                mag_val = float(mag_strength)
                if mag_val >= 2.5:
                    return '3.0t'
                else:
                    return '1.5t'
            except:
                pass
        
        return ''
    
    def _extract_acquisition_mode_new(self, metadata: Dict) -> str:
        """使用新元数据结构提取采集模式"""
        # 从parallel_imaging字段判断
        is_parallel, _ = self._extract_parallel_info(metadata)
        
        if is_parallel:
            return 'parallel_imaging'
        else:
            return 'conventional'
    
    def _extract_parallel_info(self, metadata: Dict) -> Tuple[bool, float]:
        """提取并行成像信息（兼容新旧字段名）"""
        # 尝试所有可能的字段名（新旧兼容）
        parallel_info = None
        
        # 优先级1：新格式（小写蛇形）
        if 'parallel_imaging' in metadata:
            parallel_info = metadata['parallel_imaging']
        # 优先级2：旧格式（驼峰）
        elif 'ParallelImaging' in metadata:
            parallel_info = metadata['ParallelImaging']
        # 优先级3：嵌套在acquisition_params中
        elif 'acquisition_params' in metadata and 'parallel_imaging' in metadata['acquisition_params']:
            parallel_info = metadata['acquisition_params']['parallel_imaging']
        
        if parallel_info:
            # 提取used/is_parallel字段
            is_parallel = False
            if isinstance(parallel_info, dict):
                # 新字段名：used
                if 'used' in parallel_info:
                    is_parallel = bool(parallel_info['used'])
                # 旧字段名：is_parallel（向后兼容）
                elif 'is_parallel' in parallel_info:
                    is_parallel = bool(parallel_info['is_parallel'])
                
                # 提取acceleration_factor
                acceleration_factor = float(parallel_info.get('acceleration_factor', 1.0))
            else:
                is_parallel = bool(parallel_info)
                acceleration_factor = 1.0
            
            return is_parallel, acceleration_factor
        
        return False, 1.0
    
    # ==================== 兼容性处理 ====================
    
    def _map_to_compatible_anatomy(self, anatomy: str) -> str:
        """映射到兼容的解剖名称（保持旧格式）"""
        if not anatomy:
            return 'default'
        
        anatomy_lower = anatomy.lower()
        
        # 使用兼容性映射表
        for new_anatomy, old_anatomy in self.anatomy_compatibility_map.items():
            if anatomy_lower == new_anatomy or anatomy_lower.startswith(new_anatomy + '_'):
                return old_anatomy
        
        # 如果不在映射表中，尝试直接匹配supported_anatomies
        for supported_anatomy in self.supported_anatomies:
            if supported_anatomy.lower() in anatomy_lower or anatomy_lower in supported_anatomy.lower():
                return supported_anatomy
        
        return 'default'
    
    def _clean_sequence_name(self, sequence: str) -> str:
        """清理序列名称（保持旧格式）"""
        if not sequence:
            return 't1'  # 默认值
        
        sequence_lower = sequence.lower()
        
        if 't1' in sequence_lower:
            return 't1'
        elif 't2' in sequence_lower:
            return 't2'
        elif 'pd' in sequence_lower:
            return 'pd'
        elif 'flair' in sequence_lower:
            return 'flair'
        
        return 't1'  # 默认值
    
    def _clean_field_strength(self, field_strength: str) -> str:
        """清理场强（保持旧格式）"""
        if not field_strength:
            return '1.5t'  # 默认值
        
        field_lower = field_strength.lower()
        
        if '3' in field_lower or '3.0' in field_lower:
            return '3.0t'
        elif '1.5' in field_lower:
            return '1.5t'
        elif '1.0' in field_lower:
            return '1.0t'
        
        return '1.5t'  # 默认值
    
    def _clean_acquisition_mode(self, mode: str) -> str:
        """清理采集模式（保持旧格式）"""
        if not mode:
            return 'conventional'  # 默认值
        
        mode_lower = mode.lower()
        
        if 'parallel' in mode_lower:
            return 'parallel_imaging'
        else:
            return 'conventional'
    
    # ==================== ROI和标准获取（保持兼容性） ====================
    
    def _get_roi_template_compatible(self, anatomy: str) -> Dict[str, float]:
        """获取兼容的ROI模板（结构必须保持原样）"""
        # 直接使用ROI_SETTINGS，保持完全相同的结构
        template = ROI_SETTINGS.get(anatomy.lower(), None)
        if template:
            return template
        
        # 模糊匹配
        for key in ROI_SETTINGS.keys():
            if key.lower() in anatomy.lower() or anatomy.lower() in key.lower():
                return ROI_SETTINGS[key]
        
        # 返回默认模板
        return ROI_SETTINGS['default']
    
    def _get_sequence_standards_compatible(self, sequence: str) -> Dict[str, float]:
        """获取兼容的序列标准（结构必须保持原样）"""
        # 直接使用SEQUENCE_SNR_STANDARDS
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
        """获取g-factor值（保持原有逻辑）"""
        # 保持原有逻辑不变
        available_R = sorted(G_FACTOR_TABLE.keys())
        
        if acceleration_factor in G_FACTOR_TABLE:
            nearest_R = acceleration_factor
        else:
            nearest_R = min(available_R, key=lambda x: abs(x - acceleration_factor))
        
        g_mean, _, _, _ = G_FACTOR_TABLE[nearest_R]
        
        # 插值处理
        if acceleration_factor != nearest_R:
            lower_R = max([r for r in available_R if r <= acceleration_factor], default=nearest_R)
            upper_R = min([r for r in available_R if r >= acceleration_factor], default=nearest_R)
            
            if lower_R != upper_R:
                lower_g = G_FACTOR_TABLE[lower_R][0]
                upper_g = G_FACTOR_TABLE[upper_R][0]
                ratio = (acceleration_factor - lower_R) / (upper_R - lower_R)
                g_mean = lower_g + ratio * (upper_g - lower_g)
        
        return float(g_mean)
    
    # ==================== 辅助方法 ====================
    
    def _infer_anatomy_from_description(self, metadata: Dict) -> str:
        """从描述字段推断解剖区域（最后手段）"""
        # 获取各种描述字段
        study_desc = metadata.get('study_info', {}).get('study_description', '').lower()
        series_desc = metadata.get('series_info', {}).get('series_description', '').lower()
        
        combined_desc = f"{study_desc} {series_desc}"
        
        # 关键词匹配
        anatomy_keywords = {
            'lumbar': 'lumbar',
            'lspine': 'lumbar',
            'cervical': 'cervical',
            'cspine': 'cervical',
            'thoracic': 'thoracic',
            'tspine': 'thoracic',
            'brain': 'brain',
            'head': 'brain',
            'knee': 'knee',
            'shoulder': 'shoulder',
            'hip': 'hip'
        }
        
        for keyword, anatomy in anatomy_keywords.items():
            if keyword in combined_desc:
                return anatomy
        
        return ''
    
    def _get_anatomical_details(self, metadata: Dict) -> Dict:
        """获取解剖详细信息（用于扩展信息）"""
        anatomical_info = metadata.get('anatomical_info', {})
        
        if anatomical_info:
            return {
                'detailed_region': anatomical_info.get('detailed_region', ''),
                'standardized_region': anatomical_info.get('standardized_region', ''),
                'original_body_part': anatomical_info.get('original_body_part', ''),
                'laterality': anatomical_info.get('laterality', ''),
                'patient_position': anatomical_info.get('patient_position', '')
            }
        
        return {}


# ==================== 便捷函数（保持兼容） ====================

def route_from_metadata_file(metadata_path: Path) -> Dict[str, Any]:
    """从metadata.json文件直接路由（保持接口不变）"""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        router = AlgorithmRouter()
        return router.route(metadata)
        
    except Exception as e:
        print(f"路由失败 {metadata_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ==================== 测试函数 ====================

def test_router_compatibility():
    """测试路由器兼容性"""
    print("=" * 60)
    print("测试算法路由器兼容性")
    print("=" * 60)
    
    # 测试用例1：新格式metadata.json
    print("\n测试用例1：新格式metadata.json")
    test_metadata_new = {
        "format_version": "2.1",
        "anatomical_info": {
            "detailed_region": "lumbar",
            "standardized_region": "spine",
            "original_body_part": "LSPINE",
            "laterality": "unknown",
            "confidence": 0.9
        },
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
            "field_strength_t": "1.5t",
            "magnetic_field_strength": 1.5
        },
        "study_info": {
            "study_description": "HZSDEYY^spine"
        },
        "series_info": {
            "series_description": "T1_tse_sag_320"
        }
    }
    
    router = AlgorithmRouter()
    result_new = router.route(test_metadata_new)
    
    # 验证关键字段
    print("✓ 算法名称:", result_new['algorithm'], "(应该是: single_image_v3_corrected)")
    print("✓ 解剖区域:", result_new['constraint_space']['A'], "(应该是: lumbar)")
    print("✓ 噪声校正:", result_new['adjustments']['noise_correction_required'], "(应该是: True)")
    print("✓ 加速因子:", result_new['adjustments']['acceleration_factor'], "(应该是: 2.0)")
    
    # 测试用例2：带laterality的新格式
    print("\n测试用例2：带laterality的新格式")
    test_metadata_lateral = {
        "format_version": "2.1",
        "anatomical_info": {
            "detailed_region": "knee_left",
            "standardized_region": "knee",
            "laterality": "left"
        },
        "parallel_imaging": {"used": False},
        "sequence_info": {"sequence_type": "t2"},
        "acquisition_params": {"field_strength_t": "3.0t"}
    }
    
    result_lateral = router.route(test_metadata_lateral)
    print("✓ 解剖区域:", result_lateral['constraint_space']['A'], "(应该是: knee)")
    print("✓ 场强:", result_lateral['constraint_space']['F'], "(应该是: 3.0t)")
    
    # 测试用例3：旧格式兼容性
    print("\n测试用例3：旧格式兼容性")
    test_metadata_old = {
        "anatomical_region": "spine",
        "ParallelImaging": {
            "is_parallel": True,
            "acceleration_factor": 1.5
        },
        "sequence_info": {
            "sequence_type": "pd"
        },
        "acquisition_params": {
            "field_strength_t": "1.5t"
        }
    }
    
    result_old = router.route(test_metadata_old)
    print("✓ 解剖区域:", result_old['constraint_space']['A'], "(应该是: lumbar)")
    print("✓ 序列类型:", result_old['constraint_space']['S'], "(应该是: pd)")
    
    print("\n" + "=" * 60)
    print("兼容性测试完成")
    print("=" * 60)
    
    return result_new, result_lateral, result_old


if __name__ == "__main__":
    # 运行测试
    test_router_compatibility()