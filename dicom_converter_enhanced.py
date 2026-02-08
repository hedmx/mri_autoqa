#!/usr/bin/env python3
"""
Enhanced DICOM to NIfTI Converter with Standardized Metadata
Version: 2.3 - Optimized with Enhanced Metadata Strategy
优化要点：
1. 保持代码2的标准化输出结构
2. 集成代码1的高级并行采集检测和Sequence类型判断
3. 优化元数据信息数据策略：物理参数、采样技术参数、解剖部位精准信息
4. 隐私保护：不采集病人姓名
5. 新增分层解剖信息采集
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import pydicom
import nibabel as nib
import traceback
import re
import logging

# ==================== 导入统一配置 ====================
try:
    from config import (
        # 文件配置
        SCAN_FILENAME,
        METADATA_FILENAME,
        
        # 标准化映射表
        ANATOMICAL_REGION_MAPPING,
        ANATOMICAL_DETAILED_MAPPING,
        SEQUENCE_TYPE_MAPPING,
        SEQUENCE_SUBTYPE_MAPPING,
        FIELD_STRENGTH_MAPPING,
        
        # 新增：参数提取配置
        PHYSICAL_PARAMETERS_TO_EXTRACT,
        SAMPLING_PARAMETERS_TO_EXTRACT,
        COIL_INFO_FIELDS,
        VENDOR_INFO_FIELDS,
        
        # 并行成像配置
        PARALLEL_IMAGING_METHOD_MAPPING,
        VENDOR_PARALLEL_FIELDS,
        
        # 质量控制配置
        PARAMETER_COMPLETENESS_THRESHOLD,
        CRITICAL_PARAMETER_WEIGHTS,
        CONFIDENCE_LEVELS,
        
        # 硬约束配置
        HARD_CONSTRAINTS,
        
        # 错误处理配置
        ERROR_HANDLING,
        
        # 版本配置
        VERSION_CONFIG
    )
    
    # 从错误处理配置中提取错误码
    ERROR_CODES = ERROR_HANDLING.get('error_codes', {
        'FILE_NOT_FOUND': 'E001',
        'DICOM_READ_ERROR': 'E002',
        'INSUFFICIENT_DATA': 'E003',
        'IMAGE_TOO_LARGE': 'E004',
        'INVALID_DIMENSIONS': 'E005',
        'METADATA_MISSING': 'E006',
        'CONVERSION_ERROR': 'E007',
        'HARD_CONSTRAINT_VIOLATED': 'E008'
    })
    
except ImportError as e:
    print(f"Error: Could not import config.py - {e}")
    print("Make sure config.py exists in the same directory")
    exit(1)

# ==================== 核心类定义 ====================

class VendorAwareParallelDetector:
    """厂商感知的Parallel Imaging检测器"""
    
    VENDOR_CONFIGS = {
        'SIEMENS': {
            'patterns': ['SIEMENS', 'Siemens'],
            'standard_fields': [
                'ParallelReductionFactorInPlane',
                'ParallelAcquisitionTechnique'
            ],
            'private_tags': [
                (0x0051, 0x1011),  # PATModeText (关键!)
                (0x0019, 0x100a),  # 可能包含Acceleration Factor
                (0x0019, 0x109a),  # 可能的PAT标签
                (0x0051, 0x100c),  # 可能的MatrixCoilMode
            ],
            'detection_methods': [
                'parse_pat_text',      # 解析p2, p3等
                'check_private_tags',  # 检查私有标签
                'check_matrix_mode',   # 检查MatrixCoilMode
                'infer_from_matrices'  # 从矩阵推断
            ]
        },
        'GE': {
            'patterns': ['GE', 'GENERAL ELECTRIC', 'G.E.'],
            'standard_fields': [
                'AccelerationFactorPE',
                'AssetRFactors',
                'ParallelImaging',
                'ParallelReductionFactorInPlane',
                'AccelerationFactor',
                'ARC'
            ],
            'private_tags': [],
            'detection_methods': [
                'check_acceleration_factor',
                'check_asset_rfactors',
                'check_parallel_imaging'
            ]
        },
        'PHILIPS': {
            'patterns': ['PHILIPS', 'Philips'],
            'standard_fields': [
                'SENSE',
                'ParallelReductionFactor',
                'PartialFourier'
            ],
            'private_tags': [],
            'detection_methods': [
                'check_sense_field',
                'check_partial_fourier'
            ]
        },
        'TOSHIBA': {
            'patterns': ['TOSHIBA', 'Toshiba'],
            'standard_fields': [
                'ParallelImagingFactor',
                'SpeedUpFactor'
            ],
            'private_tags': [],
            'detection_methods': ['check_speedup_factor']
        },
        'HITACHI': {
            'patterns': ['HITACHI', 'Hitachi'],
            'standard_fields': ['ParallelImaging'],
            'private_tags': [],
            'detection_methods': ['check_standard_fields']
        }
    }
    
    @staticmethod
    def identify_vendor(ds) -> str:
        """识别设备厂商"""
        manufacturer = str(getattr(ds, 'Manufacturer', '')).upper()
        
        for vendor, config in VendorAwareParallelDetector.VENDOR_CONFIGS.items():
            for pattern in config['patterns']:
                if pattern.upper() in manufacturer:
                    return vendor
        
        # 从其他字段推断
        if hasattr(ds, 'ManufacturersModelName'):
            model = str(ds.ManufacturersModelName).upper()
            if 'SIEMENS' in model or 'AERA' in model or 'SKYRA' in model or 'VERIO' in model:
                return 'SIEMENS'
            elif 'GE' in model or 'SIGNA' in model:
                return 'GE'
            elif 'PHILIPS' in model or 'INGENIA' in model:
                return 'PHILIPS'
        
        return 'UNKNOWN'
    
    @staticmethod
    def detect(ds) -> Dict[str, Any]:
        """主检测方法"""
        vendor = VendorAwareParallelDetector.identify_vendor(ds)
        
        result = {
            'vendor': vendor,
            'is_parallel': False,
            'acceleration_factor': 1.0,
            'technique': 'none',
            'confidence': 0.0,
            'detection_methods': []
        }
        
        # 根据厂商调用相应的检测方法
        if vendor == 'SIEMENS':
            VendorAwareParallelDetector._detect_siemens(ds, result)
        elif vendor == 'GE':
            VendorAwareParallelDetector._detect_ge(ds, result)
        elif vendor == 'PHILIPS':
            VendorAwareParallelDetector._detect_philips(ds, result)
        else:
            VendorAwareParallelDetector._detect_generic(ds, result)
        
        # 标准化方法名称
        if result['technique'] != 'none' and result['technique'] in PARALLEL_IMAGING_METHOD_MAPPING:
            result['technique'] = PARALLEL_IMAGING_METHOD_MAPPING[result['technique']]
        
        return result
    
    @staticmethod
    def _detect_siemens(ds, result) -> Dict[str, Any]:
        """检测Siemens设备的Parallel Imaging"""
        # 方法1: 检查PATModeText私有标签 (关键!)
        pat_tag = (0x0051, 0x1011)
        if pat_tag in ds:
            pat_value = ds[pat_tag].value
            if pat_value:
                try:
                    pat_str = pat_value.decode('latin-1', errors='ignore') if isinstance(pat_value, bytes) else str(pat_value)
                    # 查找p2, p3等模式
                    match = re.search(r'[Pp](\d+(?:\.\d+)?)', pat_str)
                    if match:
                        factor = float(match.group(1))
                        if factor > 1.0:
                            result['is_parallel'] = True
                            result['acceleration_factor'] = factor
                            result['confidence'] = 0.9
                            result['detection_methods'].append('PATModeText')
                except Exception:
                    pass
        
        # 方法2: 检查标准字段
        if hasattr(ds, 'ParallelReductionFactorInPlane'):
            try:
                factor = float(ds.ParallelReductionFactorInPlane)
                if factor > 1.0:
                    result['is_parallel'] = True
                    result['acceleration_factor'] = max(result['acceleration_factor'], factor)
                    result['confidence'] = max(result['confidence'], 0.8)
                    result['detection_methods'].append('StandardField')
            except:
                pass
        
        # 方法3: 检查MatrixCoilMode
        if hasattr(ds, 'MatrixCoilMode'):
            mode = str(ds.MatrixCoilMode).upper()
            if 'GRAPPA' in mode:
                result['technique'] = 'GRAPPA'
                result['is_parallel'] = True
                result['confidence'] = max(result['confidence'], 0.85)
                result['detection_methods'].append('MatrixCoilMode')
            elif 'SENSE' in mode:
                result['technique'] = 'SENSE'
                result['is_parallel'] = True
                result['confidence'] = max(result['confidence'], 0.85)
                result['detection_methods'].append('MatrixCoilMode')
        
        # 方法4: 从矩阵推断
        if not result['is_parallel']:
            factor = VendorAwareParallelDetector._infer_from_matrices(ds)
            if factor > 1.0:
                result['is_parallel'] = True
                result['acceleration_factor'] = factor
                result['confidence'] = 0.7
                result['detection_methods'].append('MatrixInference')
        
        return result
    
    @staticmethod
    def _detect_ge(ds, result) -> Dict[str, Any]:
        """检测GE设备的Parallel Imaging"""
        # 方法1: 检查AccelerationFactorPE
        if hasattr(ds, 'AccelerationFactorPE'):
            try:
                factor = float(ds.AccelerationFactorPE)
                if factor > 1.0:
                    result['is_parallel'] = True
                    result['acceleration_factor'] = factor
                    result['confidence'] = 0.9
                    result['detection_methods'].append('AccelerationFactorPE')
            except:
                pass
        
        # 方法2: 检查AssetRFactors (GE ASSET技术)
        if hasattr(ds, 'AssetRFactors'):
            try:
                factors = ds.AssetRFactors
                if isinstance(factors, (list, tuple)) and len(factors) >= 2:
                    factor = float(factors[0])  # 通常第一个是相位编码加速
                    if factor > 1.0:
                        result['is_parallel'] = True
                        result['acceleration_factor'] = max(result['acceleration_factor'], factor)
                        result['confidence'] = max(result['confidence'], 0.9)
                        result['technique'] = 'ASSET'
                        result['detection_methods'].append('AssetRFactors')
            except:
                pass
        
        # 方法3: 检查ParallelImaging字段
        if hasattr(ds, 'ParallelImaging'):
            value = str(ds.ParallelImaging).upper()
            if 'YES' in value or 'ON' in value or 'ASSET' in value:
                result['is_parallel'] = True
                result['confidence'] = max(result['confidence'], 0.8)
                result['detection_methods'].append('ParallelImaging')
        
        # 方法4: 检查ARC (GE的ARC技术)
        if hasattr(ds, 'ARC'):
            try:
                arc_value = str(ds.ARC).upper()
                if 'YES' in arc_value or 'ON' in arc_value:
                    result['is_parallel'] = True
                    result['technique'] = 'ARC'
                    result['confidence'] = max(result['confidence'], 0.85)
                    result['detection_methods'].append('ARC')
            except:
                pass
        
        return result
    
    @staticmethod
    def _detect_philips(ds, result) -> Dict[str, Any]:
        """检测Philips设备的Parallel Imaging"""
        # 方法1: 检查SENSE字段
        if hasattr(ds, 'SENSE'):
            sense_value = str(ds.SENSE).upper()
            if 'YES' in sense_value or 'ON' in sense_value:
                result['is_parallel'] = True
                result['technique'] = 'SENSE'
                result['confidence'] = 0.9
                result['detection_methods'].append('SENSE')
            
            # 尝试从SENSE值提取Acceleration Factor
            match = re.search(r'(\d+(?:\.\d+)?)', sense_value)
            if match:
                factor = float(match.group(1))
                if factor > 1.0:
                    result['acceleration_factor'] = factor
        
        # 方法2: 检查PartialFourier (可能相关)
        if hasattr(ds, 'PartialFourier'):
            try:
                pf = float(ds.PartialFourier)
                if pf < 1.0 and pf > 0.5:  # 部分傅里叶可能暗示Parallel
                    result['is_parallel'] = True
                    result['confidence'] = max(result['confidence'], 0.6)
                    result['detection_methods'].append('PartialFourier')
            except:
                pass
        
        return result
    
    @staticmethod
    def _detect_generic(ds, result) -> Dict[str, Any]:
        """通用检测方法（用于Unknown厂商）"""
        # 检查所有可能的标准字段
        standard_fields = [
            'ParallelReductionFactorInPlane',
            'AccelerationFactorPE',
            'ParallelImaging',
            'SENSE',
            'AssetRFactors',
            'ARC'
        ]
        
        for field in standard_fields:
            if hasattr(ds, field):
                try:
                    value = getattr(ds, field)
                    if isinstance(value, (int, float)):
                        if float(value) > 1.0:
                            result['is_parallel'] = True
                            result['acceleration_factor'] = max(result['acceleration_factor'], float(value))
                            result['detection_methods'].append(field)
                    elif isinstance(value, str):
                        if any(keyword in value.upper() for keyword in ['YES', 'ON', 'GRAPPA', 'SENSE', 'ASSET']):
                            result['is_parallel'] = True
                            result['detection_methods'].append(field)
                except:
                    pass
        
        # 从矩阵推断
        if not result['is_parallel']:
            factor = VendorAwareParallelDetector._infer_from_matrices(ds)
            if factor > 1.0:
                result['is_parallel'] = True
                result['acceleration_factor'] = factor
                result['detection_methods'].append('MatrixInference')
                result['confidence'] = 0.6
        
        return result
    
    @staticmethod
    def _infer_from_matrices(ds) -> float:
        """从采集矩阵和重建矩阵推断Acceleration Factor"""
        try:
            if hasattr(ds, 'AcquisitionMatrix') and hasattr(ds, 'Rows'):
                acq_matrix = ds.AcquisitionMatrix
                recon_rows = ds.Rows
                
                if isinstance(acq_matrix, (list, tuple)) and len(acq_matrix) >= 4:
                    # AcquisitionMatrix格式: [Frequency, 相位, ?, ?]
                    acq_pe = acq_matrix[1] if acq_matrix[1] != 0 else acq_matrix[3]
                    
                    if acq_pe > 0 and recon_rows > 0 and acq_pe < recon_rows:
                        factor = recon_rows / acq_pe
                        # 检查是否为整数或常见的小数
                        common_factors = [1.5, 1.75, 2.0, 2.5, 3.0]
                        for cf in common_factors:
                            if abs(factor - cf) < 0.1:
                                return cf
                        return round(factor, 2)
        except:
            pass
        
        return 1.0


class SequenceTypeDetector:
    """Sequence Type检测器（增强版）"""
    
    @staticmethod
    def detect_sequence_type(ds) -> Tuple[str, str]:
        """
        检测Sequence Type和子类型
        
        返回: (sequence_type, sequence_subtype)
        """
        # 从多个字段获取序列描述
        series_desc = str(getattr(ds, 'SeriesDescription', '')).upper()
        protocol_name = str(getattr(ds, 'ProtocolName', '')).upper()
        manufacturer_seq_name = str(getattr(ds, 'SequenceName', '')).upper()
        
        combined_text = f"{series_desc} {protocol_name} {manufacturer_seq_name}"
        
        # 1. 使用config.py的映射（优先）
        sequence_type = 't1'  # 默认值
        if SEQUENCE_TYPE_MAPPING:
            for dicom_term, standard_key in SEQUENCE_TYPE_MAPPING.items():
                if dicom_term in combined_text:
                    sequence_type = standard_key
                    break
        
        # 2. 如果映射未找到，使用TR/TE逻辑判断
        if sequence_type == 't1':  # 如果还是默认值
            if hasattr(ds, 'EchoTime') and hasattr(ds, 'RepetitionTime'):
                try:
                    te = float(ds.EchoTime)  # 单位已经是ms
                    tr = float(ds.RepetitionTime)  # 单位已经是ms
                    
                    # 根据TR/TE判断Sequence Type
                    if te < 30 and tr < 1000:  # T1特征
                        sequence_type = 't1'
                    elif te > 50 and tr > 1500:  # T2特征
                        sequence_type = 't2'
                    elif te < 30 and tr > 1500:  # PD特征
                        sequence_type = 'pd'
                    elif te > 80 and tr > 2000:  # FLAIR特征
                        sequence_type = 'flair'
                except:
                    pass
        
        # 3. 确定子类型
        subtype = 'standard'
        if SEQUENCE_SUBTYPE_MAPPING:
            for dicom_term, standard_subtype in SEQUENCE_SUBTYPE_MAPPING.items():
                if dicom_term in combined_text:
                    subtype = standard_subtype
                    break
        
        # 4. 如果没有匹配，使用启发式规则
        if subtype == 'standard':
            if 'MPRAGE' in combined_text:
                subtype = 'mprage'
            elif 'SPGR' in combined_text:
                subtype = 'spgr'
            elif 'FSE' in combined_text or 'TSE' in combined_text:
                subtype = 'fse_tse'
            elif 'GRE' in combined_text or 'GR' in combined_text:
                subtype = 'gre'
            elif 'HASTE' in combined_text:
                subtype = 'haste'
        
        return sequence_type, subtype
    
    @staticmethod
    def detect_fat_suppression(ds) -> bool:
        """检测脂肪抑制"""
        # 检查SequenceVariant
        if hasattr(ds, 'SequenceVariant'):
            variant = str(ds.SequenceVariant).upper()
            if any(fs in variant for fs in ['FS', 'SP', 'SPIR', 'FAT SAT']):
                return True
        
        # 检查专用字段
        if hasattr(ds, 'FatSuppression'):
            fs = str(ds.FatSuppression).upper()
            if fs in ['YES', 'ON', 'SPECTRAL']:
                return True
        
        # 检查Sequence描述
        if hasattr(ds, 'SeriesDescription'):
            desc = str(ds.SeriesDescription).lower()
            if any(fs in desc for fs in ['fs', 'fat', 'fat_sat', 'fatsat']):
                return True
        
        return False


class AnatomicalRegionStandardizer:
    """解剖区域标准化器（增强版）"""
    
    @staticmethod
    def standardize(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取并标准化解剖区域（返回多层结构）"""
        
        # 1. 从多个字段获取原始信息
        body_part = str(getattr(ds, 'BodyPartExamined', '')).upper().strip()
        series_desc = str(getattr(ds, 'SeriesDescription', '')).upper()
        protocol_name = str(getattr(ds, 'ProtocolName', '')).upper()
        
        # 2. 确定标准化区域（使用映射表）
        standardized_region = 'default'
        
        # 优先级1: BodyPartExamined
        if body_part and body_part in ANATOMICAL_REGION_MAPPING:
            standardized_region = ANATOMICAL_REGION_MAPPING[body_part]
        
        # 优先级2: SeriesDescription中的解剖术语
        if standardized_region == 'default':
            for term, region in ANATOMICAL_REGION_MAPPING.items():
                if term != 'UNKNOWN' and term in series_desc:
                    standardized_region = region
                    break
        
        # 优先级3: ProtocolName中的解剖术语
        if standardized_region == 'default':
            for term, region in ANATOMICAL_REGION_MAPPING.items():
                if term != 'UNKNOWN' and term in protocol_name:
                    standardized_region = region
                    break
        
        # 3. 确定详细区域（使用详细映射）
        detailed_region = 'default'
        
        # 从BodyPart推断详细区域
        if body_part and body_part in ANATOMICAL_DETAILED_MAPPING:
            detailed_region = ANATOMICAL_DETAILED_MAPPING[body_part]
        
        # 如果详细区域还是default，尝试从描述推断
        if detailed_region == 'default':
            combined_text = f"{body_part} {series_desc} {protocol_name}"
            
            # 检查详细映射表中的所有术语
            for term, region in ANATOMICAL_DETAILED_MAPPING.items():
                if term != 'UNKNOWN' and term in combined_text:
                    detailed_region = region
                    break
            
            # 如果还是没有找到，使用标准化区域
            if detailed_region == 'default':
                detailed_region = standardized_region
        
        # 4. 确定左右侧（如果适用）
        laterality = AnatomicalRegionStandardizer._detect_laterality(
            body_part, series_desc, protocol_name
        )
        
        # 5. 如果检测到左右侧，更新详细区域
        if laterality != 'bilateral' and detailed_region in ['knee', 'shoulder', 'hip', 'ankle', 'wrist', 'elbow']:
            detailed_region = f"{detailed_region}_{laterality}"
        
        # 6. 获取患者体位
        patient_position = str(getattr(ds, 'PatientPosition', 'UNKNOWN'))
        
        # 7. 获取图像方向信息
        image_orientation = {
            'text': str(getattr(ds, 'ImageOrientationText', 'UNKNOWN')),
            'dicom': list(getattr(ds, 'ImageOrientationPatientDICOM', [])) if hasattr(ds, 'ImageOrientationPatientDICOM') else []
        }
        
        # 8. 计算置信度
        confidence = AnatomicalRegionStandardizer._calculate_confidence(
            body_part, series_desc, protocol_name
        )
        
        return {
            'standardized_region': standardized_region,
            'detailed_region': detailed_region,
            'original_body_part': body_part,
            'laterality': laterality,
            'patient_position': patient_position,
            'image_orientation': image_orientation,
            'confidence': confidence
        }
    
    @staticmethod
    def _detect_laterality(body_part: str, series_desc: str, protocol_name: str) -> str:
        """检测左右侧"""
        combined = f"{body_part} {series_desc} {protocol_name}".upper()
        
        # 左/右侧关键词
        left_indicators = ['LEFT', 'L_', '_L', 'LT', 'LFT', 'SINISTER', 'L ', ' L']
        right_indicators = ['RIGHT', 'R_', '_R', 'RT', 'RGT', 'DEXTER', 'R ', ' R']
        bilateral_indicators = ['BILATERAL', 'BOTH', 'BIL', 'B_']
        
        # 检查双侧
        for indicator in bilateral_indicators:
            if indicator in combined:
                return 'bilateral'
        
        # 检查左侧
        left_count = sum(1 for indicator in left_indicators if indicator in combined)
        right_count = sum(1 for indicator in right_indicators if indicator in combined)
        
        if left_count > 0 and right_count == 0:
            return 'left'
        elif right_count > 0 and left_count == 0:
            return 'right'
        elif left_count > 0 and right_count > 0:
            return 'bilateral'
        else:
            return 'unknown'
    
    @staticmethod
    def _calculate_confidence(body_part: str, series_desc: str, protocol_name: str) -> float:
        """计算解剖区域识别的置信度"""
        confidence = 0.0
        
        # BodyPartExamined明确存在：+0.4
        if body_part and body_part not in ['UNKNOWN', '', 'NONE']:
            confidence += 0.4
        
        # SeriesDescription包含解剖术语：+0.3
        has_anatomical_term = False
        for term in ANATOMICAL_REGION_MAPPING.keys():
            if term != 'UNKNOWN' and term in series_desc.upper():
                has_anatomical_term = True
                break
        
        if has_anatomical_term:
            confidence += 0.3
        
        # ProtocolName包含解剖术语：+0.2
        has_protocol_term = False
        for term in ANATOMICAL_REGION_MAPPING.keys():
            if term != 'UNKNOWN' and term in protocol_name.upper():
                has_protocol_term = True
                break
        
        if has_protocol_term:
            confidence += 0.2
        
        # 左右侧明确：+0.1
        if AnatomicalRegionStandardizer._detect_laterality(body_part, series_desc, protocol_name) in ['left', 'right']:
            confidence += 0.1
        
        return min(confidence, 1.0)


class PrivacyAwarePatientInfoExtractor:
    """隐私感知的患者信息提取器"""
    
    @staticmethod
    def extract(ds: pydicom.Dataset) -> Dict[str, str]:
        """提取去隐私化的患者信息"""
        patient_id = str(getattr(ds, 'PatientID', 'UNKNOWN')).strip()
        
        # 生成匿名ID（可选加盐哈希）
        import hashlib
        salt = "MEDICAL_ANON_SALT_2024"
        anonymized_id = hashlib.sha256(f"{patient_id}{salt}".encode()).hexdigest()[:16]
        
        return {
            'patient_id': patient_id,
            'anonymized_id': f"PT_{patient_id}_ANON",  # 或使用哈希值
            'patient_sex': str(getattr(ds, 'PatientSex', 'U')).strip(),
            'patient_age': str(getattr(ds, 'PatientAge', '000Y')).strip(),
            'patient_weight': str(getattr(ds, 'PatientWeight', 'UNKNOWN')).strip() if hasattr(ds, 'PatientWeight') else 'UNKNOWN'
            # 注意：不包含PatientName
        }


class OptimizedMetadataBuilder:
    """优化版元数据构建器"""
    
    @staticmethod
    def build_from_dicom(ds: pydicom.Dataset, dicom_files: List[Path]) -> Dict[str, Any]:
        """从DICOM构建标准化元数据"""
        
        metadata = {
            # 版本信息
            'format_version': VERSION_CONFIG.get('metadata_schema_version', '2.1'),
            'generated_date': datetime.now().isoformat(),
            'conversion_tool': f"MRI_AutoQA_Converter_{VERSION_CONFIG.get('current_version', '2.3')}",
            
            # 核心信息（隐私保护）
            'patient_info': PrivacyAwarePatientInfoExtractor.extract(ds),
            'study_info': OptimizedMetadataBuilder._extract_study_info(ds),
            'series_info': OptimizedMetadataBuilder._extract_series_info(ds),
            
            # 增强的参数组
            'acquisition_params': OptimizedMetadataBuilder._extract_acquisition_params(ds),
            'physical_parameters': OptimizedMetadataBuilder._extract_physical_parameters(ds),
            'sampling_parameters': OptimizedMetadataBuilder._extract_sampling_parameters(ds),
            'sequence_info': OptimizedMetadataBuilder._extract_sequence_info(ds),
            'parallel_imaging': OptimizedMetadataBuilder._extract_parallel_imaging(ds),
            
            # 增强的解剖和图像信息
            'anatomical_info': AnatomicalRegionStandardizer.standardize(ds),
            'coil_info': OptimizedMetadataBuilder._extract_coil_info(ds),
            'vendor_info': OptimizedMetadataBuilder._extract_vendor_info(ds),
            'image_characteristics': OptimizedMetadataBuilder._extract_image_characteristics(ds, dicom_files),
            
            # 质量标志（增强）
            'quality_flags': OptimizedMetadataBuilder._calculate_quality_flags(ds, dicom_files),
            
            # 转换信息
            'conversion_info': OptimizedMetadataBuilder._extract_conversion_info(dicom_files)
        }
        
        return metadata
    
    @staticmethod
    def _extract_study_info(ds: pydicom.Dataset) -> Dict[str, str]:
        """提取研究信息"""
        return {
            'study_date': str(getattr(ds, 'StudyDate', 'UNKNOWN')).strip(),
            'study_time': str(getattr(ds, 'StudyTime', 'UNKNOWN')).strip(),
            'study_description': str(getattr(ds, 'StudyDescription', 'UNKNOWN')).strip(),
            'accession_number': str(getattr(ds, 'AccessionNumber', 'UNKNOWN')).strip(),
            'referring_physician': str(getattr(ds, 'ReferringPhysicianName', 'UNKNOWN')).strip() if hasattr(ds, 'ReferringPhysicianName') else 'UNKNOWN'
        }
    
    @staticmethod
    def _extract_series_info(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取序列信息"""
        return {
            'series_number': str(getattr(ds, 'SeriesNumber', 'UNKNOWN')).strip(),
            'series_description': str(getattr(ds, 'SeriesDescription', 'UNKNOWN')).strip(),
            'modality': str(getattr(ds, 'Modality', 'MR')).strip(),
            'protocol_name': str(getattr(ds, 'ProtocolName', 'UNKNOWN')).strip(),
            'raw_sequence_name': str(getattr(ds, 'SequenceName', 'UNKNOWN')).strip(),
            'scanning_sequence': str(getattr(ds, 'ScanningSequence', 'UNKNOWN')).strip() if hasattr(ds, 'ScanningSequence') else 'UNKNOWN',
            'sequence_variant': str(getattr(ds, 'SequenceVariant', 'UNKNOWN')).strip() if hasattr(ds, 'SequenceVariant') else 'UNKNOWN'
        }
    
    @staticmethod
    def _extract_acquisition_params(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取采集参数"""
        # 场强标准化
        field_strength = float(getattr(ds, 'MagneticFieldStrength', 1.5))
        
        # 查找最接近的标准场强
        field_strength_t = '1.5t'  # 默认值
        if FIELD_STRENGTH_MAPPING:
            available_strengths = list(FIELD_STRENGTH_MAPPING.keys())
            if available_strengths:
                closest_strength = min(available_strengths, key=lambda x: abs(x - field_strength))
                field_strength_t = FIELD_STRENGTH_MAPPING[closest_strength]
        
        # 像素间距处理
        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        if isinstance(pixel_spacing, pydicom.multival.MultiValue):
            pixel_spacing = [float(ps) for ps in pixel_spacing]
        elif isinstance(pixel_spacing, (list, tuple)):
            pixel_spacing = [float(ps) for ps in pixel_spacing]
        else:
            pixel_spacing = [1.0, 1.0]
        
        # 检查脂肪抑制
        fat_suppressed = SequenceTypeDetector.detect_fat_suppression(ds)
        
        return {
            'field_strength_t': field_strength_t,
            'magnetic_field_strength': field_strength,
            'tr_ms': float(getattr(ds, 'RepetitionTime', 0.0)),
            'te_ms': float(getattr(ds, 'EchoTime', 0.0)),
            'flip_angle_deg': float(getattr(ds, 'FlipAngle', 0.0)),
            'slice_thickness_mm': float(getattr(ds, 'SliceThickness', 0.0)),
            'pixel_spacing_mm': pixel_spacing,
            'echo_train_length': int(getattr(ds, 'EchoTrainLength', 1)),
            'number_of_averages': float(getattr(ds, 'NumberOfAverages', 1.0)),
            'fat_suppressed': fat_suppressed,
            'contrast_agent_used': str(getattr(ds, 'ContrastBolusAgent', 'NO')).strip() if hasattr(ds, 'ContrastBolusAgent') else 'UNKNOWN'
        }
    
    @staticmethod
    def _extract_physical_parameters(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取物理参数"""
        params = {}
        
        for param_name in PHYSICAL_PARAMETERS_TO_EXTRACT:
            if hasattr(ds, param_name):
                try:
                    value = getattr(ds, param_name)
                    # 转换为标准格式
                    key_name = param_name.lower()
                    if isinstance(value, (list, pydicom.multival.MultiValue)):
                        params[key_name] = [float(v) for v in value]
                    elif isinstance(value, (int, float)):
                        params[key_name] = float(value)
                    else:
                        params[key_name] = str(value)
                except:
                    params[param_name.lower()] = 'ERROR'
        
        # 特殊处理场强和频率（如果在主参数中已提取，这里可以跳过）
        if 'magnetic_field_strength' not in params and hasattr(ds, 'MagneticFieldStrength'):
            params['magnetic_field_strength'] = float(ds.MagneticFieldStrength)
        
        if 'imaging_frequency' not in params and hasattr(ds, 'ImagingFrequency'):
            params['imaging_frequency'] = float(ds.ImagingFrequency)
        
        # 添加额外的物理参数
        if hasattr(ds, 'EchoSpacing'):
            try:
                params['echo_spacing'] = float(ds.EchoSpacing)
            except:
                pass
        
        if hasattr(ds, 'TotalScanTime'):
            try:
                params['total_scan_time'] = float(ds.TotalScanTime)
            except:
                pass
        
        return params
    
    @staticmethod
    def _extract_sampling_parameters(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取采样参数"""
        params = {}
        
        for param_name in SAMPLING_PARAMETERS_TO_EXTRACT:
            if hasattr(ds, param_name):
                try:
                    value = getattr(ds, param_name)
                    # 转换为标准格式
                    key_name = param_name.lower()
                    if isinstance(value, (list, pydicom.multival.MultiValue)):
                        params[key_name] = [float(v) for v in value]
                    elif isinstance(value, (int, float)):
                        params[key_name] = float(value)
                    else:
                        params[key_name] = str(value)
                except:
                    params[param_name.lower()] = 'ERROR'
        
        # 添加相位编码方向
        if hasattr(ds, 'InPlanePhaseEncodingDirection'):
            params['phase_encoding_direction'] = str(ds.InPlanePhaseEncodingDirection)
        elif hasattr(ds, 'PhaseEncodingDirection'):
            params['phase_encoding_direction'] = str(ds.PhaseEncodingDirection)
        
        # 添加切片编码方向
        if hasattr(ds, 'SliceEncodingDirection'):
            params['slice_encoding_direction'] = str(ds.SliceEncodingDirection)
        
        return params
    
    @staticmethod
    def _extract_sequence_info(ds: pydicom.Dataset) -> Dict[str, str]:
        """提取序列信息（集成高级检测）"""
        # 使用SequenceTypeDetector检测序列类型
        sequence_type, subtype = SequenceTypeDetector.detect_sequence_type(ds)
        
        # 获取Manufacturer Sequence Name
        manufacturer_sequence_name = str(getattr(ds, 'SequenceName', 'UNKNOWN')).strip()
        
        # 获取脉冲序列细节
        pulse_sequence_details = str(getattr(ds, 'PulseSequenceDetails', 'UNKNOWN')).strip() if hasattr(ds, 'PulseSequenceDetails') else 'UNKNOWN'
        
        return {
            'sequence_type': sequence_type,
            'sequence_subtype': subtype,
            'manufacturer_sequence_name': manufacturer_sequence_name,
            'pulse_sequence_details': pulse_sequence_details,
            'scanning_sequence': str(getattr(ds, 'ScanningSequence', 'UNKNOWN')).strip() if hasattr(ds, 'ScanningSequence') else 'UNKNOWN',
            'sequence_variant': str(getattr(ds, 'SequenceVariant', 'UNKNOWN')).strip() if hasattr(ds, 'SequenceVariant') else 'UNKNOWN'
        }
    
    @staticmethod
    def _extract_parallel_imaging(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取并行成像信息（集成高级检测）"""
        # 使用VendorAwareParallelDetector检测
        parallel_result = VendorAwareParallelDetector.detect(ds)
        
        # 转换为标准化格式
        if parallel_result['is_parallel']:
            method = parallel_result['technique'].upper() if parallel_result['technique'] != 'none' else 'GRAPPA'
            if method == 'NONE':
                method = 'GRAPPA'  # 默认
        else:
            method = 'NONE'
        
        # 收集厂商特定信息
        vendor_specific = {}
        if parallel_result['vendor'] in VENDOR_PARALLEL_FIELDS:
            vendor_config = VENDOR_PARALLEL_FIELDS[parallel_result['vendor']]
            for field in vendor_config.get('standard', []):
                if hasattr(ds, field):
                    try:
                        vendor_specific[field.lower()] = str(getattr(ds, field))
                    except:
                        pass
        
        return {
            'used': parallel_result['is_parallel'],
            'acceleration_factor': float(parallel_result['acceleration_factor']),
            'method': method,
            'vendor': parallel_result['vendor'],
            'confidence': float(parallel_result['confidence']),
            'detection_methods': parallel_result['detection_methods'],
            'vendor_specific': vendor_specific if vendor_specific else {}
        }
    
    @staticmethod
    def _extract_coil_info(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取线圈信息"""
        coil_info = {}
        
        for field in COIL_INFO_FIELDS:
            if hasattr(ds, field):
                try:
                    value = getattr(ds, field)
                    coil_info[field.lower()] = str(value)
                except:
                    coil_info[field.lower()] = 'ERROR'
        
        return coil_info
    
    @staticmethod
    def _extract_vendor_info(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取厂商信息"""
        vendor_info = {}
        
        for field in VENDOR_INFO_FIELDS:
            if hasattr(ds, field):
                try:
                    value = getattr(ds, field)
                    vendor_info[field.lower()] = str(value)
                except:
                    vendor_info[field.lower()] = 'ERROR'
        
        # 添加机构地址信息（如果可用）
        if hasattr(ds, 'InstitutionAddress'):
            vendor_info['institution_address'] = str(ds.InstitutionAddress)
        
        if hasattr(ds, 'InstitutionalDepartmentName'):
            vendor_info['department_name'] = str(ds.InstitutionalDepartmentName)
        
        return vendor_info
    
    @staticmethod
    def _extract_image_characteristics(ds: pydicom.Dataset, dicom_files: List[Path]) -> Dict[str, Any]:
        """提取图像特性"""
        # 获取图像类型
        image_type = []
        if hasattr(ds, 'ImageType'):
            if isinstance(ds.ImageType, (list, pydicom.multival.MultiValue)):
                image_type = [str(item) for item in ds.ImageType]
            else:
                image_type = [str(ds.ImageType)]
        
        # 获取采集类型
        mr_acquisition_type = str(getattr(ds, 'MRAcquisitionType', 'UNKNOWN'))
        
        # 获取比特信息
        bits_allocated = int(getattr(ds, 'BitsAllocated', 16))
        bits_stored = int(getattr(ds, 'BitsStored', 12))
        high_bit = int(getattr(ds, 'HighBit', bits_stored - 1))
        
        # 获取像素表示
        pixel_representation = int(getattr(ds, 'PixelRepresentation', 0))
        
        return {
            'matrix_size': [
                int(getattr(ds, 'Rows', 256)),
                int(getattr(ds, 'Columns', 256))
            ],
            'num_slices': len(dicom_files),
            'bits_allocated': bits_allocated,
            'bits_stored': bits_stored,
            'high_bit': high_bit,
            'pixel_representation': pixel_representation,
            'image_type': image_type,
            'mr_acquisition_type': mr_acquisition_type,
            'window_center': float(getattr(ds, 'WindowCenter', 0.0)) if hasattr(ds, 'WindowCenter') else 0.0,
            'window_width': float(getattr(ds, 'WindowWidth', 0.0)) if hasattr(ds, 'WindowWidth') else 0.0
        }
    
    @staticmethod
    def _calculate_quality_flags(ds: pydicom.Dataset, dicom_files: List[Path]) -> Dict[str, Any]:
        """计算质量标志"""
        # 检查关键参数是否存在
        critical_params = [
            'RepetitionTime', 'EchoTime', 'MagneticFieldStrength',
            'PixelSpacing', 'SliceThickness', 'FlipAngle',
            'Rows', 'Columns'
        ]
        
        missing_params = []
        for param in critical_params:
            if not hasattr(ds, param):
                missing_params.append(param)
        
        # 计算参数完整性
        total_params = len(critical_params)
        missing_count = len(missing_params)
        completeness = 1.0 - (missing_count / total_params) if total_params > 0 else 0.0
        
        # 确定置信度等级
        confidence_level = 'PENDING'
        for level, threshold in CONFIDENCE_LEVELS.items():
            if completeness >= threshold:
                confidence_level = level
                break
        
        # 检查警告条件
        warnings = []
        
        # 检查切片数
        if len(dicom_files) < 3:
            warnings.append(f"Low number of slices: {len(dicom_files)}")
        
        # 检查矩阵尺寸
        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
            rows = int(ds.Rows)
            cols = int(ds.Columns)
            if rows < 64 or cols < 64:
                warnings.append(f"Small matrix size: {rows}x{cols}")
        
        # 检查TR/TE合理性
        if hasattr(ds, 'RepetitionTime') and hasattr(ds, 'EchoTime'):
            try:
                tr = float(ds.RepetitionTime)
                te = float(ds.EchoTime)
                if tr <= 0 or te <= 0:
                    warnings.append(f"Invalid TR/TE values: TR={tr}, TE={te}")
                elif te > tr:
                    warnings.append(f"TE ({te}) greater than TR ({tr})")
            except:
                pass
        
        return {
            'is_valid': completeness >= PARAMETER_COMPLETENESS_THRESHOLD,
            'hard_constraint_violations': [],
            'warnings': warnings,
            'confidence_level': confidence_level,
            'algorithm_route': 'standard',
            'parameter_completeness': completeness,
            'missing_critical_parameters': missing_params
        }
    
    @staticmethod
    def _extract_conversion_info(dicom_files: List[Path]) -> Dict[str, Any]:
        """提取转换信息"""
        # 计算总文件大小
        total_size = 0
        for file_path in dicom_files:
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        return {
            'source_dicom_count': len(dicom_files),
            'total_dicom_size_bytes': total_size,
            'conversion_date': datetime.now().isoformat(),
            'converter_version': VERSION_CONFIG.get('current_version', '2.3')
        }


class HardConstraintValidator:
    """硬约束验证器（使用config.py配置）"""
    
    @staticmethod
    def validate_dicom_files(dicom_files: List[Path]) -> Tuple[bool, List[str], Optional[str]]:
        """
        验证DICOM文件是否满足硬约束
        
        返回: (是否通过, 违反列表, 错误码)
        """
        violations = []
        
        # 1. 文件存在性检查
        if not dicom_files:
            return False, ["No DICOM files found"], ERROR_CODES['FILE_NOT_FOUND']
        
        for file_path in dicom_files:
            if not file_path.exists():
                violations.append(f"File not found: {file_path}")
        
        if violations:
            return False, violations, ERROR_CODES['FILE_NOT_FOUND']
        
        # 2. 文件大小检查
        total_size = sum(f.stat().st_size for f in dicom_files if f.exists())
        max_file_size = HARD_CONSTRAINTS.get('max_total_size_mb', 2048) * 1024 * 1024
        
        if total_size > max_file_size:
            violations.append(f"Total file size too large: {total_size/(1024*1024):.1f}MB")
        
        if violations:
            return False, violations, ERROR_CODES['IMAGE_TOO_LARGE']
        
        return True, [], None
    
    @staticmethod
    def validate_dicom_dataset(ds: pydicom.Dataset, dicom_files: List[Path]) -> Tuple[bool, List[str], Optional[str]]:
        """
        验证DICOM数据集是否满足硬约束
        
        返回: (是否通过, 违反列表, 错误码)
        """
        violations = []
        
        # 1. 必需DICOM标签检查（使用config.py配置）
        required_tags = HARD_CONSTRAINTS.get('required_dicom_tags', [])
        for tag, name in required_tags:
            if not hasattr(ds, name):
                violations.append(f"Missing required DICOM tag: {name}")
        
        if violations:
            return False, violations, ERROR_CODES['METADATA_MISSING']
        
        # 2. 图像维度检查
        if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
            rows = int(getattr(ds, 'Rows', 0))
            cols = int(getattr(ds, 'Columns', 0))
            num_slices = len(dicom_files)
            
            total_pixels = rows * cols * max(1, num_slices)
            
            # 检查最小像素数
            min_pixels = HARD_CONSTRAINTS.get('min_total_pixels', 1000)
            if total_pixels < min_pixels:
                violations.append(f"Insufficient pixels: {total_pixels} < {min_pixels}")
            
            # 检查最大像素数
            max_pixels = HARD_CONSTRAINTS.get('max_total_pixels', 100000000)
            if total_pixels > max_pixels:
                violations.append(f"Too many pixels: {total_pixels} > {max_pixels}")
            
            # 检查切片数范围
            min_slices = HARD_CONSTRAINTS.get('min_slices', 1)
            if num_slices < min_slices:
                violations.append(f"Insufficient slices: {num_slices} < {min_slices}")
        
        if violations:
            return False, violations, ERROR_CODES['INVALID_DIMENSIONS']
        
        return True, [], None


class DICOMToNIfTIConverter:
    """DICOM到NIfTI转换器"""
    
    @staticmethod
    def convert(dicom_files: List[Path], output_nii_path: Path) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        转换DICOM文件为NIfTI
        
        返回: (是否成功, 错误信息, 转换统计)
        """
        try:
            # 1. 加载DICOM文件
            slices = []
            for file_path in dicom_files:
                try:
                    ds = pydicom.dcmread(file_path, force=True)
                    slices.append(ds)
                except Exception as e:
                    return False, f"Failed to read DICOM file {file_path}: {str(e)}", None
            
            if not slices:
                return False, "No valid DICOM slices found", None
            
            # 2. 排序切片
            slices = DICOMToNIfTIConverter._sort_dicom_slices(slices)
            
            # 3. 提取像素数据
            pixel_arrays = []
            for slice_ds in slices:
                if hasattr(slice_ds, 'pixel_array'):
                    pixel_data = slice_ds.pixel_array.astype(np.float32)
                    
                    # 应用Rescale Slope和Intercept
                    if hasattr(slice_ds, 'RescaleSlope') and hasattr(slice_ds, 'RescaleIntercept'):
                        pixel_data = pixel_data * slice_ds.RescaleSlope + slice_ds.RescaleIntercept
                    
                    pixel_arrays.append(pixel_data)
                else:
                    return False, f"Missing pixel data in slice", None
            
            # 4. 创建3D体积
            if len(pixel_arrays) > 1:
                volume = np.stack(pixel_arrays, axis=-1)
            else:
                volume = pixel_arrays[0]
            
            # 5. 创建NIfTI图像
            affine = DICOMToNIfTIConverter._create_affine_matrix(slices[0])
            
            nii_img = nib.Nifti1Image(volume, affine)
            
            # 6. 保存NIfTI文件
            output_nii_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nii_img, output_nii_path)
            
            # 7. 收集转换统计
            stats = {
                'num_slices_converted': len(slices),
                'volume_shape': list(volume.shape),
                'data_type': str(volume.dtype),
                'file_size_bytes': output_nii_path.stat().st_size if output_nii_path.exists() else 0,
                'conversion_time': datetime.now().isoformat(),
                'affine_matrix': affine.tolist() if isinstance(affine, np.ndarray) else affine
            }
            
            return True, None, stats
            
        except MemoryError as e:
            return False, f"Memory error during conversion: {str(e)}", None
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            return False, error_msg, None
    
    @staticmethod
    def _sort_dicom_slices(slices: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
        """排序DICOM切片"""
        try:
            # 首先尝试按ImagePositionPatient的Z坐标排序
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            try:
                # 然后尝试按InstanceNumber排序
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except:
                # 保持原始顺序
                pass
        
        return slices
    
    @staticmethod
    def _create_affine_matrix(ds: pydicom.Dataset) -> np.ndarray:
        """从DICOM创建仿射矩阵"""
        affine = np.eye(4)
        
        try:
            # 像素间距
            if hasattr(ds, 'PixelSpacing'):
                ps = ds.PixelSpacing
                if isinstance(ps, pydicom.multival.MultiValue):
                    affine[0, 0] = float(ps[0])
                    affine[1, 1] = float(ps[1])
                elif isinstance(ps, (list, tuple)):
                    affine[0, 0] = float(ps[0])
                    affine[1, 1] = float(ps[1])
            
            # 切片厚度
            if hasattr(ds, 'SliceThickness'):
                affine[2, 2] = float(ds.SliceThickness)
            else:
                affine[2, 2] = 1.0
                    
        except Exception:
            # 使用默认值
            affine[0, 0] = 1.0
            affine[1, 1] = 1.0
            affine[2, 2] = 1.0
        
        return affine


class ConversionLogger:
    """转换日志记录器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        self.main_log_path = log_dir / "conversion_log.json"
        self.error_log_path = log_dir / "error_log.json"
        self.stats_log_path = log_dir / "conversion_stats.json"
        
        self.log_entries = []
        self.error_entries = []
        
        # 统计信息
        self.stats = {
            'conversion_version': VERSION_CONFIG.get('current_version', '2.3'),
            'start_time': datetime.now().isoformat(),
            'total_scans_processed': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'skipped_scans': 0,
            'total_processing_time_seconds': 0.0,
            'hard_constraint_violations': {},
            'vendor_distribution': {},
            'anatomical_region_distribution': {},
            'sequence_type_distribution': {}
        }
    
    def log_success(self, patient_id: str, scan_name: str, metadata: Dict, 
                   conversion_stats: Dict, processing_time: float):
        """记录成功转换"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'SUCCESS',
            'patient_id': patient_id,
            'scan_name': scan_name,
            'anatomical_region': metadata.get('anatomical_info', {}).get('detailed_region', 'unknown'),
            'sequence_type': metadata.get('sequence_info', {}).get('sequence_type', 'unknown'),
            'parallel_imaging': metadata.get('parallel_imaging', {}).get('used', False),
            'acceleration_factor': metadata.get('parallel_imaging', {}).get('acceleration_factor', 1.0),
            'processing_time_seconds': round(processing_time, 2),
            'conversion_stats': conversion_stats,
            'confidence_level': metadata.get('quality_flags', {}).get('confidence_level', 'PENDING')
        }
        
        self.log_entries.append(entry)
        self.stats['successful_conversions'] += 1
        self.stats['total_scans_processed'] += 1
        
        # 更新分布统计
        vendor = metadata.get('vendor_info', {}).get('manufacturer', 'UNKNOWN')
        self.stats['vendor_distribution'][vendor] = self.stats['vendor_distribution'].get(vendor, 0) + 1
        
        anat_region = metadata.get('anatomical_info', {}).get('detailed_region', 'unknown')
        self.stats['anatomical_region_distribution'][anat_region] = self.stats['anatomical_region_distribution'].get(anat_region, 0) + 1
        
        seq_type = metadata.get('sequence_info', {}).get('sequence_type', 'unknown')
        self.stats['sequence_type_distribution'][seq_type] = self.stats['sequence_type_distribution'].get(seq_type, 0) + 1
    
    def log_skip(self, patient_id: str, scan_name: str, violations: List[str], 
                error_code: str, processing_time: float):
        """记录跳过扫描"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'SKIPPED',
            'patient_id': patient_id,
            'scan_name': scan_name,
            'reason': 'HARD_CONSTRAINT_VIOLATED',
            'error_code': error_code,
            'violations': violations,
            'processing_time_seconds': round(processing_time, 2)
        }
        
        self.error_entries.append(entry)
        self.stats['skipped_scans'] += 1
        self.stats['total_scans_processed'] += 1
        
        # 统计约束违反类型
        for violation in violations:
            if "pixels" in violation.lower():
                key = 'pixel_count'
            elif "size" in violation.lower():
                key = 'file_size'
            elif "slice" in violation.lower():
                key = 'slice_count'
            elif "tag" in violation.lower():
                key = 'dicom_tag'
            else:
                key = 'other'
            
            self.stats['hard_constraint_violations'][key] = self.stats['hard_constraint_violations'].get(key, 0) + 1
    
    def log_error(self, patient_id: str, scan_name: str, error_message: str,
                 error_code: str, processing_time: float):
        """记录转换错误"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'ERROR',
            'patient_id': patient_id,
            'scan_name': scan_name,
            'error_code': error_code,
            'error_message': error_message,
            'processing_time_seconds': round(processing_time, 2)
        }
        
        self.error_entries.append(entry)
        self.stats['failed_conversions'] += 1
        self.stats['total_scans_processed'] += 1
        
        # 统计错误类型
        self.stats['hard_constraint_violations'][error_code] = self.stats['hard_constraint_violations'].get(error_code, 0) + 1
    
    def update_processing_time(self, processing_time: float):
        """更新总处理时间"""
        self.stats['total_processing_time_seconds'] += processing_time
    
    def save_logs(self):
        """保存所有日志"""
        # 更新结束时间
        self.stats['end_time'] = datetime.now().isoformat()
        
        # 计算成功率
        if self.stats['total_scans_processed'] > 0:
            self.stats['success_rate'] = self.stats['successful_conversions'] / self.stats['total_scans_processed']
            if self.stats['successful_conversions'] > 0:
                self.stats['avg_processing_time'] = self.stats['total_processing_time_seconds'] / self.stats['successful_conversions']
        
        # 保存主日志
        if self.log_entries or self.stats['total_scans_processed'] > 0:
            with open(self.main_log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'conversion_summary': self.stats,
                    'successful_conversions': self.log_entries
                }, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存错误日志
        if self.error_entries:
            # 统计错误信息
            error_codes_summary = {}
            error_types_summary = {}
            
            for entry in self.error_entries:
                # 统计错误码
                error_code = entry['error_code']
                error_codes_summary[error_code] = error_codes_summary.get(error_code, 0) + 1
                
                # 统计错误类型
                error_type = entry['action']
                error_types_summary[error_type] = error_types_summary.get(error_type, 0) + 1
            
            error_summary = {
                'total_errors': len(self.error_entries),
                'error_codes': error_codes_summary,
                'error_types': error_types_summary
            }
            
            with open(self.error_log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'error_summary': error_summary,
                    'error_details': self.error_entries
                }, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存统计
        with open(self.stats_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False, default=str)


class SmartDirectoryScanner:
    """智能目录扫描器"""
    
    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
    
    def find_dicom_directories(self) -> List[Path]:
        """查找包含DICOM文件的目录"""
        dicom_dirs = set()
        
        for root, _, files in os.walk(self.input_dir):
            root_path = Path(root)
            
            # 检查是否有DICOM文件
            has_dicom = False
            for file in files:
                file_path = root_path / file
                if self._is_dicom_file(file_path):
                    has_dicom = True
                    break
            
            if has_dicom:
                dicom_dirs.add(root_path)
        
        return sorted(dicom_dirs)
    
    def find_dicom_files_in_directory(self, directory: Path) -> List[Path]:
        """查找目录中的所有DICOM文件"""
        dicom_files = []
        
        # 查找所有可能包含DICOM数据的文件
        for pattern in ['*.dcm', '*.DCM', '*.IMA', '*']:
            for file_path in directory.glob(pattern):
                if file_path.is_file() and self._is_dicom_file(file_path):
                    dicom_files.append(file_path)
        
        return sorted(dicom_files)
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """检查文件是否为DICOM文件"""
        if not file_path.is_file():
            return False
        
        # 检查文件大小
        min_file_size = HARD_CONSTRAINTS.get('min_file_size_kb', 1) * 1024
        if file_path.stat().st_size < min_file_size:
            return False
        
        try:
            # 检查DICOM前缀（标准DICOM文件）
            with open(file_path, 'rb') as f:
                f.seek(128)
                if f.read(4) == b'DICM':
                    return True
            
            # 尝试读取DICOM文件（非标准但包含DICOM数据）
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
            return hasattr(ds, 'Modality')
            
        except:
            return False


class EnhancedDICOMConverter:
    """增强版DICOM转换器主类"""
    
    def __init__(self, input_dir: str, output_dir: str, verbose: bool = True, 
                 log_dir: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # 设置日志目录
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = self.output_dir / "logs"
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.scanner = SmartDirectoryScanner(self.input_dir)
        self.validator = HardConstraintValidator()
        self.logger = ConversionLogger(self.log_dir)
        
        # 统计信息
        self.conversion_stats = {
            'converter_version': VERSION_CONFIG.get('current_version', '2.3'),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'start_time': datetime.now().isoformat(),
            'total_dicom_dirs_found': 0,
            'successfully_converted': 0,
            'skipped_due_to_constraints': 0,
            'failed_conversions': 0,
            'vendor_distribution': {},
            'parallel_imaging_stats': {
                'total_parallel_scans': 0,
                'siemens': 0,
                'ge': 0,
                'philips': 0,
                'other': 0
            },
            'anatomical_region_stats': {},
            'sequence_type_stats': {}
        }
        
        if self.verbose:
            self._print_header()
    
    def convert_all(self) -> Dict[str, Any]:
        """转换所有DICOM目录"""
        print("=" * 70)
        print("🚀 BATCH DICOM TO NIFTI CONVERSION")
        print("=" * 70)
        print("🔄 Conversion in progress...")
        print("-" * 70)
        
        # 1. 扫描DICOM目录
        dicom_dirs = self.scanner.find_dicom_directories()
        self.conversion_stats['total_dicom_dirs_found'] = len(dicom_dirs)
        
        print(f"📊 Found {len(dicom_dirs)} DICOM directories to process")
        
        # 2. 处理每个目录
        for dir_idx, dicom_dir in enumerate(dicom_dirs, 1):
            start_time = datetime.now()
            rel_path = dicom_dir.relative_to(self.input_dir)
            print(f"\n[{dir_idx}/{len(dicom_dirs)}] 🔄 Processing: {rel_path}")
            
            try:
                # 处理单个DICOM目录
                success = self._process_dicom_directory(dicom_dir)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                self.logger.update_processing_time(processing_time)
                
                if success:
                    print(f"  ✅ Success ({processing_time:.1f}s)")
                else: 
                    print(f"  ⚠️  Skipped/Failed ({processing_time:.1f}s)")
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                self.logger.update_processing_time(processing_time)
                
                error_msg = f"Unexpected error: {str(e)}"
                self.logger.log_error('UNKNOWN', str(dicom_dir), error_msg, 
                                    ERROR_CODES['CONVERSION_ERROR'], processing_time)
                
                print(f"  ❌ Error: {str(e)[:100]}... ({processing_time:.1f}s)")
            
            # 进度显示
            if dir_idx % 5 == 0:
                progress = dir_idx / len(dicom_dirs) * 100
                print(f"\n📈 Progress: {dir_idx}/{len(dicom_dirs)} ({progress:.1f}%)")
                print("-" * 50)
        
        # 3. 保存日志和生成摘要
        self.logger.save_logs()
        
        # 更新统计信息
        self.conversion_stats['end_time'] = datetime.now().isoformat()
        self.conversion_stats['successfully_converted'] = self.logger.stats['successful_conversions']
        self.conversion_stats['skipped_due_to_constraints'] = self.logger.stats['skipped_scans']
        self.conversion_stats['failed_conversions'] = self.logger.stats['failed_conversions']
        
        # 4. 打印摘要
        self._print_summary()
        
        return self.conversion_stats
    
    def _process_dicom_directory(self, dicom_dir: Path) -> bool:
        """处理单个DICOM目录"""
        start_time = datetime.now()
        
        # 1. 查找DICOM文件
        dicom_files = self.scanner.find_dicom_files_in_directory(dicom_dir)
        if not dicom_files:
            if self.verbose:
                print(f"  ⚠️ No DICOM files found")
            return False
        
        # 2. 硬约束验证（文件层面）
        files_valid, file_violations, error_code = self.validator.validate_dicom_files(dicom_files)
        if not files_valid:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            patient_id = 'UNKNOWN'
            scan_name = str(dicom_dir.relative_to(self.input_dir))
            
            self.logger.log_skip(patient_id, scan_name, file_violations, 
                               error_code, processing_time)
            
            if self.verbose and file_violations:
                for violation in file_violations[:2]:
                    print(f"  ⚠️ {violation}")
            
            return False
        
        # 3. 读取第一个DICOM文件获取元数据
        try:
            first_ds = pydicom.dcmread(dicom_files[0], force=True, stop_before_pixels=True)
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            patient_id = 'UNKNOWN'
            scan_name = str(dicom_dir.relative_to(self.input_dir))
            
            self.logger.log_error(patient_id, scan_name, 
                                f"Failed to read DICOM file: {str(e)}",
                                ERROR_CODES['DICOM_READ_ERROR'], processing_time)
            return False
        
        # 4. 硬约束验证（数据层面）
        data_valid, data_violations, error_code = self.validator.validate_dicom_dataset(first_ds, dicom_files)
        if not data_valid:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            patient_id = str(getattr(first_ds, 'PatientID', 'UNKNOWN'))
            scan_name = str(dicom_dir.relative_to(self.input_dir))
            
            self.logger.log_skip(patient_id, scan_name, data_violations,
                               error_code, processing_time)
            
            if self.verbose and data_violations:
                for violation in data_violations[:2]:
                    print(f"  ⚠️ {violation}")
            
            return False
        
        # 5. 构建标准化元数据
        metadata = OptimizedMetadataBuilder.build_from_dicom(first_ds, dicom_files)
        
        # 6. 更新统计信息
        vendor = VendorAwareParallelDetector.identify_vendor(first_ds)
        self.conversion_stats['vendor_distribution'][vendor] = self.conversion_stats['vendor_distribution'].get(vendor, 0) + 1
        
        # 解剖区域统计
        anat_region = metadata.get('anatomical_info', {}).get('detailed_region', 'unknown')
        self.conversion_stats['anatomical_region_stats'][anat_region] = self.conversion_stats['anatomical_region_stats'].get(anat_region, 0) + 1
        
        # 序列类型统计
        seq_type = metadata.get('sequence_info', {}).get('sequence_type', 'unknown')
        self.conversion_stats['sequence_type_stats'][seq_type] = self.conversion_stats['sequence_type_stats'].get(seq_type, 0) + 1
        
        # 并行成像统计
        parallel_info = metadata.get('parallel_imaging', {})
        if parallel_info.get('used', False):
            self.conversion_stats['parallel_imaging_stats']['total_parallel_scans'] += 1
            vendor_lower = vendor.lower()
            if vendor_lower in self.conversion_stats['parallel_imaging_stats']:
                self.conversion_stats['parallel_imaging_stats'][vendor_lower] += 1
        
        # 7. 确定输出路径
        rel_path = dicom_dir.relative_to(self.input_dir)
        output_scan_dir = self.output_dir / rel_path
        output_nii_path = output_scan_dir / SCAN_FILENAME
        output_metadata_path = output_scan_dir / METADATA_FILENAME
        
        # 8. 转换为NIfTI
        conversion_success, conversion_error, conversion_stats = DICOMToNIfTIConverter.convert(
            dicom_files, output_nii_path
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if not conversion_success:
            patient_id = str(getattr(first_ds, 'PatientID', 'UNKNOWN'))
            scan_name = str(dicom_dir.relative_to(self.input_dir))
            
            self.logger.log_error(patient_id, scan_name, conversion_error,
                                ERROR_CODES['CONVERSION_ERROR'], processing_time)
            return False
        
        # 9. 保存元数据
        output_scan_dir.mkdir(parents=True, exist_ok=True)
        with open(output_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # 10. 记录成功
        patient_id = str(getattr(first_ds, 'PatientID', 'UNKNOWN'))
        scan_name = str(dicom_dir.relative_to(self.input_dir))
        
        self.logger.log_success(patient_id, scan_name, metadata, 
                              conversion_stats, processing_time)
        
        if self.verbose:
            # 显示关键信息
            anat_info = metadata.get('anatomical_info', {})
            detailed_region = anat_info.get('detailed_region', 'unknown')
            standardized_region = anat_info.get('standardized_region', 'unknown')
            
            sequence_info = metadata.get('sequence_info', {})
            sequence_type = sequence_info.get('sequence_type', 'unknown')
            sequence_subtype = sequence_info.get('sequence_subtype', 'unknown')
            
            image_chars = metadata.get('image_characteristics', {})
            matrix_size = image_chars.get('matrix_size', [0, 0])
            num_slices = image_chars.get('num_slices', 0)
            
            parallel_used = parallel_info.get('used', False)
            
            print(f"  📍 Region: {detailed_region} ({standardized_region})")
            print(f"  🧬 Sequence: {sequence_type} ({sequence_subtype})")
            print(f"  📐 Size: {matrix_size[0]}×{matrix_size[1]}×{num_slices}")
            print(f"  ⚡ Parallel Imaging: {'Yes' if parallel_used else 'No'}")
            if parallel_used:
                print(f"  📊 Acceleration Factor: {parallel_info.get('acceleration_factor', 1.0)}")
            print(f"  🎯 Confidence: {metadata.get('quality_flags', {}).get('confidence_level', 'PENDING')}")
        
        return True
    
    def _print_header(self):
        """打印标题"""
        print("=" * 70)
        print("ENHANCED DICOM TO NIfTI CONVERTER")
        print("=" * 70)
        print(f"Version: {VERSION_CONFIG.get('current_version', '2.3')}")
        print(f"Converter Type: Optimized Metadata Strategy")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Log directory: {self.log_dir}")
        print("-" * 70)
    
    def _print_summary(self):
        """打印转换摘要"""
        print("\n" + "=" * 70)
        print("📊 CONVERSION SUMMARY")
        print("=" * 70)
        
        total = self.conversion_stats['total_dicom_dirs_found']
        success = self.conversion_stats['successfully_converted']
        skipped = self.conversion_stats['skipped_due_to_constraints']
        failed = self.conversion_stats['failed_conversions']
        
        print(f"Total DICOM directories scanned: {total}")
        
        if total > 0:
            print(f"✅ Successfully converted: {success} ({success/total*100:.1f}%)")
            print(f"⚠️  Skipped (hard constraints): {skipped} ({skipped/total*100:.1f}%)")
            print(f"❌ Failed conversions: {failed} ({failed/total*100:.1f}%)")
        else:
            print("No DICOM directories found")
        
        # 厂商分布
        if self.conversion_stats['vendor_distribution']:
            print("\nVendor distribution:")
            for vendor, count in sorted(self.conversion_stats['vendor_distribution'].items()):
                print(f"  {vendor}: {count}")
        
        # 解剖区域分布
        if self.conversion_stats['anatomical_region_stats']:
            print("\nAnatomical region distribution (top 5):")
            sorted_regions = sorted(self.conversion_stats['anatomical_region_stats'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for region, count in sorted_regions:
                print(f"  {region}: {count}")
        
        # 序列类型分布
        if self.conversion_stats['sequence_type_stats']:
            print("\nSequence type distribution:")
            for seq_type, count in sorted(self.conversion_stats['sequence_type_stats'].items()):
                print(f"  {seq_type}: {count}")
        
        # Parallel Imaging统计
        parallel_stats = self.conversion_stats['parallel_imaging_stats']
        if parallel_stats['total_parallel_scans'] > 0:
            print(f"\nParallel Imaging scans: {parallel_stats['total_parallel_scans']} ({parallel_stats['total_parallel_scans']/max(success, 1)*100:.1f}%)")
            for vendor in ['siemens', 'ge', 'philips', 'other']:
                if parallel_stats.get(vendor, 0) > 0:
                    print(f"  {vendor.capitalize()}: {parallel_stats[vendor]}")
        
        if 'start_time' in self.conversion_stats and 'end_time' in self.conversion_stats:
            try:
                start = datetime.fromisoformat(self.conversion_stats['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(self.conversion_stats['end_time'].replace('Z', '+00:00'))
                total_time = (end - start).total_seconds()
                
                print(f"\n⏱️  Total processing time: {total_time:.1f}s")
                
                if success > 0:
                    avg_time = total_time / success
                    print(f"📈 Average time per successful conversion: {avg_time:.1f}s")
            except:
                pass
        
        print("\n📁 Output structure:")
        print(f"  NIfTI files: {self.output_dir}/<patient>/<scan>/{SCAN_FILENAME}")
        print(f"  Metadata files: {self.output_dir}/<patient>/<scan>/{METADATA_FILENAME}")
        print(f"  Log files: {self.log_dir}/")
        print("=" * 70)


# ==================== 主程序入口 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced DICOM to NIfTI Converter with Optimized Metadata Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s --input data --output converted_data
  %(prog)s --input data --output converted_data --verbose
  %(prog)s --input data --output converted_data --log-dir logs

Features:
  • Enhanced metadata output (format_version: 2.1)
  • Advanced parallel imaging detection for Siemens, GE, Philips
  • TR/TE-based sequence type detection
  • Detailed anatomical region mapping (multi-level)
  • Physical and sampling parameter extraction
  • Privacy protection (no patient names)
  • Hard constraint validation

Output files:
  NIfTI: <output>/<patient>/<scan>/{SCAN_FILENAME}
  Metadata: <output>/<patient>/<scan>/{METADATA_FILENAME}
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default='data',
                       help='Input directory containing DICOM files (default: data)')
    parser.add_argument('--output', '-o', type=str, default='converted_data',
                       help=f'Output directory for NIfTI files (default: converted_data)')
    parser.add_argument('--log-dir', '-l', type=str, default=None,
                       help='Log directory (default: <output>/logs)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Disable all output except errors')
    
    args = parser.parse_args()
    
    # 设置详细级别
    verbose = args.verbose and not args.quiet
    
    # 创建转换器并执行转换
    converter = EnhancedDICOMConverter(
        input_dir=args.input,
        output_dir=args.output,
        log_dir=args.log_dir,
        verbose=verbose
    )
    
    try:
        stats = converter.convert_all()
        
        # 返回退出码
        if stats['total_dicom_dirs_found'] > 0 and stats['successfully_converted'] == 0:
            return 1  # 有数据但没有成功转换
        else:
            return 0  # 成功
        
    except KeyboardInterrupt:
        if verbose:
            print("\n\nConversion interrupted by user")
        return 130
    except Exception as e:
        if verbose:
            print(f"\n\nFatal error: {str(e)}")
            traceback.print_exc()
        return 2


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)