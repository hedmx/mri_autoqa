#!/usr/bin/env python3
"""
Enhanced DICOM to NIfTI Converter with Standardized Metadata
Version: 2.2 - Integrated with Advanced Parallel Imaging Detection
优化要点：
1. 保持代码2的标准化输出结构
2. 集成代码1的高级并行采集检测和Sequence类型判断
3. 保持与metadata.json示例完全一致的输出格式
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
        SEQUENCE_TYPE_MAPPING,
        FIELD_STRENGTH_MAPPING,
        
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
    """厂商感知的Parallel Imaging检测器（从代码1移植，保持原逻辑）"""
    
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
    """Sequence Type检测器（从代码1移植，但适配标准化输出）"""
    
    @staticmethod
    def detect_sequence_type(ds) -> Tuple[str, str]:
        """
        检测Sequence Type和子类型
        
        返回: (sequence_type, sequence_subtype)
        """
        # 默认值
        sequence_type = 't1'
        subtype = 'standard'
        
        # 从多个字段获取序列描述
        series_desc = str(getattr(ds, 'SeriesDescription', '')).upper()
        protocol_name = str(getattr(ds, 'ProtocolName', '')).upper()
        combined_text = f"{series_desc} {protocol_name}"
        
        # 使用config.py的映射（优先）
        if SEQUENCE_TYPE_MAPPING:
            for dicom_term, standard_key in SEQUENCE_TYPE_MAPPING.items():
                if dicom_term in combined_text:
                    sequence_type = standard_key
                    break
        
        # 如果映射未找到，使用TR/TE逻辑判断（从代码1移植）
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
                except:
                    pass
        
        # 确定子类型
        if 'MPRAGE' in combined_text:
            subtype = 'mprage'
        elif 'SPGR' in combined_text:
            subtype = 'spgr'
        elif 'FSE' in combined_text or 'TSE' in combined_text:
            subtype = 'fse_tse'
        elif 'GRE' in combined_text or 'GR' in combined_text:
            subtype = 'gre'
        
        return sequence_type, subtype
    
    @staticmethod
    def detect_fat_suppression(ds) -> bool:
        """检测脂肪抑制（从代码1移植）"""
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
    """解剖区域标准化器（从代码1移植，但适配标准化输出）"""
    
    @staticmethod
    def standardize(ds: pydicom.Dataset) -> str:
        """提取并标准化解剖区域"""
        # 从多个字段获取解剖信息
        series_desc = str(getattr(ds, 'SeriesDescription', '')).upper()
        protocol_name = str(getattr(ds, 'ProtocolName', '')).upper()
        body_part = str(getattr(ds, 'BodyPartExamined', '')).upper()
        
        combined_text = f"{series_desc} {protocol_name} {body_part}"
        
        # 使用config.py映射（优先）
        if ANATOMICAL_REGION_MAPPING:
            for dicom_term, standard_key in ANATOMICAL_REGION_MAPPING.items():
                if dicom_term in combined_text:
                    return standard_key
        
        # 从描述中推断（从代码1移植）
        desc = str(ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else '').lower()
        if 'lumbar' in desc or 'lspine' in desc:
            return 'lumbar'
        elif 'cervical' in desc or 'cspine' in desc:
            return 'cervical'
        elif 'thoracic' in desc or 'tspine' in desc:
            return 'thoracic'
        elif 'brain' in desc or 'head' in desc:
            return 'brain'
        elif 'knee' in desc:
            return 'knee'
        elif 'shoulder' in desc:
            return 'shoulder'
        elif 'hip' in desc:
            return 'hip'
        
        # 默认返回
        return 'default'


class StandardizedMetadataBuilder:
    """标准化元数据构建器（使用config.py配置，集成高级检测）"""
    
    @staticmethod
    def build_from_dicom(ds: pydicom.Dataset, dicom_files: List[Path]) -> Dict[str, Any]:
        """从DICOM构建标准化元数据"""
        
        metadata = {
            # 版本信息
            'format_version': VERSION_CONFIG.get('metadata_schema_version', '2.0'),
            'generated_date': datetime.now().isoformat(),
            'conversion_tool': 'MRI_AutoQA_Converter_v2.2',
            
            # 核心信息
            'patient_info': StandardizedMetadataBuilder._extract_patient_info(ds),
            'study_info': StandardizedMetadataBuilder._extract_study_info(ds),
            'series_info': StandardizedMetadataBuilder._extract_series_info(ds),
            
            # 采集和分析参数
            'acquisition_params': StandardizedMetadataBuilder._extract_acquisition_params(ds),
            'sequence_info': StandardizedMetadataBuilder._extract_sequence_info(ds),
            'parallel_imaging': StandardizedMetadataBuilder._extract_parallel_imaging(ds),
            
            # 解剖和图像信息
            'anatomical_region': StandardizedMetadataBuilder._extract_anatomical_region(ds),
            'image_characteristics': StandardizedMetadataBuilder._extract_image_characteristics(ds, dicom_files),
            
            # 质量标志（为后续分析准备）
            'quality_flags': {
                'is_valid': True,
                'hard_constraint_violations': [],
                'warnings': [],
                'confidence_level': 'PENDING',
                'algorithm_route': 'standard'
            },
            
            # 转换信息
            'conversion_info': {
                'source_dicom_count': len(dicom_files),
                'conversion_date': datetime.now().isoformat()
            }
        }
        
        return metadata
    
    @staticmethod
    def _extract_patient_info(ds: pydicom.Dataset) -> Dict[str, str]:
        """提取患者信息"""
        return {
            'patient_id': str(getattr(ds, 'PatientID', 'UNKNOWN')).strip(),
            'patient_name': str(getattr(ds, 'PatientName', 'UNKNOWN')).strip(),
            'patient_sex': str(getattr(ds, 'PatientSex', 'U')).strip(),
            'patient_age': str(getattr(ds, 'PatientAge', '000Y')).strip()
        }
    
    @staticmethod
    def _extract_study_info(ds: pydicom.Dataset) -> Dict[str, str]:
        """提取研究信息"""
        return {
            'study_date': str(getattr(ds, 'StudyDate', 'UNKNOWN')).strip(),
            'study_time': str(getattr(ds, 'StudyTime', 'UNKNOWN')).strip(),
            'study_description': str(getattr(ds, 'StudyDescription', 'UNKNOWN')).strip(),
            'accession_number': str(getattr(ds, 'AccessionNumber', 'UNKNOWN')).strip()
        }
    
    @staticmethod
    def _extract_series_info(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取序列信息"""
        return {
            'series_number': str(getattr(ds, 'SeriesNumber', 'UNKNOWN')).strip(),
            'series_description': str(getattr(ds, 'SeriesDescription', 'UNKNOWN')).strip(),
            'modality': str(getattr(ds, 'Modality', 'MR')).strip(),
            'protocol_name': str(getattr(ds, 'ProtocolName', 'UNKNOWN')).strip()
        }
    
    @staticmethod
    def _extract_acquisition_params(ds: pydicom.Dataset) -> Dict[str, Any]:
        """提取采集参数（使用config.py的标准化映射）"""
        # 场强标准化
        field_strength = float(getattr(ds, 'MagneticFieldStrength', 1.5))
        
        # 查找最接近的标准场强
        field_strength_t = '1.5t'  # 默认值
        if FIELD_STRENGTH_MAPPING:
            available_strengths = list(FIELD_STRENGTH_MAPPING.keys())
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
        
        return {
            'field_strength_t': field_strength_t,
            'tr_ms': float(getattr(ds, 'RepetitionTime', 0.0)),
            'te_ms': float(getattr(ds, 'EchoTime', 0.0)),
            'flip_angle_deg': float(getattr(ds, 'FlipAngle', 0.0)),
            'slice_thickness_mm': float(getattr(ds, 'SliceThickness', 0.0)),
            'pixel_spacing_mm': pixel_spacing,
            'echo_train_length': int(getattr(ds, 'EchoTrainLength', 1)),
            'number_of_averages': float(getattr(ds, 'NumberOfAverages', 1.0))
        }
    
    @staticmethod
    def _extract_sequence_info(ds: pydicom.Dataset) -> Dict[str, str]:
        """提取序列信息（集成高级检测）"""
        # 使用SequenceTypeDetector检测序列类型
        sequence_type, subtype = SequenceTypeDetector.detect_sequence_type(ds)
        
        # 获取Manufacturer Sequence Name
        manufacturer_sequence_name = str(getattr(ds, 'SequenceName', 'UNKNOWN')).strip()
        
        # 如果SequenceName包含已知信息，可以进一步调整
        seq_name_upper = manufacturer_sequence_name.upper()
        if 'T1' in seq_name_upper and sequence_type != 't1':
            sequence_type = 't1'
        elif 'T2' in seq_name_upper and sequence_type != 't2':
            sequence_type = 't2'
        elif 'PD' in seq_name_upper and sequence_type != 'pd':
            sequence_type = 'pd'
        
        return {
            'sequence_type': sequence_type,
            'sequence_subtype': subtype,
            'manufacturer_sequence_name': manufacturer_sequence_name
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
        
        return {
            'used': parallel_result['is_parallel'],
            'acceleration_factor': float(parallel_result['acceleration_factor']),
            'method': method
        }
    
    @staticmethod
    def _extract_anatomical_region(ds: pydicom.Dataset) -> str:
        """提取并标准化解剖区域（集成高级检测）"""
        return AnatomicalRegionStandardizer.standardize(ds)
    
    @staticmethod
    def _extract_image_characteristics(ds: pydicom.Dataset, dicom_files: List[Path]) -> Dict[str, Any]:
        """提取图像特性"""
        return {
            'matrix_size': [
                int(getattr(ds, 'Rows', 256)),
                int(getattr(ds, 'Columns', 256))
            ],
            'num_slices': len(dicom_files),
            'bits_allocated': int(getattr(ds, 'BitsAllocated', 16)),
            'bits_stored': int(getattr(ds, 'BitsStored', 12))
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
                'conversion_time': datetime.now().isoformat()
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
            'conversion_version': VERSION_CONFIG.get('current_version', '2.2'),
            'start_time': datetime.now().isoformat(),
            'total_scans_processed': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'skipped_scans': 0,
            'total_processing_time_seconds': 0.0,
            'hard_constraint_violations': {}
        }
    
    def log_success(self, patient_id: str, scan_name: str, metadata: Dict, 
                   conversion_stats: Dict, processing_time: float):
        """记录成功转换"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'SUCCESS',
            'patient_id': patient_id,
            'scan_name': scan_name,
            'anatomical_region': metadata.get('anatomical_region', 'unknown'),
            'sequence_type': metadata.get('sequence_info', {}).get('sequence_type', 'unknown'),
            'parallel_imaging': metadata.get('parallel_imaging', {}).get('used', False),
            'acceleration_factor': metadata.get('parallel_imaging', {}).get('acceleration_factor', 1.0),
            'processing_time_seconds': round(processing_time, 2),
            'conversion_stats': conversion_stats
        }
        
        self.log_entries.append(entry)
        self.stats['successful_conversions'] += 1
        self.stats['total_scans_processed'] += 1
    
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
            'converter_version': VERSION_CONFIG.get('current_version', '2.2'),
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
            }
        }
        
        if self.verbose:
            self._print_header()
    
    def convert_all(self) -> Dict[str, Any]:
        """转换所有DICOM目录"""
        # ==================== 新增：批量转换开始提示 ====================
        
        print("=" * 70)
        print("🚀 BATCH DICOM TO NIFTI CONVERSION")
        print("=" * 70)
        print("🔄 Conversion in progress...")
        print("-" * 70)
        # =============================================================
        
        # 1. 扫描DICOM目录
        dicom_dirs = self.scanner.find_dicom_directories()
        self.conversion_stats['total_dicom_dirs_found'] = len(dicom_dirs)
        
        
        print(f"📊 Found {len(dicom_dirs)} DICOM directories to process")
        
        # 2. 处理每个目录
        for dir_idx, dicom_dir in enumerate(dicom_dirs, 1):
            start_time = datetime.now()
            rel_path = dicom_dir.relative_to(self.input_dir)
            print(f"\n[{dir_idx}/{len(dicom_dirs)}] 🔄 Processing: {rel_path}")
            # ========================================================
            
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
            
            # ==================== 新增：进度显示 ====================
            if  dir_idx % 5 == 0:
                progress = dir_idx / len(dicom_dirs) * 100
                print(f"\n📈 Progress: {dir_idx}/{len(dicom_dirs)} ({progress:.1f}%)")
                print("-" * 50)
            # ====================================================
        
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
        metadata = StandardizedMetadataBuilder.build_from_dicom(first_ds, dicom_files)
        
        # 6. 更新统计信息
        vendor = VendorAwareParallelDetector.identify_vendor(first_ds)
        self.conversion_stats['vendor_distribution'][vendor] = self.conversion_stats['vendor_distribution'].get(vendor, 0) + 1
        
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
            anatomical_region = metadata.get('anatomical_region', 'unknown')
            sequence_type = metadata.get('sequence_info', {}).get('sequence_type', 'unknown')
            matrix_size = metadata.get('image_characteristics', {}).get('matrix_size', [0, 0])
            num_slices = metadata.get('image_characteristics', {}).get('num_slices', 0)
            parallel_used = parallel_info.get('used', False)
            
            print(f"  📍 Region: {anatomical_region}")
            print(f"  🧬 Sequence: {sequence_type}")
            print(f"  📐 Size: {matrix_size[0]}×{matrix_size[1]}×{num_slices}")
            print(f"  ⚡ Parallel Imaging: {'Yes' if parallel_used else 'No'}")
            if parallel_used:
                print(f"  📊 Acceleration Factor: {parallel_info.get('acceleration_factor', 1.0)}")
        
        return True
    
    def _print_header(self):
        """打印标题"""
        print("=" * 70)
        print("ENHANCED DICOM TO NIfTI CONVERTER")
        print("=" * 70)
        print(f"Version: {VERSION_CONFIG.get('current_version', '2.2')}")
        print(f"Converter Type: Integrated with Advanced Parallel Imaging Detection")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Log directory: {self.log_dir}")
        print("-" * 70)
    
    def _print_summary(self):
        """打印转换摘要"""
        #if not self.verbose:
            #return
        
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
        description='Enhanced DICOM to NIfTI Converter with Standardized Metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s --input data --output converted_data
  %(prog)s --input data --output converted_data --verbose
  %(prog)s --input data --output converted_data --log-dir logs

Features:
  • Standardized metadata output (format_version: 2.0)
  • Advanced parallel imaging detection for Siemens, GE, Philips
  • TR/TE-based sequence type detection
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