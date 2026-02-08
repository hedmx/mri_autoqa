#!/usr/bin/env python3
"""
MRI AutoQA Configuration File - 完整整合版
包含所有系统配置参数，兼容所有模块
版本: 3.0 (整合版)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
import platform
import sys
from typing import Dict, Any, List
#============================================================================
# 新增：背景选择配置
#============================================================================
BACKGROUND_SELECTION = {
    'quality_thresholds': {
        'A': {'max_mean': 6.0, 'max_std': 3.0, 'max_cv': 0.7},
        'B': {'max_mean': 12.0, 'max_std': 6.0, 'max_cv': 1.0},
        'C': {'max_mean': 20.0, 'max_std': 10.0, 'max_cv': 1.5}
    },
    'emergency_estimates': {
        'low_signal': {'mean': 1.5, 'std': 1.0},
        'medium_signal': {'mean': 2.5, 'std': 1.5},
        'high_signal': {'mean': 4.0, 'std': 2.0}
    }
}
# ============================================================================
# 1. 字体配置 - 解决中文显示问题（从副本2）
# ============================================================================
system = platform.system()

if system == "Windows":
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    matplotlib.rcParams['axes.unicode_minus'] = False
elif system == "Darwin":  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    matplotlib.rcParams['axes.unicode_minus'] = False
elif system == "Linux":
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'AR PL UMing CN']
    matplotlib.rcParams['axes.unicode_minus'] = False

matplotlib.rcParams['font.family'] = 'sans-serif'

# ============================================================================
# 2. 文件系统配置（整合两个版本）
# ============================================================================

# 默认目录路径（从原文件）
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "converted_data"      # 输入数据目录
OUTPUT_DIR = PROJECT_ROOT / "autoqa_results"     # 输出结果目录
LOG_DIR = PROJECT_ROOT / "logs"                  # 日志目录

# 文件命名约定（两个版本都有，保持一致）
SCAN_FILENAME = "scan.nii.gz"                    # NIfTI文件标准名称
METADATA_FILENAME = "metadata.json"              # 元数据文件标准名称
QA_REPORT_FILENAME = "qa_report.json"            # 质量分析报告文件名
VISUALIZATION_FILENAME = "visualization.png"     # 可视化文件名
SUMMARY_FILENAME = "quality_summary.txt"         # 文本摘要文件名

# ============================================================================
# 3. DICOM映射配置（优化版）
# ============================================================================

# 1. 解剖区域映射 - 分层次结构
ANATOMICAL_REGION_MAPPING = {
    # 一级映射：DICOM术语 -> 标准化区域（保持向后兼容）
    'SPINE': 'spine',
    'LSPINE': 'lumbar',
    'CSPINE': 'cervical',
    'TSPINE': 'thoracic',
    'LUMBAR': 'lumbar',
    'CERVICAL': 'cervical',
    'THORACIC': 'thoracic',
    
    # 头部相关
    'BRAIN': 'brain',
    'HEAD': 'brain',
    'CEREBRUM': 'brain',
    'CEREBELLUM': 'brain',
    
    # 关节相关
    'KNEE': 'knee',
    'SHOULDER': 'shoulder',
    'HIP': 'hip',
    'ANKLE': 'ankle',
    'WRIST': 'wrist',
    'ELBOW': 'elbow',
    
    # 腹部/胸部
    'ABDOMEN': 'abdomen',
    'LIVER': 'liver',
    'KIDNEY': 'kidney',
    'CHEST': 'chest',
    'LUNG': 'lung',
    'HEART': 'heart',
    'BREAST': 'breast',
    
    # 默认回退
    'UNKNOWN': 'default'
}

# 新增：详细解剖区域映射（用于新结构中的 detailed_region）
ANATOMICAL_DETAILED_MAPPING = {
    # 脊柱细分
    'LSPINE': 'lumbar',
    'LUMBAR': 'lumbar',
    'CSPINE': 'cervical',
    'CERVICAL': 'cervical',
    'TSPINE': 'thoracic',
    'THORACIC': 'thoracic',
    
    # 脊柱具体节段（如果有）
    'L1': 'lumbar_L1',
    'L2': 'lumbar_L2',
    'L3': 'lumbar_L3',
    'L4': 'lumbar_L4',
    'L5': 'lumbar_L5',
    'S1': 'sacral_S1',
    
    # 头部细分
    'BRAIN': 'brain',
    'BRAINSTEM': 'brainstem',
    'CEREBELLUM': 'cerebellum',
    'CEREBRUM': 'cerebrum',
    
    # 关节细分
    'KNEE_LEFT': 'knee_left',
    'KNEE_RIGHT': 'knee_right',
    'KNEE_BILATERAL': 'knee_bilateral',
    'SHOULDER_LEFT': 'shoulder_left',
    'SHOULDER_RIGHT': 'shoulder_right',
    
    # 默认
    'UNKNOWN': 'default'
}

# 2. 序列类型映射 - 增强版
SEQUENCE_TYPE_MAPPING = {
    # T1类
    'T1': 't1',
    'T1W': 't1',
    'T1_': 't1',
    'T1W_': 't1',
    'T1W_3D': 't1',
    
    # T2类
    'T2': 't2',
    'T2W': 't2',
    'T2_': 't2',
    'T2W_': 't2',
    
    # PD类
    'PD': 'pd',
    'PDFS': 'pd',
    'PDW': 'pd',
    
    # 特殊序列
    'FLAIR': 'flair',
    'STIR': 'stir',
    'DWI': 'dwi',
    'DWI_': 'dwi',
    'ADC': 'adc',
    'SWI': 'swi',
    'TOF': 'tof',
    'MRA': 'tof',
    
    # 快速/涡轮序列（根据TR/TE进一步判断）
    'TSE': 't1',      # 需要结合TR/TE判断
    'FSE': 't1',      # 需要结合TR/TE判断
    'TURBO': 't1',    # 需要结合TR/TE判断
    'HASTE': 't2',    # 通常为T2
    'GRASE': 't2',    # 通常为T2
    
    # 梯度回波
    'GRE': 't1',      # 通常是T1
    'GR': 't1',
    'SPGR': 't1',
    'MPRAGE': 't1',
    'VIBE': 't1',
    
    # 默认
    'UNKNOWN': 't1'
}

# 新增：序列子类型映射（用于新结构中的 sequence_subtype）
SEQUENCE_SUBTYPE_MAPPING = {
    'TSE': 'fse_tse',
    'FSE': 'fse_tse',
    'TURBO': 'fse_tse',
    'HASTE': 'haste',
    'GRASE': 'grase',
    'GRE': 'gre',
    'GR': 'gre',
    'SPGR': 'spgr',
    'MPRAGE': 'mprage',
    'VIBE': 'vibe',
    'FLASH': 'flash',
    'FISP': 'fisp',
    'PSIF': 'psif',
    'TRUE FISP': 'true_fisp',
    'DESS': 'dess'
}

# 3. 场强映射
FIELD_STRENGTH_MAPPING = {
    0.23: '0.2t',
    0.35: '0.3t',
    0.5: '0.5t',
    1.0: '1.0t',
    1.5: '1.5t',
    3.0: '3.0t',
    7.0: '7.0t',
    9.4: '9.4t',
    11.7: '11.7t'
}

# ============================================================================
# 4. 新增：物理参数和采样参数配置
# ============================================================================

# 需要提取的物理参数列表（用于质量控制和后处理）
PHYSICAL_PARAMETERS_TO_EXTRACT = [
    'MagneticFieldStrength',      # 磁场强度 (T)
    'ImagingFrequency',           # 成像频率 (MHz)
    'SAR',                        # 比吸收率 (W/kg)
    'DwellTime',                  # 驻留时间 (s)
    'PixelBandwidth',             # 像素带宽 (Hz/pixel)
    'TxRefAmp',                   # 发射参考幅度 (V)
    'ShimSetting',                # 匀场设置
    'SpacingBetweenSlices',       # 层间距 (mm)
    'EchoTrainLength',            # 回波链长度
    'PercentPhaseFOV',            # 相位FOV百分比
    'PercentSampling',            # 采样百分比
    'PhaseResolution',            # 相位分辨率
    'PhaseOversampling',          # 相位过采样
    'EchoSpacing',                # 回波间隔
    'BandwidthPerPixelPhaseEncode',  # 相位编码带宽
    'WaterFatShift',              # 水脂位移
    'TotalScanTime',              # 总扫描时间
]

# 需要提取的采样/几何参数列表
SAMPLING_PARAMETERS_TO_EXTRACT = [
    'AcquisitionMatrix',           # 采集矩阵
    'AcquisitionMatrixPE',         # 相位编码采集矩阵
    'ReconMatrixPE',               # 相位编码重建矩阵
    'BaseResolution',              # 基础分辨率
    'PhaseEncodingSteps',          # 相位编码步数
    'RefLinesPE',                  # 相位编码参考线
    'SliceTiming',                 # 切片时间
    'NumberOfPhaseEncodingSteps',  # 相位编码步数
    'Rows',                        # 行数
    'Columns',                     # 列数
    'InPlanePhaseEncodingDirection',  # 相位编码方向
    'PhaseEncodingDirection',      # 相位编码方向
    'SliceEncodingDirection',      # 切片编码方向
    'ImageOrientationPatient',     # 患者图像方向
    'ImagePositionPatient',        # 患者图像位置
]

# 需要提取的线圈信息
COIL_INFO_FIELDS = [
    'ReceiveCoilName',            # 接收线圈名称
    'ReceiveCoilActiveElements',  # 接收线圈激活单元
    'TransmitCoilName',           # 发射线圈名称
    'CoilCombinationMethod',      # 线圈组合方法
    'MulticoilElementName',       # 多线圈单元名称
]

# 需要提取的厂商/设备信息
VENDOR_INFO_FIELDS = [
    'Manufacturer',               # 厂商
    'ManufacturersModelName',     # 设备型号
    'DeviceSerialNumber',         # 设备序列号
    'StationName',                # 站点名称
    'SoftwareVersions',           # 软件版本
    'ProtocolName',               # 协议名称
    'InstitutionName',            # 机构名称
    'InstitutionalDepartmentName', # 部门名称
    'ConsistencyInfo',            # 一致性信息
]

# ============================================================================
# 5. 新增：并行成像配置
# ============================================================================

# 并行成像方法标准化映射
PARALLEL_IMAGING_METHOD_MAPPING = {
    'GRAPPA': 'GRAPPA',
    'SENSE': 'SENSE',
    'ASSET': 'ASSET',
    'ARC': 'ARC',
    'mSENSE': 'SENSE',
    'SMASH': 'SMASH',
    'PILS': 'PILS',
    'CAIPIRINHA': 'CAIPIRINHA',
    'BLAZE': 'BLAZE',
    'HyperSense': 'HyperSense',
}

# 厂商特定的并行成像字段映射
VENDOR_PARALLEL_FIELDS = {
    'SIEMENS': {
        'standard': ['ParallelReductionFactorInPlane', 'ParallelAcquisitionTechnique'],
        'private': ['PATModeText', 'MatrixCoilMode'],
    },
    'GE': {
        'standard': ['AccelerationFactorPE', 'AssetRFactors', 'ARC'],
        'private': [],
    },
    'PHILIPS': {
        'standard': ['SENSE', 'ParallelReductionFactor'],
        'private': [],
    },
    'TOSHIBA': {
        'standard': ['ParallelImagingFactor', 'SpeedUpFactor'],
        'private': [],
    },
}

# ============================================================================
# 6. 新增：质量控制参数
# ============================================================================

# 参数完整性检查阈值
PARAMETER_COMPLETENESS_THRESHOLD = 0.8  # 80%的参数完整即认为质量高

# 关键参数权重（用于质量评分）
CRITICAL_PARAMETER_WEIGHTS = {
    'RepetitionTime': 0.15,
    'EchoTime': 0.15,
    'MagneticFieldStrength': 0.10,
    'PixelSpacing': 0.10,
    'SliceThickness': 0.10,
    'FlipAngle': 0.08,
    'MatrixSize': 0.08,
    'NumberOfSlices': 0.08,
    'SequenceType': 0.08,
    'AnatomicalRegion': 0.08,
}

# 置信度等级定义
CONFIDENCE_LEVELS = {
    'HIGH': 0.8,      # 80-100% 参数完整，厂商识别明确
    'MEDIUM': 0.5,    # 50-80% 参数完整
    'LOW': 0.3,       # 30-50% 参数完整
    'PENDING': 0.0,   # 待人工检查
}

# ============================================================================
# 7. 版本配置更新
# ============================================================================

VERSION_CONFIG = {
    'current_version': '2.3',  # 更新版本号
    'metadata_schema_version': '2.1',
    'min_required_config_version': '2.0',
    'changelog': {
        '2.3': [
            '新增分层解剖区域映射',
            '增强物理参数提取',
            '添加采样参数配置',
            '改进序列子类型检测',
            '增强质量控制参数'
        ]
    }
}
# ============================================================================
# 新增：统一ROI搜索策略配置（信号ROI优化专用）
# ============================================================================

UNIFIED_SEARCH_STRATEGY = {
    # 多层搜索框架
    'search_layers': [
        {
            'name': 'primary',
            'display_name': '主搜索',
            'max_cv_multiplier': 0.8,           # 相对于区域max_allowed_cv的倍数
            'search_radius_multiplier': 1.0,    # 搜索半径乘数
            'required_score': 0.7,              # 最低综合评分
            'min_pixels': 30,                   # 最小像素数
            'search_step_ratio': 0.03,          # 搜索步长比例（相对于图像尺寸）
            'description': '高质量搜索：严格的均匀性要求'
        },
        {
            'name': 'extended',
            'display_name': '扩展搜索',
            'max_cv_multiplier': 1.2,           # 放宽CV要求
            'search_radius_multiplier': 1.5,    # 扩大搜索范围
            'required_score': 0.5,              # 降低评分要求
            'min_pixels': 25,                   # 减少像素数要求
            'search_step_ratio': 0.04,          # 增加步长以加快搜索
            'description': '中等质量搜索：放宽要求以找到可用区域'
        }
    ],
    
    # 智能回退策略
    'fallback_strategy': {
        'method': 'hybrid',                     # hybrid:混合 | fixed:固定位置 | best:最佳候选
        'display_name': '智能混合回退',
        
        # 质量阈值
        'quality_thresholds': {
            'acceptable_cv': 0.35,              # 可接受候选点的最大CV
            'fixed_position_cv': 0.40,          # 固定位置的最大CV阈值
            'min_signal_mean_factor': 0.6,      # 最小信号均值因子（相对于期望值）
            'max_signal_mean_factor': 1.4,      # 最大信号均值因子
        },
        
        # 选择规则
        'selection_rules': {
            'prefer_acceptable_over_fixed': True,   # 优先选择勉强可用的候选点
            'min_pixels_for_acceptable': 20,        # 勉强可用点的最小像素数
            'max_fixed_position_tries': 3,          # 最多尝试几个固定位置
            'require_anatomically_reasonable': True # 要求解剖位置合理
        },
        
        'description': '智能回退：优先选择质量尚可的候选点，其次尝试固定位置'
    },
    
    # 统一评分系统
    'scoring_system': {
        'display_name': '综合评分系统',
        
        # 权重分配（总和为1.0）
        'weights': {
            'uniformity': 0.50,   # 均匀性（CV相关） - 最重要
            'intensity': 0.20,    # 信号强度合理性
            'centrality': 0.15,   # 中心性（距离目标位置）
            'size': 0.15          # ROI尺寸合理性
        },
        
        # 评分计算公式（字符串格式，运行时eval）
        'formulas': {
            'uniformity': '1.0 / (cv + 0.05) if cv > 0 else 1.0',
            'intensity': '1.0 - min(1.0, abs(mean - target_center) / max(target_center, 50.0))',
            'centrality': 'max(0.0, 1.0 - (abs(y_dist) + abs(x_dist)) / 2.0)',
            'size': 'min(1.0, pixel_count / 100.0)'
        },
        
        # 评分阈值
        'score_thresholds': {
            'excellent': 0.85,
            'good': 0.70,
            'fair': 0.50,
            'poor': 0.30
        },
        
        'description': '统一评分：综合考虑均匀性、强度、位置和尺寸'
    },
    
    # 搜索算法参数
    'algorithm_params': {
        'grid_search': {
            'min_grid_step': 3,           # 最小网格步长（像素）
            'max_grid_step': 10,          # 最大网格步长
            'adaptive_step': True,        # 是否自适应步长
        },
        'roi_variations': {
            'size_factors': [0.8, 1.0, 1.2],      # ROI尺寸变化因子
            'shape_factors': [1.0],               # 形状因子（保留用于未来扩展）
            'min_roi_ratio': 0.05,                # 最小ROI尺寸比例
            'max_roi_ratio': 0.12,                # 最大ROI尺寸比例
        },
        'validation': {
            'validate_connectivity': True,        # 验证区域连通性
            'min_compactness': 0.7,               # 最小紧凑度
            'max_edge_distance_ratio': 0.1,       # 最大边缘距离比例
        }
    },
    
    'description': '统一的ROI搜索策略框架，所有解剖部位共享相同的搜索逻辑，但使用个性化参数'
}

# ============================================================================
# 4. ROI模板配置（整合副本2的详细解剖区域，增加搜索策略引用）
# ============================================================================

ROI_SETTINGS = {
    # --- 基本区域（统一为动态搜索格式）---
    'default': {
        # 动态搜索参数
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.5,
        'search_radius_x_ratio': 0.15,
        'search_radius_y_ratio': 0.15,
        
        # 目标组织信号特征
        'expected_signal_min': 50,
        'expected_signal_max': 150,
        'max_allowed_cv': 0.25,
        
        # ROI尺寸参数
        'roi_size_ratio': 0.08,
        'description': '默认动态搜索ROI',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'image_center',
                'display_name': '图像中心',
                'y_ratio': 0.50,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.08,
                'description': '默认回退位置：图像中心'
            }
        ],
        'signal_range_reference': {
            'percentile_reference': 75,
            'min_range_width': 40,
            'use_robust_statistics': True
        }
    },
    'lumbar': {
        # 动态搜索参数（已验证有效）
        'search_center_y_ratio': 0.6,
        'search_center_x_ratio': 0.5,
        'search_radius_y_ratio': 0.18,
        'search_radius_x_ratio': 0.10,
        
        # 目标组织信号特征（基于数据分析优化）
        'expected_signal_min': 90,
        'expected_signal_max': 155,
        'max_allowed_cv': 0.25,
        
        # ROI尺寸参数（关键优化）
        'roi_size_ratio': 0.08,
        'description': '腰椎椎体松质骨动态搜索（优化尺寸：0.08）',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'primary_fallback',
                'display_name': '主要回退位置',
                'y_ratio': 0.60,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.08,
                'description': '腰椎主要回退位置：椎体中心'
            },
            {
                'name': 'upper_fallback',
                'display_name': '上侧回退位置', 
                'y_ratio': 0.55,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.07,
                'description': '腰椎上侧回退位置：考虑个体解剖变异'
            },
            {
                'name': 'lower_fallback',
                'display_name': '下侧回退位置',
                'y_ratio': 0.65,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.09,
                'description': '腰椎下侧回退位置：考虑个体解剖变异'
            }
        ],
        'signal_range_reference': {
            'percentile_reference': 75,
            'min_range_width': 40,
            'use_robust_statistics': True
        }
    },
    'brain': {
        # 转换为动态搜索格式
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.4,
        'search_radius_x_ratio': 0.10,
        'search_radius_y_ratio': 0.10,
        
        # 大脑信号特征（T1加权）
        'expected_signal_min': 80,
        'expected_signal_max': 120,
        'max_allowed_cv': 0.20,
        
        'roi_size_ratio': 0.08,
        'description': '大脑中心动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'brain_center',
                'display_name': '大脑中心',
                'y_ratio': 0.45,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.07,
                'description': '大脑中心回退位置'
            },
            {
                'name': 'brain_slightly_high',
                'display_name': '大脑稍高位置',
                'y_ratio': 0.40,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.07,
                'description': '大脑稍高位置回退'
            }
        ],
        'signal_range_reference': {
            'percentile_reference': 75,
            'min_range_width': 35,
            'use_robust_statistics': True
        }
    },
    'cervical': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.55,
        'search_radius_x_ratio': 0.08,
        'search_radius_y_ratio': 0.08,
        
        'expected_signal_min': 70,
        'expected_signal_max': 110,
        'max_allowed_cv': 0.22,
        
        'roi_size_ratio': 0.07,
        'description': '颈椎动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'cervical_center',
                'display_name': '颈椎中心',
                'y_ratio': 0.55,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.06,
                'description': '颈椎中心回退位置'
            }
        ],
        'signal_range_reference': {
            'percentile_reference': 75,
            'min_range_width': 35,
            'use_robust_statistics': True
        }
    },
    'thoracic': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.58,
        'search_radius_x_ratio': 0.10,
        'search_radius_y_ratio': 0.10,
        
        'expected_signal_min': 80,
        'expected_signal_max': 115,
        'max_allowed_cv': 0.21,
        
        'roi_size_ratio': 0.08,
        'description': '胸椎动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'thoracic_center',
                'display_name': '胸椎中心',
                'y_ratio': 0.58,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.07,
                'description': '胸椎中心回退位置'
            }
        ],
        'signal_range_reference': {
            'percentile_reference': 75,
            'min_range_width': 35,
            'use_robust_statistics': True
        }
    },
    
    # --- 其他区域统一转换 ---
    'spine': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.6,
        'search_radius_x_ratio': 0.12,
        'search_radius_y_ratio': 0.12,
        
        'expected_signal_min': 75,
        'expected_signal_max': 120,
        'max_allowed_cv': 0.22,
        
        'roi_size_ratio': 0.08,
        'description': '通用脊柱区域动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'spine_center',
                'display_name': '脊柱中心',
                'y_ratio': 0.60,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.08,
                'description': '通用脊柱中心回退位置'
            }
        ],
        'signal_range_reference': {
            'percentile_reference': 75,
            'min_range_width': 40,
            'use_robust_statistics': True
        }
    },
    
    # 大脑细分
    'cerebrum': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.5,
        'search_radius_x_ratio': 0.08,
        'search_radius_y_ratio': 0.08,
        
        'expected_signal_min': 85,
        'expected_signal_max': 125,
        'max_allowed_cv': 0.18,
        
        'roi_size_ratio': 0.08,
        'description': '大脑半球中心动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'cerebrum_center',
                'display_name': '大脑半球中心',
                'y_ratio': 0.50,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.08,
                'description': '大脑半球中心回退位置'
            }
        ]
    },
    'cerebellum': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.65,
        'search_radius_x_ratio': 0.06,
        'search_radius_y_ratio': 0.06,
        
        'expected_signal_min': 70,
        'expected_signal_max': 110,
        'max_allowed_cv': 0.20,
        
        'roi_size_ratio': 0.06,
        'description': '小脑中心动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'cerebellum_center',
                'display_name': '小脑中心',
                'y_ratio': 0.65,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.06,
                'description': '小脑中心回退位置'
            }
        ]
    },
    'cortex': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.5,
        'search_radius_x_ratio': 0.06,
        'search_radius_y_ratio': 0.06,
        
        'expected_signal_min': 90,
        'expected_signal_max': 130,
        'max_allowed_cv': 0.19,
        
        'roi_size_ratio': 0.06,
        'description': '大脑皮层动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'cortex_center',
                'display_name': '皮层中心',
                'y_ratio': 0.50,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.06,
                'description': '大脑皮层中心回退位置'
            }
        ]
    },
    'wm': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.5,
        'search_radius_x_ratio': 0.07,
        'search_radius_y_ratio': 0.07,
        
        'expected_signal_min': 95,  # 白质信号较高
        'expected_signal_max': 135,
        'max_allowed_cv': 0.17,
        
        'roi_size_ratio': 0.08,
        'description': '白质动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'wm_center',
                'display_name': '白质中心',
                'y_ratio': 0.50,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.08,
                'description': '白质中心回退位置'
            }
        ]
    },
    'gm': {
        'search_center_x_ratio': 0.52,
        'search_center_y_ratio': 0.5,
        'search_radius_x_ratio': 0.05,
        'search_radius_y_ratio': 0.05,
        
        'expected_signal_min': 80,
        'expected_signal_max': 120,
        'max_allowed_cv': 0.20,
        
        'roi_size_ratio': 0.06,
        'description': '灰质动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'gm_center',
                'display_name': '灰质中心',
                'y_ratio': 0.52,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.06,
                'description': '灰质中心回退位置'
            }
        ]
    },
    
    # 脊柱细分
    'spinal_cord': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.5,
        'search_radius_x_ratio': 0.03,
        'search_radius_y_ratio': 0.03,
        
        'expected_signal_min': 60,
        'expected_signal_max': 100,
        'max_allowed_cv': 0.25,
        
        'roi_size_ratio': 0.04,
        'description': '脊髓中心动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'spinal_cord_center',
                'display_name': '脊髓中心',
                'y_ratio': 0.50,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.04,
                'description': '脊髓中心回退位置'
            }
        ]
    },
    'disc': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.55,
        'search_radius_x_ratio': 0.04,
        'search_radius_y_ratio': 0.04,
        
        'expected_signal_min': 65,
        'expected_signal_max': 105,
        'max_allowed_cv': 0.23,
        
        'roi_size_ratio': 0.05,
        'description': '椎间盘动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'disc_center',
                'display_name': '椎间盘中心',
                'y_ratio': 0.55,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.05,
                'description': '椎间盘中心回退位置'
            }
        ]
    },
    
    # 关节相关
    'knee': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.5,
        'search_radius_x_ratio': 0.08,
        'search_radius_y_ratio': 0.08,
        
        'expected_signal_min': 70,
        'expected_signal_max': 115,
        'max_allowed_cv': 0.21,
        
        'roi_size_ratio': 0.08,
        'description': '膝关节中心动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'knee_center',
                'display_name': '膝关节中心',
                'y_ratio': 0.50,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.08,
                'description': '膝关节中心回退位置'
            }
        ]
    },
    'shoulder': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.45,
        'search_radius_x_ratio': 0.06,
        'search_radius_y_ratio': 0.06,
        
        'expected_signal_min': 65,
        'expected_signal_max': 110,
        'max_allowed_cv': 0.22,
        
        'roi_size_ratio': 0.07,
        'description': '肩关节动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'shoulder_center',
                'display_name': '肩关节中心',
                'y_ratio': 0.45,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.07,
                'description': '肩关节中心回退位置'
            }
        ]
    },
    'hip': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.55,
        'search_radius_x_ratio': 0.07,
        'search_radius_y_ratio': 0.07,
        
        'expected_signal_min': 75,
        'expected_signal_max': 120,
        'max_allowed_cv': 0.20,
        
        'roi_size_ratio': 0.08,
        'description': '髋关节中心动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'hip_center',
                'display_name': '髋关节中心',
                'y_ratio': 0.55,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.08,
                'description': '髋关节中心回退位置'
            }
        ]
    },
    
    # 器官相关
    'liver': {
        'search_center_x_ratio': 0.4,
        'search_center_y_ratio': 0.4,
        'search_radius_x_ratio': 0.07,
        'search_radius_y_ratio': 0.07,
        
        'expected_signal_min': 60,
        'expected_signal_max': 100,
        'max_allowed_cv': 0.23,
        
        'roi_size_ratio': 0.08,
        'description': '肝右叶动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'liver_center',
                'display_name': '肝脏中心',
                'y_ratio': 0.40,
                'x_ratio': 0.40,
                'roi_size_ratio': 0.08,
                'description': '肝右叶中心回退位置'
            }
        ]
    },
    'kidney': {
        'search_center_x_ratio': 0.35,
        'search_center_y_ratio': 0.45,
        'search_radius_x_ratio': 0.05,
        'search_radius_y_ratio': 0.05,
        
        'expected_signal_min': 55,
        'expected_signal_max': 95,
        'max_allowed_cv': 0.24,
        
        'roi_size_ratio': 0.07,
        'description': '右肾动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'kidney_center',
                'display_name': '肾脏中心',
                'y_ratio': 0.45,
                'x_ratio': 0.35,
                'roi_size_ratio': 0.07,
                'description': '右肾中心回退位置'
            }
        ]
    },
    'heart': {
        'search_center_x_ratio': 0.55,
        'search_center_y_ratio': 0.45,
        'search_radius_x_ratio': 0.05,
        'search_radius_y_ratio': 0.05,
        
        'expected_signal_min': 70,
        'expected_signal_max': 110,
        'max_allowed_cv': 0.22,
        
        'roi_size_ratio': 0.06,
        'description': '心脏动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'heart_center',
                'display_name': '心脏中心',
                'y_ratio': 0.45,
                'x_ratio': 0.55,
                'roi_size_ratio': 0.06,
                'description': '心脏中心回退位置'
            }
        ]
    },
    
    # 回退区域
    'unknown': {
        'search_center_x_ratio': 0.5,
        'search_center_y_ratio': 0.5,
        'search_radius_x_ratio': 0.10,
        'search_radius_y_ratio': 0.10,
        
        'expected_signal_min': 50,
        'expected_signal_max': 150,
        'max_allowed_cv': 0.30,
        
        'roi_size_ratio': 0.08,
        'description': '未知区域动态搜索',
        
        # 新增：搜索策略引用和固定位置
        'search_strategy_ref': 'UNIFIED_SEARCH_STRATEGY',
        'fallback_positions': [
            {
                'name': 'unknown_center',
                'display_name': '未知区域中心',
                'y_ratio': 0.50,
                'x_ratio': 0.50,
                'roi_size_ratio': 0.08,
                'description': '未知区域中心回退位置'
            }
        ]
    },
}

# ============================================================================
# 5. SNR标准配置（整合两个版本）
# ============================================================================
SEQUENCE_SNR_STANDARDS = {
    # T1加权序列（基于您的腰椎数据优化）
    't1': {
        'excellent': 60.0,    # 从30.0提高到60.0（前25%）
        'good': 45.0,         # 从20.0提高到45.0（中间50%）
        'fair': 30.0,         # 从10.0提高到30.0（后25%）
        'poor': 15.0,         # 从5.0提高到15.0
        'description': 'T1加权序列（基于腰椎数据优化）'
    },
    't2': {
        'excellent': 50.0,    # 从25.0提高到50.0
        'good': 35.0,
        'fair': 20.0,
        'poor': 10.0,
        'description': 'T2加权序列'
    },
    'pd': {
        'excellent': 55.0,    # 从28.0提高到55.0
        'good': 40.0,
        'fair': 25.0,
        'poor': 12.0,
        'description': '质子密度加权序列'
    },
    'flair': {
        'excellent': 45.0,    # 从22.0提高到45.0
        'good': 30.0,
        'fair': 18.0,
        'poor': 9.0,
        'description': 'FLAIR序列'
    },
    # 从副本2新增的序列类型
    't2star': {
        'excellent': 45.0,
        'good': 30.0,
        'fair': 18.0,
        'poor': 9.0,
        'description': 'T2*加权序列'
    },
    'swi': {
        'excellent': 40.0,
        'good': 25.0,
        'fair': 15.0,
        'poor': 7.5,
        'description': '磁敏感加权成像'
    },
    'dwi': {
        'excellent': 35.0,    # DWI通常SNR较低
        'good': 22.0,
        'fair': 12.0,
        'poor': 6.0,
        'description': '弥散加权成像'
    },
    'default': {
        'excellent': 50.0,
        'good': 35.0,
        'fair': 20.0,
        'poor': 10.0,
        'description': '默认序列标准'
    }
}

# ============================================================================
# 6. 并行成像g-factor表（整合副本2的更细粒度）
# ============================================================================

G_FACTOR_TABLE = {
    1.0: (1.00, 1.00, 1.00, '无加速'),
    1.2: (1.04, 1.02, 1.06, '轻度加速'),
    1.5: (1.10, 1.07, 1.13, '中等加速'),
    1.8: (1.14, 1.10, 1.18, '中高度加速'),
    2.0: (1.18, 1.14, 1.22, '标准2倍加速'),
    2.5: (1.26, 1.20, 1.32, '2.5倍加速'),
    3.0: (1.34, 1.27, 1.41, '3倍加速'),
    4.0: (1.48, 1.39, 1.57, '4倍加速')
}

# ============================================================================
# 7. 置信度权重配置（从原文件 - 重要！）
# ============================================================================

CONFIDENCE_WEIGHTS = {
    'total': {
        'C_ROI': 0.40,      # ROI放置质量
        'C_noise': 0.30,    # 噪声估计可靠性
        'C_image': 0.20,    # 图像内在质量
        'C_algorithm': 0.10 # 算法执行质量
    },
    'roi': {
        'S_intensity': 0.40,    # 信号强度
        'S_uniformity': 0.30,   # 均匀性
        'S_edge': 0.20,        # 边缘距离
        'S_anatomy': 0.10      # 解剖位置合理性
    },
    'noise': {
        'S_bg_size': 0.30,     # 背景区域大小
        'S_bg_mean': 0.40,     # 背景均值
        'S_bg_uniformity': 0.30 # 背景均匀性
    },
    'image': {
        'S_contrast': 0.40,    # 对比度
        'S_artifact': 0.30,    # 伪影
        'S_integrity': 0.30    # 图像完整性
    },
    'algorithm': {
        'S_convergence': 0.50,  # 收敛性
        'S_stability': 0.30,    # 稳定性
        'S_completeness': 0.20  # 完整性
    }
}

# ============================================================================
# 8. 质量评分权重配置（从原文件 - 重要！）
# ============================================================================

QUALITY_SCORE_WEIGHTS = {
    'total': {
        'snr_quality': 0.25,      # SNR质量 (25%)
        'cnr_quality': 0.25,      # CNR质量 (25%)
        'noise_quality': 0.20,    # 噪声质量 (20%)
        'artifact_free': 0.20,    # 伪影水平 (20%)
        'image_integrity': 0.10   # 图像完整性 (10%)
    },
    'snr_quality': {
        'snr_value': 0.70,        # SNR数值
        'snr_stability': 0.30     # SNR稳定性
    },
    'cnr_quality': {
        'cnr_value': 0.60,        # CNR数值
        'cnr_contrast': 0.40      # 对比度合理性
    },
    'noise_quality': {
        'uniformity_cv': 0.60,    # 均匀性CV
        'distribution': 0.40      # 分布特性
    },
    'artifact_free': {
        'motion_artifact': 0.40,  # 运动伪影
        'gibbs_artifact': 0.30,   # Gibbs伪影
        'wrap_artifact': 0.20,    # 卷褶伪影
        'chemical_shift': 0.10    # 化学位移
    }
}

# ============================================================================
# 9. CNR组织对模板（从原文件 - 包含default很重要）
# ============================================================================

ANATOMY_TISSUE_PAIRS = {
    "lumbar": [
        {
            "name": "vertebral_body_vs_disc",
            "roi1_offset": (-20, 0),
            "roi2_offset": (20, 0),
            "roi_size": 20,
            "clinical_significance": "评估脊柱退行性病变",
            "description": "椎体 vs 椎间盘"
        },
        {
            "name": "spinal_cord_vs_csf",
            "roi1_offset": (0, -15),
            "roi2_offset": (0, 15),
            "roi_size": 15,
            "clinical_significance": "评估脊髓和脑脊液对比",
            "description": "脊髓 vs 脑脊液"
        }
    ],
    "brain": [
        {
            "name": "gray_matter_vs_white_matter",
            "roi1_offset": (-15, 0),
            "roi2_offset": (15, 0),
            "roi_size": 15,
            "clinical_significance": "评估灰质白质对比",
            "description": "灰质 vs 白质"
        },
        {
            "name": "cortex_vs_csf",
            "roi1_offset": (-10, -10),
            "roi2_offset": (10, 10),
            "roi_size": 12,
            "clinical_significance": "评估皮质和脑脊液对比",
            "description": "皮质 vs 脑脊液"
        }
    ],
    "cervical": [
        {
            "name": "cord_vs_csf",
            "roi1_offset": (0, -12),
            "roi2_offset": (0, 12),
            "roi_size": 12,
            "clinical_significance": "评估颈髓和脑脊液对比",
            "description": "颈髓 vs 脑脊液"
        }
    ],
    "spine": [  # 添加spine的默认配置
        {
            "name": "center_vs_periphery",
            "roi1_offset": (0, 0),
            "roi2_offset": (25, 0),
            "roi_size": 15,
            "clinical_significance": "评估脊柱中心与周边组织对比",
            "description": "脊柱中心区域 vs 周边区域"
        }
    ],
    "default": [
        {
            "name": "center_vs_periphery",
            "roi1_offset": (0, 0),
            "roi2_offset": (25, 0),
            "roi_size": 15,
            "clinical_significance": "评估中心与周边组织对比",
            "description": "中心区域 vs 周边区域"
        }
    ]
}

# ============================================================================
# 10. CSV字段配置（从原文件，与系统兼容）
# ============================================================================

CSV_FIELDS = [
    'patient_id',
    'scan_name',
    'anatomical_region',
    'sequence_type',
    'field_strength_t',
    'parallel_imaging',
    'acceleration_factor',
    'snr_value',
    'snr_db',
    'snr_rating',
    'cnr_value',
    'cnr_rating',
    'cnr_tissue_pair',
    'noise_uniformity_cv',
    'noise_uniformity_rating',
    'signal_mean',
    'noise_std',
    'g_factor_applied',
    'background_pixels',
    'roi_mean',
    'roi_std',
    'algorithm_confidence',
    'image_quality',
    'validation_status',
    'analysis_date',
    'analysis_status'
]

# ============================================================================
# 11. 批量处理配置（整合两个版本）
# ============================================================================

BATCH_PROCESSING = {
    'max_workers': 4,                    # 最大并行工作数
    'skip_on_error': True,               # 出错时跳过继续处理
    'max_retries': 1,                    # 最大重试次数
    'timeout_per_scan': 300,             # 每扫描超时时间（秒）
    'memory_limit_gb': 3.0,              # 内存限制（GB）
    'log_level': 'INFO',                 # 日志级别
    'progress_update_interval': 10       # 进度更新间隔（扫描数）
}

# ============================================================================
# 12. 硬约束阈值（从原文件 - 非常重要！）
# ============================================================================

HARD_CONSTRAINTS = {
    'file_system': {
        'max_file_size_gb': 3.0,      # <3GB
        'min_file_size_kb': 1,        # >1KB
    },
    'image_data': {
        'min_slices': 1,              # ≥1切片
        'min_total_pixels': 1000,     # 1K像素
        'max_total_pixels': 100000000, # 100M像素
        'allowed_dimensions': [2, 3], # 2D或3D
    },
    'metadata': {
        'required_fields_m1': [        # M1级必需字段
            'patient_id',
            'anatomical_region',
            'sequence_type',
            'field_strength_t',
            'matrix_size',
            'num_slices',
            'pixel_spacing_mm'
        ]
    },
    'system_resources': {
        'max_memory_gb_per_scan': 3.0,  # <3GB/扫描
        'max_processing_time_sec': 300, # <300秒/扫描
    },
    # 为 dicom_converter 添加DICOM特定约束
    'dicom': {
        'required_dicom_tags': [
            ('Modality', 'Modality'),
            ('PatientID', 'PatientID'),
            ('StudyDate', 'StudyDate'),
            ('SeriesDescription', 'SeriesDescription'),
            ('Rows', 'Rows'),
            ('Columns', 'Columns'),
            ('PixelSpacing', 'PixelSpacing')
        ]
    }
}

# ============================================================================
# 13. 错误处理配置（为 dicom_converter_enhanced.py 添加）
# ============================================================================

ERROR_HANDLING = {
    'error_codes': {
        'FILE_NOT_FOUND': 'E001',
        'DICOM_READ_ERROR': 'E002',
        'INSUFFICIENT_DATA': 'E003',
        'IMAGE_TOO_LARGE': 'E004',
        'INVALID_DIMENSIONS': 'E005',
        'METADATA_MISSING': 'E006',
        'CONVERSION_ERROR': 'E007',
        'HARD_CONSTRAINT_VIOLATED': 'E008'
    },
    'log_levels': {
        'ERROR': 'ERROR',
        'WARNING': 'WARNING',
        'INFO': 'INFO',
        'DEBUG': 'DEBUG'
    }
}

# ============================================================================
# 14. 版本配置（为 dicom_converter_enhanced.py 添加）
# ============================================================================

VERSION_CONFIG = {
    'metadata_schema_version': '2.0',
    'current_version': '3.0',
    'compatible_versions': ['2.0', '2.1', '2.2', '3.0'],
    'conversion_tool': 'MRI_AutoQA_Converter_v3.0'
}

# ============================================================================
# 15. 可视化配置（从原文件 - 完整配置）
# ============================================================================

VISUALIZATION_CONFIG = {
    'figure_size': (18, 12),  # 英寸
    'dpi': 150,
    'output_format': 'png',
    
    # 颜色方案
    'colors': {
        'confidence': {
            'HIGH': '#4CAF50',
            'MEDIUM': '#FFC107', 
            'LOW': '#FF9800',
            'FAILED': '#F44336'
        },
        'snr_rating': {
            'EXCELLENT': '#2E7D32',
            'GOOD': '#7CB342',
            'FAIR': '#FFB300',
            'POOR': '#FF7043',
            'UNACCEPTABLE': '#D32F2F'
        },
        'roi': {
            'signal': '#00C853',
            'background': '#FF5252',
            'cnr_roi1': '#2979FF',
            'cnr_roi2': '#FF4081'
        }
    },
    
    # 字体设置（已在上方全局配置）
    'fonts': {
        'family': matplotlib.rcParams['font.sans-serif'][0] if matplotlib.rcParams['font.sans-serif'] else 'DejaVu Sans',
        'title_size': 10,
        'label_size': 9,
        'tick_size': 8,
        'annotation_size': 8
    }
}

# 子图标题（英文）
SUBPLOT_TITLES = {
    1: 'Original Image with ROI Annotation',
    2: 'SNR Analysis', 
    3: 'CNR Analysis',
    4: 'Noise Uniformity Analysis',
    5: 'Artifact Detection',
    6: 'Comprehensive Quality Score',
    7: 'Signal ROI Histogram',
    8: 'Background Noise Histogram',
    9: 'Confidence Assessment',
    10: 'Score Components Breakdown',
    11: 'Acquisition Information',
    12: 'Processing Information'
}

# ============================================================================
# 16. 质量评估阈值（从原文件）
# ============================================================================

QUALITY_THRESHOLDS = {
    'snr': {
        'excellent': 30.0,
        'good': 20.0,
        'fair': 10.0,
        'poor': 5.0
    },
    'cnr': {
        'excellent': 5.0,
        'good': 3.0,
        'fair': 1.5,
        'poor': 0.5
    },
    'confidence': {
        'high': 0.80,
        'medium': 0.60,
        'low': 0.40,
        'failed': 0.00
    },
    'noise_uniformity': {
        'excellent': 0.3,
        'good': 0.5,
        'fair': 0.7,
        'poor': 1.0
    }
}

# ============================================================================
# 17. 系统配置（从原文件）
# ============================================================================

SYSTEM_CONFIG = {
    'software_version': '3.0',
    'min_python_version': (3, 8),
    'required_packages': [
        'numpy>=1.21.0',
        'nibabel>=4.0.0',
        'matplotlib>=3.5.0',
        'pandas>=1.4.0',
        'scipy>=1.8.0',
        'psutil>=5.9.0',
        'pydicom>=2.3.0'  # 为 dicom_converter 添加
    ],
    'optional_packages': [
        'scikit-image>=0.19.0',
        'scikit-learn>=1.1.0',
        'seaborn>=0.11.0'
    ],
    'supported_formats': ['.nii.gz', '.nii', '.dcm'],
    'max_concurrent_processes': 4,
    'environment': {
        'os': system,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
}

# ============================================================================
# 18. 工具函数（从原文件，新增搜索策略相关函数）
# ============================================================================

def get_config_summary():
    """获取配置摘要"""
    import sys
    
    return {
        'project_root': str(PROJECT_ROOT),
        'input_dir': str(INPUT_DIR),
        'output_dir': str(OUTPUT_DIR),
        'software_version': SYSTEM_CONFIG['software_version'],
        'supported_anatomies': list(ROI_SETTINGS.keys()),
        'supported_sequences': list(SEQUENCE_SNR_STANDARDS.keys()),
        'csv_fields_count': len(CSV_FIELDS),
        'batch_processing': BATCH_PROCESSING,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'search_strategy_enabled': 'UNIFIED_SEARCH_STRATEGY' in globals(),
        'regions_with_fallback': sum(1 for region in ROI_SETTINGS.values() if 'fallback_positions' in region)
    }


def get_region_search_strategy(anatomical_region: str) -> Dict[str, Any]:
    """获取指定解剖区域的搜索策略配置"""
    region_cfg = ROI_SETTINGS.get(anatomical_region, ROI_SETTINGS["default"])
    
    # 构建完整的策略配置
    strategy = {
        'region_specific': region_cfg,
        'unified_framework': UNIFIED_SEARCH_STRATEGY,
        'effective_params': {
            'search_center': (
                region_cfg.get('search_center_y_ratio', 0.5),
                region_cfg.get('search_center_x_ratio', 0.5)
            ),
            'search_radius': (
                region_cfg.get('search_radius_y_ratio', 0.15),
                region_cfg.get('search_radius_x_ratio', 0.10)
            ),
            'expected_signal_range': (
                region_cfg.get('expected_signal_min', 50),
                region_cfg.get('expected_signal_max', 150)
            ),
            'max_allowed_cv': region_cfg.get('max_allowed_cv', 0.25),
            'roi_size_ratio': region_cfg.get('roi_size_ratio', 0.08),
            'has_fallback_positions': 'fallback_positions' in region_cfg
        }
    }
    
    return strategy


def validate_search_config() -> List[str]:
    """验证搜索配置的完整性"""
    issues = []
    
    # 检查统一策略是否存在
    if 'UNIFIED_SEARCH_STRATEGY' not in globals():
        issues.append("❌ 缺少UNIFIED_SEARCH_STRATEGY配置")
        return issues
    
    # 检查统一策略的结构
    required_strategy_keys = ['search_layers', 'fallback_strategy', 'scoring_system']
    for key in required_strategy_keys:
        if key not in UNIFIED_SEARCH_STRATEGY:
            issues.append(f"❌ UNIFIED_SEARCH_STRATEGY缺少'{key}'配置")
    
    # 检查评分权重总和为1.0
    if 'scoring_system' in UNIFIED_SEARCH_STRATEGY:
        weights = UNIFIED_SEARCH_STRATEGY['scoring_system'].get('weights', {})
        if weights:
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                issues.append(f"❌ 评分权重总和不为1.0: {weight_sum:.3f}")
    
    # 检查所有区域是否有回退位置
    for region_name, region_cfg in ROI_SETTINGS.items():
        if 'fallback_positions' not in region_cfg:
            issues.append(f"⚠️ 区域 '{region_name}' 缺少fallback_positions配置")
        else:
            # 检查回退位置有效性
            for i, pos in enumerate(region_cfg['fallback_positions']):
                required_keys = ['y_ratio', 'x_ratio', 'roi_size_ratio']
                for key in required_keys:
                    if key not in pos:
                        issues.append(f"❌ 区域 '{region_name}' 回退位置{i}缺少'{key}'")
    
    # 检查搜索层配置
    if 'search_layers' in UNIFIED_SEARCH_STRATEGY:
        for i, layer in enumerate(UNIFIED_SEARCH_STRATEGY['search_layers']):
            if 'max_cv_multiplier' not in layer:
                issues.append(f"❌ 搜索层{i}缺少'max_cv_multiplier'配置")
            if 'search_radius_multiplier' not in layer:
                issues.append(f"❌ 搜索层{i}缺少'search_radius_multiplier'配置")
    
    return issues


def validate_config():
    """验证配置有效性"""
    issues = []
    
    # 检查必要目录
    required_dirs = [INPUT_DIR, OUTPUT_DIR, LOG_DIR]
    for dir_path in required_dirs:
        if not dir_path.parent.exists():
            issues.append(f"父目录不存在: {dir_path.parent}")
    
    # 检查配置完整性
    required_configs = ['ROI_SETTINGS', 'SEQUENCE_SNR_STANDARDS', 'CSV_FIELDS', 
                       'HARD_CONSTRAINTS', 'ANATOMICAL_REGION_MAPPING']
    
    for config_name in required_configs:
        if config_name not in globals() or not globals()[config_name]:
            issues.append(f"配置缺失或为空: {config_name}")
    
    # 检查必要的映射
    for region in ROI_SETTINGS.keys():
        if region not in ANATOMY_TISSUE_PAIRS and region != 'default' and region != 'unknown':
            # 警告但不作为错误
            issues.append(f"警告: {region} 没有对应的ANATOMY_TISSUE_PAIRS配置")
    
    # 验证搜索配置
    search_issues = validate_search_config()
    issues.extend(search_issues)
    
    return issues


# 添加sys模块导入
import sys

if __name__ == "__main__":
    # 测试配置
    print("MRI AutoQA Configuration Test - 整合版 v3.0 (含搜索策略)")
    print("=" * 60)
    
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"{key:25s}: {value}")
    
    issues = validate_config()
    if issues:
        print("\n配置验证结果:")
        for issue in issues:
            print(f"  {issue}")
        
        # 统计错误和警告
        errors = sum(1 for issue in issues if issue.startswith("❌"))
        warnings = sum(1 for issue in issues if issue.startswith("⚠️") or issue.startswith("警告:"))
        
        if errors == 0:
            print(f"\n✓ 必要配置验证通过 (有{warnings}个警告)")
        else:
            print(f"\n✗ 存在{errors}个错误需要修复，{warnings}个警告")
    else:
        print("\n✓ 所有配置验证通过")
    
    # 显示搜索策略详情
    if 'UNIFIED_SEARCH_STRATEGY' in globals():
        print("\n🔍 搜索策略详情:")
        strategy = UNIFIED_SEARCH_STRATEGY
        print(f"  搜索层数: {len(strategy['search_layers'])}")
        print(f"  回退策略: {strategy['fallback_strategy']['method']}")
        print(f"  评分权重: {strategy['scoring_system']['weights']}")
        
        # 显示示例区域的配置
        if 'lumbar' in ROI_SETTINGS:
            lumbar_cfg = ROI_SETTINGS['lumbar']
            print(f"\n📊 腰椎区域示例:")
            print(f"  搜索中心: ({lumbar_cfg['search_center_y_ratio']}, {lumbar_cfg['search_center_x_ratio']})")
            print(f"  信号范围: [{lumbar_cfg['expected_signal_min']}, {lumbar_cfg['expected_signal_max']}]")
            print(f"  最大CV: {lumbar_cfg['max_allowed_cv']}")
            print(f"  回退位置数: {len(lumbar_cfg['fallback_positions'])}")