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
# 3. DICOM映射配置（为 dicom_converter_enhanced.py 添加）
# ============================================================================

# DICOM术语 -> 标准解剖区域
ANATOMICAL_REGION_MAPPING = {
    # 脊柱相关（根据你的 metadata.json 推测）
    'SPINE': 'spine',
    'LSPINE': 'lumbar',
    'CSPINE': 'cervical',
    'TSPINE': 'thoracic',
    'LUMBAR': 'lumbar',
    'CERVICAL': 'cervical',
    'THORACIC': 'thoracic',
    'T1_TSE_SAG': 'spine',    # 根据你的 metadata.json: "T1_tse_sag_320"
    'T2_TSE_SAG': 'spine',
    
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

# DICOM术语 -> 标准序列类型
SEQUENCE_TYPE_MAPPING = {
    'T1': 't1',
    'T1W': 't1',
    'T1_': 't1',
    'T2': 't2',
    'T2W': 't2',
    'T2_': 't2',
    'PD': 'pd',
    'PDFS': 'pd',
    'FLAIR': 'flair',
    'STIR': 'stir',
    'DWI': 'dwi',
    'ADC': 'adc',
    'SWI': 'swi',
    'TOF': 'tof',
    'TSE': 't1',  # T1 TSE
    'FSE': 't1',  # T1 FSE
}

# 场强映射（实际场强 -> 标准场强）
FIELD_STRENGTH_MAPPING = {
    0.5: '0.5t',
    1.0: '1.0t',
    1.5: '1.5t',  # 根据你的 metadata.json
    3.0: '3.0t',
    7.0: '7.0t'
}

# ============================================================================
# 4. ROI模板配置（整合副本2的详细解剖区域）
# ============================================================================

ROI_SETTINGS = {
    # --- 基本区域（从原文件保留）---
    'default': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.5,
        'size_ratio': 0.3,
        'description': '默认中心ROI'
    },
    'lumbar': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.6,
        'size_ratio': 0.25,
        'description': '腰椎椎体'
    },
    'brain': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.4,
        'size_ratio': 0.2,
        'description': '大脑中心'
    },
    'cervical': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.55,
        'size_ratio': 0.22,
        'description': '颈椎'
    },
    'thoracic': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.58,
        'size_ratio': 0.24,
        'description': '胸椎'
    },
    
    # --- 从副本2添加的详细解剖区域 ---
    # 通用脊柱
    'spine': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.6,
        'size_ratio': 0.25,
        'description': '通用脊柱区域'
    },
    
    # 大脑细分（从副本2）
    'cerebrum': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.5,
        'size_ratio': 0.12,
        'description': '大脑半球中心'
    },
    'cerebellum': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.65,
        'size_ratio': 0.08,
        'description': '小脑中心'
    },
    'cortex': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.5,
        'size_ratio': 0.10,
        'description': '大脑皮层'
    },
    'wm': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.5,
        'size_ratio': 0.10,
        'description': '白质'
    },
    'gm': {
        'center_x_ratio': 0.52,
        'center_y_ratio': 0.5,
        'size_ratio': 0.08,
        'description': '灰质'
    },
    
    # 脊柱细分（从副本2）
    'spinal_cord': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.5,
        'size_ratio': 0.05,
        'description': '脊髓中心'
    },
    'disc': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.55,
        'size_ratio': 0.06,
        'description': '椎间盘'
    },
    
    # 关节相关（从副本2 - 选择常用）
    'knee': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.5,
        'size_ratio': 0.10,
        'description': '膝关节中心'
    },
    'shoulder': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.45,
        'size_ratio': 0.09,
        'description': '肩关节'
    },
    'hip': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.55,
        'size_ratio': 0.10,
        'description': '髋关节中心'
    },
    
    # 器官相关（从副本2 - 选择常用）
    'liver': {
        'center_x_ratio': 0.4,
        'center_y_ratio': 0.4,
        'size_ratio': 0.08,
        'description': '肝右叶'
    },
    'kidney': {
        'center_x_ratio': 0.35,
        'center_y_ratio': 0.45,
        'size_ratio': 0.07,
        'description': '右肾'
    },
    'heart': {
        'center_x_ratio': 0.55,
        'center_y_ratio': 0.45,
        'size_ratio': 0.07,
        'description': '心脏'
    },
    
    # 回退区域
    'unknown': {
        'center_x_ratio': 0.5,
        'center_y_ratio': 0.5,
        'size_ratio': 0.10,
        'description': '未知区域默认ROI'
    }
}

# ============================================================================
# 5. SNR标准配置（整合两个版本）
# ============================================================================

SEQUENCE_SNR_STANDARDS = {
    # 从副本2扩展序列类型，但保持原文件的阈值（更保守）
    't1': {
        'excellent': 30.0,
        'good': 20.0,
        'fair': 10.0,
        'poor': 5.0,
        'description': 'T1加权序列'
    },
    't2': {
        'excellent': 25.0,  # 原文件阈值
        'good': 15.0,
        'fair': 8.0,
        'poor': 4.0,
        'description': 'T2加权序列'
    },
    'pd': {
        'excellent': 28.0,
        'good': 18.0,
        'fair': 9.0,
        'poor': 4.5,
        'description': '质子密度加权序列'
    },
    'flair': {
        'excellent': 22.0,
        'good': 14.0,
        'fair': 7.0,
        'poor': 3.5,
        'description': 'FLAIR序列'
    },
    # 从副本2新增的序列类型
    't2star': {
        'excellent': 25.0,
        'good': 15.0,
        'fair': 8.0,
        'poor': 4.0,
        'description': 'T2*加权序列'
    },
    'swi': {
        'excellent': 25.0,
        'good': 15.0,
        'fair': 8.0,
        'poor': 4.0,
        'description': '磁敏感加权成像'
    },
    'dwi': {
        'excellent': 25.0,
        'good': 15.0,
        'fair': 8.0,
        'poor': 4.0,
        'description': '弥散加权成像'
    },
    'default': {
        'excellent': 25.0,
        'good': 15.0,
        'fair': 8.0,
        'poor': 4.0,
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
# 18. 工具函数（从原文件）
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
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}"
    }


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
    
    return issues


# 添加sys模块导入
import sys

if __name__ == "__main__":
    # 测试配置
    print("MRI AutoQA Configuration Test - 整合版 v3.0")
    print("=" * 50)
    
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"{key:25s}: {value}")
    
    issues = validate_config()
    if issues:
        print("\n配置验证结果:")
        for issue in issues:
            if issue.startswith("警告:"):
                print(f"  ⚠️  {issue}")
            else:
                print(f"  ❌ {issue}")
        
        if all(issue.startswith("警告:") for issue in issues):
            print("\n✓ 所有必要配置验证通过（只有警告）")
        else:
            print("\n✗ 存在配置问题需要修复")
    else:
        print("\n✓ 所有配置验证通过")