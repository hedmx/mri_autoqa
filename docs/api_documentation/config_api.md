
### docs/api_documentation/config_api.md

# Configuration API / 配置API

## 中文 (Chinese):

### 全局配置
# 从 config.py 导入
from config import (
     # 文件系统配置
    SCAN_FILENAME, METADATA_FILENAME, QA_REPORT_FILENAME,
    
    # DICOM映射配置
    ANATOMICAL_REGION_MAPPING, SEQUENCE_TYPE_MAPPING, FIELD_STRENGTH_MAPPING,
    
    # ROI模板配置
    ROI_SETTINGS,
    
    # SNR标准配置
    SEQUENCE_SNR_STANDARDS,
    
    # 并行成像g-factor表
    G_FACTOR_TABLE,
    
    # 置信度权重配置
    CONFIDENCE_WEIGHTS,
    
    # 质量评分权重配置
    QUALITY_SCORE_WEIGHTS,
    
    # CNR组织对模板
    ANATOMY_TISSUE_PAIRS,
    
    # CSV字段配置
    CSV_FIELDS,
    
    # 批量处理配置
    BATCH_PROCESSING,
    
    # 硬约束阈值
    HARD_CONSTRAINTS,
    
    # 可视化配置
    VISUALIZATION_CONFIG,
    
    # 质量评估阈值
    QUALITY_THRESHOLDS,
    
    # 系统配置
    SYSTEM_CONFIG
)

### ROI设置示例
ROI_SETTINGS = {

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

}

}
## English:

### Global Configuration

### Import from config.py

from config import (

# File system configuration

SCAN_FILENAME, METADATA_FILENAME, QA_REPORT_FILENAME,

# DICOM mapping configuration

ANATOMICAL_REGION_MAPPING, SEQUENCE_TYPE_MAPPING, FIELD_STRENGTH_MAPPING,

# ROI template configuration

ROI_SETTINGS,

# SNR standard configuration

SEQUENCE_SNR_STANDARDS,

# Parallel imaging g-factor table

G_FACTOR_TABLE,

# Confidence weight configuration

CONFIDENCE_WEIGHTS,

# Quality score weight configuration

QUALITY_SCORE_WEIGHTS,

# CNR tissue pair templates

ANATOMY_TISSUE_PAIRS,

# CSV field configuration

CSV_FIELDS,

# Batch processing configuration

BATCH_PROCESSING,

# Hard constraint thresholds

HARD_CONSTRAINTS,

# Visualization configuration

VISUALIZATION_CONFIG,

# Quality assessment thresholds

QUALITY_THRESHOLDS,

# System configuration

SYSTEM_CONFIG

)

### ROI Settings Example

ROI_SETTINGS = {

'lumbar': {

'center_x_ratio': 0.5,

'center_y_ratio': 0.6,

'size_ratio': 0.25,

'description': 'Lumbar vertebral body'

},

'brain': {

'center_x_ratio': 0.5,

'center_y_ratio': 0.4,

'size_ratio': 0.2,

'description': 'Brain center'

}

}