# Output Formats / 输出格式

## 中文 (Chinese):

### JSON 质量报告
每个分析的扫描产生一个全面的 JSON 报告，具有以下结构：

json
{
  "analysis_info": {
    "date": "时间戳",
    "software": "版本",
    "algorithm_version": "算法类型",
    "scan_id": "患者/扫描标识符"
  },
  "acquisition": {
    "patient_id": "标识符",
    "scan_name": "扫描名称",
    "anatomical_region": "区域",
    "sequence_type": "类型",
    "field_strength": "强度",
    "parallel_imaging": 布尔值,
    "acceleration_factor": 数字
  },
  "snr_results": {
    "traditional": {...},
    "recommended": {...},
    "rayleigh_correction": {...}
  },
  "quality_assessment": {
    "snr_rating": {...},
    "cnr_analysis": {...},
    "noise_uniformity": {...},
    "confidence_scores": {...},
    "quality_scores": {...}
  }
}

### CSV 摘要文件

批量分析结果的表格格式：

patient_id,scan_name,anatomical_region,sequence_type,field_strength_t,parallel_imaging,acceleration_factor,snr_value,snr_db,snr_rating,cnr_value,cnr_rating,noise_uniformity_cv,quality_score_total,confidence_score,analysis_status

### 可视化输出

- 个体扫描质量报告 (PNG)
- 批量统计摘要 (PNG/PDF)
- ROI 叠加图像
- 直方图图表

## English:

### JSON Quality Reports

Each analyzed scan produces a comprehensive JSON report with the following structure:

{

"analysis_info": {

"date": "timestamp",

"software": "version",

"algorithm_version": "algorithm_type",

"scan_id": "patient/scan_identifier"

},

"acquisition": {

"patient_id": "identifier",

"scan_name": "scan_name",

"anatomical_region": "region",

"sequence_type": "type",

"field_strength": "strength",

"parallel_imaging": boolean,

"acceleration_factor": number

},

"snr_results": {

"traditional": {...},

"recommended": {...},

"rayleigh_correction": {...}

},

"quality_assessment": {

"snr_rating": {...},

"cnr_analysis": {...},

"noise_uniformity": {...},

"confidence_scores": {...},

"quality_scores": {...}

}

}

### CSV Summary File

Tabular format for batch analysis results:
patient_id,scan_name,anatomical_region,sequence_type,field_strength_t,parallel_imaging,acceleration_factor,snr_value,snr_db,snr_rating,cnr_value,cnr_rating,noise_uniformity_cv,quality_score_total,confidence_score,analysis_status

### Visualization Outputs

- Individual scan quality reports (PNG)
- Batch statistical summaries (PNG/PDF)
- ROI overlay images
- Histogram plots

