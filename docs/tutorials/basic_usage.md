

### docs/tutorials/basic_usage.md
``markdown
# Basic Usage Tutorial / 基础使用教程

## 中文 (Chinese):

### 第一步: 数据准备
``bash
# 创建数据目录
mkdir data/

# 将DICOM文件按以下结构组织
data/
└── patient_001/
    └── scan_001/
        ├── *.dcm (DICOM文件)
        └── ...



### 第二步: DICOM转换
# 将DICOM转换为NIfTI格式
python dicom_converter.py --input data --output converted_data

### 第三步: 批量分析
# 运行一键分析
python autoqa_launcher.py --input converted_data --output autoqa_results

# 或使用详细参数
python autoqa_launcher.py --input my_data --output my_results --skip-vis --force-clean


### 第四步: 查看结果

分析完成后，结果将保存在 `autoqa_results/` 目录中：

- `batch_report_YYYYMMDD_HHMMSS/`: 带时间戳的完整报告
- `00_executive_summary/`: 执行摘要
- `01_detailed_data/`: 详细数据CSV
- `02_visualizations/`: 可视化图表
- `03_quality_analysis/`: 质量分析报告

## English:

### Step 1: Data Preparation
# Create data directory
mkdir data/

# Organize DICOM files in the following structure
data/
└── patient_001/
    └── scan_001/
        ├── *.dcm (DICOM files)
        └── ...
### Step 2: DICOM Conversion
# Convert DICOM to NIfTI format
python dicom_converter.py --input data --output converted_data

### Step 3: Batch Analysis

# Run one-click analysis
python autoqa_launcher.py --input converted_data --output autoqa_results

# Or with detailed parameters
python autoqa_launcher.py --input my_data --output my_results --skip-vis --force-clean


### Step 4: View Results

After analysis, results will be saved in the `autoqa_results/` directory:

- `batch_report_YYYYMMDD_HHMMSS/`: Timestamped complete reports
- `00_executive_summary/`: Executive summaries
- `01_detailed_data/`: Detailed data CSV
- `02_visualizations/`: Visualization charts
- `03_quality_analysis/`: Quality analysis reports