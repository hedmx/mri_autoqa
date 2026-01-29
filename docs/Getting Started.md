# Getting Started with MRI AutoQA / MRI AutoQA 快速入门

## 中文 (Chinese):

### 先决条件
- Python 3.8+
- 至少 8GB 内存
- 兼容的 DICOM/NIfTI 文件

### 安装
bash
pip install numpy nibabel matplotlib pandas scipy pydicom

### 基础使用
# 1. 准备数据
mkdir data/
# 将 DICOM 文件放在 data/patient_id/scan_name/ 中

# 2. 转换 DICOM 到 NIfTI
python dicom_converter.py --input data --output converted_data

# 3. 运行分析
python autoqa_launcher.py --input converted_data --output results

### 期望输出

- JSON 格式的质量报告
- CSV 格式的统计摘要
- 可视化图表
- 批量处理日志

## English:
### Prerequisites
- Python 3.8+
- At least 8GB RAM
- Compatible DICOM/NIfTI files

### Installation
bash
pip install numpy nibabel matplotlib pandas scipy pydicom

### Basic Usage
# 1. Prepare your data
mkdir data/
# Place DICOM files in data/patient_id/scan_name/

# 2. Convert DICOM to NIfTI
python dicom_converter.py --input data --output converted_data

# 3. Run analysis
python autoqa_launcher.py --input converted_data --output results

### Expected Output

- Quality reports in JSON format
- Statistical summaries in CSV
- Visualization plots
- Batch processing logs