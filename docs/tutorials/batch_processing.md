
### docs/tutorials/batch_processing.md
``markdown
# Batch Processing Tutorial / 批量处理教程

## 中文 (Chinese):

### 批量处理流程
``python
from batch_processor import BatchProcessor

# 创建批量处理器
processor = BatchProcessor(
    input_dir="converted_data/",
    output_dir="autoqa_results/",
    verbose=True
)

# 处理所有扫描
stats = processor.process_all()

# 查看统计信息
print(f"成功处理: {stats['successful']}/{stats['total_scans']}")

### 数据发现机制

批量处理器自动发现以下结构的扫描：
input_directory/
└── patient_id/
    └── scan_name/
        ├── scan.nii.gz
        └── metadata.json
### 输出文件

- `quality_summary.csv`: 批量处理结果汇总
- `patient_id/scan_name/qa_report.json`: 每个扫描的详细报告
- `patient_id/scan_name/visualization.png`: 可视化报告
# 配置错误处理

BATCH_PROCESSING = {

'max_workers': 4, # 最大并行工作数

'skip_on_error': True, # 出错时跳过继续处理

'max_retries': 1, # 最大重试次数

'timeout_per_scan': 300, # 每扫描超时时间（秒）

'memory_limit_gb': 3.0, # 内存限制（GB）

'log_level': 'INFO', # 日志级别

'progress_update_interval': 10 # 进度更新间隔（扫描数）

}

## English:

### Batch Processing Flow
from batch_processor import BatchProcessor

  

# Create batch processor

processor = BatchProcessor(

input_dir="converted_data/",

output_dir="autoqa_results/",

verbose=True

)

  

# Process all scans

stats = processor.process_all()

  

# View statistics

print(f"Successfully processed: {stats['successful']}/{stats['total_scans']}")

### Data Discovery Mechanism

The batch processor automatically discovers scans in the following structure:

input_directory/
└── patient_id/
    └── scan_name/
        ├── scan.nii.gz
        └── metadata.json


### Output Files

- `quality_summary.csv`: Batch processing result summary
- `patient_id/scan_name/qa_report.json`: Detailed report for each scan
- `patient_id/scan_name/visualization.png`: Visualization report
# Configure error handling

BATCH_PROCESSING = {

'max_workers': 4, # Maximum parallel workers

'skip_on_error': True, # Skip on error and continue processing

'max_retries': 1, # Maximum retry attempts

'timeout_per_scan': 300, # Timeout per scan (seconds)

'memory_limit_gb': 3.0, # Memory limit (GB)

'log_level': 'INFO', # Log level

'progress_update_interval': 10 # Progress update interval (scans)

}