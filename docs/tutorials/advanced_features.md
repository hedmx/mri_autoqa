
### docs/tutorials/advanced_features.md
``markdown
# Advanced Features Tutorial / 高级功能教程

## 中文 (Chinese):

### 自定义ROI设置
您可以修改 `config.py` 中的 ROI_SETTINGS 来自定义ROI位置：

``python
ROI_SETTINGS = {
    'custom_region': {
        'center_x_ratio': 0.45,  # X轴中心位置比例
        'center_y_ratio': 0.55,  # Y轴中心位置比例
        'size_ratio': 0.22,      # ROI大小比例
        'description': '自定义解剖区域'
    }
}

### 并行处理

使用批处理器的并行功能：
from batch_processor import BatchProcessor
processor = BatchProcessor()
processor.process_all() # 将使用BATCH_PROCESSING配置中的max_workers

### 高级质量评估

访问详细的质量评估结果：
from quality_metrics import QualityMetricsCalculator

calculator = QualityMetricsCalculator()

results = calculator.calculate_all_metrics(

image=image_data,

metadata=scan_metadata,

signal_roi=signal_roi_info,

background_roi=background_roi_info

)

# 访问详细的置信度评估

confidence = results['confidence_scores']

quality_scores = results['quality_scores']

from quality_metrics import QualityMetricsCalculator

## English:

### Custom ROI Settings

You can modify the ROI_SETTINGS in `config.py` to customize ROI positions:  

ROI_SETTINGS = {

'custom_region': {

'center_x_ratio': 0.45, # Center position ratio on X-axis

'center_y_ratio': 0.55, # Center position ratio on Y-axis

'size_ratio': 0.22, # ROI size ratio

'description': 'Custom anatomical region'

}

}

### Parallel Processing

Use the parallel functionality of the batch processor:
from batch_processor import BatchProcessor
processor = BatchProcessor()
processor.process_all() # 将使用BATCH_PROCESSING配置中的max_workers

### Advanced Quality Assessment

Access detailed quality assessment results:
from quality_metrics import QualityMetricsCalculator

calculator = QualityMetricsCalculator()

results = calculator.calculate_all_metrics(

image=image_data,

metadata=scan_metadata,

signal_roi=signal_roi_info,

background_roi=background_roi_info

)

  

# Access detailed confidence assessment

confidence = results['confidence_scores']

quality_scores = results['quality_scores']

