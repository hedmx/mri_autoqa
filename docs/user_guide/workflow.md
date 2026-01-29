# Analysis Workflow / 分析工作流程

## 中文 (Chinese):

### 完整工作流程
1. **数据准备**: 收集 DICOM 文件并按患者/扫描组织
2. **DICOM 转换**: 将 DICOM 转换为 NIfTI 格式并生成元数据
3. **算法路由**: 根据元数据确定适当的分析方法
4. **质量分析**: 执行 SNR、CNR 和其他质量指标计算
5. **置信度评估**: 计算四维度置信度得分
6. **质量评分**: 计算五维度质量评分
7. **结果输出**: 生成 JSON 报告和 CSV 摘要
8. **可视化**: 生成质量报告可视化图表

### 单扫描分析流程

输入 NIfTI 图像  
↓  
阶段 1: 全局诊断扫描  
↓  
阶段 2: 背景选择  
↓  
阶段 3: 噪声估计  
↓  
阶段 4: 信号ROI选择  
↓  
SNR 计算  
↓  
质量指标计算  
↓  
置信度评估  
↓  
生成报告


## English:

### Complete Workflow
1. **Data Preparation**: Collect DICOM files organized by patient/scan
2. **DICOM Conversion**: Convert DICOM to NIfTI format and generate metadata
3. **Algorithm Routing**: Determine appropriate analysis method based on metadata
4. **Quality Analysis**: Execute SNR, CNR, and other quality metric calculations
5. **Confidence Assessment**: Calculate four-dimensional confidence scores
6. **Quality Scoring**: Calculate five-dimensional quality scores
7. **Result Output**: Generate JSON reports and CSV summaries
8. **Visualization**: Generate quality report visualizations

### Single Scan Analysis Flow

Input NIfTI Image  
↓  
Phase 1: Global Diagnostic Scan  
↓  
Phase 2: Background Selection  
↓  
Phase 3: Noise Estimation  
↓  
Phase 4: Signal ROI Selection  
↓  
SNR Calculation  
↓  
Quality Metrics Calculation  
↓  
Confidence Assessment  
↓  
Generate Report