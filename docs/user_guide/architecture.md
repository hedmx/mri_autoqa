# System Architecture / 系统架构

## 中文 (Chinese):

### 组件概述

#### 核心组件

single_imagine.py # 主分析引擎  
├── AlgorithmRouter # 路由到适当的算法  
├── SingleImageV3Core # 核心SNR分析  
├── QualityMetricsCalculator # 质量评估计算  

#### 数据流

输入 DICOM 文件  
↓  
DICOM 转换器  
↓  
NIfTI 文件 + 元数据  
↓  
算法路由器  
↓  
适当分析模块  
↓  
质量指标计算  
↓  
JSON 报告 + CSV 输出  
↓  
可视化生成

### 模块描述

#### 算法路由器
- 根据 A/S/F/M 约束空间确定适当的分析方法
- 处理解剖区域检测
- 管理序列类型识别
- 处理并行成像信息

#### 单图像V3引擎
- 实现高级单图像SNR分析
- 使用椭圆ROI自适应定位
- 应用瑞利分布校正
- 提供综合质量指标

#### 质量指标计算器
- 计算四维度置信度得分
- 计算五维度质量得分
- 执行基于组织对模板的CNR分析
- 评估噪声均匀性和伪影

## English:

### Component Overview

#### Core Components

single_imagine.py # Main analysis engine  
├── AlgorithmRouter # Routes to appropriate algorithms  
├── SingleImageV3Core # Core SNR analysis  
├── QualityMetricsCalculator # Quality assessments  
#### Data Flow

Input DICOM Files  
↓  
DICOM Converter  
↓  
NIfTI Files + Metadata  
↓  
Algorithm Router  
↓  
Appropriate Analysis Module  
↓  
Quality Metrics Calculation  
↓  
JSON Reports + CSV Output  
↓  
Visualization Generation

### Module Descriptions

#### Algorithm Router
- Determines appropriate analysis based on A/S/F/M constraint space
- Handles anatomical region detection
- Manages sequence type identification
- Processes parallel imaging information

#### Single Image V3 Engine
- Implements advanced single-image SNR analysis
- Uses elliptical ROI adaptive positioning
- Applies Rayleigh distribution correction
- Provides comprehensive quality metrics

#### Quality Metrics Calculator
- Calculates four-dimensional confidence scores
- Computes five-dimensional quality scores
- Performs CNR analysis using tissue pair templates
- Evaluates noise uniformity and artifacts


