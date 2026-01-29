# Algorithm Details / 算法详解

## 中文 (Chinese):

### 单图像SNR分析算法 (V3)

#### 阶段 1: 全局诊断扫描
- 识别图像中的候选背景区域
- 评估角落区域的空气/背景特征
- 根据均值和标准差阈值过滤区域

#### 阶段 2: 背景选择
- 选择最优空气背景区域
- 基于均值、标准差和大小标准的评分
- 如果没有空气区域，回退到最低均值区域

#### 阶段 3: 噪声估计
- 应用瑞利分布校正
- 使用多种估计方法 (MLE, IQR, MAD)
- 一致性方法确保稳健估计

#### 阶段 4: 信号ROI选择
- 基于解剖区域的椭圆ROI定位
- 强度引导的ROI位置微调
- ROI内均匀性优化

### 质量评估系统

#### 置信度评分 (四维度)
1. **ROI质量 (C_ROI)**: 40% 权重
   - 信号强度 (40%)
   - 均匀性 (30%)
   - 边缘距离 (20%)
   - 解剖适当性 (10%)

2. **噪声可靠性 (C_noise)**: 30% 权重
   - 背景大小 (30%)
   - 背景均值 (40%)
   - 背景均匀性 (30%)

3. **图像质量 (C_image)**: 20% 权重
   - 对比度 (40%)
   - 伪影 (30%)
   - 完整性 (30%)

4. **算法质量 (C_algorithm)**: 10% 权重
   - 收敛性 (50%)
   - 稳定性 (30%)
   - 完整性 (20%)

#### 质量评分 (五维度)
- SNR质量: 25%
- CNR质量: 25%
- 噪声质量: 20%
- 无伪影: 20%
- 图像完整性: 10%

## English:

### Single Image SNR Analysis Algorithm (V3)

#### Phase 1: Global Diagnostic Scan
- Identifies candidate background regions in image
- Evaluates corner regions for air/background characteristics
- Filters regions based on mean and std thresholds

#### Phase 2: Background Selection
- Selects optimal air background region
- Uses scoring based on mean, std, and size criteria
- Falls back to lowest mean regions if no air regions found

#### Phase 3: Noise Estimation
- Applies Rayleigh distribution correction
- Uses multiple estimation methods (MLE, IQR, MAD)
- Consensus approach for robust estimation

#### Phase 4: Signal ROI Selection
- Elliptical ROI positioning based on anatomical region
- Intensity-guided fine-tuning of ROI position
- Uniformity optimization within ROI

### Quality Assessment System

#### Confidence Scoring (Four Dimensions)
1. **ROI Quality (C_ROI)**: 40% weight
   - Signal intensity (40%)
   - Uniformity (30%)
   - Edge distance (20%)
   - Anatomical appropriateness (10%)

2. **Noise Reliability (C_noise)**: 30% weight
   - Background size (30%)
   - Background mean (40%)
   - Background uniformity (30%)

3. **Image Quality (C_image)**: 20% weight
   - Contrast (40%)
   - Artifacts (30%)
   - Integrity (30%)

4. **Algorithm Quality (C_algorithm)**: 10% weight
   - Convergence (50%)
   - Stability (30%)
   - Completeness (20%)

### Quality Scoring (Five Dimensions)
- SNR Quality: 25%
- CNR Quality: 25%
- Noise Quality: 20%
- Artifact-Free: 20%
- Image Integrity: 10%


