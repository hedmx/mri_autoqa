# MRI AutoQA Documentation / MRI AutoQA 文档

## 中文 (Chinese):

#  MRI AutoQA：开源 MRI 图像全自动质量评估平台
### Open-Source Automated MRI Quality Assessment Platform


---
 **MRI AutoQA** 是一个面向临床与科研的端到端开源平台，致力于实现磁共振成像（MRI）质量的全自动、客观化评估。
##  摘要 / Abstract

**中文**：
磁共振成像（MRI）的质量是保障诊断准确性和科研数据可靠性的基石。然而，传统的人工质量评估（QA）方法效率低下、主观性强且难以规模化。为此，我们开发并开源了 **MRI AutoQA**，旨在实现从原始 DICOM 数据到标准化、多维度、可视化质量报告的全流程自动化分析。

**English**:
Magnetic Resonance Imaging (MRI) quality is the cornerstone of diagnostic accuracy and research reliability. However, traditional manual Quality Assessment (QA) methods are inefficient, subjective, and difficult to scale. To address this, we developed **MRI AutoQA**, an end-to-end open-source platform designed to automate the entire process from raw DICOM data to standardized, multi-dimensional, and visual quality reports.

---

##  核心特性 / Key Features

- ** 端到端自动化**：一键完成 DICOM 解析、NIfTI 转换、指标计算与报告生成。
- ** 多维度量化指标**：
    - **信噪比 (SNR)**：采用 V3 引擎与瑞利（Rayleigh）分布校正，无偏估计。
    - **对比噪声比 (CNR)**：基于解剖模板的自动计算。
    - **均匀性 (Uniformity)**：量化噪声空间分布。
    - **伪影检测**：识别运动、Gibbs 等伪影严重程度。
- ** 智能上下文感知**：根据解剖区域（脑、脊柱等）、序列类型（T1/T2/PD）及场强（1.5T/3.0T）动态调整分析策略。
- ** 标准化可视化**：生成包含 12 个子图的 PNG 报告及 JSON 结构化数据。
- ** 科研级鲁棒性**：基于最大似然估计（MLE）、四分位距（IQR）等共识机制筛选噪声区域。

---

##  性能验证 / Performance Validation

我们在包含 **10,607** 例 MRI 扫描的本地数据集上进行了验证，覆盖多种解剖部位与场强。

| 指标 (Metric) | 平均值 (Mean) | 中位数 (Median) | 处理速度 (Time) |
| :--- | :--- | :--- | :--- |
| **信噪比 (SNR)** | 128.24 | 86.57 | **5.54 秒/例** |
| **对比度 (CNR)** | 23.85 | 9.58 | (Intel i3, 32GB RAM) |
| **质量评分 (Score)** | 0.831 | 0.897 | |
| **置信度 (Confidence)** | 0.728 | 0.738 | |

> **结果说明**：平台成功处理了所有案例，展现了极高的鲁棒性与效率。


## 目录 / Table of Contents
- [快速入门](getting_started.md) - 快速开始指南
- [安装指南](installation.md) - 设置说明
- [用户指南](user_guide/) - 完整使用文档
- [API参考](api_documentation/) - 技术参考
- [教程](tutorials/) - 逐步指南
- [示例](examples/) - 实际示例
- [开发者指南](developer_guide/) - 为贡献者提供

## 快速链接 / Quick Links
- [系统要求](installation.md#system-requirements) / [System Requirements](installation.md#system-requirements)
- [基础使用](tutorials/basic_usage.md) / [Basic Usage](tutorials/basic_usage.md)
- [故障排除](tutorials/troubleshooting.md) / [Troubleshooting](tutorials/troubleshooting.md)
- [贡献指南](developer_guide/contributing.md) / [Contributing](developer_guide/contributing.md)

## English:

Welcome to the comprehensive documentation for MRI AutoQA, an open-source automated system for medical imaging quality analysis.

## Table of Contents
- [Getting Started](getting_started.md) - Quick start guide
- [Installation](installation.md) - Setup instructions
- [User Guide](user_guide/) - Complete usage documentation
- [API Reference](api_documentation/) - Technical reference
- [Tutorials](tutorials/) - Step-by-step guides
- [Examples](examples/) - Practical examples
- [Developer Guide](developer_guide/) - For contributors

## Quick Links
- [System Requirements](installation.md#system-requirements)
- [Basic Usage](tutorials/basic_usage.md)
- [Troubleshooting](tutorials/troubleshooting.md)
- [Contributing](developer_guide/contributing.md)



