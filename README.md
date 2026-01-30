
# MRI AutoQA Documentation / MRI AutoQA 文档

---

## 中文 (Chinese)

# MRI AutoQA：开源磁共振影像全自动质量评估平台
### 开源全自动磁共振成像质量评估平台

**MRI AutoQA** 是一个面向临床与科研的端到端开源平台，致力于实现磁共振成像（MRI）质量的全自动、客观化评估。

## 摘要 / Abstract

**中文**：
磁共振成像（MRI）的质量是保障诊断准确性和科研数据可靠性的基石。然而，传统的人工质量评估（QA）方法效率低下、主观性强且难以规模化。为此，我们开发并开源了 **MRI AutoQA**，旨在实现从原始 DICOM 数据到标准化、多维度、可视化质量报告的全流程自动化分析。

**English**：
Magnetic Resonance Imaging (MRI) quality is the cornerstone of diagnostic accuracy and research data reliability. However, traditional manual quality assessment (QA) methods are inefficient, subjective, and difficult to scale. To address this, we developed **MRI AutoQA**, an end-to-end open-source platform designed to automate the entire analysis pipeline from raw DICOM data to standardized, multi-dimensional, and visual quality reports.

---

## 核心特性 / Key Features

- **端到端自动化**：一键完成 DICOM 解析、NIfTI 转换、指标计算与报告生成。
- **多维度量化指标**：
    - **信噪比 (SNR)**：采用 V3 引擎与瑞利（Rayleigh）分布校正，实现无偏估计。
    - **对比噪声比 (CNR)**：基于解剖模板的自动计算。
    - **均匀性 (Uniformity)**：量化噪声空间分布。
    - **伪影检测**：识别运动伪影、吉布斯伪影等严重程度。
- **智能上下文感知**：根据解剖区域（脑、脊柱等）、序列类型（T1/T2/PD）及场强（1.5T/3.0T）动态调整分析策略。
- **标准化可视化**：生成包含 12 个子图的 PNG 质量报告及 JSON 结构化数据。
- **科研级鲁棒性**：基于最大似然估计（MLE）、四分位距（IQR）等共识机制筛选噪声区域。

---

## 性能验证 / Performance Validation

我们在本地MRI影像中心进行了平台部署，并对覆盖多种解剖部位与场强的 **10,607** 个 MRI 扫描的数据进行了DICOM数据转换与批量自动化质量分析验证。

| 指标 | 平均值 | 中位数 | 处理速度 |
| :--- | :--- | :--- | :--- |
| **信噪比 (SNR)** | 128.24 | 86.57 | **5.54 秒/例** |
| **对比噪声比 (CNR)** | 23.85 | 9.58 | (Intel i3, 32GB RAM) |
| **质量评分** | 0.831 | 0.897 | |
| **置信度** | 0.728 | 0.738 | |

> **结果说明**：平台成功处理了所有案例，展现了极高的鲁棒性与运行效率。

---

## 目录 / Table of Contents

### 用户文档
- [快速入门](docs/getting_started.md) - 快速开始指南
- [安装指南](docs/installation.md) - 环境与依赖设置说明
- [用户指南](docs/user_guide/) - 完整使用文档
- [教程](docs/tutorials/) - 逐步操作指南
- [示例](docs/examples/) - 实际使用示例
- [故障排除](docs/tutorials/troubleshooting.md) - 常见问题与解决方法

### 开发者文档
- [API 参考](docs/api_documentation/) - 接口技术参考
- [开发者指南](docs/developer_guide/) - 为贡献者提供
- [贡献指南](docs/developer_guide/contributing.md) - 参与开发与提交指南
- [版本说明](changelog.md) - 版本更新记录

---

## 快速链接 / Quick Links

### 起步
- [系统要求](docs/installation.md#系统要求) / [System Requirements](docs/installation.md#system-requirements)
- [基础使用](docs/tutorials/basic_usage.md) / [Basic Usage](docs/tutorials/basic_usage.md)

### 支持
- [常见问题（FAQ）](docs/tutorials/faq.md) / [FAQ](docs/tutorials/faq.md)
- [故障排除](docs/tutorials/troubleshooting.md) / [Troubleshooting](docs/tutorials/troubleshooting.md)

### 社区
- [代码仓库](https://github.com/hedmx/mri-autoqa) / [Repository](https://github.com/hedmx/mri-autoqa)
- [问题反馈](https://github.com/hedmx/mri-autoqa/issues) / [Issue Tracker](https://github.com/hedmx/mri-autoqa/issues)
- [讨论区](https://github.com/hedmx/mri-autoqa/discussions) / [Discussions](https://github.com/hedmx/mri-autoqa/discussions)

---

## English

# MRI AutoQA: Open-Source Automated MRI Quality Assessment Platform

**MRI AutoQA** is an end-to-end open-source platform for clinical and research use, dedicated to achieving fully automated and objective quality assessment of Magnetic Resonance Imaging (MRI).

## Abstract

**English**:
Magnetic Resonance Imaging (MRI) quality is the cornerstone of diagnostic accuracy and research data reliability. However, traditional manual quality assessment (QA) methods are inefficient, subjective, and difficult to scale. To address this, we developed **MRI AutoQA**, an end-to-end open-source platform designed to automate the entire analysis pipeline from raw DICOM data to standardized, multi-dimensional, and visual quality reports.

**中文**：
磁共振成像（MRI）的质量是保障诊断准确性和科研数据可靠性的基石。然而，传统的人工质量评估（QA）方法效率低下、主观性强且难以规模化。为此，我们开发并开源了 **MRI AutoQA**，旨在实现从原始 DICOM 数据到标准化、多维度、可视化质量报告的全流程自动化分析。

---

## Key Features

- **End-to-End Automation**: One-click processing for DICOM parsing, NIfTI conversion, metric computation, and report generation.
- **Multi-Dimensional Quantitative Metrics**:
    - **Signal-to-Noise Ratio (SNR)**: Unbiased estimation using V3 engine with Rayleigh distribution correction.
    - **Contrast-to-Noise Ratio (CNR)**: Automated calculation based on anatomical templates.
    - **Uniformity**: Quantification of noise spatial distribution.
    - **Artifact Detection**: Identification of motion, Gibbs, and other artifact severities.
- **Intelligent Context Awareness**: Dynamically adjusts analysis strategies based on anatomical region (brain, spine, etc.), sequence type (T1/T2/PD), and field strength (1.5T/3.0T).
- **Standardized Visualization**: Generates PNG quality reports with 12 subplots and JSON-structured data.
- **Research-Grade Robustness**: Noise region selection based on consensus mechanisms including Maximum Likelihood Estimation (MLE) and Interquartile Range (IQR).

---

## Performance Validation

We have deployed the platform at a local MRI imaging center and conducted validation through DICOM data conversion and batch automated quality analysis on a dataset of **10,607** MRI scans covering various anatomical regions and field strengths.

| Metric | Mean | Median | Processing Speed |
| :--- | :--- | :--- | :--- |
| **SNR** | 128.24 | 86.57 | **5.54 sec/case** |
| **CNR** | 23.85 | 9.58 | (Intel i3, 32GB RAM) |
| **Quality Score** | 0.831 | 0.897 | |
| **Confidence** | 0.728 | 0.738 | |

> **Interpretation**: The platform successfully processed all cases, demonstrating high robustness and efficiency.

---

## Table of Contents

### User Documentation
- [Getting Started](docs/getting_started.md) - Quick start guide
- [Installation](docs/installation.md) - Environment and dependency setup
- [User Guide](docs/user_guide/) - Complete usage documentation
- [Tutorials](docs/tutorials/) - Step-by-step guides
- [Examples](docs/examples/) - Practical examples
- [Troubleshooting](docs/tutorials/troubleshooting.md) - Common issues and solutions

### Developer Documentation
- [API Reference](docs/api_documentation/) - Technical interface reference
- [Developer Guide](docs/developer_guide/) - For contributors
- [Contributing Guide](docs/developer_guide/contributing.md) - Development participation and submission guidelines
- [Changelog](changelog.md) - Version history

---

## Quick Links

### Getting Started
- [System Requirements](docs/installation.md#system-requirements)
- [Basic Usage](docs/tutorials/basic_usage.md)

### Support
- [FAQ](docs/tutorials/faq.md)
- [Troubleshooting](docs/tutorials/troubleshooting.md)

### Community
- [Repository](https://github.com/hedmx/mri-autoqa)
- [Issue Tracker](https://github.com/hedmx/mri-autoqa/issues)
- [Discussions](https://github.com/hedmx/mri-autoqa/discussions)

---

## 许可证 / License

本项目采用 [MIT 许可证](LICENSE) 开源。  
This project is open-source under the [MIT License](LICENSE).
