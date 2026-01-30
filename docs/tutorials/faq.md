
### docs/faq.md
```markdown
# Frequently Asked Questions / 常见问题

## 中文 (Chinese):

### Q: MRI AutoQA 支持哪些序列类型？
A: 支持 T1、T2、PD、FLAIR 等常见 MRI 序列类型。

### Q: 需要多少扫描才能运行分析？
A: 只需要单个扫描即可运行 SNR 分析，无需重复扫描。

### Q: 系统支持哪些厂商的设备？
A: 支持 Siemens、GE、Philips 等主流 MRI 设备的并行成像检测。

### Q: 输出格式是什么？
A: 提供 JSON 报告和 CSV 摘要两种格式。

### Q: 如何自定义 ROI 位置？
A: 在 config.py 中修改 ROI_SETTINGS 配置。

### Q: 分析失败怎么办？
A: 检查 logs/ 目录中的日志文件，查看具体错误信息。

## English:

### Q: What sequence types does MRI AutoQA support?
A: Supports common MRI sequence types including T1, T2, PD, FLAIR.

### Q: How many scans are needed to run analysis?
A: Only a single scan is needed to run SNR analysis, no repeated scans required.

### Q: Which vendor equipment does the system support?
A: Supports parallel imaging detection for mainstream MRI equipment from Siemens, GE, Philips.

### Q: What are the output formats?
A: Provides both JSON reports and CSV summaries.

### Q: How do I customize ROI positions?
A: Modify the ROI_SETTINGS configuration in config.py.

### Q: What if analysis fails?
A: Check log files in the logs/ directory for specific error information.


