
### docs/tutorials/troubleshooting.md
``markdown
# Troubleshooting Guide / 故障排除指南

## 中文 (Chinese):

### 常见问题

#### 安装问题
**问题**: 安装后出现导入错误
**解决方案**: 确保所有依赖项都已安装
``bash
pip install -r requirements.txt


#### DICOM转换问题

**问题**: "未找到DICOM文件"  
**解决方案**: 验证目录结构和文件权限

- 检查: `data/patient_id/scan_name/*.dcm`
- 确保文件有 .dcm 扩展名

#### 内存错误

**问题**: 分析过程中出现内存不足错误  
**解决方案**:

- 减少批次大小
- 关闭其他应用程序
- 如果可能，增加虚拟内存

#### 分析失败

**问题**: 某些扫描分析失败  
**解决方案**: 检查 `logs/` 目录中的日志

- 查找特定错误代码
- 验证DICOM文件完整性
- 检查硬件要求

### 性能优化

- 使用SSD存储以获得更快的I/O
- 增加可用RAM
- 为大数据集处理较小的批次
- 在处理过程中监控系统资源

### 支持资源

- GitHub Issues: 错误报告和功能请求
- 电子邮件: [mailto:3532370@qq.com]
- 社区论坛: 即将推出

## English:

### Common Issues

#### Installation Problems

**Issue**: Import errors after installation  
**Solution**: Ensure all dependencies are installed
``bash
pip install -r requirements.txt

#### DICOM Conversion Issues

**Issue**: "No DICOM files found"  
**Solution**: Verify directory structure and file permissions

- Check: `data/patient_id/scan_name/*.dcm`
- Ensure files have .dcm extension

#### Memory Errors

**Issue**: OutOfMemoryError during analysis  
**Solution**:

- Reduce batch size
- Close other applications
- Increase virtual memory if possible

#### Analysis Failures

**Issue**: Some scans fail analysis  
**Solution**: Check logs in `logs/` directory

- Look for specific error codes
- Verify DICOM file integrity
- Check hardware requirements

### Performance Optimization

- Use SSD storage for faster I/O
- Increase available RAM
- Process smaller batches for large datasets
- Monitor system resources during processing

### Support Resources

- GitHub Issues: Bug reports and feature requests
- Email: [mailto:3532370@qq.com]
- Community Forum: Coming soon