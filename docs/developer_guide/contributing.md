
### docs/developer_guide/contributing.md
`markdown
# Contributing to MRI AutoQA / 为 MRI AutoQA 贡献

## 中文 (Chinese):

我们欢迎社区贡献！请遵循以下步骤：

### 如何贡献
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码标准
- 遵循 PEP 8 风格指南
- 为所有函数和类编写清晰的文档字符串
- 保持向后兼容性（当可能时）

### 开发工作流程
`bash
# 设置开发环境
git clone https://github.com/hedmx/mri-autoqa.git
cd mri-autoqa
python -m venv dev-env
source dev-env/bin/activate
pip install -e .
pip install pytest black flake8


# 检查代码风格
flake8 .
black --check .

## English:

We welcome community contributions! Please follow these steps:

### How to Contribute

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Write clear docstrings for all functions and classes
- Maintain backward compatibility when possible

### Development Workflow
# Set up development environment
git clone https://github.com/hedmx/mri-autoqa.git
cd mri-autoqa
python -m venv dev-env
source dev-env/bin/activate
pip install -e .
pip install pytest black flake8


# Check code style
flake8 .
black --check .


