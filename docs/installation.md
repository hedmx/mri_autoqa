
### docs/installation.md

# Installation Guide / 安装指南

## 中文 (Chinese):

### 系统要求
#### 最低要求
- 操作系统: Windows 10+, macOS 10.14+, Linux
- CPU: Intel/AMD x86_64 处理器
- 内存: 8GB 最低，16GB 推荐
- 存储: 1GB 空闲空间 + 数据存储要求
- Python: 3.8 或更高版本

#### 推荐要求
- CPU: 多核处理器 (4+ 核心)
- 内存: 16GB 或更多
- 存储: 推荐 SSD 驱动器

### 安装方法

#### 方法 1: 从源码安装 (推荐)

# 克隆仓库
git clone https://github.com/hedmx/mri-autoqa.git
cd mri-autoqa

# 创建虚拟环境
python -m venv mri-autoqa-env
source mri-autoqa-env/bin/activate  # Linux/macOS
# 或
mri-autoqa-env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

#### 方法 2: 直接安装
pip install numpy nibabel matplotlib pandas scipy psutil pydicom scikit-image seaborn

### 验证

python -c "import nibabel; import matplotlib; print('Installation successful!')"

## English:

### System Requirements

#### Minimum Requirements

- Operating System: Windows 10+, macOS 10.14+, Linux
- CPU: Intel/AMD x86_64 processor
- RAM: 8GB minimum, 16GB recommended
- Storage: 1GB free space + data storage requirements
- Python: 3.8 or higher

#### Recommended Requirements

- CPU: Multi-core processor (4+ cores)
- RAM: 16GB or more
- Storage: SSD drive recommended

### Installation Methods

#### Method 1: From Source (Recommended)
# Clone the repository
git clone https://github.com/hedmx/mri-autoqa.git
cd mri-autoqa

# Create virtual environment
python -m venv mri-autoqa-env
source mri-autoqa-env/bin/activate  # Linux/macOS
# or
mri-autoqa-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

#### Method 2: Direct Installation

pip install numpy nibabel matplotlib pandas scipy psutil pydicom scikit-image seaborn

### Verification

python -c "import nibabel; import matplotlib; print('Installation successful!')"
