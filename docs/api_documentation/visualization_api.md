
### docs/api_documentation/visualization_api.md

# Visualization API / 可视化API

## 中文 (Chinese):

### VisualizationConfig 类

class VisualizationConfig:
    # 图形尺寸和DPI
    FIGURE_SIZE = (18, 12)  # 英寸
    DPI = 150
    
    # 布局配置
    LAYOUT = {
        'nrows': 4,
        'ncols': 3,
        'gridspec_kw': {
            'wspace': 0.29,
            'hspace': 0.39,
            'width_ratios': [1, 1, 1],
            'height_ratios': [1, 1, 1, 0.8],
            'top': 0.95,
            'bottom': 0.07
        }
    }
    
    # 颜色配置
    COLOR_SCHEME = {
        'confidence': {
            'HIGH': '#4CAF50',      # 绿色
            'MEDIUM': '#FFC107',    # 黄色
            'LOW': '#FF9800',       # 橙色
            'FAILED': '#F44336'     # 红色
        },
        'snr_rating': {
            'EXCELLENT': '#2E7D32',   # 深绿
            'GOOD': '#7CB342',        # 浅绿
            'FAIR': '#FFB300',        # 琥珀色
            'POOR': '#FF7043',        # 深橙
            'UNACCEPTABLE': '#D32F2F' # 深红
        }
    }
    
    # 字体配置
    FONT_CONFIG = {
        'family': 'DejaVu Sans',
        'title_size': 10,
        'axis_label_size': 9,
        'tick_label_size': 8,
        'annotation_size': 8,
        'table_font_size': 8
    }


### 可视化函数

def create_visualization_for_scan(result: Dict[str, Any], image: np.ndarray, output_dir: str, filename: str) -> bool

def visualize_batch_results(csv_path: str, output_dir: str) -> Dict[str, bool]
## English:

### VisualizationConfig Class
class VisualizationConfig:

# Figure size and DPI

FIGURE_SIZE = (18, 12) # inches

DPI = 150

# Layout configuration

LAYOUT = {

'nrows': 4,

'ncols': 3,

'gridspec_kw': {

'wspace': 0.29,

'hspace': 0.39,

'width_ratios': [1, 1, 1],

'height_ratios': [1, 1, 1, 0.8],

'top': 0.95,

'bottom': 0.07

}

}

# Color configuration

COLOR_SCHEME = {

'confidence': {

'HIGH': '#4CAF50', # Green

'MEDIUM': '#FFC107', # Yellow

'LOW': '#FF9800', # Orange

'FAILED': '#F44336' # Red

},

'snr_rating': {

'EXCELLENT': '#2E7D32', # Dark green

'GOOD': '#7CB342', # Light green

'FAIR': '#FFB300', # Amber

'POOR': '#FF7043', # Deep orange

'UNACCEPTABLE': '#D32F2F' # Deep red

}

}

# Font configuration

FONT_CONFIG = {

'family': 'DejaVu Sans',

'title_size': 10,

'axis_label_size': 9,

'tick_label_size': 8,

'annotation_size': 8,

'table_font_size': 8

}

### Visualization Functions

def create_visualization_for_scan(result: Dict[str, Any], image: np.ndarray, output_dir: str, filename: str) -> bool

def visualize_batch_results(csv_path: str, output_dir: str) -> Dict[str, bool]