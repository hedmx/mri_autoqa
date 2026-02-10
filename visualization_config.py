
#!/usr/bin/env python3
"""
Visualization Configuration - Based on Engineering Document
12-subplot layout design with English labels
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
from typing import Dict, Any, Tuple, List
import colorsys

# ============================================================================
# 全局可视化配置
# ============================================================================

class VisualizationConfig:
    """可视化配置类 - 工程文件第12-14页"""
    
    # 1. 报告尺寸和DPI（工程文件第2页）
    FIGURE_SIZE = (18, 12)  # 英寸
    DPI = 150
    FIGURE_FORMAT = 'png'
    
    # 2. 布局配置（工程文件第12页）
    LAYOUT = {
        'nrows': 4,
        'ncols': 3,
        'gridspec_kw': {
            'wspace': 0.29,
            'hspace': 0.39,
            'width_ratios': [1, 1, 1],
            'height_ratios': [1.5, 1, 1, 0.8],
            'top': 0.95,      # 新增：减小顶部间距
            'bottom': 0.07    # 新增：减小底部间距
        }
    }
    
    # 3. 颜色配置（工程文件第14页）
    COLOR_SCHEME = {
        # 置信度等级颜色编码
        'confidence': {
            'HIGH': '#4CAF50',      # 绿色
            'MEDIUM': '#FFC107',    # 黄色
            'LOW': '#FF9800',       # 橙色
            'FAILED': '#F44336'     # 红色
        },
        # SNR评级颜色
        'snr_rating': {
            'EXCELLENT': '#2E7D32',   # 深绿
            'GOOD': '#7CB342',        # 浅绿
            'FAIR': '#FFB300',        # 琥珀色
            'POOR': '#FF7043',        # 深橙
            'UNACCEPTABLE': '#D32F2F' # 深红
        },
        # ROI颜色
        'roi': {
            'signal': '#00C853',     # 信号ROI - 绿色
            'background': '#FF5252', # 背景ROI - 红色
            'cnr_roi1': '#2979FF',   # CNR ROI 1 - 蓝色
            'cnr_roi2': '#FF4081'    # CNR ROI 2 - 粉色
        },
        # 图表颜色
        'charts': {
            'primary': '#2196F3',    # 主蓝色
            'secondary': '#4CAF50',  # 绿色
            'tertiary': '#FF9800',   # 橙色
            'gray': '#9E9E9E'        # 灰色
        }
    }
    
    # 4. 字体配置
    FONT_CONFIG = {
        'family': 'DejaVu Sans',
        'title_size': 10,
        'axis_label_size': 9,
        'tick_label_size': 8,
        'annotation_size': 8,
        'table_font_size': 8
    }
    
    # 5. 子图配置（工程文件第12-14页）
    SUBPLOT_CONFIGS = {
        # 子图1：原始图像 + ROI标注
        1: {
            'title': 'Original Image with ROI Annotation',
            'title_position': (0.5, 0.95),
            'roi_linewidth': 2,
            'roi_alpha': 0.7,
            'anatomy_fontsize': 9,
            'low_confidence_dash': [6, 2]  # 低置信度时ROI用虚线
        },
        
        # 子图2：SNR分析
        2: {
            'title': 'SNR Analysis',
            'bar_width': 0.6,
            'show_dB': True,
            'show_rating': True,
            'rating_fontsize': 9
        },
        
        # 子图3：CNR分析
        3: {
            'title': 'CNR Analysis',
            'show_all_pairs': True,
            'best_cnr_highlight': True,
            'clinical_significance_fontsize': 8
        },
        

        # 子图4：伪影检测
        4: {
            'title': 'Artifact Detection',
            'heatmap_cmap': 'hot',
            'artifact_types': ['Motion', 'Gibbs', 'Wrap-around', 'Chemical Shift'],
            'artifact_score_fontsize': 9
        },
        
        # 子图5：综合质量评分
        5: {
            'title': 'Comprehensive Quality Score',
            'radar_angles': np.linspace(0, 2*np.pi, 4, endpoint=False).tolist(),
            'radar_categories': ['ROI Placement', 'Noise Reliability', 'Image Quality', 'Algorithm Execution'],
            'radar_fill_alpha': 0.25
        },
        
        # 子图6：信号直方图
        6: {
            'title': 'Signal ROI Histogram',
            'bins': 50,
            'show_fit': True,
            'show_statistics': True,
            'norm_fit_color': '#FF5252',
            'norm_fit_alpha': 0.6
        },
        
        # 子图7：噪声直方图
        7: {
            'title': 'Background Noise Histogram',
            'bins': 30,
            'show_reference_lines': True,
            'reference_line_color': '#9C27B0',
            'reference_line_alpha': 0.5
        },
        
        # 子图8：置信度评估
        8: {
            'title': 'Confidence Assessment',
            'show_four_dimensions': True,
            'show_total_confidence': True,
            'dimension_labels': ['ROI Quality', 'Noise Reliability', 'Image Quality', 'Algorithm Quality'],
            'bar_width': 0.7
        },
        
        # 子图9：评分组件
        9: {
            'title': 'Score Components Breakdown',
            'stacked_bar': True,
            'show_weights': True,
            'component_labels': ['Intensity', 'Uniformity', 'Edge Distance', 'Anatomy']
        },
        
        # 子图10：采集信息
        10: {
            'title': 'Acquisition Information',
            'table_rows': 8,
            'table_columns': 2,
            'table_fontsize': 8,
            'table_cell_height': 0.08,
            'header_color': '#E3F2FD'
        },
        
        # 子图11：处理信息
        11: {
            'title': 'Processing Information',
            'show_status_icon': True,
            'status_icon_size': 12,
            'show_constraint_check': True,
            'constraint_symbols': {'PASS': '✓', 'FAIL': '✗', 'WARNING': '⚠'}
        }
    }
    
    # 6. 顶部状态栏配置（工程文件第14页）
    STATUS_BAR_CONFIG = {
        'height': 0.025,  # 减小高度（原来是 0.03）
        'background_color': '#F5F5F5',
        'border_color': '#BDBDBD',
        'border_width': 1,
        'fontsize': 9,
        'padding': 0.01,
        'sections': [
            {'label': 'MRI Quality Analysis Report', 'width': 0.25},
            {'label': 'Patient:', 'field': 'patient_id', 'width': 0.15},
            {'label': 'Scan:', 'field': 'scan_name', 'width': 0.15},
            {'label': 'Confidence:', 'field': 'confidence_level', 'width': 0.15},
            {'label': 'Quality Grade:', 'field': 'quality_grade', 'width': 0.15},
            {'label': 'Processing Status:', 'field': 'processing_status', 'width': 0.15}
        ]
    }
    
    # 7. 问题摘要配置
    ISSUE_SUMMARY_CONFIG = {
        'max_issues': 3,
        'issue_prefix': '• ',
        'suggestion_prefix': 'Recommendation: ',
        'fontsize': 8,
        'line_spacing': 1.2,
        'warning_color': '#FF9800',
        'critical_color': '#F44336'
    }
    
    @classmethod
    def get_subplot_position(cls, subplot_index: int) -> Tuple[int, int]:
        """获取子图在网格中的位置"""
        row = (subplot_index - 1) // cls.LAYOUT['ncols']
        col = (subplot_index - 1) % cls.LAYOUT['ncols']
        return row, col
    
    @classmethod
    def get_color_for_confidence(cls, confidence_score: float) -> str:
        """根据置信度分数获取颜色"""
        if confidence_score >= 0.80:
            return cls.COLOR_SCHEME['confidence']['HIGH']
        elif confidence_score >= 0.60:
            return cls.COLOR_SCHEME['confidence']['MEDIUM']
        elif confidence_score >= 0.40:
            return cls.COLOR_SCHEME['confidence']['LOW']
        else:
            return cls.COLOR_SCHEME['confidence']['FAILED']
    
    @classmethod
    def get_confidence_level(cls, confidence_score: float) -> str:
        """根据置信度分数获取等级"""
        if confidence_score >= 0.80:
            return 'HIGH'
        elif confidence_score >= 0.60:
            return 'MEDIUM'
        elif confidence_score >= 0.40:
            return 'LOW'
        else:
            return 'FAILED'
    
    @classmethod
    def get_color_for_snr_rating(cls, rating: str) -> str:
        """根据SNR评级获取颜色"""
        return cls.COLOR_SCHEME['snr_rating'].get(rating.upper(), '#9E9E9E')
    
    @classmethod
    def get_color_for_cnr_rating(cls, cnr_value: float) -> str:
        """根据CNR值获取颜色"""
        if cnr_value >= 5.0:
            return cls.COLOR_SCHEME['snr_rating']['EXCELLENT']
        elif cnr_value >= 3.0:
            return cls.COLOR_SCHEME['snr_rating']['GOOD']
        elif cnr_value >= 1.5:
            return cls.COLOR_SCHEME['snr_rating']['FAIR']
        elif cnr_value >= 0.5:
            return cls.COLOR_SCHEME['snr_rating']['POOR']
        else:
            return cls.COLOR_SCHEME['snr_rating']['UNACCEPTABLE']
    
    @classmethod
    def get_snr_rating(cls, snr_value: float) -> str:
        """根据SNR值获取评级"""
        if snr_value >= 30.0:
            return 'EXCELLENT'
        elif snr_value >= 20.0:
            return 'GOOD'
        elif snr_value >= 10.0:
            return 'FAIR'
        elif snr_value >= 5.0:
            return 'POOR'
        else:
            return 'UNACCEPTABLE'
    
    @classmethod
    def get_quality_grade(cls, total_score: float) -> str:
        """根据总分获取质量等级"""
        if total_score >= 0.90:
            return 'A'
        elif total_score >= 0.75:
            return 'B'
        elif total_score >= 0.60:
            return 'C'
        elif total_score >= 0.40:
            return 'D'
        else:
            return 'F'
    
    @classmethod
    def get_quality_grade_color(cls, grade: str) -> str:
        """根据质量等级获取颜色"""
        grade_colors = {
            'A': '#4CAF50',  # 绿色
            'B': '#8BC34A',  # 浅绿
            'C': '#FFC107',  # 黄色
            'D': '#FF9800',  # 橙色
            'F': '#F44336'   # 红色
        }
        return grade_colors.get(grade.upper(), '#9E9E9E')
    
    @classmethod
    def get_processing_status_color(cls, status: str) -> str:
        """根据处理状态获取颜色"""
        status_colors = {
            'COMPLETED': '#4CAF50',
            'IN_PROGRESS': '#2196F3',
            'FAILED': '#F44336',
            'SKIPPED': '#FF9800',
            'PENDING': '#9E9E9E'
        }
        return status_colors.get(status.upper(), '#9E9E9E')
    
    @classmethod
    def get_anatomy_color(cls, anatomy: str) -> str:
        """根据解剖区域获取颜色"""
        anatomy_colors = {
            'brain': '#2196F3',
            'lumbar': '#4CAF50',
            'cervical': '#9C27B0',
            'thoracic': '#FF9800',
            'abdomen': '#FF5722',
            'pelvis': '#795548'
        }
        return anatomy_colors.get(anatomy.lower(), '#9E9E9E')
    @classmethod
    def get_color_for_quality_score(cls, score: float) -> str:
        """根据质量分数获取颜色"""
        if score >= 0.90:
            return cls.COLOR_SCHEME['snr_rating']['EXCELLENT']
        elif score >= 0.75:
            return cls.COLOR_SCHEME['snr_rating']['GOOD']
        elif score >= 0.60:
            return cls.COLOR_SCHEME['snr_rating']['FAIR']
        elif score >= 0.40:
            return cls.COLOR_SCHEME['snr_rating']['POOR']
        else:
            return cls.COLOR_SCHEME['snr_rating']['UNACCEPTABLE']
    
    @classmethod
    def get_quality_level(cls, score: float) -> str:
        """根据质量分数获取等级"""
        if score >= 0.90:
            return 'EXCELLENT'
        elif score >= 0.75:
            return 'GOOD'
        elif score >= 0.60:
            return 'FAIR'
        elif score >= 0.40:
            return 'POOR'
        else:
            return 'UNACCEPTABLE'

# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("测试 VisualizationConfig 类...")
    
    # 测试置信度等级
    test_scores = [0.90, 0.70, 0.50, 0.30]
    for score in test_scores:
        level = VisualizationConfig.get_confidence_level(score)
        color = VisualizationConfig.get_color_for_confidence(score)
        print(f"置信度 {score:.2f}: {level} ({color})")
    
    # 测试SNR评级
    test_snrs = [35.0, 25.0, 15.0, 8.0, 3.0]
    for snr in test_snrs:
        rating = VisualizationConfig.get_snr_rating(snr)
        color = VisualizationConfig.get_color_for_snr_rating(rating)
        print(f"SNR {snr:.1f}: {rating} ({color})")
    
    # 测试质量等级
    test_scores = [0.95, 0.82, 0.65, 0.45, 0.25]
    for score in test_scores:
        grade = VisualizationConfig.get_quality_grade(score)
        color = VisualizationConfig.get_quality_grade_color(grade)
        print(f"总分 {score:.2f}: {grade} ({color})")
    
    print("✅ 所有测试通过！")
