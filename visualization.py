#!/usr/bin/env python3
"""
Main Visualization Generator - 12-subplot Quality Report
English version, following engineering document specifications
修复版：修复雷达图维度不匹配问题
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from typing import Dict, Any, Tuple, List, Optional
import json
from datetime import datetime
import warnings

from visualization_config import VisualizationConfig

plt.rcParams.update({
    'font.family': VisualizationConfig.FONT_CONFIG['family'],
    'axes.titlesize': VisualizationConfig.FONT_CONFIG['title_size'],
    'axes.labelsize': VisualizationConfig.FONT_CONFIG['axis_label_size'],
    'xtick.labelsize': VisualizationConfig.FONT_CONFIG['tick_label_size'],
    'ytick.labelsize': VisualizationConfig.FONT_CONFIG['tick_label_size'],
    'legend.fontsize': VisualizationConfig.FONT_CONFIG['annotation_size']
})


class MRIQualityVisualizer:
    """
    MRI Quality Report Visualizer
    Generates 12-subplot report as per engineering document
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.fig = None
        self.axes = None
        self.current_result = None
        self.current_image = None
        
    def create_visualization(self, 
                           analysis_result: Dict[str, Any], 
                           image_data: np.ndarray,
                           output_path: str) -> bool:
        """
        Create complete 12-subplot visualization
        
        Args:
            analysis_result: Complete analysis results
            image_data: 2D image array
            output_path: Output PNG file path
            
        Returns:
            Success status
        """
        try:
            self.current_result = analysis_result
            self.current_image = image_data
            
            # 1. Create figure with gridspec
            self._create_figure_layout()
            
            # 2. Add status bar
            self._add_status_bar()
            
            # 3. Create all 12 subplots
            self._create_subplot_1()  # Original Image + ROI
            self._create_subplot_2()  # SNR Analysis
            self._create_subplot_3()  # CNR Analysis
            self._create_subplot_4()  # Noise Uniformity
            self._create_subplot_5()  # Artifact Detection
            self._create_subplot_6()  # Comprehensive Quality Score
            self._create_subplot_7()  # Signal Histogram
            self._create_subplot_8()  # Noise Histogram
            self._create_subplot_9()  # Confidence Assessment
            self._create_subplot_10() # Score Components
            self._create_subplot_11() # Acquisition Info
            self._create_subplot_12() # Processing Info
            
            # 4. Add issue summary if confidence is low
            confidence_score = analysis_result.get('quality_assessment', {}).get(
                'overall_confidence', {}).get('score', 1.0)
            if confidence_score < 0.7:
                self._add_issue_summary(confidence_score)
            
            # 5. Adjust layout and save
            #plt.tight_layout(rect=[0, 0.03, 1, 0.99])  # 上移图形
            plt.subplots_adjust(
                top=0.94,     # 减小顶部边距（原来可能更大）
                bottom=0.07,  # 稍微减小底部边距
                left=0.06,
                right=0.97,
                wspace=0.28,
                hspace=0.38
            )
            plt.savefig(output_path, 
                       dpi=VisualizationConfig.DPI,
                       bbox_inches='tight',
                       facecolor='white')
            
            if self.verbose:
                print(f"Visualization saved to: {output_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            print(f"Visualization creation failed: {e}")
            import traceback
            traceback.print_exc()
            if self.fig:
                plt.close()
            return False
    
    def _create_figure_layout(self):
        """创建图形和子图布局"""
        self.fig = plt.figure(figsize=VisualizationConfig.FIGURE_SIZE)
        
        gs = gridspec.GridSpec(
            VisualizationConfig.LAYOUT['nrows'],
            VisualizationConfig.LAYOUT['ncols'],
            **VisualizationConfig.LAYOUT['gridspec_kw']
        )
        
        self.axes = []
        for i in range(1, 13):
            row, col = VisualizationConfig.get_subplot_position(i)
            ax = self.fig.add_subplot(gs[row, col])
            self.axes.append(ax)
    
    def _add_status_bar(self):
        """添加顶部状态栏（工程文件第14页）"""
        result = self.current_result
        config = VisualizationConfig.STATUS_BAR_CONFIG
        
        # 创建状态栏背景
        status_bar = plt.axes([0, 0.97, 1, config['height']], facecolor=config['background_color'])
        status_bar.set_xlim(0, 1)
        status_bar.set_ylim(0, 1)
        
        # 隐藏边框和坐标轴
        for spine in status_bar.spines.values():
            spine.set_edgecolor(config['border_color'])
            spine.set_linewidth(config['border_width'])
        
        status_bar.set_xticks([])
        status_bar.set_yticks([])
        
        # 添加状态信息
        x_pos = 0.01
        for section in config['sections']:
            label = section['label']
            
            if 'field' in section:
                field_name = section['field']
                if field_name == 'confidence_level':
                    confidence_score = result.get('quality_assessment', {}).get(
                        'overall_confidence', {}).get('score', 0)
                    value = VisualizationConfig.get_confidence_level(confidence_score)
                    color = VisualizationConfig.get_color_for_confidence(confidence_score)
                elif field_name == 'quality_grade':
                    snr_rating = result.get('quality_assessment', {}).get(
                        'snr_rating', {}).get('level', 'UNKNOWN')
                    value = snr_rating
                    color = VisualizationConfig.get_color_for_snr_rating(snr_rating)
                elif field_name == 'processing_status':
                    value = result.get('analysis_status', 'UNKNOWN')
                    color = VisualizationConfig.get_processing_status_color(value)
                else:
                    value = result.get('acquisition', {}).get(field_name, 'N/A')
                    color = 'black'
                
                text = f"{label} {value}"
            else:
                text = label
                color = 'black'
            
            status_bar.text(x_pos, 0.5, text, 
                          fontsize=config['fontsize'],
                          va='center',
                          color=color)
            
            x_pos += section['width']
    
    def _add_issue_summary(self, confidence_score: float):
        """添加问题摘要（当置信度<0.7时）"""
        result = self.current_result
        config = VisualizationConfig.ISSUE_SUMMARY_CONFIG
        
        # 创建问题摘要区域
        issue_ax = plt.axes([0.7, 0.02, 0.3, 0.05], facecolor='#FFF3E0')
        issue_ax.set_xlim(0, 1)
        issue_ax.set_ylim(0, 1)
        issue_ax.set_xticks([])
        issue_ax.set_yticks([])
        
        # 添加边框
        for spine in issue_ax.spines.values():
            spine.set_edgecolor('#FFB74D')
            spine.set_linewidth(1)
        
        # 标题
        title_color = config['warning_color'] if confidence_score >= 0.4 else config['critical_color']
        issue_ax.text(0.05, 0.7, 'Primary Issues (Confidence < 0.7):',
                     fontsize=config['fontsize'],
                     color=title_color,
                     weight='bold')
        
        # 提取问题（简化示例）
        issues = []
        
        # 检查ROI信号强度
        roi_stats = result.get('roi_info', {}).get('signal', {}).get('statistics', {})
        roi_mean = roi_stats.get('mean', 0)
        if roi_mean < 50:  # 示例阈值
            issues.append("Low ROI signal intensity")
        
        # 检查背景区域大小
        bg_stats = result.get('roi_info', {}).get('background', {}).get('statistics', {})
        bg_pixels = bg_stats.get('pixel_count', 0)
        if bg_pixels < 500:
            issues.append("Small background region")
        
        # 检查噪声均匀性
        uniformity = result.get('quality_assessment', {}).get('noise_uniformity', {})
        cv = uniformity.get('cv', 0)
        if cv > 0.5:
            issues.append("Non-uniform noise distribution")
        
        # 限制问题数量
        issues = issues[:config['max_issues']]
        
        # 显示问题
        y_pos = 0.45
        for issue in issues:
            issue_ax.text(0.05, y_pos, f"{config['issue_prefix']}{issue}",
                         fontsize=config['fontsize'],
                         color='black')
            y_pos -= 0.2
        
        # 添加建议
        if issues:
            issue_ax.text(0.01, -0.2, 
                         f"{config['suggestion_prefix']}Check ROI placement and ensure sufficient tissue inclusion",
                         fontsize=config['fontsize'] - 1,
                         color='#388E3C',
                         style='italic')
    def _get_quality_level(self, score: float) -> str:
        """获取质量等级（与置信度分开）"""
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
    
    def _get_color_for_quality_score(self, score: float) -> str:
        """获取质量分数的颜色"""
        if score >= 0.90:
            return '#2E7D32'  # 深绿 - EXCELLENT
        elif score >= 0.75:
            return '#7CB342'  # 绿 - GOOD
        elif score >= 0.60:
            return '#FFB300'  # 琥珀色 - FAIR
        elif score >= 0.40:
            return '#FF7043'  # 深橙 - POOR
        else:
            return '#D32F2F'   # 深红 - UNACCEPTABLE
    # ============================================================================
    # 子图1：原始图像 + ROI标注（工程文件第12页）
    # ============================================================================
    def _create_subplot_1(self):
        """Subplot 1: Original Image with ROI Annotation"""
        ax = self.axes[0]
        config = VisualizationConfig.SUBPLOT_CONFIGS[1]
        
        # 显示原始图像
        if self.current_image is not None:
            ax.imshow(self.current_image, cmap='gray', aspect='auto')
        
        # 获取ROI信息
        result = self.current_result
        signal_roi = result.get('roi_info', {}).get('signal', {})
        background_roi = result.get('roi_info', {}).get('background', {})
        
        # 绘制信号ROI（绿色椭圆）
        if 'coordinates' in signal_roi:
            coords = signal_roi['coordinates']
            y1, y2, x1, x2 = coords
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # 检查置信度
            confidence = result.get('quality_assessment', {}).get('overall_confidence', {}).get('score', 1.0)
            line_style = '--' if confidence < 0.7 else '-'
            
            ellipse = Ellipse((center_x, center_y), width, height,
                            edgecolor=VisualizationConfig.COLOR_SCHEME['roi']['signal'],
                            facecolor='none',
                            linewidth=config['roi_linewidth'],
                            linestyle=line_style,
                            alpha=config['roi_alpha'])
            ax.add_patch(ellipse)
            
            # 添加标签
            ax.text(center_x, center_y - height/2 - 5, 'Signal ROI',
                   color=VisualizationConfig.COLOR_SCHEME['roi']['signal'],
                   fontsize=config['anatomy_fontsize'],
                   ha='center',
                   weight='bold')
        
        # 绘制背景ROI（红色矩形）
        if 'coordinates' in background_roi:
            coords = background_roi['coordinates']
            y1, y2, x1, x2 = coords
            
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                           edgecolor=VisualizationConfig.COLOR_SCHEME['roi']['background'],
                           facecolor='none',
                           linewidth=config['roi_linewidth'],
                           alpha=config['roi_alpha'])
            ax.add_patch(rect)
            
            # 添加标签
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ax.text(center_x, center_y - (y2-y1)/2 - 5, 'Background ROI',
                   color=VisualizationConfig.COLOR_SCHEME['roi']['background'],
                   fontsize=config['anatomy_fontsize'],
                   ha='center',
                   weight='bold')
        
        # 添加解剖部位信息
        anatomy = result.get('acquisition', {}).get('anatomical_region', 'Unknown')
        ax.text(0.01, -0.06, f'Anatomy: {anatomy.upper()}',
               transform=ax.transAxes,
               fontsize=config['anatomy_fontsize'],
               color='white',
               weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        
        # 添加图像尺寸
        if self.current_image is not None:
            h, w = self.current_image.shape
            ax.text(0.02, 0.02, f'Size: {w} × {h} px',
                   transform=ax.transAxes,
                   fontsize=config['anatomy_fontsize'] - 1,
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        
        ax.set_title(config['title'])
        ax.axis('off')
    
    # ============================================================================
    # 子图2：SNR分析（工程文件第12页）
    # ============================================================================
    def _create_subplot_2(self):
        """Subplot 2: SNR Analysis"""
        ax = self.axes[1]
        config = VisualizationConfig.SUBPLOT_CONFIGS[2]
        
        result = self.current_result
        snr_results = result.get('snr_results', {})
        
        # 获取SNR值
        snr_raw = snr_results.get('traditional', {}).get('snr', 0)
        snr_corrected = snr_results.get('recommended', {}).get('snr', 0)
        #snr_db = 20 * np.log10(snr_corrected) if snr_corrected > 0 else -np.inf
        # 【优先使用rayleigh_correction中的数据】
        rayleigh_correction = snr_results.get('rayleigh_correction', {})
        if rayleigh_correction:
            snr_raw = rayleigh_correction.get('snr_raw', snr_raw)
            snr_corrected = rayleigh_correction.get('snr_corrected', snr_corrected)
            improvement_percent = rayleigh_correction.get('improvement_percent', 0)
            correction_factor = rayleigh_correction.get('correction_factor', 1.0)
            correction_applied = True
        else:
            improvement_percent = ((snr_corrected - snr_raw) / snr_raw * 100) if snr_raw > 0 else 0
            correction_factor = 1.0
            correction_applied = False
        # 获取评级
        snr_rating = result.get('quality_assessment', {}).get('snr_rating', {}).get('level', 'UNKNOWN')
        
        # 创建条形图
        categories = ['Raw SNR', 'Corrected SNR']
        values = [snr_raw, snr_corrected]
        colors = [VisualizationConfig.COLOR_SCHEME['charts']['gray'],
                 VisualizationConfig.get_color_for_snr_rating(snr_rating)]
        
        bars = ax.bar(categories, values, width=config['bar_width'], color=colors)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{val:.1f}',
                   ha='center', va='bottom',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'])
            # 在原始SNR柱上标注方法
            if i == 0:
                method_note = "Intelligent Background"
                ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                       method_note, rotation=90,
                       ha='center', va='center',
                       fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'] - 2,
                       color='black', alpha=0.7)
        # 添加dB值（如果显示）
        snr_db = 20 * np.log10(snr_corrected) if snr_corrected > 0 else -np.inf
        if config['show_dB'] and not np.isinf(snr_db):
            text_offset = height * 0.07
            ax.text(1, snr_corrected + text_offset,
                   f'({snr_db:.1f} dB)',
                   ha='center', va='bottom',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'] - 1,
                   color='gray')
        
        # 添加评级（如果显示）
        if config['show_rating']:
            rating_text = f'Rating: {snr_rating}'
            if correction_applied:
                rating_text += f' (Rayleigh Corrected)'
            ax.text(0.5, -0.15, f'Rating: {snr_rating}',
                   transform=ax.transAxes,
                   ha='center',
                   fontsize=config['rating_fontsize'],
                   color=colors[1],
                   weight='bold')
        # 【新增】添加校正信息
        if correction_applied and improvement_percent > 0:
            info_text = f'Correction factor: {correction_factor:.2f}\nImprovement: +{improvement_percent:.1f}%'
            ax.text(0.5, -0.25, info_text,
                   transform=ax.transAxes,
                   ha='center',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'] - 1,
                   color='gray',
                   style='italic')
        ax.set_ylabel('SNR Value')
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(values) * 1.2 if values else 50)
    
    # ============================================================================
    # 子图3：CNR分析（工程文件第12页）
    # ============================================================================
    def _create_subplot_3(self):
        """Subplot 3: CNR Analysis"""
        ax = self.axes[2]
        config = VisualizationConfig.SUBPLOT_CONFIGS[3]
        
        result = self.current_result
        cnr_analysis = result.get('quality_assessment', {}).get('cnr_analysis', {})
        tissue_pairs = cnr_analysis.get('tissue_pairs', [])
        best_cnr = cnr_analysis.get('best_cnr', {})
        
        if not tissue_pairs:
            # 如果没有CNR数据，显示占位符
            ax.text(0.5, 0.5, 'CNR Analysis\nNot Available',
                   ha='center', va='center',
                   fontsize=10, color='gray')
            ax.set_title(config['title'])
            ax.axis('off')
            return
        
        # 创建条形图显示所有组织对的CNR
        pair_names = []
        display_names_list = []  # 显示名称（修复变量名）
        cnr_values = []
        colors = []
        
        for pair in tissue_pairs:
            pair_names.append(pair.get('description', pair.get('name', 'Unknown')))
            raw_description = pair.get('description', pair.get('name', 'Unknown'))
           
            # 【新增】格式化显示名称 - 分三行
            if ' vs ' in raw_description:
                parts = raw_description.split(' vs ')
                if len(parts) == 2:
                    # 格式：第一行
                    line1 = parts[0].strip()
                    # 第二行：'vs'
                    line2 = 'vs'
                    # 第三行
                    line3 = parts[1].strip()
                    display_name = f'{line1}\n{line2}\n{line3}'
                else:
                    display_name = raw_description
            else:
                display_name = raw_description
        
            display_names_list.append(display_name)  # 使用正确的变量名
            cnr_value = pair.get('cnr_value', 0)
            cnr_values.append(cnr_value)
            
            # 如果是最佳CNR，使用特殊颜色
            if best_cnr and pair.get('name') == best_cnr.get('name'):
                colors.append(VisualizationConfig.COLOR_SCHEME['charts']['primary'])
            else:
                colors.append(VisualizationConfig.COLOR_SCHEME['charts']['gray'])
        
        bars = ax.barh(display_names_list, cnr_values, color=colors, height=0.5)
        
        # 添加数值标签
        for bar, value in zip(bars, cnr_values):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}',
                   ha='left', va='center',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'])
        
        # 突出显示最佳CNR
        if config['best_cnr_highlight'] and best_cnr:
            best_index = next((i for i, pair in enumerate(tissue_pairs) 
                             if pair.get('name') == best_cnr.get('name')), -1)
            if best_index >= 0:
                bars[best_index].set_edgecolor('black')
                bars[best_index].set_linewidth(2)
                
                # 添加临床意义说明
                if 'clinical_significance' in best_cnr:
                    ax.text(0.02, 0.005, f"Clinical: {best_cnr['clinical_significance']}",
                           transform=ax.transAxes,
                           fontsize=config['clinical_significance_fontsize'],
                           style='italic')
        
        ax.set_xlabel('CNR Value')
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, max(cnr_values) * 1.3 if cnr_values else 10)
    
    # ============================================================================
    # 子图4：噪声均匀性（工程文件第12页）
    # ============================================================================
    def _create_subplot_4(self):
        """Subplot 4: Noise Uniformity Analysis"""
        ax = self.axes[3]
        config = VisualizationConfig.SUBPLOT_CONFIGS[4]
        
        result = self.current_result
        noise_uniformity = result.get('quality_assessment', {}).get('noise_uniformity', {})
        cv = noise_uniformity.get('cv', 0)
        uniformity_rating = noise_uniformity.get('uniformity_rating', 'UNKNOWN')
        
        # 创建箱线图
        bg_stats = result.get('roi_info', {}).get('background', {}).get('statistics', {})
        bg_mean = bg_stats.get('mean', 0)
        bg_std = bg_stats.get('std', 0)
        
        # 模拟多个背景区域的噪声分布
        np.random.seed(42)
        noise_samples = []
        for i in range(4):  # 4个角落区域
            region_noise = np.random.normal(bg_mean, bg_std * (1 + i*0.1), 100)
            noise_samples.append(region_noise)
        
        # 创建箱线图
        boxplot = ax.boxplot(noise_samples, 
                            patch_artist=True,
                            showfliers=config['boxplot_showfliers'])
        
        # 设置箱线图颜色
        for patch in boxplot['boxes']:
            patch.set_facecolor(VisualizationConfig.COLOR_SCHEME['charts']['secondary'])
            patch.set_alpha(0.7)
        
        # 添加CV值和评级
        if config['show_cv']:
            ax.text(0.02, 0.98, f'CV = {cv:.3f}',
                   transform=ax.transAxes,
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'],
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        if config['show_rating']:
            ax.text(0.98, 0.98, uniformity_rating,
                   transform=ax.transAxes,
                   ha='right',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'],
                   color=VisualizationConfig.get_color_for_cnr_rating(cv*10),  # 缩放CV用于颜色映射
                   weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('Background Region')
        ax.set_ylabel('Noise Value')
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(['TL', 'TR', 'BL', 'BR'])  # 左上、右上、左下、右下
    
    # ============================================================================
    # 子图5：伪影检测（工程文件第12-13页）- 修复版
    # ============================================================================
    def _create_subplot_5(self):
        """Subplot 5: Artifact Detection - FIXED VERSION"""
        ax = self.axes[4]
        config = VisualizationConfig.SUBPLOT_CONFIGS[5]
        
        # 清除当前轴
        ax.clear()
        
        result = self.current_result
        quality_assessment = result.get('quality_assessment', {})
        
        # 获取伪影得分（示例）
        artifact_types = ['Motion', 'Gibbs', 'Wrap-around', 'Chemical Shift']
        
        artifact_scores = {
            'Motion': quality_assessment.get('artifact_motion', 0.8),
            'Gibbs': quality_assessment.get('artifact_gibbs', 0.9),
            'Wrap-around': quality_assessment.get('artifact_wrap', 0.95),
            'Chemical Shift': quality_assessment.get('artifact_chemical', 0.85)
        }
        
        # 创建条形图代替雷达图，避免维度问题
        categories = list(artifact_scores.keys())
        values = list(artifact_scores.values())
        
        bars = ax.bar(categories, values, 
                     color=VisualizationConfig.COLOR_SCHEME['charts']['tertiary'],
                     alpha=0.7)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}',
                   ha='center', va='bottom',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'])
        
        # 添加平均得分
        avg_score = np.mean(values)
        ax.axhline(y=avg_score, color='red', linestyle='--', alpha=0.5, 
                  label=f'Average: {avg_score:.2f}')
        
        ax.set_ylabel('Artifact Score')
        ax.set_title(config['title'])
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'] - 1)
        ax.grid(True, alpha=0.3)
    
    # ============================================================================
    # 子图6：综合质量评分（工程文件第13页）- 修复版
    # ============================================================================
    def _create_subplot_6(self):
        """Subplot 6: Comprehensive Quality Score - FIXED VERSION"""
        ax = self.axes[5]
        config = VisualizationConfig.SUBPLOT_CONFIGS[6]
        
        # 清除当前轴
        ax.clear()
        
        result = self.current_result
        quality_scores = result.get('quality_assessment', {}).get('quality_scores', {})
        # 检查质量评分数据是否存在
        if not quality_scores or 'dimensions' not in quality_scores:
            # 如果没有质量评分数据，显示错误信息
            ax.text(0.5, 0.5, 'Quality Scores\nNot Available',
                   ha='center', va='center',
                   fontsize=10, color='gray')
            ax.set_title(config['title'])
            ax.axis('off')
            return

        # 从质量评分数据中提取维度分数
        dim_data = quality_scores['dimensions']
        # 定义要显示的维度（使用你定义的5个维度）
        dimensions = ['SNR Quality', 'CNR Quality', 'Noise Quality', 'Artifact Free', 'Image Integrity']
        dimension_keys = ['snr_quality', 'cnr_quality', 'noise_quality', 'artifact_free', 'image_integrity']
    
        scores = []
        for key in dimension_keys:
            score_data = dim_data.get(key, {})
            score = score_data.get('score', 0.7)  # 默认值0.7
            scores.append(score)
    
        # 创建条形图
        x_pos = np.arange(len(dimensions))
    
        # 根据分数选择颜色（可以使用现有的颜色映射）
        colors = [VisualizationConfig.get_color_for_quality_score(score) for score in scores]
    
        bars = ax.bar(x_pos, scores, color=colors, alpha=0.7)
    
        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.2f}',
                   ha='center', va='bottom',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'])
    
        # 设置x轴标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dimensions, rotation=15, ha='right')
    
        # 添加总分和评级
        total_score = quality_scores.get('total_score', np.mean(scores))
        quality_level = self._get_quality_level(total_score)
    
        # 在图表下方显示总分
        ax.text(0.5, -0.2, f'Quality Score: {total_score:.2f} ({quality_level})',
               transform=ax.transAxes,
               ha='center',
               fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'],
               color=VisualizationConfig.get_color_for_quality_score(total_score),
               weight='bold')
    
        ax.set_ylabel('Quality Score')
        ax.set_title(config['title'])
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # 子图7：信号直方图（工程文件第13页）
    # ============================================================================
    def _create_subplot_7(self):
        """Subplot 7: Signal ROI Histogram"""
        ax = self.axes[6]
        config = VisualizationConfig.SUBPLOT_CONFIGS[7]
        
        result = self.current_result
        signal_stats = result.get('roi_info', {}).get('signal', {}).get('statistics', {})
        
        mean_val = signal_stats.get('mean', 100)
        std_val = signal_stats.get('std', 15)
        pixel_count = signal_stats.get('pixel_count', 1000)
        
        # 生成直方图数据
        np.random.seed(42)
        if pixel_count > 0:
            signal_data = np.random.normal(mean_val, std_val, min(pixel_count, 10000))
        else:
            signal_data = np.random.normal(100, 20, 1000)
        
        # 绘制直方图
        n, bins, patches = ax.hist(signal_data, bins=config['bins'],
                                  color=VisualizationConfig.COLOR_SCHEME['charts']['primary'],
                                  alpha=0.7,
                                  density=True)
        
        # 添加正态分布拟合曲线
        if config['show_fit'] and std_val > 0:
            from scipy.stats import norm
            x = np.linspace(min(signal_data), max(signal_data), 100)
            y = norm.pdf(x, mean_val, std_val)
            ax.plot(x, y, color=config['norm_fit_color'],
                   linewidth=2, alpha=config['norm_fit_alpha'])
        
        # 添加统计信息
        if config['show_statistics']:
            info_text = (f'Mean: {mean_val:.1f}\n'
                        f'Std: {std_val:.1f}\n'
                        f'N: {pixel_count}')
            
            ax.text(0.98, 0.98, info_text,
                   transform=ax.transAxes,
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'],
                   va='top', ha='right',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('Signal Intensity')
        ax.set_ylabel('Probability Density')
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3)
    
    # ============================================================================
    # 子图8：噪声直方图（工程文件第13页）
    # ============================================================================
    def _create_subplot_8(self):
        """Subplot 8: Background Noise Histogram"""
        ax = self.axes[7]
        config = VisualizationConfig.SUBPLOT_CONFIGS[8]
        
        result = self.current_result
        noise_stats = result.get('roi_info', {}).get('background', {}).get('statistics', {})
        
        mean_val = noise_stats.get('mean', 2)
        std_val = noise_stats.get('std', 0.5)
        pixel_count = noise_stats.get('pixel_count', 500)
        
        # 生成噪声数据
        np.random.seed(42)
        if pixel_count > 0:
            noise_data = np.random.normal(mean_val, std_val, min(pixel_count, 5000))
        else:
            noise_data = np.random.normal(2, 0.5, 1000)
        
        # 绘制直方图
        n, bins, patches = ax.hist(noise_data, bins=config['bins'],
                                  color=VisualizationConfig.COLOR_SCHEME['charts']['secondary'],
                                  alpha=0.7,
                                  density=True)
        
        # 添加参考线（理论噪声范围）
        if config['show_reference_lines']:
            # 理论噪声范围：mean ± 2*std
            lower_bound = mean_val - 2 * std_val
            upper_bound = mean_val + 2 * std_val
            
            ax.axvline(lower_bound, color=config['reference_line_color'],
                      linestyle='--', alpha=config['reference_line_alpha'])
            ax.axvline(upper_bound, color=config['reference_line_color'],
                      linestyle='--', alpha=config['reference_line_alpha'])
            
            ax.axvspan(lower_bound, upper_bound,
                      alpha=0.1, color=config['reference_line_color'])
            
            # 添加标签
            ax.text(upper_bound, ax.get_ylim()[1] * 0.9,
                   f'±2σ range', rotation=90,
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'] - 1,
                   color=config['reference_line_color'])
        
        # 添加噪声统计
        info_text = (f'Noise Mean: {mean_val:.2f}\n'
                    f'Noise Std: {std_val:.2f}')
        
        ax.text(0.98, 0.98, info_text,
               transform=ax.transAxes,
               fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'],
               va='top', ha='right',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('Noise Value')
        ax.set_ylabel('Probability Density')
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3)
    
    # ============================================================================
    # 子图9：置信度评估（工程文件第13页）
    # ============================================================================
    def _create_subplot_9(self):
        """Subplot 9: Confidence Assessment"""
        ax = self.axes[8]
        config = VisualizationConfig.SUBPLOT_CONFIGS[9]
        
        result = self.current_result
        confidence_scores = result.get('quality_assessment', {}).get('confidence_scores', {})
        
        if not confidence_scores or 'dimensions' not in confidence_scores:
            # 示例数据
            dimension_scores = [0.85, 0.72, 0.68, 0.90]
            dimension_labels = config['dimension_labels']
        else:
            dim_data = confidence_scores['dimensions']
            dimension_scores = [
                dim_data.get('C_ROI', {}).get('score', 0),
                dim_data.get('C_noise', {}).get('score', 0),
                dim_data.get('C_image', {}).get('score', 0),
                dim_data.get('C_algorithm', {}).get('score', 0)
            ]
            dimension_labels = config['dimension_labels']
        
        # 创建条形图
        x_pos = np.arange(len(dimension_scores))
        colors = [VisualizationConfig.get_color_for_confidence(score) for score in dimension_scores]
        
        bars = ax.bar(x_pos, dimension_scores, width=config['bar_width'], color=colors)
        
        # 添加数值标签
        for bar, score in zip(bars, dimension_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.2f}',
                   ha='center', va='bottom',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'])
        
        # 设置x轴标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dimension_labels, rotation=15, ha='right')
        
        # 添加总体置信度
        if config['show_total_confidence']:
            total_score = confidence_scores.get('total_score', np.mean(dimension_scores))
            confidence_level = VisualizationConfig.get_confidence_level(total_score)
            
            ax.text(0.5, -0.25, f'Overall Confidence: {total_score:.2f} ({confidence_level})',
                   transform=ax.transAxes,
                   ha='center',
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'],
                   color=VisualizationConfig.get_color_for_confidence(total_score),
                   weight='bold')
        
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 1.1)
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # 子图10：评分组件（工程文件第13页）
    # ============================================================================
    def _create_subplot_10(self):
        """Subplot 10: Score Components Breakdown"""
        ax = self.axes[9]
        config = VisualizationConfig.SUBPLOT_CONFIGS[10]
        
        # 示例数据 - ROI质量评分组件
        components = ['Intensity', 'Uniformity', 'Edge Distance', 'Anatomy']
        component_scores = [0.9, 0.8, 0.7, 0.6]  # 示例分数
        component_weights = [0.4, 0.3, 0.2, 0.1]  # 工程文件权重
        
        # 计算加权分数
        weighted_scores = [score * weight for score, weight in zip(component_scores, component_weights)]
        
        # 分组条形图
        x_pos = np.arange(len(components))
        width = 0.35
        
        ax.bar(x_pos - width/2, component_scores, width,
              label='Actual Score', 
              color=VisualizationConfig.COLOR_SCHEME['charts']['primary'],
              alpha=0.7)
        ax.bar(x_pos + width/2, component_weights, width,
              label='Weight', 
              color=VisualizationConfig.COLOR_SCHEME['charts']['gray'],
              alpha=0.7)
        
        # 添加加权总分
        total_weighted = sum(weighted_scores)
        ax.axhline(y=total_weighted, color='red', linestyle='--', 
                  label=f'Weighted Total: {total_weighted:.2f}')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(components, rotation=15)
        ax.set_ylabel('Score / Weight')
        ax.set_title(config['title'])
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'] - 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # 子图11：采集信息（工程文件第13页）
    # ============================================================================
    def _create_subplot_11(self):
        """Subplot 11: Acquisition Information Table"""
        ax = self.axes[10]
        config = VisualizationConfig.SUBPLOT_CONFIGS[11]
        
        result = self.current_result
        acquisition = result.get('acquisition', {})
        
        # 准备表格数据
        table_data = [
            ['Patient ID', acquisition.get('patient_id', 'N/A')],
            ['Scan Name', acquisition.get('scan_name', 'N/A')],
            ['Anatomical Region', acquisition.get('anatomical_region', 'N/A')],
            ['Sequence Type', acquisition.get('sequence_type', 'N/A')],
            ['Field Strength', acquisition.get('field_strength', 'N/A')],
            ['Parallel Imaging', 'Yes' if acquisition.get('parallel_imaging') else 'No'],
            ['Acceleration Factor', f"{acquisition.get('acceleration_factor', 1.0):.1f}" 
             if acquisition.get('parallel_imaging') else 'N/A'],
            ['Acquisition Mode', acquisition.get('acquisition_mode', 'N/A')]
        ]
        
        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=['Parameter', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.4, 0.6])
        
        # 格式化表格
        table.auto_set_font_size(False)
        table.set_fontsize(config['table_fontsize'])
        table.scale(1, 1.5)
        
        # 设置表格样式
        for i in range(len(table_data) + 1):  # +1 用于表头
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor(config['header_color'])
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('white')
        
        # 隐藏坐标轴
        ax.axis('off')
        ax.set_title(config['title'], pad=20)
    
    # ============================================================================
    # 子图12：处理信息（工程文件第13-14页）
    # ============================================================================
    def _create_subplot_12(self):
        """Subplot 12: Processing Information"""
        ax = self.axes[11]
        config = VisualizationConfig.SUBPLOT_CONFIGS[12]
        
        result = self.current_result
        analysis_info = result.get('analysis_info', {})
        validation_info = result.get('validation_info', {})
        
        # 准备信息文本
        info_lines = [
            f"Software: {analysis_info.get('software', 'MRI_AutoQA')}",
            f"Algorithm: {analysis_info.get('algorithm_version', 'v1.0')}",
            f"Analysis Date: {analysis_info.get('date', 'N/A')}",
            f"Processing Time: {result.get('processing_time_seconds', 'N/A')} sec",
            "",
            f"Metadata Valid: {'✓' if validation_info.get('metadata_valid') else '✗'}",
            f"Image Valid: {'✓' if validation_info.get('image_valid') else '✗'}",
            f"Constraints Valid: {'✓' if validation_info.get('constraint_space_valid') else '✗'}"
        ]
        
        # 显示处理信息
        y_pos = 0.95
        for line in info_lines:
            if line == "":
                y_pos -= 0.08
                continue
                
            color = 'black'
            if '✓' in line:
                color = '#4CAF50'
            elif '✗' in line:
                color = '#F44336'
            
            ax.text(0.05, y_pos, line,
                   transform=ax.transAxes,
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'],
                   color=color,
                   verticalalignment='top')
            y_pos -= 0.12
        
        # 添加状态图标
        if config['show_status_icon']:
            status = result.get('analysis_status', 'UNKNOWN')
            if status == 'COMPLETED':
                status_symbol = '✓'
                status_color = '#4CAF50'
            elif status == 'FAILED':
                status_symbol = '✗'
                status_color = '#F44336'
            else:
                status_symbol = '⏳'
                status_color = '#FF9800'
            
            ax.text(0.85, 0.9, status_symbol,
                   transform=ax.transAxes,
                   fontsize=config['status_icon_size'] * 3,
                   color=status_color,
                   weight='bold',
                   ha='center')
            
            ax.text(0.85, 0.75, status,
                   transform=ax.transAxes,
                   fontsize=VisualizationConfig.FONT_CONFIG['annotation_size'],
                   color=status_color,
                   ha='center',
                   weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(config['title'])


# ============================================================================
# 便捷函数
# ============================================================================

def create_visualization_for_scan(analysis_result: Dict[str, Any],
                                 image_data: np.ndarray,
                                 output_dir: str,
                                 filename: str = "visualization.png") -> bool:
    """
    为单次扫描创建可视化报告的便捷函数
    
    Args:
        analysis_result: 分析结果字典
        image_data: 2D图像数据
        output_dir: 输出目录
        filename: 输出文件名
        
    Returns:
        成功状态
    """
    from pathlib import Path
    
    output_path = Path(output_dir) / filename
    visualizer = MRIQualityVisualizer(verbose=True)
    
    return visualizer.create_visualization(analysis_result, image_data, str(output_path))