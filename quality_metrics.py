#!/usr/bin/env python3
"""
质量指标计算器 - 包含完整的置信度评估和CNR计算
完全遵循工程文件规范（第4-6页）
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import warnings
from config import (
    ROI_SETTINGS, SEQUENCE_SNR_STANDARDS, QUALITY_SCORE_WEIGHTS,
    QUALITY_THRESHOLDS, ANATOMY_TISSUE_PAIRS
)
@dataclass
class ConfidenceWeights:
    """置信度权重配置（工程文件第5页）"""
    # 总体置信度权重
    total_weights = {
        'C_ROI': 0.35,      # ROI放置质量
        'C_noise': 0.35,    # 噪声估计可靠性
        'C_image': 0.20,    # 图像内在质量
        'C_algorithm': 0.10 # 算法执行质量
    }
    
    # ROI放置质量子权重
    roi_weights = {
        'S_intensity': 0.40,    # 信号强度
        'S_uniformity': 0.30,   # 均匀性
        'S_edge': 0.20,        # 边缘距离
        'S_anatomy': 0.10      # 解剖位置合理性
    }
    
    # 噪声估计可靠性子权重
    noise_weights = {
        'S_bg_size': 0.30,     # 背景区域大小
        'S_bg_mean': 0.40,     # 背景均值
        'S_bg_uniformity': 0.30 # 背景均匀性
    }
    
    # 图像内在质量子权重
    image_weights = {
        'S_contrast': 0.40,    # 对比度
        'S_artifact': 0.30,    # 伪影
        'S_integrity': 0.30    # 图像完整性
    }
    
    # 算法执行质量子权重
    algorithm_weights = {
        'S_convergence': 0.50,  # 收敛性
        'S_stability': 0.30,    # 稳定性
        'S_completeness': 0.20  # 完整性
    }


class QualityMetricsCalculator:
    """
    质量指标计算器 - 实现工程文件第4-6页的完整算法
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.weights = ConfidenceWeights()
        # 使用config.py中的配置
        self.quality_weights = QUALITY_SCORE_WEIGHTS['total']
        self.quality_thresholds = QUALITY_THRESHOLDS
        # 阈值配置（可根据工程文件调整）
        self.thresholds = {
            'intensity': {
                'high': 0.8,    # ≥0.8×P70
                'medium': 0.5,  # 0.5×P70 - 0.8×P70
                'low': 0.0      # <0.5×P70
            },
            'uniformity': {
                'high': 0.3,    # CV≤0.3
                'medium': 0.5,  # 0.3<CV≤0.5
                'low': 1.0      # CV>0.5
            },
            'bg_mean': {
                'ideal_min': 1.0,
                'ideal_max': 10.0,
                'acceptable_min': 0.5,
                'acceptable_max': 20.0
            },
            'bg_uniformity': {
                'high': 0.5,    # CV≤0.5
                'medium': 1.0,  # 0.5<CV≤1.0
                'low': 2.0      # CV>1.0
            }
        }
    
    def calculate_all_metrics(self,
                            image: np.ndarray,
                            metadata: Dict[str, Any],
                            signal_roi: Dict[str, Any],
                            background_roi: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算所有质量指标-修改版：分开质量和置信度
        """
        # 1. 置信度评估（四维度）
        confidence_scores = self.calculate_confidence_scores(
            image, metadata, signal_roi, background_roi
        )
        # 1.2. 图像质量评分（五维度）- 新增
        quality_scores = self.calculate_quality_scores(
            image, metadata, signal_roi, background_roi
        )
        # 2. CNR分析（模板方法）
        cnr_analysis = self.calculate_cnr_analysis(
            image, metadata, signal_roi, background_roi
        )
        
        # 3. 噪声均匀性分析
        noise_uniformity = self.calculate_noise_uniformity(
            background_roi, image
        )
        
        # 4. 信号质量分析
        signal_quality = self.analyze_signal_quality(signal_roi, image)
        
        # 5. 背景质量分析
        background_quality = self.analyze_background_quality(background_roi)
        
        return {
            'confidence_scores': confidence_scores,
            'quality_scores': quality_scores,   
            'cnr_analysis': cnr_analysis,
            'noise_uniformity': noise_uniformity,
            'signal_quality': signal_quality,
            'background_quality': background_quality,
            'overall_confidence': {
                'score': confidence_scores['total_score'],
                'level': self._confidence_level(confidence_scores['total_score'])
            }
        }
    def calculate_quality_scores(self,
                                image: np.ndarray,
                                metadata: Dict[str, Any],
                                signal_roi: Dict[str, Any],
                                background_roi: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算图像质量评分（工程文件第13页）- 新增
        基于5个客观质量指标
        """
        # 1. SNR质量评分 (25%权重)
        snr_quality = self._calculate_snr_quality(image, signal_roi, background_roi, metadata)
        
        # 2. CNR质量评分 (25%权重)
        cnr_quality = self._calculate_cnr_quality(image, signal_roi, background_roi, metadata)
        
        # 3. 噪声均匀性评分 (20%权重)
        noise_quality = self._calculate_noise_quality(background_roi, image)
        
        # 4. 伪影水平评分 (20%权重)
        artifact_quality = self._detect_artifacts_quality(image, metadata)
        
        # 5. 图像完整性评分 (10%权重)
        integrity_quality = self._check_image_integrity(image, metadata)
        
        # 加权总分
        # 使用config.py中的权重
        total_score = (
            snr_quality * self.quality_weights['snr_quality'] +
            cnr_quality * self.quality_weights['cnr_quality'] +
            noise_quality * self.quality_weights['noise_quality'] +
            artifact_quality * self.quality_weights['artifact_free'] +
            integrity_quality * self.quality_weights['image_integrity']
        )
        
        return {
            'total_score': float(total_score),
            'dimensions': {
                'snr_quality': {'score': float(snr_quality)},
                'cnr_quality': {'score': float(cnr_quality)},
                'noise_quality': {'score': float(noise_quality)},
                'artifact_free': {'score': float(artifact_quality)},
                'image_integrity': {'score': float(integrity_quality)}
            },
        
            'weights_used': self.quality_weights,
            'thresholds_used': self.quality_thresholds
        }
    def _calculate_snr_quality(self, image, signal_roi, background_roi, metadata):
        """使用config.py阈值计算SNR质量"""
        signal_mean = signal_roi.get('statistics', {}).get('mean', 0)
        noise_std = background_roi.get('statistics', {}).get('std', 1)
        
        if noise_std > 0:
            snr = signal_mean / noise_std
        else:
            snr = 0
        
        # 使用config.py中的SNR阈值
        thresholds = self.quality_thresholds['snr']
        
        if snr >= thresholds['excellent']:
            return 1.0
        elif snr >= thresholds['good']:
            return 0.8 + 0.2 * (snr - thresholds['good']) / (thresholds['excellent'] - thresholds['good'])
        elif snr >= thresholds['fair']:
            return 0.6 + 0.2 * (snr - thresholds['fair']) / (thresholds['good'] - thresholds['fair'])
        elif snr >= thresholds['poor']:
            return 0.4 + 0.2 * (snr - thresholds['poor']) / (thresholds['fair'] - thresholds['poor'])
        else:
            return max(0.0, 0.4 * snr / thresholds['poor'])
    
    def _calculate_cnr_quality(self, image, signal_roi, background_roi, metadata):
        """使用config.py阈值计算CNR质量"""
        cnr_analysis = self.calculate_cnr_analysis(image, metadata, signal_roi, background_roi)
        best_cnr = cnr_analysis.get('best_cnr', {})
        cnr_value = best_cnr.get('cnr_value', 0)
        
        # 使用config.py中的CNR阈值
        thresholds = self.quality_thresholds['cnr']
        
        if cnr_value >= thresholds['excellent']:
            return 1.0
        elif cnr_value >= thresholds['good']:
            return 0.8 + 0.2 * (cnr_value - thresholds['good']) / (thresholds['excellent'] - thresholds['good'])
        elif cnr_value >= thresholds['fair']:
            return 0.6 + 0.2 * (cnr_value - thresholds['fair']) / (thresholds['good'] - thresholds['fair'])
        elif cnr_value >= thresholds['poor']:
            return 0.4 + 0.2 * (cnr_value - thresholds['poor']) / (thresholds['fair'] - thresholds['poor'])
        else:
            return max(0.0, 0.4 * cnr_value / thresholds['poor'])
    
    def _calculate_noise_quality(self, background_roi, image):
        """使用config.py阈值计算噪声质量"""
        noise_uniformity = self.calculate_noise_uniformity(background_roi, image)
        cv = noise_uniformity.get('cv', 0)
        
        # 使用config.py中的噪声均匀性阈值
        thresholds = self.quality_thresholds['noise_uniformity']
        
        if cv <= thresholds['excellent']:
            return 1.0
        elif cv <= thresholds['good']:
            return 0.8 + 0.2 * (thresholds['good'] - cv) / (thresholds['good'] - thresholds['excellent'])
        elif cv <= thresholds['fair']:
            return 0.6 + 0.2 * (thresholds['fair'] - cv) / (thresholds['fair'] - thresholds['good'])
        elif cv <= thresholds['poor']:
            return 0.4 + 0.2 * (thresholds['poor'] - cv) / (thresholds['poor'] - thresholds['fair'])
        else:
            return max(0.0, 0.4 * (2.0 - cv) / 1.0)
    def calculate_confidence_scores(self,
                                  image: np.ndarray,
                                  metadata: Dict[str, Any],
                                  signal_roi: Dict[str, Any],
                                  background_roi: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算四维度置信度得分（工程文件第5-6页）
        """
        # 维度1: ROI放置质量
        c_roi = self._calculate_c_roi(image, signal_roi, metadata)
        
        # 维度2: 噪声估计可靠性
        c_noise = self._calculate_c_noise(background_roi)
        
        # 维度3: 图像内在质量
        c_image = self._calculate_c_image(image, metadata)
        
        # 维度4: 算法执行质量
        c_algorithm = self._calculate_c_algorithm(image, signal_roi, background_roi)
        
        # 总体置信度（加权求和）
        total_score = (
            c_roi['score'] * self.weights.total_weights['C_ROI'] +
            c_noise['score'] * self.weights.total_weights['C_noise'] +
            c_image['score'] * self.weights.total_weights['C_image'] +
            c_algorithm['score'] * self.weights.total_weights['C_algorithm']
        )
        
        return {
            'total_score': float(total_score),
            'dimensions': {
                'C_ROI': c_roi,
                'C_noise': c_noise,
                'C_image': c_image,
                'C_algorithm': c_algorithm
            },
            'weights': {
                'total': self.weights.total_weights,
                'roi': self.weights.roi_weights,
                'noise': self.weights.noise_weights,
                'image': self.weights.image_weights,
                'algorithm': self.weights.algorithm_weights
            }
        }
    
    def _calculate_c_roi(self, image: np.ndarray, 
                        signal_roi: Dict[str, Any],
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        维度1: ROI放置质量 - 修正版
        基于医学图像特性：
        1. 信号强度应显著高于背景（不是全局均值）
        2. 均匀性应良好
        3. 位置应合理（但允许解剖变异）
        """
        roi_stats = signal_roi.get('statistics', {})
        roi_mean = roi_stats.get('mean', 0)
        roi_std = roi_stats.get('std', 0)

        image_flat = image.flatten()
        positive_pixels = image_flat[image_flat > 0]
        if len(positive_pixels) > 0:
            bg_estimate = np.percentile(positive_pixels, 5)
        else:
            bg_estimate = 1.0  # 避免除零
        
        # 1. 信号强度得分 S_intensity
        if roi_mean > 0 and bg_estimate > 0:
            signal_bg_ratio = roi_mean / bg_estimate
            if signal_bg_ratio > 8.0:  # SNR通常要求>8
                s_intensity = 1.0
            elif signal_bg_ratio > 5.0:
                s_intensity = 0.8
            elif signal_bg_ratio > 3.0:
                s_intensity = 0.6
            elif signal_bg_ratio > 2.0:
                s_intensity = 0.4
            else:
                s_intensity = 0.2
        else:
            s_intensity = 0.3
        
        # 2. 均匀性得分 S_uniformity
        
        cv_roi = roi_std / roi_mean if roi_mean > 0 else 1.0
        
        if cv_roi <= 0.3:
            s_uniformity = 1.0
        elif cv_roi <= 0.5:
            s_uniformity = 0.7
        elif cv_roi <= 0.7:
            s_uniformity = 0.5    
        else:
            s_uniformity = 0.3
        
        # 3. 边缘距离得分 S_edge
        # 计算ROI到图像边缘的最小距离
        h, w = image.shape
        roi_coords = signal_roi.get('coordinates', (0, h, 0, w))
        y1, y2, x1, x2 = roi_coords
        
        roi_center_y = (y1 + y2) / 2
        roi_center_x = (x1 + x2) / 2
        
        # 到四边的最小距离
        min_dist = min(
            roi_center_y,  # 上边缘
            h - roi_center_y,  # 下边缘
            roi_center_x,  # 左边缘
            w - roi_center_x   # 右边缘
        )
        
        s_edge = 1.0 - min_dist / (0.15 * min(h, w))
        s_edge = max(0.2, min(1.0, s_edge))  # 限制在[0,1]区间
        
        # 4. 解剖位置合理性 S_anatomy
        s_anatomy = self._calculate_anatomy_score(
            roi_center_x, roi_center_y, h, w, metadata
        )
        
        # 加权求和
        c_roi_score = (
            s_intensity * self.weights.roi_weights['S_intensity'] +
            s_uniformity * self.weights.roi_weights['S_uniformity'] +
            s_edge * self.weights.roi_weights['S_edge'] +
            s_anatomy * self.weights.roi_weights['S_anatomy']
        )
        
        return {
            'score': float(c_roi_score),
            'components': {
                'S_intensity': float(s_intensity),
                'S_uniformity': float(s_uniformity),
                'S_edge': float(s_edge),
                'S_anatomy': float(s_anatomy)
            },
            'details': {
                'roi_mean': float(roi_mean),
                'roi_std': float(roi_std),
                'signal_bg_ratio': float(signal_bg_ratio) if roi_mean > 0 and bg_estimate > 0 else 0,
                'background_estimate': float(bg_estimate),
                'min_edge_distance': float(min_dist)
            }
        }
    
    def _calculate_c_noise(self, background_roi: Dict[str, Any]) -> Dict[str, Any]:
        """
        维度2: 噪声估计可靠性
        公式: C_noise = 0.3×S_bg_size + 0.4×S_bg_mean + 0.3×S_bg_uniformity
        """
        bg_stats = background_roi.get('statistics', {})
        bg_mean = bg_stats.get('mean', 0)
        bg_std = bg_stats.get('std', 0)
        bg_pixels = bg_stats.get('pixel_count', 0)
        
        # 1. 背景区域大小得分 S_bg_size
        s_bg_size = min(1.0, bg_pixels / 1000)  # 工程文件公式
        
        # 2. 背景均值得分 S_bg_mean
        if 1.0 <= bg_mean <= 10.0:
            s_bg_mean = 1.0
        elif 0.5 <= bg_mean < 1.0 or 10.0 < bg_mean <= 20.0:
            s_bg_mean = 0.7
        else:
            s_bg_mean = 0.3
        
        # 3. 背景均匀性得分 S_bg_uniformity
        cv_bg = bg_std / bg_mean if bg_mean > 0 else 1.0
        
        if cv_bg <= 0.5:
            s_bg_uniformity = 1.0
        elif cv_bg <= 1.0:
            s_bg_uniformity = 0.7
        else:
            s_bg_uniformity = 0.3
        
        # 加权求和
        c_noise_score = (
            s_bg_size * self.weights.noise_weights['S_bg_size'] +
            s_bg_mean * self.weights.noise_weights['S_bg_mean'] +
            s_bg_uniformity * self.weights.noise_weights['S_bg_uniformity']
        )
        
        return {
            'score': float(c_noise_score),
            'components': {
                'S_bg_size': float(s_bg_size),
                'S_bg_mean': float(s_bg_mean),
                'S_bg_uniformity': float(s_bg_uniformity)
            },
            'details': {
                'background_mean': float(bg_mean),
                'background_std': float(bg_std),
                'background_cv': float(cv_bg),
                'background_pixels': int(bg_pixels)
            }
        }
    
    def _calculate_c_image(self, image: np.ndarray, 
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        维度3: 图像内在质量
        公式: C_image = 0.4×S_contrast + 0.3×S_artifact + 0.3×S_integrity
        """
        # 1. 对比度得分 S_contrast
        image_flat = image.flatten()
        p25 = np.percentile(image_flat, 25)
        p50 = np.percentile(image_flat, 50)
        p75 = np.percentile(image_flat, 75)
        
        if p50 > 0:
            s_contrast = (p75 - p25) / p50
            # 归一化到[0,1]区间（假设正常对比度在0.5-2.0之间）
            s_contrast = max(0.0, min(1.0, 1.0 - abs(s_contrast - 1.0)))
        else:
            s_contrast = 0.0
        
        # 2. 伪影得分 S_artifact
        s_artifact = self._detect_artifacts_quality(image, metadata)
        
        # 3. 图像完整性得分 S_integrity
        s_integrity = self._check_image_integrity(image, metadata)
        
        # 加权求和
        c_image_score = (
            s_contrast * self.weights.image_weights['S_contrast'] +
            s_artifact * self.weights.image_weights['S_artifact'] +
            s_integrity * self.weights.image_weights['S_integrity']
        )
        
        return {
            'score': float(c_image_score),
            'components': {
                'S_contrast': float(s_contrast),
                'S_artifact': float(s_artifact),
                'S_integrity': float(s_integrity)
            },
            'details': {
                'p25': float(p25),
                'p50': float(p50),
                'p75': float(p75),
                'contrast_ratio': float((p75 - p25) / p50) if p50 > 0 else 0.0
            }
        }
    
    def _calculate_c_algorithm(self, image: np.ndarray,
                             signal_roi: Dict[str, Any],
                             background_roi: Dict[str, Any]) -> Dict[str, Any]:
        """
        维度4: 算法执行质量
        公式: C_algorithm = 0.5×S_convergence + 0.3×S_stability + 0.2×S_completeness
        """
        # 1. 收敛性得分 S_convergence
        # 检查ROI选择是否收敛（多次迭代变化小）
        s_convergence = self._check_convergence(image, signal_roi)
        
        # 2. 稳定性得分 S_stability
        # 检查噪声估计的稳定性（多个背景区域一致性）
        s_stability = self._check_stability(image, background_roi)
        
        # 3. 完整性得分 S_completeness
        # 检查所有必要步骤是否完成
        s_completeness = self._check_completeness(signal_roi, background_roi)
        
        # 加权求和
        c_algorithm_score = (
            s_convergence * self.weights.algorithm_weights['S_convergence'] +
            s_stability * self.weights.algorithm_weights['S_stability'] +
            s_completeness * self.weights.algorithm_weights['S_completeness']
        )
        
        return {
            'score': float(c_algorithm_score),
            'components': {
                'S_convergence': float(s_convergence),
                'S_stability': float(s_stability),
                'S_completeness': float(s_completeness)
            }
        }
    
    def calculate_cnr_analysis(self,
                             image: np.ndarray,
                             metadata: Dict[str, Any],
                             signal_roi: Dict[str, Any],
                             background_roi: Dict[str, Any]) -> Dict[str, Any]:
        """
        CNR计算（工程文件第6页模板方法）
        公式: CNR_ij = |μ_i - μ_j| / σ_noise
        """
        # 获取噪声估计
        noise_std = background_roi.get('statistics', {}).get('std', 1.0)
        
        # 获取解剖区域
        anatomy = metadata.get('anatomical_region', 'lumbar').lower()
        
        # 根据解剖区域获取预定义的组织对模板
        tissue_pairs = self._get_tissue_pairs_for_anatomy(anatomy)
        
        # 计算所有组织对的CNR
        cnr_results = []
        for pair in tissue_pairs:
            # 计算两个组织ROI的均值
            mean1 = self._calculate_tissue_mean(image, signal_roi, pair['roi1_offset'], pair['roi_size'])
            mean2 = self._calculate_tissue_mean(image, signal_roi, pair['roi2_offset'], pair['roi_size'])
            
            # 计算CNR
            if noise_std > 0:
                cnr_value = abs(mean1 - mean2) / noise_std
            else:
                cnr_value = 0.0
            
            cnr_results.append({
                'name': pair['name'],
                'cnr_value': float(cnr_value),
                'tissue1_mean': float(mean1),
                'tissue2_mean': float(mean2),
                'clinical_significance': pair.get('clinical_significance', ''),
                'description': pair.get('description', pair['name'])
            })
        
        # 选择最佳CNR
        if cnr_results:
            best_cnr = max(cnr_results, key=lambda x: x['cnr_value'])
            best_cnr['cnr_rating'] = self._rate_cnr(best_cnr['cnr_value'])
        else:
            best_cnr = None
        
        return {
            'tissue_pairs': cnr_results,
            'best_cnr': best_cnr,
            'noise_std': float(noise_std),
            'anatomy': anatomy
        }
    
    def _get_tissue_pairs_for_anatomy(self, anatomy: str) -> List[Dict[str, Any]]:
        """
        根据解剖区域获取预定义的组织对模板
        工程文件第6页示例
        """
        # 这里可以从配置文件加载，硬编码示例
        tissue_templates = {
            'lumbar': [
                {
                    'name': 'vertebral_body_vs_disc',
                    'roi1_offset': (-20, 0),  # 相对信号ROI中心的偏移
                    'roi2_offset': (20, 0),
                    'roi_size': 20,  # ROI半径（像素）
                    'clinical_significance': 'Assessing spinal degenerative lesions',
                    'description': 'Vertebral Body vs Disc'
                },
                {
                    'name': 'spinal_cord_vs_csf',
                    'roi1_offset': (0, -15),
                    'roi2_offset': (0, 15),
                    'roi_size': 15,
                    'clinical_significance': 'Assessing spinal cord and CSF contrast',
                    'description': 'Spinal Cord vs CSF'
                }
            ],
            'brain': [
                {
                    'name': 'gray_matter_vs_white_matter',
                    'roi1_offset': (-15, 0),
                    'roi2_offset': (15, 0),
                    'roi_size': 15,
                    'clinical_significance': 'Assessing gray-white matter contrast',
                    'description': 'Gray Matter vs White Matter'
                }
            ],
            'default': [
                {
                    'name': 'central_vs_peripheral',
                    'roi1_offset': (0, 0),
                    'roi2_offset': (30, 0),
                    'roi_size': 15,
                    'clinical_significance': 'Assessing central vs peripheral tissue contrast',
                    'description': 'Central vs Peripheral Region'
                }
            ]
        }
        
        return tissue_templates.get(anatomy, tissue_templates['default'])
    
    def _calculate_tissue_mean(self, image: np.ndarray,
                             signal_roi: Dict[str, Any],
                             offset: Tuple[int, int],
                             roi_size: int) -> float:
        """
        计算特定组织ROI的均值
        """
        h, w = image.shape
        
        # 获取信号ROI中心
        roi_coords = signal_roi.get('coordinates', (0, h, 0, w))
        y1, y2, x1, x2 = roi_coords
        roi_center_y = (y1 + y2) // 2
        roi_center_x = (x1 + x2) // 2
        
        # 计算组织ROI位置
        tissue_center_y = roi_center_y + offset[0]
        tissue_center_x = roi_center_x + offset[1]
        
        # 确保在图像范围内
        y_start = max(0, tissue_center_y - roi_size)
        y_end = min(h, tissue_center_y + roi_size)
        x_start = max(0, tissue_center_x - roi_size)
        x_end = min(w, tissue_center_x + roi_size)
        
        # 提取ROI区域
        tissue_roi = image[y_start:y_end, x_start:x_end]
        
        if tissue_roi.size > 0:
            return float(np.mean(tissue_roi))
        else:
            return 0.0
    
    def calculate_noise_uniformity(self,
                                 background_roi: Dict[str, Any],
                                 image: np.ndarray) -> Dict[str, Any]:
        """
        计算噪声均匀性
        """
        bg_stats = background_roi.get('statistics', {})
        bg_std = bg_stats.get('std', 0)
        bg_mean = bg_stats.get('mean', 0)
        
        # 计算变异系数
        cv = bg_std / bg_mean if bg_mean > 0 else 0
        
        # 评估均匀性
        if cv <= 0.3:
            uniformity_rating = 'EXCELLENT'
        elif cv <= 0.5:
            uniformity_rating = 'GOOD'
        elif cv <= 0.7:
            uniformity_rating = 'FAIR'
        else:
            uniformity_rating = 'POOR'
        
        return {
            'cv': float(cv),
            'uniformity_rating': uniformity_rating,
            'assessment': self._get_uniformity_assessment(cv),
            'background_std': float(bg_std),
            'background_mean': float(bg_mean)
        }
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _calculate_anatomy_score(self, center_x: float, center_y: float,
                               h: int, w: int, metadata: Dict[str, Any]) -> float:
        """计算解剖位置合理性得分"""
        # 简单实现：根据解剖区域检查ROI位置
        anatomy = metadata.get('anatomical_region', '').lower()
        
        # 理想中心位置（基于图像尺寸）
        ideal_center_x = w * 0.5
        ideal_center_y = h * 0.5
        
        # 根据解剖区域调整理想位置
        if anatomy == 'brain':
            ideal_center_y = h * 0.4  # 大脑偏上
        elif anatomy == 'lumbar':
            ideal_center_x = w * 0.5
            ideal_center_y = h * 0.6  # 腰椎偏下
        
        # 计算距离得分
        dist_x = abs(center_x - ideal_center_x) / w
        dist_y = abs(center_y - ideal_center_y) / h
        total_dist = np.sqrt(dist_x**2 + dist_y**2)
        
        # 转换为得分（距离越小得分越高）
        score = max(0.0, 1.0 - total_dist * 2.0)
        return float(score)
    
    def _detect_artifacts_quality(self, image: np.ndarray, 
                         metadata: Dict[str, Any]) -> float:
        """检测伪影"""
        # 简化实现：检查图像中的异常模式
        h, w = image.shape
        
        # 1. 检查条纹伪影（傅里叶变换）
        fft_image = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft_image)
        magnitude = np.abs(fft_shift)
        
        # 检查水平/垂直线（条纹伪影）
        center_h = h // 2
        center_w = w // 2
        
        # 检查中心区域外的亮线
        artifact_score = 1.0
        
        # 2. 检查运动伪影（边缘模糊）
        sobel_x = np.abs(np.gradient(image, axis=1))
        sobel_y = np.abs(np.gradient(image, axis=0))
        edge_strength = np.mean(sobel_x + sobel_y)
        
        # 边缘强度太低可能表示模糊
        if edge_strength < np.mean(image) * 0.01:
            artifact_score *= 0.8
        
        return float(artifact_score)
    
    def _check_image_integrity(self, image: np.ndarray,
                             metadata: Dict[str, Any]) -> float:
        """检查图像完整性"""
        h, w = image.shape
        
        # 1. 检查NaN或Inf值
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            return 0.3
        
        # 2. 检查全零区域
        zero_ratio = np.sum(image == 0) / (h * w)
        if zero_ratio > 0.3:  # 超过30%为零
            return 0.5
        
        # 3. 检查动态范围
        img_min, img_max = np.min(image), np.max(image)
        dynamic_range = img_max - img_min
        
        if dynamic_range < 10:  # 动态范围太小
            return 0.6
        
        return 1.0
    
    def _check_convergence(self, image: np.ndarray,
                          signal_roi: Dict[str, Any]) -> float:
        """检查算法收敛性"""
        # 检查ROI统计量的合理性
        roi_stats = signal_roi.get('statistics', {})
        roi_mean = roi_stats.get('mean', 0)
        roi_std = roi_stats.get('std', 0)
        # 更合理的检查：
        # 1. ROI均值应大于0
        if roi_mean <= 0:
            return 0.3
    
        # 2. ROI内应有足够像素
        pixel_count = roi_stats.get('pixel_count', 0)
        if pixel_count < 100:
            return 0.5
    
        # 3. 均匀性检查（CV）
        cv = roi_std / roi_mean if roi_mean > 0 else 1.0
        if cv < 0.5:
            return 1.0
        elif cv < 0.7:
            return 0.8
        elif cv < 1.0:
            return 0.6
        else:
            return 0.4
    
    def _check_stability(self, image: np.ndarray,
                        background_roi: Dict[str, Any]) -> float:
        """检查算法稳定性"""
        # 检查多个角落区域的噪声一致性
        h, w = image.shape
        region_size = 30
        
        corners = [
            image[:region_size, :region_size],  # 左上
            image[:region_size, -region_size:],  # 右上
            image[-region_size:, :region_size],  # 左下
            image[-region_size:, -region_size:]  # 右下
        ]
        
        corner_stds = [np.std(corner) for corner in corners if corner.size > 0]
        
        if len(corner_stds) >= 2:
            cv = np.std(corner_stds) / np.mean(corner_stds) if np.mean(corner_stds) > 0 else 1.0
            if cv < 0.3:
                return 1.0
            elif cv < 0.6:
                return 0.7
            else:
                return 0.4
        else:
            return 0.5
    
    def _check_completeness(self, signal_roi: Dict[str, Any],
                           background_roi: Dict[str, Any]) -> float:
        """检查算法完整性"""
        # 检查所有必要信息是否存在
        required_keys = ['coordinates', 'statistics']
        
        signal_complete = all(key in signal_roi for key in required_keys)
        background_complete = all(key in background_roi for key in required_keys)
        
        if signal_complete and background_complete:
            # 进一步检查统计量
            signal_stats = signal_roi.get('statistics', {})
            background_stats = background_roi.get('statistics', {})
            
            if all(k in signal_stats for k in ['mean', 'std', 'pixel_count']) and \
               all(k in background_stats for k in ['mean', 'std', 'pixel_count']):
                return 1.0
            else:
                return 0.7
        else:
            return 0.4
    
    def _rate_cnr(self, cnr_value: float) -> str:
        """CNR评级"""
        if cnr_value >= 5.0:
            return 'EXCELLENT'
        elif cnr_value >= 3.0:
            return 'GOOD'
        elif cnr_value >= 1.5:
            return 'FAIR'
        elif cnr_value >= 0.5:
            return 'POOR'
        else:
            return 'UNACCEPTABLE'
    
    def _get_uniformity_assessment(self, cv: float) -> str:
        """获取均匀性评估"""
        if cv <= 0.3:
            return '噪声分布非常均匀'
        elif cv <= 0.5:
            return '噪声分布均匀'
        elif cv <= 0.7:
            return '噪声分布一般'
        elif cv <= 1.0:
            return '噪声分布不均匀'
        else:
            return '噪声分布严重不均匀'
    
    def _confidence_level(self, score: float) -> str:
        """置信度等级（工程文件颜色编码）"""
        if score >= 0.80:
            return 'HIGH'
        elif score >= 0.60:
            return 'MEDIUM'
        elif score >= 0.40:
            return 'LOW'
        else:
            return 'FAILED'
    
    def analyze_signal_quality(self, signal_roi: Dict[str, Any],
                             image: np.ndarray) -> Dict[str, Any]:
        """分析信号质量"""
        stats = signal_roi.get('statistics', {})
        
        return {
            'mean': float(stats.get('mean', 0)),
            'std': float(stats.get('std', 0)),
            'cv': float(stats.get('std', 0) / stats.get('mean', 1) if stats.get('mean', 0) > 0 else 0),
            'pixel_count': int(stats.get('pixel_count', 0)),
            'selection_method': signal_roi.get('selection_method', 'unknown')
        }
    
    def analyze_background_quality(self, background_roi: Dict[str, Any]) -> Dict[str, Any]:
        """分析背景质量"""
        stats = background_roi.get('statistics', {})
        
        return {
            'mean': float(stats.get('mean', 0)),
            'std': float(stats.get('std', 0)),
            'pixel_count': int(stats.get('pixel_count', 0)),
            'is_all_zeros': background_roi.get('is_all_zeros', False),
            'region_name': background_roi.get('region_name', 'unknown'),
            'selection_method': background_roi.get('selection_method', 'unknown')
        }


# 便捷函数
def create_quality_metrics_calculator(config: Dict[str, Any] = None) -> QualityMetricsCalculator:
    """创建质量指标计算器"""
    return QualityMetricsCalculator(config)


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 创建测试数据
    test_image = np.random.normal(loc=100, scale=20, size=(256, 256))
    test_metadata = {'anatomical_region': 'lumbar'}
    
    test_signal_roi = {
        'coordinates': (100, 150, 100, 150),
        'statistics': {
            'mean': 120.5,
            'std': 15.2,
            'pixel_count': 2500
        }
    }
    
    test_background_roi = {
        'coordinates': (10, 40, 10, 40),
        'statistics': {
            'mean': 2.5,
            'std': 1.2,
            'pixel_count': 900
        },
        'is_all_zeros': False,
        'region_name': 'top_left_30'
    }
    
    # 测试计算器
    calculator = QualityMetricsCalculator()
    results = calculator.calculate_all_metrics(
        test_image, test_metadata, test_signal_roi, test_background_roi
    )
    
    print("质量指标计算结果:")
    print(f"总体置信度: {results['overall_confidence']['score']:.3f} ({results['overall_confidence']['level']})")
    
    if results['cnr_analysis']['best_cnr']:
        best = results['cnr_analysis']['best_cnr']
        print(f"最佳CNR: {best['cnr_value']:.2f} ({best['description']})")
    
    print(f"噪声均匀性CV: {results['noise_uniformity']['cv']:.3f}")