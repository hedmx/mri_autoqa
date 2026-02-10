#!/usr/bin/env python3
"""
Parallel Imaging噪声校正器
基于Acceleration FactorR和g-factor校正单图像法噪声估计
"""

import numpy as np
from typing import Dict, Any, Tuple
from config import G_FACTOR_TABLE


class ParallelCorrector:
    """
    Parallel Imaging噪声校正器
    应用公式: σ_parallel = σ_single_image / (g × √R)
    """
    
    def __init__(self, conservativeness: str = 'standard'):
        """
        Args:
            conservativeness: 保守程度
                - 'standard': 标准g-factor
                - 'conservative': 保守估计（使用较大g-factor）
                - 'minimal': 最小估计（使用较小g-factor）
        """
        self.conservativeness = conservativeness
        self.available_R = sorted(G_FACTOR_TABLE.keys())
    
    def apply_correction(self, 
                        noise_std: float, 
                        acceleration_factor: float,
                        noise_quality: str = 'unknown') -> Dict[str, Any]:
        """
        应用Parallel Imaging校正
        
        Args:
            noise_std: 单图像法估计的Noise Standard Deviation
            acceleration_factor: Acceleration FactorR
            noise_quality: 噪声QualityAssessment（影响g-factor选择）
            
        Returns:
            校正后的噪声统计
        """
        # 获取g-factor
        g_factor, g_description = self._get_g_factor(acceleration_factor, noise_quality)
        
        # 应用校正公式
        if acceleration_factor > 1.0 and g_factor > 0:
            corrected_noise = noise_std / (g_factor * np.sqrt(acceleration_factor))
            correction_factor = 1.0 / (g_factor * np.sqrt(acceleration_factor))
        else:
            corrected_noise = noise_std
            correction_factor = 1.0
        
        # 计算校正比例
        if noise_std > 0:
            correction_percent = (1.0 - correction_factor) * 100
        else:
            correction_percent = 0.0
        
        return {
            'noise_std_original': float(noise_std),
            'noise_std_corrected': float(corrected_noise),
            'acceleration_factor': float(acceleration_factor),
            'g_factor': float(g_factor),
            'g_description': g_description,
            'correction_factor': float(correction_factor),
            'correction_percent': float(correction_percent),
            'formula': 'σ_corrected = σ_original / (g × √R)',
            'conservativeness': self.conservativeness,
            'noise_quality': noise_quality
        }
    
    def _get_g_factor(self, acceleration_factor: float, 
                     noise_quality: str) -> Tuple[float, str]:
        """
        获取g-factor值
        
        Args:
            acceleration_factor: Acceleration FactorR
            noise_quality: 噪声Quality（影响保守程度选择）
            
        Returns:
            (g_factor, description)
        """
        # 根据噪声Quality调整保守程度
        if noise_quality == 'poor':
            conservativeness = 'conservative'
        elif noise_quality == 'fair':
            conservativeness = 'standard'
        else:
            conservativeness = 'minimal'
        
        # 查找最近的R值
        if acceleration_factor in G_FACTOR_TABLE:
            nearest_R = acceleration_factor
        else:
            nearest_R = min(self.available_R, key=lambda x: abs(x - acceleration_factor))
        
        g_mean, g_min, g_max, description = G_FACTOR_TABLE[nearest_R]
        
        # 插值Processing
        if acceleration_factor != nearest_R:
            lower_R = max([r for r in self.available_R if r <= acceleration_factor], default=nearest_R)
            upper_R = min([r for r in self.available_R if r >= acceleration_factor], default=nearest_R)
            
            if lower_R != upper_R:
                # 线性插值
                lower_g = G_FACTOR_TABLE[lower_R][0]
                upper_g = G_FACTOR_TABLE[upper_R][0]
                ratio = (acceleration_factor - lower_R) / (upper_R - lower_R)
                g_mean = lower_g + ratio * (upper_g - lower_g)
                g_min = g_mean * 0.97
                g_max = g_mean * 1.03
                description = f"插值估计 (R={acceleration_factor:.1f})"
        
        # 基于保守性选择
        if conservativeness == 'standard':
            g_factor = g_mean
        elif conservativeness == 'conservative':
            g_factor = g_max
        elif conservativeness == 'minimal':
            g_factor = g_min
        else:
            g_factor = g_mean
        
        return g_factor, description
    
    def estimate_noise_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        估计噪声Quality（用于g-factor选择）
        通过分析四个角落区域的噪声一致性
        
        Args:
            image: MRI图像
            
        Returns:
            噪声Quality分析
        """
        h, w = image.shape
        region_size = min(30, h//10, w//10)
        
        # 定义四个角落区域
        regions = {
            'top_left': image[:region_size, :region_size],
            'top_right': image[:region_size, -region_size:],
            'bottom_left': image[-region_size:, :region_size],
            'bottom_right': image[-region_size:, -region_size:]
        }
        
        # 计算各区域Noise Standard Deviation
        noise_stds = {}
        for name, region in regions.items():
            if region.size > 0:
                # 单图像法噪声估计：直接使用Standard Deviation
                noise_std = np.std(region)
                noise_stds[name] = noise_std
        
        if not noise_stds:
            return {
                'quality': 'unknown',
                'unevenness_ratio': 1.0,
                'cv': 0.0,
                'mean_noise': 0.0,
                'std_noise': 0.0
            }
        
        std_values = list(noise_stds.values())
        mean_std = np.mean(std_values)
        std_std = np.std(std_values)
        
        # 计算Coefficient of Variation
        if mean_std > 0:
            cv = std_std / mean_std
        else:
            cv = 0.0
        
        # AssessmentQuality
        if cv < 0.2:
            quality = 'excellent'
        elif cv < 0.4:
            quality = 'good'
        elif cv < 0.6:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'unevenness_ratio': float(cv + 1.0),  # 转换为比率
            'cv': float(cv),
            'mean_noise': float(mean_std),
            'std_noise': float(std_std),
            'corner_stds': noise_stds,
            'n_regions': len(std_values)
        }
    
    def calculate_snr_with_correction(self,
                                     signal_mean: float,
                                     noise_std_original: float,
                                     acceleration_factor: float,
                                     noise_quality: str = 'unknown') -> Dict[str, Any]:
        """
        计算校正后的SNR
        
        Args:
            signal_mean: Signal Mean
            noise_std_original: 原始Noise Standard Deviation
            acceleration_factor: Acceleration FactorR
            noise_quality: 噪声Quality
            
        Returns:
            SNR分析结果
        """
        # 应用校正
        correction_result = self.apply_correction(
            noise_std_original, acceleration_factor, noise_quality
        )
        
        noise_std_corrected = correction_result['noise_std_corrected']
        
        # 计算SNR
        if noise_std_corrected > 0:
            snr_corrected = signal_mean / noise_std_corrected
            snr_db_corrected = 20 * np.log10(snr_corrected) if snr_corrected > 0 else -np.inf
        else:
            snr_corrected = 0
            snr_db_corrected = -np.inf
        
        # 计算原始SNR（用于对比）
        if noise_std_original > 0:
            snr_original = signal_mean / noise_std_original
        else:
            snr_original = 0
        
        # 计算差异
        if snr_original > 0:
            snr_difference = ((snr_corrected - snr_original) / snr_original) * 100
        else:
            snr_difference = 0.0
        
        return {
            'snr_original': float(snr_original),
            'snr_corrected': float(snr_corrected),
            'snr_db_corrected': float(snr_db_corrected),
            'snr_difference_percent': float(snr_difference),
            'correction': correction_result,
            'recommendation': self._generate_recommendation(snr_difference, noise_quality)
        }
    
    def _generate_recommendation(self, snr_diff: float, noise_quality: str) -> str:
        """生成建议"""
        recommendations = []
        
        # 基于SNR差异
        if snr_diff < -20:
            recommendations.append("Parallel Imaging显著降低SNR，传统估计可能高估约20%以上")
        elif snr_diff < -10:
            recommendations.append("Parallel Imaging对SNR有中等影响（约10-20%）")
        elif snr_diff < -5:
            recommendations.append("Parallel Imaging对SNR有轻微影响（约5-10%）")
        else:
            recommendations.append("Parallel Imaging影响较小")
        
        # 基于噪声Quality
        if noise_quality == 'poor':
            recommendations.append("噪声分布不均匀，建议检查Parallel Imaging参数")
        elif noise_quality == 'fair':
            recommendations.append("噪声分布一般，已应用保守校正")
        
        return "; ".join(recommendations)


# Test函数
if __name__ == "__main__":
    # 创建Test图像
    test_image = np.random.normal(loc=10.0, scale=2.0, size=(256, 256))
    
    # Test校正器
    corrector = ParallelCorrector()
    
    # Test噪声QualityAssessment
    noise_quality = corrector.estimate_noise_quality(test_image)
    print("噪声QualityAssessment:")
    print(f"  Quality等级: {noise_quality['quality']}")
    print(f"  Coefficient of Variation: {noise_quality['cv']:.3f}")
    
    # TestSNR校正
    snr_result = corrector.calculate_snr_with_correction(
        signal_mean=100.0,
        noise_std_original=2.0,
        acceleration_factor=2.0,
        noise_quality='good'
    )
    
    print("\nSNR校正结果:")
    print(f"  原始SNR: {snr_result['snr_original']:.1f}")
    print(f"  校正SNR: {snr_result['snr_corrected']:.1f}")
    print(f"  差异: {snr_result['snr_difference_percent']:+.1f}%")
    print(f"  建议: {snr_result['recommendation']}")