#!/usr/bin/env python3
"""单图像法V3核心引擎 - 椭圆ROI自适应定位版 (兼容可视化)
只保留核心SNR算法，移除可视化和其他辅助功能
专为集成到MRI质量分析系统设计
"""

import numpy as np
import nibabel as nib
from typing import Dict, Optional, Tuple, Any
import datetime
import warnings
from config import ROI_SETTINGS  # 👈 导入新的配置

class SingleImageV3Core:
    """ 单图像法V3核心算法 仅包含SNR计算相关功能 """

    # 物理标准
    AIR_CRITERIA = {
        'max_mean': 12.0,
        'max_std': 6.0,
        'max_single_pixel': 25.0,
        'min_pixels': 30,
        'allow_all_zeros': False,
        'min_mean_for_realistic_noise': 0.5,
    }

    CLINICAL_SNR_STANDARDS = {
        'EXCELLENT': 30.0,
        'GOOD': 20.0,
        'FAIR': 10.0,
        'POOR': 5.0,
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.decision_log = []
        self.image_stats = None
        warnings.filterwarnings('ignore', category=UserWarning)

    # ========================================================================
    # 公共接口
    # ========================================================================
    def analyze_image(self, image: np.ndarray, anatomical_region: str = "default") -> Dict[str, Any]:
        """ 分析单张图像的主入口 """
        if self.verbose:
            print("=" * 50)
            print("Single Image V3 Core Analysis")
            print("=" * 50)

        # 1. 全局诊断扫描
        diagnostic = self._stage1_global_diagnostic(image)
        # 2. 背景选择
        background = self._stage2_background_selection(diagnostic, image)
        # 3. 噪声估计
        noise_result = self._stage3_noise_estimation(background, image)
        # 4. 信号ROI选择 (椭圆 + 强度引导微调)
        signal = self._stage4_signal_roi_selection(image, anatomical_region=anatomical_region)
        # 【新增】获取原始背景std
        background_std_raw = background['statistics']['std']
        signal_mean = signal['statistics']['mean']
        # 【新增】计算两种SNR
        # 原始SNR：基于智能选择的背景区域原始std
        if background_std_raw > 0:
            snr_raw = signal_mean / background_std_raw
        else:
            snr_raw = 0
        # 校正后SNR：基于瑞利校正的噪声估计
        noise_std_corrected = noise_result['noise_std']
        if noise_std_corrected > 0:
            snr_corrected = signal_mean / noise_std_corrected
        else:
            snr_corrected = 0
        # 【新增】计算校正因子
        if background_std_raw > 0:
            correction_factor = noise_std_corrected / background_std_raw
            improvement_percent = ((snr_corrected - snr_raw) / snr_raw * 100) if snr_raw > 0 else 0
        else:
            correction_factor = 1.0
            improvement_percent = 0.0  
        # 返回与主引擎兼容的结构
        return {
            'snr_results': {
                'snr_raw': float(snr_raw),                    # 原始SNR
                'snr_corrected': float(snr_corrected),        # 瑞利校正后SNR
                'snr_final': float(snr_corrected),            # 最终使用的SNR（=校正后）
                'signal_mean': float(signal_mean),
                'noise_std_raw': float(background_std_raw),    # 原始背景std
                'noise_std_corrected': float(noise_std_corrected),  # 瑞利校正噪声
                'noise_std_final': float(noise_std_corrected),      # 最终使用的噪声
                'correction_factor': float(correction_factor),      # 校正因子
                'improvement_percent': float(improvement_percent),  # 改善百分比
                'correction_method': noise_result['estimation_method']  # 校正方法
            },
            'signal_selection': signal,
            'background_selection': background,
            'version': 'v3_ellipse_adaptive'
        }  
       
    def analyze_nifti(self, nifti_path: str, anatomical_region: str = "default") -> Dict[str, Any]:
        """ 从NIfTI文件分析 """
        nii = nib.load(nifti_path)
        data = nii.get_fdata()
        if len(data.shape) == 3:
            mid_slice = data.shape[2] // 2
            slice_img = data[:, :, mid_slice]
        elif len(data.shape) == 2:
            slice_img = data
        else:
            raise ValueError(f"Unsupported NIfTI shape: {data.shape}")
        return self.analyze_image(slice_img, anatomical_region=anatomical_region)

    # ========================================================================
    # 阶段1-3: 背景与噪声 (保持不变)
    # ========================================================================
    def _stage1_global_diagnostic(self, img: np.ndarray) -> Dict[str, Any]:
        self.image_stats = self._get_image_statistics(img)
        h, w = img.shape
        all_regions = {}
        for size in [20, 30, 40, 50]:
            corners = self._generate_corner_regions(h, w, size)
            for name, coords in corners:
                region_data = self._analyze_region(img, coords)
                all_regions[f"{name}_{size}"] = region_data
        valid_regions = {}
        for region_name, region_data in all_regions.items():
            meets_criteria, _ = self._check_air_criteria(region_data)
            if meets_criteria:
                valid_regions[region_name] = region_data
        return {
            'total_candidates': len(all_regions),
            'valid_air_regions': len(valid_regions),
            'valid_regions': valid_regions,
            'image_statistics': self.image_stats
        }

    def _generate_corner_regions(self, h: int, w: int, size: int) -> list:
        return [
            ('top_left', (0, size, 0, size)),
            ('top_right', (0, size, w-size, w)),
            ('bottom_left', (h-size, h, 0, size)),
            ('bottom_right', (h-size, h, w-size, w))
        ]

    def _analyze_region(self, img: np.ndarray, coords: Tuple) -> Dict[str, Any]:
        y1, y2, x1, x2 = coords
        region = img[y1:y2, x1:x2]
        flat_values = region.flatten()
        is_all_zeros = np.all(np.abs(flat_values) < 1e-10)
        if is_all_zeros:
            return {
                'coordinates': coords,
                'mean': 0.0,
                'std': 0.0,
                'pixel_count': len(flat_values),
                'is_all_zeros': True,
                'is_too_pure': False
            }
        mean_val = float(np.mean(flat_values))
        std_val = float(np.std(flat_values))
        is_too_pure = mean_val < self.AIR_CRITERIA['min_mean_for_realistic_noise']
        return {
            'coordinates': coords,
            'mean': mean_val,
            'std': std_val,
            'pixel_count': len(flat_values),
            'is_all_zeros': False,
            'is_too_pure': is_too_pure
        }

    def _check_air_criteria(self, region_data: Dict) -> Tuple[bool, str]:
        if region_data.get('is_all_zeros', False):
            return False, "region is all zeros"
        if region_data.get('is_too_pure', False):
            return False, "region too pure"
        if region_data['mean'] > self.AIR_CRITERIA['max_mean']:
            return False, f"mean too high ({region_data['mean']:.2f})"
        if region_data['std'] > self.AIR_CRITERIA['max_std']:
            return False, f"std too high ({region_data['std']:.2f})"
        if region_data['pixel_count'] < self.AIR_CRITERIA['min_pixels']:
            return False, f"too few pixels ({region_data['pixel_count']})"
        return True, "meets criteria"

    def _stage2_background_selection(self, diagnostic: Dict, img: np.ndarray) -> Dict[str, Any]:
        valid_regions = diagnostic['valid_regions']
        if valid_regions:
            selected_region = self._select_best_region(valid_regions)
            meets_criteria = True
            selection_method = "optimal_air_background"
            confidence = "HIGH"
        else:
            selected_region = self._fallback_background_selection(img)
            meets_criteria = False
            selection_method = "fallback_lowest_mean"
            confidence = "MEDIUM"
        y1, y2, x1, x2 = selected_region['coordinates']
        background_values = img[y1:y2, x1:x2].flatten()
        return {
            'region_name': selected_region.get('name', 'selected_region'),
            'coordinates': selected_region['coordinates'],
            'statistics': {
                'mean': selected_region['mean'],
                'std': selected_region['std'],
                'pixel_count': selected_region['pixel_count']
            },
            'is_all_zeros': selected_region.get('is_all_zeros', False),
            'is_too_pure': selected_region.get('is_too_pure', False),
            'meets_air_criteria': meets_criteria,
            'confidence': confidence,
            'selection_method': selection_method,
            'background_values': background_values.tolist()
        }

    def _select_best_region(self, valid_regions: Dict) -> Dict:
        scored_regions = []
        for name, data in valid_regions.items():
            mean_val = data['mean']
            std_val = data['std']
            mean_score = 0.1
            if 1.0 <= mean_val <= 5.0:
                mean_score = 1.0 - abs(mean_val - 3.0) / 4.0
            elif 0.5 <= mean_val < 1.0:
                mean_score = 0.3 + (mean_val - 0.5) * 0.4
            elif 5.0 < mean_val <= 10.0:
                mean_score = 0.5 - (mean_val - 5.0) * 0.1

            std_score = 0.1
            if 1.0 <= std_val <= 3.0:
                std_score = 1.0 - abs(std_val - 2.0) / 2.0
            elif 0.5 <= std_val < 1.0:
                std_score = 0.3 + (std_val - 0.5) * 1.4
            elif 3.0 < std_val <= 6.0:
                std_score = 1.0 - (std_val - 3.0) * 0.2

            size_score = min(1.0, np.log(data['pixel_count'] + 1) / 5.0)
            total_score = (mean_score * 0.5 + std_score * 0.3 + size_score * 0.2)
            scored_regions.append((total_score, name, data))
        scored_regions.sort(key=lambda x: x[0], reverse=True)
        best_score, best_name, best_data = scored_regions[0]
        return {'name': best_name, 'score': best_score, **best_data}

    def _fallback_background_selection(self, img: np.ndarray) -> Dict:
        h, w = img.shape
        region_size = 30
        test_positions = [(0, 0), (0, w-region_size), (h-region_size, 0), (h-region_size, w-region_size), (h//2, w//2)]
        test_regions = []
        for y_start, x_start in test_positions:
            y1 = max(0, y_start)
            y2 = min(h, y_start + region_size)
            x1 = max(0, x_start)
            x2 = min(w, x_start + region_size)
            if y2 > y1 and x2 > x1:
                region = img[y1:y2, x1:x2]
                mean_val = np.mean(region)
                std_val = np.std(region)
                test_regions.append({
                    'name': f"fallback_{y_start}_{x_start}",
                    'coordinates': (y1, y2, x1, x2),
                    'mean': mean_val,
                    'std': std_val,
                    'pixel_count': region.size,
                    'is_all_zeros': False,
                    'is_too_pure': mean_val < 0.5
                })
        if test_regions:
            non_pure = [r for r in test_regions if not r['is_too_pure']]
            if non_pure:
                selected = min(non_pure, key=lambda x: x['mean'])
            else:
                selected = min(test_regions, key=lambda x: x['mean'])
            return selected
        else:
            flat_img = img.flatten()
            threshold = np.percentile(flat_img, 5)
            low_pixels = flat_img[flat_img < threshold]
            return {
                'name': "final_fallback_percentile",
                'coordinates': (0, h, 0, w),
                'mean': float(np.mean(low_pixels)) if len(low_pixels) > 0 else 0.0,
                'std': float(np.std(low_pixels)) if len(low_pixels) > 0 else 0.0,
                'pixel_count': len(low_pixels),
                'is_all_zeros': False,
                'is_too_pure': False
            }

    def _stage3_noise_estimation(self, background: Dict, img: np.ndarray) -> Dict[str, Any]:
        bg_values = np.array(background['background_values'])
        bg_values = bg_values[bg_values > np.finfo(float).eps]
        # 【确保】获取原始背景统计
        bg_mean = float(np.mean(bg_values)) if len(bg_values) > 0 else 0.0
        bg_std_raw = float(np.std(bg_values)) if len(bg_values) > 0 else 0.0
        if len(bg_values) == 0:
            return self._estimate_noise_from_zero_background(img)

        try:
            sigma_mle_squared = np.sum(bg_values ** 2) / (2 * len(bg_values))
            noise_mle = np.sqrt(sigma_mle_squared)
            mle_valid = True
        except Exception as e:
            noise_mle = np.nan
            mle_valid = False

        q75 = np.percentile(bg_values, 75)
        q25 = np.percentile(bg_values, 25)
        iqr = q75 - q25
        noise_iqr = iqr / 1.349 if iqr > 0 else np.std(bg_values)

        median_val = np.median(bg_values)
        abs_diff = np.abs(bg_values - median_val)
        mad_raw = np.median(abs_diff)
        noise_mad = mad_raw / 0.6745 if mad_raw > 1e-10 else np.nan

        estimates = []
        if mle_valid and np.isfinite(noise_mle) and noise_mle > 0:
            estimates.append(('mle', noise_mle))
        if np.isfinite(noise_iqr) and noise_iqr > 0:
            estimates.append(('iqr', noise_iqr))
        if np.isfinite(noise_mad) and noise_mad > 0:
            estimates.append(('mad', noise_mad))

        if len(estimates) == 0:
            final_noise = self._estimate_noise_from_pure_background(img, bg_values)['noise_std']
            estimation_method = 'fallback_hybrid'
        elif len(estimates) == 1:
            final_noise = estimates[0][1]
            estimation_method = f"single_{estimates[0][0]}"
        else:
            noise_vals = [val for _, val in estimates]
            final_noise = float(np.median(noise_vals))
            if 'mle' in [name for name, _ in estimates] and final_noise == noise_mle:
                estimation_method = 'primary_mle_with_consensus'
            else:
                estimation_method = 'consensus_robust'

        final_noise = self._apply_realistic_noise_bounds(final_noise, img)
        bg_mean = float(np.mean(bg_values))
        bg_std = float(np.std(bg_values))

        return {
            'noise_std': final_noise,          # 瑞利校正后的噪声
            'background_std_raw': bg_std_raw,   # 【新增】原始背景std
            'background_mean': bg_mean,
            'estimation_method': estimation_method,
            'confidence': 'HIGH' if 'mle' in estimation_method else 'MEDIUM',
            'pixel_count': len(bg_values),
        }

    def _estimate_noise_from_zero_background(self, img: np.ndarray) -> Dict[str, Any]:
        img_range = np.max(img) - np.min(img)
        img_std = np.std(img)
        img_mean = np.mean(img)
        estimates = [img_range * 0.02, img_std * 0.03, 1.5 if img_mean < 50 else 2.0 if img_mean < 200 else 2.5]
        empirical_noise = np.median(estimates)
        empirical_noise = max(1.0, min(empirical_noise, 4.0))
        return {
            'noise_std': float(empirical_noise),
            'estimation_method': 'empirical_zero_background',
            'confidence': 'MEDIUM',
            'background_mean': 0.0,
            'background_std': 0.0,
            'pixel_count': 0
        }

    def _estimate_noise_from_pure_background(self, img: np.ndarray, bg_values: np.ndarray) -> Dict[str, Any]:
        bg_std = np.std(bg_values)
        if bg_std < 0.5:
            empirical_noise = max(1.0, np.std(img) * 0.025)
            method = "empirical_pure_background"
        else:
            empirical_noise = max(bg_std, 1.0)
            method = "adjusted_pure_background"
        empirical_noise = min(empirical_noise, 5.0)
        return {
            'noise_std': float(empirical_noise),
            'estimation_method': method,
            'confidence': 'MEDIUM',
            'background_mean': float(np.mean(bg_values)),
            'background_std': float(bg_std),
            'pixel_count': len(bg_values)
        }

    def _apply_realistic_noise_bounds(self, noise_estimate: float, img: np.ndarray) -> float:
        MIN_NOISE = 1.0
        MAX_NOISE = 4.0
        if noise_estimate < MIN_NOISE:
            empirical = np.std(img) * 0.025
            return max(MIN_NOISE, min(empirical, MAX_NOISE))
        elif noise_estimate > MAX_NOISE:
            reasonable_max = min(MAX_NOISE, np.std(img) * 0.05)
            return reasonable_max
        return noise_estimate

    # ========================================================================
    # 阶段4: 信号ROI选择 (已修正 - 椭圆 + 强度引导微调 + 输出兼容坐标)
    # ========================================================================
    def _create_ellipse_mask(self, shape: Tuple[int, int], center_y: int, center_x: int, radius: float) -> np.ndarray:
        """创建实心圆形掩码（椭圆特例）"""
        h, w = shape
        y, x = np.ogrid[:h, :w]
        dist_squared = (y - center_y)**2 + (x - center_x)**2
        mask = dist_squared <= (radius ** 2)
        return mask

    def _stage4_signal_roi_selection(self, img: np.ndarray, anatomical_region: str = "default") -> Dict[str, Any]:
        """基于解剖模板 + 强度引导进行自适应椭圆ROI选择"""
        h, w = img.shape
        roi_config = ROI_SETTINGS.get(anatomical_region, ROI_SETTINGS["default"])

        base_center_y = int(h * roi_config["center_y_ratio"])
        base_center_x = int(w * roi_config["center_x_ratio"])
        base_radius = int(min(h, w) * roi_config["size_ratio"] / 2.0)

        candidates = []
        # 微调网格：空间偏移 ±8px，半径缩放 ±15%
        for dy in [-8, 0, 8]:
            for dx in [-8, 0, 8]:
                for dr_factor in [0.85, 1.0, 1.15]:
                    new_center_y = np.clip(base_center_y + dy, 10, h - 10).astype(int)
                    new_center_x = np.clip(base_center_x + dx, 10, w - 10).astype(int)
                    new_radius = base_radius * dr_factor

                    if new_radius < 3:
                        continue

                    ellipse_mask = self._create_ellipse_mask((h, w), new_center_y, new_center_x, new_radius)
                    roi_pixels = img[ellipse_mask]

                    if roi_pixels.size < 20:  # 最小像素数保障
                        continue

                    mean_val = np.mean(roi_pixels)
                    std_val = np.std(roi_pixels)
                    uniformity_score = std_val / mean_val if mean_val > 1e-6 else float('inf')
                    # 防止选到低信号区：若信号低于75%分位数，则加罚
                    intensity_penalty = 1.0 / (mean_val + 1e-6) if mean_val < np.percentile(img, 75) else 0
                    total_score = uniformity_score + intensity_penalty

                    candidates.append({
                        'mask': ellipse_mask,
                        'score': total_score,
                        'statistics': {
                            'mean': float(mean_val),
                            'std': float(std_val),
                            'min': float(np.min(roi_pixels)),
                            'max': float(np.max(roi_pixels)),
                            'pixel_count': int(roi_pixels.size),
                            'uniformity': float(1.0 / uniformity_score) if uniformity_score > 1e-6 else 0.0
                        }
                    })

        if not candidates:
            return self._fallback_to_original_ellipse(img, anatomical_region)

        best_candidate = min(candidates, key=lambda x: x['score'])
        
        # 【关键修复】计算掩码的边界框作为 coordinates
        mask = best_candidate['mask']
        coords = np.argwhere(mask)
        if len(coords) > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            coordinates = (int(y1), int(y2+1), int(x1), int(x2+1)) # y2, x2 要+1，因为切片是 [y1:y2, x1:x2]
        else:
            coordinates = (0, 0, 0, 0) # fallback

        return {
            'mask': best_candidate['mask'],
            'coordinates': coordinates, # 添加这一行，兼容旧版可视化
            'statistics': best_candidate['statistics'],
            'selection_method': f'adaptive_ellipse_{anatomical_region}'
        }

    def _fallback_to_original_ellipse(self, img: np.ndarray, anatomical_region: str) -> Dict[str, Any]:
        """回退到原始椭圆模板"""
        h, w = img.shape
        roi_config = ROI_SETTINGS.get(anatomical_region, ROI_SETTINGS["default"])
        center_y = int(h * roi_config["center_y_ratio"])
        center_x = int(w * roi_config["center_x_ratio"])
        radius = (min(h, w) * roi_config["size_ratio"]) / 2.0

        ellipse_mask = self._create_ellipse_mask((h, w), center_y, center_x, radius)
        roi_pixels = img[ellipse_mask]

        mean_val = np.mean(roi_pixels)
        std_val = np.std(roi_pixels)
        uniformity = 1.0 / (std_val / mean_val) if mean_val > 1e-6 else 0.0
        
        # 【关键修复】同样为回退情况计算 coordinates
        coords = np.argwhere(ellipse_mask)
        if len(coords) > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            coordinates = (int(y1), int(y2+1), int(x1), int(x2+1))
        else:
            coordinates = (0, 0, 0, 0)

        return {
            'mask': ellipse_mask,
            'coordinates': coordinates, # 添加这一行，兼容旧版可视化
            'statistics': {
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(np.min(roi_pixels)),
                'max': float(np.max(roi_pixels)),
                'pixel_count': int(roi_pixels.size),
                'uniformity': float(uniformity)
            },
            'selection_method': f'ellipse_template_fallback_{anatomical_region}'
        }

    # ========================================================================
    # 辅助函数
    # ========================================================================
    def _get_image_statistics(self, img: np.ndarray) -> Dict[str, float]:
        flat_img = img.flatten()
        return {
            'min': float(np.min(flat_img)),
            'max': float(np.max(flat_img)),
            'mean': float(np.mean(flat_img)),
            'std': float(np.std(flat_img)),
            'median': float(np.median(flat_img)),
            'q1': float(np.percentile(flat_img, 25)),
            'q3': float(np.percentile(flat_img, 75)),
            'shape': list(img.shape)
        }

# ========================================================================
# 便捷函数
# ========================================================================
def analyze_single_image(image: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
    engine = SingleImageV3Core(verbose=verbose)
    return engine.analyze_image(image)

def analyze_single_nifti(nifti_path: str, verbose: bool = False) -> Dict[str, Any]:
    engine = SingleImageV3Core(verbose=verbose)
    return engine.analyze_nifti(nifti_path)