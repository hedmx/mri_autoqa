#!/usr/bin/env python3
"""单图像法V3核心引擎 - 信号ROI选择优化版
基于统一搜索策略配置，实现多层递进搜索
"""

import numpy as np
import nibabel as nib
from typing import Dict, Optional, Tuple, Any, List
import datetime
import warnings
from scipy import ndimage
from config import ROI_SETTINGS, UNIFIED_SEARCH_STRATEGY

class SingleImageV3Core:
    """ 单图像法V3核心算法 - 基于配置的统一搜索策略 """

    # 物理标准
    AIR_CRITERIA = {
        'max_mean': 12.0,
        'max_std': 6.0,
        'max_single_pixel': 20.0,
        'min_pixels': 30,
        'allow_all_zeros': False,
        'min_mean_for_realistic_noise': 0.5,
        'max_cv': 0.6,
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
    # 公共接口（保持不变）
    # ========================================================================
    def analyze_image(self, image: np.ndarray, anatomical_region: str = "default") -> Dict[str, Any]:
        """ 分析单张图像的主入口 """
        if self.verbose:
            print("=" * 60)
            print("Single Image V3 Core Analysis - Unified Search Strategy")
            print("=" * 60)
            print(f"图像形状: {image.shape}, 全局均值: {np.mean(image):.1f}")

        # 1. 全局诊断扫描
        diagnostic = self._stage1_global_diagnostic(image)
        # 2. 背景选择
        background = self._stage2_background_selection(diagnostic, image)
        # 3. 噪声估计
        noise_result = self._stage3_noise_estimation(background, image)
        # 4. 信号ROI选择（基于统一策略）
        signal = self._stage4_unified_signal_search(image, anatomical_region=anatomical_region)
        
        # SNR计算
        background_std_raw = background['statistics']['std']
        signal_mean = signal['statistics']['mean']
        
        if background_std_raw > 0:
            snr_raw = signal_mean / background_std_raw
        else:
            snr_raw = 0
        
        noise_std_corrected = noise_result['noise_std']
        if noise_std_corrected > 0:
            snr_corrected = signal_mean / noise_std_corrected
        else:
            snr_corrected = 0
        
        if background_std_raw > 0:
            correction_factor = noise_std_corrected / background_std_raw
            improvement_percent = ((snr_corrected - snr_raw) / snr_raw * 100) if snr_raw > 0 else 0
        else:
            correction_factor = 1.0
            improvement_percent = 0.0

        # 信号质量评估
        signal_cv = signal['statistics']['cv']
        signal_quality = self._evaluate_signal_quality(signal_mean, signal_cv)
        
        return {
            'snr_results': {
                'snr_raw': float(snr_raw),
                'snr_corrected': float(snr_corrected),
                'snr_final': float(snr_corrected),
                'signal_mean': float(signal_mean),
                'noise_std_raw': float(background_std_raw),
                'noise_std_corrected': float(noise_std_corrected),
                'noise_std_final': float(noise_std_corrected),
                'correction_factor': float(correction_factor),
                'improvement_percent': float(improvement_percent),
                'correction_method': noise_result['estimation_method']
            },
            'signal_selection': signal,
            'background_selection': background,
            'signal_quality': signal_quality,
            'version': 'v3_unified_search'
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
    # 核心：基于统一策略的信号ROI搜索
    # ========================================================================
    def _stage4_unified_signal_search(self, img: np.ndarray, anatomical_region: str = "default") -> Dict[str, Any]:
        """阶段4: 基于统一策略的信号ROI搜索"""
        
        h, w = img.shape
        
        if self.verbose:
            print(f"\n🔍 统一策略信号ROI搜索 - {anatomical_region}")
            print(f"   图像尺寸: {h}x{w}")
        
        # 1. 获取配置
        region_cfg = ROI_SETTINGS.get(anatomical_region, ROI_SETTINGS["default"])
        strategy_cfg = UNIFIED_SEARCH_STRATEGY
        
        # 2. 计算自适应信号范围
        signal_range = self._calculate_adaptive_signal_range(img, region_cfg)
        
        if self.verbose:
            print(f"   信号范围: [{signal_range[0]:.1f}, {signal_range[1]:.1f}]")
        
        # 3. 执行多层搜索
        all_candidates = []  # 收集所有候选点用于回退
        
        # 第1层：主搜索（高质量）
        if self.verbose:
            print(f"\n   🟢 第1层：主搜索")
        
        result_layer1 = self._execute_search_layer(
            img, region_cfg, strategy_cfg['search_layers'][0], 
            signal_range, 'primary'
        )
        
        if result_layer1['success']:
            result_layer1['quality_indicators'] = {
                'search_layer': 'primary',
                'cv_value': result_layer1['statistics']['cv'],
                'confidence': 'HIGH',
                'fallback_used': False
            }
            return result_layer1
        
        all_candidates.extend(result_layer1.get('candidates', []))
        
        # 第2层：扩展搜索（中等质量）
        if self.verbose:
            print(f"\n   🟡 第2层：扩展搜索")
        
        result_layer2 = self._execute_search_layer(
            img, region_cfg, strategy_cfg['search_layers'][1], 
            signal_range, 'extended'
        )
        
        if result_layer2['success']:
            result_layer2['quality_indicators'] = {
                'search_layer': 'extended',
                'cv_value': result_layer2['statistics']['cv'],
                'confidence': 'MEDIUM',
                'fallback_used': False,
                'quality_warning': True
            }
            return result_layer2
        
        all_candidates.extend(result_layer2.get('candidates', []))
        
        # 第3层：智能回退
        if self.verbose:
            print(f"\n   🔴 第3层：智能回退")
        
        fallback_result = self._execute_smart_fallback(
            img, region_cfg, all_candidates, signal_range
        )
        
        # 添加质量指示器
        fallback_result['quality_indicators'] = {
            'search_layer': 'fallback',
            'cv_value': fallback_result['statistics']['cv'],
            'confidence': 'LOW' if fallback_result['statistics']['cv'] > 0.3 else 'MEDIUM',
            'fallback_used': True,
            'fallback_type': fallback_result.get('fallback_type', 'unknown'),
            'quality_warning': True
        }
        
        return fallback_result

    def _execute_search_layer(self, img: np.ndarray, region_cfg: Dict, 
                             layer_cfg: Dict, signal_range: Tuple[float, float], 
                             layer_name: str) -> Dict[str, Any]:
        """执行单层搜索"""
        
        h, w = img.shape
        
        # 计算该层的实际参数
        max_cv = region_cfg['max_allowed_cv'] * layer_cfg['max_cv_multiplier']
        search_radius_y = int(h * region_cfg['search_radius_y_ratio'] * layer_cfg['search_radius_multiplier'])
        search_radius_x = int(w * region_cfg['search_radius_x_ratio'] * layer_cfg['search_radius_multiplier'])
        
        # 搜索中心
        center_y = int(h * region_cfg['search_center_y_ratio'])
        center_x = int(w * region_cfg['search_center_x_ratio'])
        
        # 搜索步长
        search_step = max(layer_cfg.get('min_grid_step', 3), 
                         int(min(search_radius_y, search_radius_x) * layer_cfg.get('search_step_ratio', 0.03)))
        
        if self.verbose:
            print(f"     搜索中心: ({center_y}, {center_x})")
            print(f"     搜索半径: {search_radius_y}x{search_radius_x}")
            print(f"     搜索步长: {search_step}")
            print(f"     最大CV: {max_cv:.3f}")
        
        # 执行网格搜索
        candidates = []
        
        # 生成搜索网格
        for dy in range(-search_radius_y, search_radius_y + 1, search_step):
            for dx in range(-search_radius_x, search_radius_x + 1, search_step):
                cand_y = center_y + dy
                cand_x = center_x + dx
                
                # 边界检查
                if not (30 < cand_y < h-30 and 30 < cand_x < w-30):
                    continue
                
                # 尝试不同的ROI尺寸
                for size_factor in UNIFIED_SEARCH_STRATEGY['algorithm_params']['roi_variations']['size_factors']:
                    radius = int(min(h, w) * region_cfg['roi_size_ratio'] * size_factor / 2)
                    
                    # 创建椭圆ROI
                    mask = self._create_ellipse_mask((h, w), cand_y, cand_x, radius)
                    roi_pixels = img[mask]
                    
                    # 像素数量检查
                    if roi_pixels.size < layer_cfg.get('min_pixels', 25):
                        continue
                    
                    # 计算统计
                    mean_val = np.mean(roi_pixels)
                    std_val = np.std(roi_pixels)
                    cv_val = std_val / max(mean_val, 0.1)
                    
                    # 信号范围检查
                    signal_min, signal_max = signal_range
                    if not (signal_min <= mean_val <= signal_max):
                        continue
                    
                    # CV检查
                    if cv_val > max_cv:
                        continue
                    
                    # 计算综合评分
                    score = self._calculate_candidate_score(
                        mean_val, cv_val, cand_y, cand_x, center_y, center_x,
                        search_radius_y, search_radius_x, roi_pixels.size
                    )
                    
                    # 评分阈值检查
                    if score < layer_cfg.get('required_score', 0.5):
                        continue
                    
                    candidates.append({
                        'center': (cand_y, cand_x),
                        'radius': radius,
                        'score': score,
                        'statistics': {
                            'mean': mean_val,
                            'std': std_val,
                            'cv': cv_val,
                            'pixel_count': roi_pixels.size
                        }
                    })
        
        # 如果没有候选点
        if not candidates:
            return {
                'success': False,
                'candidates': [],
                'layer_name': layer_name,
                'reason': f'No candidates found with CV<{max_cv:.3f} and score>{layer_cfg.get("required_score", 0.5)}'
            }
        
        # 选择最佳候选点
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = candidates[0]
        
        if self.verbose:
            print(f"     找到{len(candidates)}个候选点，最佳评分: {best_candidate['score']:.3f}")
            print(f"     最佳位置: {best_candidate['center']}, CV: {best_candidate['statistics']['cv']:.3f}")
        
        # 创建结果
        return self._create_signal_result(img, best_candidate, f'{layer_name}_search')

    def _execute_smart_fallback(self, img: np.ndarray, region_cfg: Dict, 
                               all_candidates: List[Dict], 
                               signal_range: Tuple[float, float]) -> Dict[str, Any]:
        """执行智能回退策略"""
        
        h, w = img.shape
        strategy_cfg = UNIFIED_SEARCH_STRATEGY['fallback_strategy']
        quality_cfg = strategy_cfg['quality_thresholds']
        
        if self.verbose:
            print(f"     总候选点数: {len(all_candidates)}")
        
        # 1. 尝试从候选点中找"勉强可用"的
        acceptable_candidates = []
        for candidate in all_candidates:
            stats = candidate['statistics']
            cv_val = stats['cv']
            
            # CV检查
            if cv_val > quality_cfg['acceptable_cv']:
                continue
            
            # 像素数量检查
            if stats['pixel_count'] < strategy_cfg['selection_rules']['min_pixels_for_acceptable']:
                continue
            
            # 信号均值检查
            signal_min, signal_max = signal_range
            if not (signal_min * 0.8 <= stats['mean'] <= signal_max * 1.2):
                continue
            
            acceptable_candidates.append(candidate)
        
        # 如果存在勉强可用的候选点，选择最好的
        if acceptable_candidates and strategy_cfg['selection_rules']['prefer_acceptable_over_fixed']:
            acceptable_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_acceptable = acceptable_candidates[0]
            
            if self.verbose:
                print(f"     找到{len(acceptable_candidates)}个勉强可用的候选点")
                print(f"     选择最佳: CV={best_acceptable['statistics']['cv']:.3f}, 评分={best_acceptable['score']:.3f}")
            
            result = self._create_signal_result(img, best_acceptable, 'acceptable_candidate_fallback')
            result['fallback_type'] = 'acceptable_candidate'
            return result
        
        # 2. 尝试固定位置
        if self.verbose:
            print(f"     尝试固定位置回退")
        
        fixed_results = []
        fallback_positions = region_cfg.get('fallback_positions', [])
        max_tries = strategy_cfg['selection_rules'].get('max_fixed_position_tries', 3)
        
        for i, pos_cfg in enumerate(fallback_positions[:max_tries]):
            # 计算实际坐标
            pos_y = int(h * pos_cfg['y_ratio'])
            pos_x = int(w * pos_cfg['x_ratio'])
            radius = int(min(h, w) * pos_cfg['roi_size_ratio'] / 2)
            
            # 边界检查
            if not (30 < pos_y < h-30 and 30 < pos_x < w-30):
                continue
            
            # 创建ROI
            mask = self._create_ellipse_mask((h, w), pos_y, pos_x, radius)
            roi_pixels = img[mask]
            
            if roi_pixels.size < 20:
                continue
            
            # 计算统计
            mean_val = np.mean(roi_pixels)
            std_val = np.std(roi_pixels)
            cv_val = std_val / max(mean_val, 0.1)
            
            # CV检查
            if cv_val > quality_cfg['fixed_position_cv']:
                continue
            
            # 评分
            score = self._calculate_candidate_score(
                mean_val, cv_val, pos_y, pos_x, 
                int(h * region_cfg.get('search_center_y_ratio', 0.5)),
                int(w * region_cfg.get('search_center_x_ratio', 0.5)),
                int(h * region_cfg.get('search_radius_y_ratio', 0.15)),
                int(w * region_cfg.get('search_radius_x_ratio', 0.10)),
                roi_pixels.size
            )
            
            fixed_results.append({
                'center': (pos_y, pos_x),
                'radius': radius,
                'score': score,
                'statistics': {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val,
                    'pixel_count': roi_pixels.size
                },
                'position_name': pos_cfg.get('name', f'position_{i}')
            })
        
        # 如果固定位置有可用的
        if fixed_results:
            fixed_results.sort(key=lambda x: x['score'], reverse=True)
            best_fixed = fixed_results[0]
            
            if self.verbose:
                print(f"     找到{len(fixed_results)}个可用的固定位置")
                print(f"     选择: {best_fixed.get('position_name', 'unknown')}, CV={best_fixed['statistics']['cv']:.3f}")
            
            result = self._create_signal_result(img, best_fixed, 'fixed_position_fallback')
            result['fallback_type'] = 'fixed_position'
            result['position_name'] = best_fixed.get('position_name')
            return result
        
        # 3. 最终手段：选择所有候选点中相对最好的
        if all_candidates:
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_overall = all_candidates[0]
            
            if self.verbose:
                print(f"     强制选择最佳候选: CV={best_overall['statistics']['cv']:.3f}")
            
            result = self._create_signal_result(img, best_overall, 'forced_best_fallback')
            result['fallback_type'] = 'forced_best'
            return result
        
        # 4. 完全失败：使用第一个固定位置（即使CV可能很高）
        if fallback_positions:
            pos_cfg = fallback_positions[0]
            pos_y = int(h * pos_cfg['y_ratio'])
            pos_x = int(w * pos_cfg['x_ratio'])
            radius = int(min(h, w) * pos_cfg['roi_size_ratio'] / 2)
            
            # 边界调整
            pos_y = max(30, min(pos_y, h-30))
            pos_x = max(30, min(pos_x, w-30))
            
            mask = self._create_ellipse_mask((h, w), pos_y, pos_x, radius)
            roi_pixels = img[mask]
            
            mean_val = np.mean(roi_pixels) if roi_pixels.size > 0 else 50.0
            std_val = np.std(roi_pixels) if roi_pixels.size > 0 else 10.0
            cv_val = std_val / max(mean_val, 0.1)
            
            if self.verbose:
                print(f"     ⚠️ 完全回退: 使用固定位置")
                print(f"     位置: ({pos_y}, {pos_x}), CV={cv_val:.3f}")
            
            candidate = {
                'center': (pos_y, pos_x),
                'radius': radius,
                'score': 0.1,
                'statistics': {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val,
                    'pixel_count': roi_pixels.size
                }
            }
            
            result = self._create_signal_result(img, candidate, 'emergency_fallback')
            result['fallback_type'] = 'emergency'
            return result
        
        # 5. 绝对失败：使用图像中心
        if self.verbose:
            print(f"     ⚠️ 绝对失败: 使用图像中心")
        
        center_y, center_x = h//2, w//2
        radius = int(min(h, w) * 0.08 / 2)
        
        mask = self._create_ellipse_mask((h, w), center_y, center_x, radius)
        roi_pixels = img[mask]
        
        mean_val = np.mean(roi_pixels) if roi_pixels.size > 0 else 50.0
        std_val = np.std(roi_pixels) if roi_pixels.size > 0 else 10.0
        cv_val = std_val / max(mean_val, 0.1)
        
        candidate = {
            'center': (center_y, center_x),
            'radius': radius,
            'score': 0.05,
            'statistics': {
                'mean': mean_val,
                'std': std_val,
                'cv': cv_val,
                'pixel_count': roi_pixels.size
            }
        }
        
        result = self._create_signal_result(img, candidate, 'absolute_fallback')
        result['fallback_type'] = 'absolute'
        return result

    def _calculate_adaptive_signal_range(self, img: np.ndarray, region_cfg: Dict) -> Tuple[float, float]:
        """计算自适应信号范围 - 基于配置和图像统计"""
        
        # 获取配置的期望范围
        config_min = region_cfg.get('expected_signal_min', 50)
        config_max = region_cfg.get('expected_signal_max', 150)
        
        # 获取图像统计
        img_percentile_75 = np.percentile(img, 75)
        
        # 使用稳健的参考值
        if 'signal_range_reference' in region_cfg:
            ref_cfg = region_cfg['signal_range_reference']
            percentile = ref_cfg.get('percentile_reference', 75)
            use_robust = ref_cfg.get('use_robust_statistics', True)
            
            if use_robust:
                # 使用百分位数避免异常值影响
                reference_value = np.percentile(img, percentile)
            else:
                reference_value = np.mean(img)
        else:
            reference_value = img_percentile_75
        
        # 基于参考值计算范围
        if region_cfg.get('max_allowed_cv', 0.25) < 0.22:
            # 对于要求严格的区域（如腰椎），使用较窄的范围
            image_based_min = reference_value * 0.7
            image_based_max = reference_value * 1.3
        else:
            # 对于一般区域，使用较宽的范围
            image_based_min = reference_value * 0.6
            image_based_max = reference_value * 1.4
        
        # 综合配置和图像统计
        final_min = max(config_min, image_based_min)
        final_max = min(config_max, image_based_max)
        
        # 确保最小范围宽度
        min_range_width = region_cfg.get('signal_range_reference', {}).get('min_range_width', 40)
        if final_max - final_min < min_range_width:
            expand = (min_range_width - (final_max - final_min)) / 2
            final_min = max(30.0, final_min - expand)
            final_max = final_max + expand
        
        return final_min, final_max

    def _calculate_candidate_score(self, mean_val: float, cv_val: float,
                                  cand_y: int, cand_x: int,
                                  target_y: int, target_x: int,
                                  search_radius_y: int, search_radius_x: int,
                                  pixel_count: int) -> float:
        """计算候选点综合评分"""
        
        weights = UNIFIED_SEARCH_STRATEGY['scoring_system']['weights']
        formulas = UNIFIED_SEARCH_STRATEGY['scoring_system']['formulas']
        
        # 1. 均匀性评分
        try:
            uniformity_score = eval(formulas['uniformity'], {
                'cv': cv_val,
                'mean': mean_val
            })
        except:
            uniformity_score = 1.0 / (cv_val + 0.05) if cv_val > 0 else 1.0
        
        # 2. 信号强度评分（使用配置的期望值作为目标）
        target_center = 100.0  # 默认值，实际应该从配置获取
        try:
            intensity_score = eval(formulas['intensity'], {
                'mean': mean_val,
                'target_center': target_center
            })
        except:
            intensity_score = 1.0 - min(1.0, abs(mean_val - target_center) / max(target_center, 50.0))
        
        # 3. 中心性评分
        y_dist = abs(cand_y - target_y) / max(search_radius_y, 1)
        x_dist = abs(cand_x - target_x) / max(search_radius_x, 1)
        try:
            centrality_score = eval(formulas['centrality'], {
                'y_dist': y_dist,
                'x_dist': x_dist
            })
        except:
            centrality_score = max(0.0, 1.0 - (y_dist + x_dist) / 2.0)
        
        # 4. 尺寸评分
        try:
            size_score = eval(formulas['size'], {
                'pixel_count': pixel_count
            })
        except:
            size_score = min(1.0, pixel_count / 100.0)
        
        # 加权综合
        total_score = (
            uniformity_score * weights['uniformity'] +
            intensity_score * weights['intensity'] +
            centrality_score * weights['centrality'] +
            size_score * weights['size']
        )
        
        return total_score

    def _create_signal_result(self, img: np.ndarray, candidate: Dict, method: str) -> Dict[str, Any]:
        """创建信号ROI结果"""
        
        center_y, center_x = candidate['center']
        radius = candidate['radius']
        stats = candidate['statistics']
        
        mask = self._create_ellipse_mask(img.shape, center_y, center_x, radius)
        roi_pixels = img[mask]
        
        # 计算坐标边界
        coords = np.argwhere(mask)
        if len(coords) > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            coordinates = (int(y1), int(y2+1), int(x1), int(x2+1))
        else:
            coordinates = (0, 0, 0, 0)
        
        # 计算均匀性
        uniformity = 1.0 / stats['cv'] if stats['cv'] > 0 else 0.0
        
        result = {
            'mask': mask,
            'coordinates': coordinates,
            'statistics': {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(np.min(roi_pixels)),
                'max': float(np.max(roi_pixels)),
                'pixel_count': int(stats['pixel_count']),
                'cv': float(stats['cv']),
                'uniformity': float(uniformity)
            },
            'selection_method': method,
            'search_info': {
                'candidate_score': float(candidate.get('score', 0.0)),
                'search_center': (center_y, center_x),
                'roi_radius': radius
            },
            'success': True
        }
        
        # 添加额外的候选点信息
        if 'position_name' in candidate:
            result['search_info']['position_name'] = candidate['position_name']
        
        return result

    def _evaluate_signal_quality(self, signal_mean: float, signal_cv: float) -> Dict[str, Any]:
        """评估信号质量"""
        if signal_cv < 0.15:
            quality = 'EXCELLENT'
            confidence = 'HIGH'
        elif signal_cv < 0.25:
            quality = 'GOOD'
            confidence = 'HIGH'
        elif signal_cv < 0.35:
            quality = 'FAIR'
            confidence = 'MEDIUM'
        elif signal_cv < 0.5:
            quality = 'POOR'
            confidence = 'MEDIUM'
        else:
            quality = 'UNACCEPTABLE'
            confidence = 'LOW'
        
        return {
            'quality': quality,
            'confidence': confidence,
            'cv_value': float(signal_cv),
            'mean_value': float(signal_mean),
            'is_acceptable': signal_cv < 0.35
        }

    # ========================================================================
    # 原有方法（保持不变）
    # ========================================================================
    
    def _create_ellipse_mask(self, shape: Tuple[int, int], center_y: int, center_x: int, radius: float) -> np.ndarray:
        h, w = shape
        y, x = np.ogrid[:h, :w]
        dist_squared = (y - center_y)**2 + (x - center_x)**2
        mask = dist_squared <= (radius ** 2)
        return mask

    # 背景检测和噪声估计部分（保持不变）
    def _stage1_global_diagnostic(self, img: np.ndarray) -> Dict[str, Any]:
        """阶段1: 全局诊断扫描"""
        # 使用之前的背景检测逻辑
        self.image_stats = self._get_image_statistics(img)
        h, w = img.shape
        
        all_regions = {}
        for size in [15, 25, 30, 40]:
            corners = self._generate_corner_regions(h, w, size)
            for name, coords in corners:
                region_data = self._analyze_region(img, coords)
                all_regions[f"{name}_{size}"] = region_data

            # 2. 新增：小块边缘检查
        edge_regions = self._generate_edge_regions_small_blocks(h, w, edge_height=10, width_divisor=20)
        for name, coords in edge_regions:
            region_data = self._analyze_region(img, coords)
            all_regions[name] = region_data

        valid_regions = {}
        for region_name, region_data in all_regions.items():
            meets_criteria, _ = self._check_air_criteria(region_data)
            if meets_criteria:
                valid_regions[region_name] = region_data
        
        if len(valid_regions) == 0:
            all_region_list = list(all_regions.items())
            all_region_list.sort(key=lambda x: (x[1]['mean'], x[1]['std']))
            
            if all_region_list:
                best_name, best_data = all_region_list[0]
                
                if best_data['mean'] < 0.1:
                    best_data['mean'] = 0.5
                if best_data['std'] < 0.1:
                    best_data['std'] = 0.5
                
                valid_regions[best_name] = best_data

        return {
            'total_candidates': len(all_regions),
            'valid_air_regions': len(valid_regions),
            'valid_regions': valid_regions,
            'image_statistics': self.image_stats
        }
    def _generate_edge_regions_small_blocks(self, h: int, w: int, edge_height: int = 10, 
                                       width_divisor: int = 20) -> list:
        """生成小块边缘区域
    
        参数:
            h: 图像高度
            w: 图像宽度
            edge_height: 边缘区域高度（像素）
            width_divisor: 宽度分割除数，决定每个小块的宽度
        """
    
        regions = []
    
        # 计算每个小块的宽度
        block_width = max(10, w // width_divisor)  # 至少10像素宽
    
        # 1. 上边缘小块
        num_top_blocks = max(1, w // block_width)
        for i in range(num_top_blocks):
            x1 = i * block_width
            x2 = min((i + 1) * block_width, w)
            regions.append(
                (f'top_edge_{i:02d}', 
                 (0, edge_height, x1, x2))
            )
    
        # 2. 下边缘小块
        for i in range(num_top_blocks):
            x1 = i * block_width
            x2 = min((i + 1) * block_width, w)
            regions.append(
                (f'bottom_edge_{i:02d}',
                 (h - edge_height, h, x1, x2))
            )
    
        # 3. 左边缘小块（需要计算高度分割）
        block_height = max(10, h // width_divisor)  # 使用相同的分割逻辑
        num_left_blocks = max(1, h // block_height)
        for i in range(num_left_blocks):
            y1 = i * block_height
            y2 = min((i + 1) * block_height, h)
            regions.append(
                (f'left_edge_{i:02d}',
                 (y1, y2, 0, edge_height))
            )
    
        # 4. 右边缘小块
        for i in range(num_left_blocks):
            y1 = i * block_height
            y2 = min((i + 1) * block_height, h)
            regions.append(
                (f'right_edge_{i:02d}',
                 (y1, y2, w - edge_height, w))
            )
    
        return regions
    def _generate_edge_regions(self, h: int, w: int, edge_width: int = 10) -> list:
        """生成边缘区域（排除角点重叠区域）
    
        参数:
            h: 图像高度
            w: 图像宽度
            edge_width: 边缘宽度（像素）
    
        返回:
            边缘区域列表，每个元素为 (区域名称, (y1, y2, x1, x2))
        """
    
        regions = []
    
        # 确保边缘宽度合理
        edge_width = min(edge_width, h//4, w//4)
    
        # 1. 上边缘中心（排除左右角点）
        if w > 2 * edge_width:  # 确保有中间区域
            regions.append(
                ('top_edge_middle', 
                 (0, edge_width, edge_width, w - edge_width))
            )
    
        # 2. 下边缘中心（排除左右角点）
        if w > 2 * edge_width:
            regions.append(
                ('bottom_edge_middle',
                 (h - edge_width, h, edge_width, w - edge_width))
            )
    
        # 3. 左边缘中心（排除上下角点）
        if h > 2 * edge_width:
            regions.append(
                ('left_edge_middle',
                 (edge_width, h - edge_width, 0, edge_width))
            )
    
        # 4. 右边缘中心（排除上下角点）
        if h > 2 * edge_width:
            regions.append(
                ('right_edge_middle',
                 (edge_width, h - edge_width, w - edge_width, w))
            )
    
        # 5. 四分之一边缘（更细的划分）
        quarter_w = w // 4
        quarter_h = h // 4
    
        # 上边缘左四分之一
        regions.append(
            ('top_edge_left_q', 
             (0, edge_width, 0, quarter_w))
        )
    
        # 上边缘右四分之一
        regions.append(
            ('top_edge_right_q',
             (0, edge_width, w - quarter_w, w))
        )
    
        # 下边缘左四分之一
        regions.append(
            ('bottom_edge_left_q',
             (h - edge_width, h, 0, quarter_w))
        )
    
        # 下边缘右四分之一
        regions.append(
            ('bottom_edge_right_q',
             (h - edge_width, h, w - quarter_w, w))
        )
    
        return regions
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

    # 背景选择、噪声估计等方法保持不变
    def _stage2_background_selection(self, diagnostic: Dict, img: np.ndarray) -> Dict[str, Any]:
        """阶段2: 背景选择"""
        # 使用之前的背景检测逻辑
        valid_regions = diagnostic['valid_regions']
        
        if valid_regions:
            selected_region = self._select_best_region(valid_regions)
            
            if selected_region:
                meets_criteria = True
                selection_method = "optimized_air_background"
                confidence = "HIGH"
            else:
                selected_region = self._improved_fallback_background(img)
                meets_criteria = False
                selection_method = "improved_fallback"
                confidence = "MEDIUM"
        else:
            selected_region = self._improved_fallback_background(img)
            meets_criteria = False
            selection_method = "improved_fallback_no_valid"
            confidence = "MEDIUM"
        
        y1, y2, x1, x2 = selected_region['coordinates']
        background_values = img[y1:y2, x1:x2].flatten()
        
        if len(background_values) == 0:
            flat_img = img.flatten()
            threshold = np.percentile(flat_img, 1)
            background_values = flat_img[flat_img <= threshold]
            
            if len(background_values) == 0:
                background_values = np.array([1.0, 1.5, 2.0, 2.5])
                selection_method = "emergency_fallback"
        
        mean_val = np.mean(background_values) if len(background_values) > 0 else 2.0
        std_val = np.std(background_values) if len(background_values) > 0 else 1.5
        
        mean_val = max(0.5, mean_val)
        std_val = max(0.5, std_val)
        
        uniformity_cv = std_val / mean_val if mean_val > 0 else 0.5
        
        selected_region['mean'] = float(mean_val)
        selected_region['std'] = float(std_val)
        selected_region['pixel_count'] = len(background_values)
        
        return {
            'region_name': selected_region.get('name', 'selected_region'),
            'coordinates': selected_region['coordinates'],
            'statistics': {
                'mean': float(mean_val),
                'std': float(std_val),
                'pixel_count': selected_region['pixel_count'],
                'uniformity_cv': float(uniformity_cv)
            },
            'is_all_zeros': False,
            'is_too_pure': mean_val < 0.5,
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
            
            mean_score = 0.0
            if 0.5 <= mean_val <= 10.0:
                mean_score = 1.0 - abs(mean_val - 3.0) / 7.0
            elif 10.0 < mean_val <= 20.0:
                mean_score = 0.4 - (mean_val - 10.0) * 0.04
            else:
                mean_score = 0.1
            
            std_score = 0.0
            if 0.5 <= std_val <= 5.0:
                std_score = 1.0 - abs(std_val - 2.5) / 4.5
            elif 5.0 < std_val <= 8.0:
                std_score = 0.3 - (std_val - 5.0) * 0.1
            else:
                std_score = 0.1
            
            if mean_val > 0.1:
                cv_val = std_val / mean_val
                if cv_val <= 0.8:
                    uniformity_score = 1.0 - cv_val * 1.25
                elif cv_val <= 1.2:
                    uniformity_score = 0.2 - (cv_val - 0.8) * 0.5
                else:
                    uniformity_score = 0.1
            else:
                uniformity_score = 0.1
            
            size_score = min(1.0, np.log(data['pixel_count'] + 1) / 5.0)
            
            total_score = (mean_score * 0.4 +
                          std_score * 0.25 +
                          uniformity_score * 0.25 +
                          size_score * 0.10)
            
            if total_score < 0.2:
                continue
            
            scored_regions.append((total_score, name, data))
        
        if not scored_regions:
            return None
        
        scored_regions.sort(key=lambda x: x[0], reverse=True)
        best_score, best_name, best_data = scored_regions[0]
        
        if best_score < 0.3:
            return None
        
        return {'name': best_name, 'score': best_score, **best_data}

    def _improved_fallback_background(self, img: np.ndarray) -> Dict:
        import numpy as np
        h, w = img.shape
        
        corner_size = 40
        corners = [
            ('top_left', (0, corner_size, 0, corner_size)),
            ('top_right', (0, corner_size, w-corner_size, w)),
            ('bottom_left', (h-corner_size, h, 0, corner_size)),
            ('bottom_right', (h-corner_size, h, w-corner_size, w))
        ]
        
        best_region = None
        best_uniformity = float('inf')
        
        for name, coords in corners:
            y1, y2, x1, x2 = coords
            if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                region = img[y1:y2, x1:x2].flatten()
                if len(region) > 20:
                    mean_val = np.mean(region)
                    std_val = np.std(region)
                    cv_val = std_val / mean_val if mean_val > 0 else float('inf')
                    
                    if mean_val < 15 and cv_val < best_uniformity:
                        best_uniformity = cv_val
                        best_region = {
                            'name': f"corner_{name}",
                            'coordinates': coords,
                            'mean': mean_val,
                            'std': std_val,
                            'pixel_count': len(region),
                            'is_all_zeros': False,
                            'is_too_pure': mean_val < 0.5,
                            'uniformity_cv': cv_val
                        }
        
        if best_region and best_uniformity < 0.5:
            return best_region
        
        flat_img = img.flatten()
        threshold = np.percentile(flat_img, 5)
        dark_pixels = flat_img[flat_img <= threshold]
        
        if len(dark_pixels) > 50:
            mean_val = np.mean(dark_pixels)
            std_val = np.std(dark_pixels)
            cv_val = std_val / mean_val if mean_val > 0 else 1.0
            
            return {
                'name': "darkest_5_percent",
                'coordinates': (0, h, 0, w),
                'mean': mean_val,
                'std': std_val,
                'pixel_count': len(dark_pixels),
                'is_all_zeros': False,
                'is_too_pure': mean_val < 0.5,
                'uniformity_cv': cv_val
            }
        
        y1, y2, x1, x2 = 0, corner_size, 0, corner_size
        region = img[y1:y2, x1:x2].flatten()
        mean_val = np.mean(region) if len(region) > 0 else 2.0
        std_val = np.std(region) if len(region) > 0 else 1.5
        
        mean_val = max(0.5, mean_val)
        std_val = max(0.5, std_val)
        
        return {
            'name': "guaranteed_corner",
            'coordinates': (y1, y2, x1, x2),
            'mean': mean_val,
            'std': std_val,
            'pixel_count': len(region),
            'is_all_zeros': False,
            'is_too_pure': mean_val < 0.5,
            'uniformity_cv': std_val / mean_val if mean_val > 0 else 0.5
        }

    # 噪声估计方法（保持不变）
    def _stage3_noise_estimation(self, background: Dict, img: np.ndarray) -> Dict[str, Any]:
        """阶段3: 噪声估计"""
        import numpy as np
        
        bg_values = np.array(background['background_values'])
        
        if len(bg_values) > 10:
            q1 = np.percentile(bg_values, 25)
            q3 = np.percentile(bg_values, 75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_mask = (bg_values >= lower_bound) & (bg_values <= upper_bound)
                filtered_values = bg_values[filtered_mask]
                
                if len(filtered_values) > max(20, len(bg_values) * 0.5):
                    bg_values = filtered_values
                    if self.verbose:
                        print(f"  背景数据过滤: {len(bg_values)} -> {len(filtered_values)} 像素")
        
        if len(bg_values) == 0:
            return self._estimate_noise_from_zero_background(img)
        
        bg_mean = float(np.mean(bg_values))
        bg_std_raw = float(np.std(bg_values))
        
        if self.verbose:
            print(f"  背景统计: 均值={bg_mean:.2f}, 原始标准差={bg_std_raw:.2f}")
        
        estimates = []
        methods_info = []
        
        try:
            sigma_mle_squared = np.sum(bg_values ** 2) / (2 * len(bg_values))
            noise_mle = np.sqrt(sigma_mle_squared)
            if np.isfinite(noise_mle) and noise_mle > 0:
                estimates.append(('mle', noise_mle))
                methods_info.append(('mle', noise_mle, 'Rayleigh MLE'))
        except Exception as e:
            if self.verbose:
                print(f"  MLE失败: {e}")
        
        try:
            q75 = np.percentile(bg_values, 75)
            q25 = np.percentile(bg_values, 25)
            iqr = q75 - q25
            if iqr > 0:
                noise_iqr = iqr / 1.349
                if np.isfinite(noise_iqr) and noise_iqr > 0:
                    estimates.append(('iqr', noise_iqr))
                    methods_info.append(('iqr', noise_iqr, 'IQR robust'))
        except Exception as e:
            if self.verbose:
                print(f"  IQR估计失败: {e}")
        
        try:
            median_val = np.median(bg_values)
            abs_diff = np.abs(bg_values - median_val)
            mad_raw = np.median(abs_diff)
            if mad_raw > 1e-10:
                noise_mad = mad_raw / 0.6745
                if np.isfinite(noise_mad) and noise_mad > 0:
                    estimates.append(('mad', noise_mad))
                    methods_info.append(('mad', noise_mad, 'MAD robust'))
        except Exception as e:
            if self.verbose:
                print(f"  MAD估计失败: {e}")
        
        if bg_std_raw > 0 and np.isfinite(bg_std_raw):
            estimates.append(('std', bg_std_raw))
            methods_info.append(('std', bg_std_raw, 'Raw std'))
        
        if len(estimates) == 0:
            fallback_result = self._estimate_noise_from_pure_background(img, bg_values)
            final_noise = fallback_result['noise_std']
            estimation_method = 'fallback_hybrid'
        elif len(estimates) == 1:
            final_noise = estimates[0][1]
            estimation_method = f"single_{estimates[0][0]}"
        else:
            noise_vals = [val for _, val in estimates]
            final_noise = float(np.median(noise_vals))
            if 'mle' in [name for name, _ in estimates]:
                estimation_method = 'consensus_with_mle'
            else:
                estimation_method = 'consensus_robust'
        
        final_noise = self._apply_realistic_noise_bounds(final_noise, img)
        correction_factor = final_noise / bg_std_raw if bg_std_raw > 0 else 1.0
        
        if self.verbose:
            print(f"  噪声估计方法: {estimation_method}")
            print(f"  最终噪声: {final_noise:.3f}")
            print(f"  校正因子: {correction_factor:.3f}")
        
        return {
            'noise_std': final_noise,
            'background_std_raw': bg_std_raw,
            'background_mean': bg_mean,
            'estimation_method': estimation_method,
            'confidence': self._get_noise_confidence(final_noise, bg_std_raw, len(bg_values)),
            'pixel_count': len(bg_values),
            'correction_factor': correction_factor,
        }

    def _get_noise_confidence(self, noise_std: float, bg_std_raw: float, pixel_count: int) -> str:
        if pixel_count < 30:
            return "LOW"
        
        if bg_std_raw > 0:
            correction_factor = noise_std / bg_std_raw
            if 0.5 < correction_factor < 1.5:
                confidence = "HIGH"
            elif 0.3 < correction_factor < 2.0:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            confidence = "MEDIUM"
        
        if noise_std < 0.5 or noise_std > 10.0:
            confidence = min(confidence, "MEDIUM")
        
        return confidence

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
        img_std = np.std(img)
        MIN_NOISE = max(0.5, img_std * 0.01)
        MAX_NOISE = min(6.0, img_std * 0.1)
        bounded_noise = max(MIN_NOISE, min(noise_estimate, MAX_NOISE))
        
        if self.verbose and abs(bounded_noise - noise_estimate) / noise_estimate > 0.5:
            print(f"  噪声边界调整: {noise_estimate:.3f} -> {bounded_noise:.3f}")
        
        return bounded_noise


# ========================================================================
# 便捷函数（保持不变）
# ========================================================================
def analyze_single_image(image: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
    engine = SingleImageV3Core(verbose=verbose)
    return engine.analyze_image(image)

def analyze_single_nifti(nifti_path: str, verbose: bool = False) -> Dict[str, Any]:
    engine = SingleImageV3Core(verbose=verbose)
    return engine.analyze_nifti(nifti_path)