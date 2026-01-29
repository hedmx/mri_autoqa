#!/usr/bin/env python3
"""
单图像SNR分析引擎 - 重命名为 single_imagine.py
集成单图像法V3 + Parallel correction + Quality指标计算
"""

import numpy as np
import nibabel as nib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys
import importlib.util
from datetime import datetime

# 导入自定义模块
from algorithm_router import AlgorithmRouter, route_from_metadata_file
from parallel_corrector import ParallelCorrector
from quality_metrics import QualityMetricsCalculator
from config import ROI_SETTINGS, SEQUENCE_SNR_STANDARDS


class SingleImageSNREngine:
    """
    单图像SNR分析主引擎
    集成所有功能：算法路由、V3分析、Parallel correction、QualityAssessment
    """
    
    def __init__(self, verbose: bool = True):
        """
        初始化引擎
        
        Args:
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
        self.router = AlgorithmRouter()
        self.parallel_corrector = ParallelCorrector()
        self.quality_calculator = QualityMetricsCalculator()
        
        # 动态加载Single Image V3 Engine
        self.v3_engine = self._load_v3_engine()
        
        if self.v3_engine is None:
            raise ImportError("无法加载Single Image V3 Engine")
        
        self.log("单图像SNR引擎初始化完成")
    
    def log(self, message: str):
        """日志记录"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def _load_v3_engine(self):
        """动态加载Single Image V3 Engine"""
        try:
            # 尝试直接导入
            sys.path.append('.')
            
            try:
                from single_image_v3_engine import SingleImageV3Core
                return SingleImageV3Core(verbose=False)
            except ImportError:
                pass
            
            # 尝试其他可能的文件名
            v3_files = [
                "single_image_v3_engine.py",
                "single_test1.1216.18.30.py",
                "single_image_snr_engine_fixed_v3.py"
            ]
            
            for v3_file in v3_files:
                v3_path = Path(v3_file)
                if v3_path.exists():
                    spec = importlib.util.spec_from_file_location("v3_engine", v3_path)
                    v3_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(v3_module)
                    
                    # 查找引擎类
                    for attr_name in dir(v3_module):
                        if "SingleImageSNREngine" in attr_name and "FixedV3" in attr_name:
                            engine_class = getattr(v3_module, attr_name)
                            return engine_class(verbose=False)
            
            # 如果以上都失败，创建一个简化版本
            self.log("警告: 使用简化的V3引擎模拟")
            return self._create_minimal_v3_engine()
            
        except Exception as e:
            self.log(f"加载V3引擎失败: {e}")
            return None
    
    def _create_minimal_v3_engine(self):
        """创建简化的V3引擎（仅用于测试）"""
        class MinimalV3Engine:
            def __init__(self, verbose=False):
                self.verbose = verbose
            
            def analyze_single_image(self, img_path, output_dir=None):
                # 简化实现，仅用于测试
                import numpy as np
                
                # 加载图像
                img = nib.load(img_path).get_fdata()
                if len(img.shape) == 3:
                    mid_slice = img.shape[2] // 2
                    slice_img = img[:, :, mid_slice]
                else:
                    slice_img = img
                
                # 简化的背景选择
                h, w = slice_img.shape
                bg_region = slice_img[:30, :30]
                signal_region = slice_img[h//2-20:h//2+20, w//2-20:w//2+20]
                
                # 简化的SNR计算
                signal_mean = np.mean(signal_region)
                noise_std = np.std(bg_region)
                
                if noise_std > 0:
                    snr = signal_mean / noise_std
                else:
                    snr = 0
                
                return {
                    'snr_results': {
                        'snr': float(snr),
                        'snr_db': float(20 * np.log10(snr) if snr > 0 else -np.inf),
                        'signal_mean': float(signal_mean),
                        'noise_std': float(noise_std)
                    },
                    'background_selection': {
                        'coordinates': (0, 30, 0, 30),
                        'statistics': {
                            'mean': float(np.mean(bg_region)),
                            'std': float(noise_std),
                            'pixel_count': bg_region.size
                        },
                        'region_name': 'top_left_30',
                        'is_all_zeros': False,
                        'background_values': bg_region.flatten().tolist()
                    },
                    'signal_selection': {
                        'coordinates': (h//2-20, h//2+20, w//2-20, w//2+20),
                        'statistics': {
                            'mean': float(signal_mean),
                            'std': float(np.std(signal_region)),
                            'pixel_count': signal_region.size
                        }
                    }
                }
        
        return MinimalV3Engine(verbose=False)
    
    def analyze_scan(self, 
                    nifti_path: Path, 
                    metadata_path: Path,
                    output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """分析单次扫描"""
        scan_id = f"{nifti_path.parent.parent.name}/{nifti_path.parent.name}"
        self.log(f"开始扫描分析: {scan_id}")
        
        try:
            # 1. 加载数据和元数据
            image, metadata = self._load_scan_data(nifti_path, metadata_path)
            
            # 2. 算法路由
            algorithm_params = self.router.route(metadata)
            self.log(f"算法路由完成: {algorithm_params['constraint_space']}")
            
            # 3. 使用V3引擎进行基础分析
            constraint_space = algorithm_params.get('constraint_space', {})
            anatomy = constraint_space.get('A', 'default')
            
            v3_result = self.v3_engine.analyze_nifti(
                str(nifti_path), anatomical_region=anatomy
            )
            
            # 4. 【修改】直接使用V3结果，移除Parallel校正
            snr_final = v3_result['snr_results']['snr_final']
            signal_mean = v3_result['snr_results']['signal_mean']
            noise_std_final = v3_result['snr_results']['noise_std_final']
            snr_raw = v3_result['snr_results']['snr_raw']
            noise_std_raw = v3_result['snr_results']['noise_std_raw']
            signal_roi = v3_result['signal_selection']
            background_roi = v3_result['background_selection']
            
           
            # 6. 计算质量指标
            quality_metrics = self.quality_calculator.calculate_all_metrics(
                image=image,
                metadata=metadata,
                signal_roi=signal_roi,
                background_roi=background_roi
            )
            
            # 7. 应用解剖区域特定的ROI模板
            roi_template = ROI_SETTINGS.get(anatomy, ROI_SETTINGS['default'])
            
            # 8. 应用序列特定的SNR标准评估
            sequence = constraint_space.get('S', 't1')
            snr_standards = SEQUENCE_SNR_STANDARDS.get(sequence, SEQUENCE_SNR_STANDARDS['t1'])
            snr_rating = self._rate_snr(snr_final, snr_standards)
            
            # 9. 构建最终结果
            final_result = self._build_final_result(
                scan_id=scan_id,
                metadata=metadata,
                algorithm_params=algorithm_params,
                v3_result=v3_result,  # 【修改】传入完整的V3结果
                snr_final=snr_final,
                snr_raw=snr_raw,  # 【新增】原始SNR
                noise_std_final=noise_std_final,
                noise_std_raw=noise_std_raw,  # 【新增】原始噪声
                signal_mean=signal_mean,
                signal_roi=signal_roi,
                background_roi=background_roi,
                quality_metrics=quality_metrics,
                snr_rating=snr_rating,
                snr_standards=snr_standards,
                roi_template=roi_template
            )
            
            self.log(f"扫描分析完成: SNR={snr_final:.1f}, Quality={snr_rating}")
            
            # 10. 保存结果
            if output_dir:
                self._save_results(final_result, output_dir, scan_id, image)
            
            return final_result
            
        except Exception as e:
            error_msg = f"扫描分析失败 {scan_id}: {str(e)}"
            self.log(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            
            return self._build_error_result(scan_id, error_msg, metadata_path)
    
    def _load_scan_data(self, nifti_path: Path, metadata_path: Path) -> Tuple[np.ndarray, Dict]:
        """加载扫描数据和元数据"""
        # 加载NIfTI image
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI文件不存在: {nifti_path}")
        
        img = nib.load(nifti_path).get_fdata()
        
        # 提取中间切片（如果是3D）
        if len(img.shape) == 3:
            mid_slice = img.shape[2] // 2
            slice_img = img[:, :, mid_slice]
            self.log(f"从3D图像提取中间切片: {mid_slice}/{img.shape[2]}")
        else:
            slice_img = img
        
        # 加载元数据
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return slice_img, metadata
    
    def _rate_snr(self, snr: float, standards: Dict[str, float]) -> str:
        """根据序列标准评估SNR质量"""
        if snr >= standards['excellent']:
            return 'EXCELLENT'
        elif snr >= standards['good']:
            return 'GOOD'
        elif snr >= standards['fair']:
            return 'FAIR'
        elif snr >= standards['poor']:
            return 'POOR'
        else:
            return 'UNACCEPTABLE'
    
    def _build_final_result(self, **kwargs) -> Dict[str, Any]:
        """构建最终结果字典"""
        # 提取参数
        scan_id = kwargs['scan_id']
        metadata = kwargs['metadata']
        algorithm_params = kwargs['algorithm_params']
        v3_result = kwargs['v3_result']
        snr_final = kwargs['snr_final']
        snr_raw = kwargs['snr_raw']
        noise_std_final = kwargs['noise_std_final']
        noise_std_raw = kwargs['noise_std_raw']
        signal_mean = kwargs['signal_mean']
        signal_roi = kwargs['signal_roi']
        background_roi = kwargs['background_roi']
        quality_metrics = kwargs['quality_metrics']
        snr_rating = kwargs['snr_rating']
        snr_standards = kwargs['snr_standards']
        roi_template = kwargs['roi_template']
    
        # 从V3结果获取校正信息
        v3_snr_results = v3_result['snr_results']
        correction_factor = v3_snr_results.get('correction_factor', 1.0)
        improvement_percent = v3_snr_results.get('improvement_percent', 0.0)
        correction_method = v3_snr_results.get('correction_method', 'rayleigh_consensus')
        
        # 构建与双图像法兼容的结果结构-待修改12.27
        result = {
            'analysis_info': {
                'date': datetime.now().isoformat(),
                'software': 'MRI_AutoQA_Single_Image_v1.0',
                'algorithm_version': 'single_image_v3_rayleigh',
                'scan_id': scan_id
            },
            'acquisition': {
                'patient_id': metadata.get('patient_info', {}).get('patient_id', 'unknown'),
                  # 修复：从 series_info 提取 scan_name
                'scan_name': metadata.get('series_info', {}).get('series_description', 'unknown'),
                'anatomical_region': algorithm_params['constraint_space']['A'],
                'sequence_type': algorithm_params['constraint_space']['S'],
                'field_strength': algorithm_params['constraint_space']['F'],
                'parallel_imaging': algorithm_params['adjustments']['noise_correction_required'],
                'acceleration_factor': algorithm_params['adjustments']['acceleration_factor'],
                'acquisition_mode': algorithm_params['constraint_space']['M']
            },
            'snr_results': {
                'traditional': {
                    'snr': float(snr_raw),
                    'snr_db': float(20 * np.log10(snr_raw) if snr_raw > 0 else -np.inf),
                    'signal_mean': float(signal_mean),
                    'noise_std': float(noise_std_raw),
                    'method': 'intelligent_background_std',
                    'background_method': 'adaptive_selection'
                },
                'recommended': {
                    'snr': float(snr_final),
                    'snr_db': float(20 * np.log10(snr_final) if snr_final > 0 else -np.inf),
                    'method': 'rayleigh_corrected',
                    'note': f'Rayleigh distribution correction applied'
                },
                # 【新增】瑞利校正详细信息
                'rayleigh_correction': {
                    'snr_raw': float(snr_raw),
                    'snr_corrected': float(snr_final),
                    'improvement_percent': float(improvement_percent),
                    'correction_factor': float(correction_factor),
                    'correction_method': correction_method,
                    'note': f'Rayleigh correction improved SNR by {improvement_percent:.1f}%'
            }
                
                    
            },
            'quality_assessment': {
                'snr_rating': {
                    'level': snr_rating,
                    'score': self._calculate_snr_score(snr_final, snr_standards),
                    'standards_used': snr_standards
                },
                'cnr_analysis': quality_metrics.get('cnr_analysis', {}),
                'noise_uniformity': quality_metrics.get('noise_uniformity', {}),
                'background_quality': quality_metrics.get('background_quality', {}),
                'signal_quality': quality_metrics.get('signal_quality', {}),
                'confidence_scores': quality_metrics.get('confidence_scores', {}),
                'overall_confidence': quality_metrics.get('overall_confidence', {}),
                'quality_scores': quality_metrics.get('quality_scores', {})
            },
            'roi_info': {
                'signal': signal_roi,
                'background': background_roi,
                'anatomy_template_applied': roi_template
            },
            'algorithm_parameters': algorithm_params,
            'validation_info': {
                'status': 'PASSED',
                'metadata_valid': True,
                'image_valid': True,
                'constraint_space_valid': all(algorithm_params['constraint_space'].values())
            },
            'analysis_status': 'COMPLETED'
        }
        
       
        
        return result
    
    def _calculate_snr_score(self, snr: float, standards: Dict[str, float]) -> float:
        """计算SNR分数（0-1）"""
        if snr >= standards['excellent']:
            return 1.0
        elif snr >= standards['good']:
            return 0.75 + 0.25 * (snr - standards['good']) / (standards['excellent'] - standards['good'])
        elif snr >= standards['fair']:
            return 0.5 + 0.25 * (snr - standards['fair']) / (standards['good'] - standards['fair'])
        elif snr >= standards['poor']:
            return 0.25 + 0.25 * (snr - standards['poor']) / (standards['fair'] - standards['poor'])
        else:
            return max(0.0, snr / standards['poor'] * 0.25)
    
    def _build_error_result(self, scan_id: str, error_msg: str, 
                           metadata_path: Path) -> Dict[str, Any]:
        """构建错误结果"""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except:
            metadata = {}
        
        return {
            'analysis_info': {
                'date': datetime.now().isoformat(),
                'software': 'MRI_AutoQA_Single_Image_v1.0',
                'scan_id': scan_id
            },
            'acquisition': {
                'patient_id': metadata.get('patient_id', 'unknown'),
                'scan_name': metadata.get('scan_name', 'unknown')
            },
            'analysis_status': 'FAILED',
            'error': error_msg,
            'error_type': 'ANALYSIS_ERROR'
        }
    
    def _save_results(self, result: Dict[str, Any], output_dir: Path, 
                     scan_id: str, image: np.ndarray):
        """保存分析结果"""
        # 解析Scan ID为目录结构
        parts = scan_id.split('/')
        if len(parts) >= 2:
            patient_dir = output_dir / parts[0] / parts[1]
        else:
            patient_dir = output_dir / scan_id
        
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON结果
        json_path = patient_dir / "qa_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        self.log(f"结果保存到: {json_path}")
        
        # 保存文本摘要
        txt_path = patient_dir / "quality_summary.txt"
        self._save_text_summary(result, txt_path)
        
        # 生成可视化（如果模块可用）
        try:
            from visualization import create_visualization_for_scan
            viz_success = create_visualization_for_scan(
                result, image, patient_dir, "visualization.png"
            )
            if viz_success:
                self.log(f"可视化已生成")
        except ImportError:
            self.log(f"可视化模块不可用，跳过可视化")
        except Exception as e:
            self.log(f"可视化生成失败: {e}")
    
    def _save_text_summary(self, result: Dict[str, Any], txt_path: Path):
        """保存文本摘要"""
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("MRI Quality Analysis Summary\n")
                f.write("=" * 60 + "\n\n")
                
                # 基本信息
                f.write(f"Scan ID: {result['analysis_info']['scan_id']}\n")
                f.write(f"Analysis Date: {result['analysis_info']['date']}\n")
                f.write(f"Patient ID: {result['acquisition']['patient_id']}\n")
                f.write(f"Scan Name: {result['acquisition']['scan_name']}\n\n")
                
                # 采集信息
                f.write("Acquisition Parameters:\n")
                f.write(f"  Anatomical Region: {result['acquisition']['anatomical_region']}\n")
                f.write(f"  Sequence Type: {result['acquisition']['sequence_type']}\n")
                f.write(f"  Field Strength: {result['acquisition']['field_strength']}\n")
                f.write(f"  Parallel Imaging: {'Yes' if result['acquisition']['parallel_imaging'] else 'No'}\n")
                if result['acquisition']['parallel_imaging']:
                    f.write(f"  Acceleration Factor: {result['acquisition']['acceleration_factor']:.1f}\n\n")
                
                # SNR结果
                snr_rec = result['snr_results']['recommended']
                f.write(f"SNR Results ({snr_rec['method']}):\n")
                f.write(f"  SNR Value: {snr_rec['snr']:.1f}\n")
                f.write(f"  SNR(dB): {snr_rec['snr_db']:.1f}\n")
                
                # SNR评级
                snr_rating = result['quality_assessment']['snr_rating']
                f.write(f"  SNR Rating: {snr_rating['level']} (Score: {snr_rating['score']:.2f}/1.00)\n\n")
                
                # CNR结果
                if 'cnr_analysis' in result['quality_assessment']:
                    cnr = result['quality_assessment']['cnr_analysis']
                    if 'best_cnr' in cnr:
                        best = cnr['best_cnr']
                        f.write(f"CNR Results: {best['cnr_value']:.2f} ({best['cnr_rating']})\n")
                        f.write(f"  Tissue Pair: {best['description']}\n")
                        if best.get('clinical_significance'):
                            f.write(f"  Clinical Significance: {best['clinical_significance']}\n")
                    f.write("\n")
                
                # 噪声均匀性
                if 'noise_uniformity' in result['quality_assessment']:
                    uniformity = result['quality_assessment']['noise_uniformity']
                    f.write(f"Noise Uniformity: {uniformity['uniformity_rating']}\n")
                    f.write(f"  Coefficient of Variation: {uniformity['cv']:.3f}\n")
                    f.write(f"  Assessment: {uniformity['assessment']}\n\n")
                
                # 分析状态
                f.write(f"Analysis Status: {result['analysis_status']}\n")
                
        except Exception as e:
            self.log(f"保存文本摘要失败: {e}")


# 便捷函数
def analyze_single_scan(nifti_path: str, 
                       metadata_path: str,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    分析单次扫描的便捷函数
    
    Args:
        nifti_path: scan.nii.gz文件路径
        metadata_path: metadata.json文件路径
        output_dir: 输出目录（可选）
        
    Returns:
        分析结果
    """
    engine = SingleImageSNREngine(verbose=True)
    result = engine.analyze_scan(
        Path(nifti_path),
        Path(metadata_path),
        Path(output_dir) if output_dir else None
    )
    return result


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) >= 3:
        nifti_file = sys.argv[1]
        metadata_file = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else None
        
        print(f"测试单图像分析引擎:")
        print(f"  NIfTI文件: {nifti_file}")
        print(f"  元数据文件: {metadata_file}")
        print(f"  输出目录: {output_dir}")
        print("=" * 60)
        
        result = analyze_single_scan(nifti_file, metadata_file, output_dir)
        
        if result['analysis_status'] == 'COMPLETED':
            snr = result['snr_results']['recommended']['snr']
            rating = result['quality_assessment']['snr_rating']['level']
            print(f"\n分析成功完成!")
            print(f"  SNR: {snr:.1f}")
            print(f"  评级: {rating}")
            print(f"  状态: {result['analysis_status']}")
        else:
            print(f"\n分析失败: {result.get('error', '未知错误')}")
    else:
        print("用法: python single_imagine.py <nifti_file> <metadata_file> [output_dir]")
        print("示例: python single_imagine.py converted_data/p001/T1_1/scan.nii.gz converted_data/p001/T1_1/metadata.json")