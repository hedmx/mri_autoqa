#!/usr/bin/env python3
"""
批量Processing器 - 从converted_data到autoqa_results的自动化流水线
简化版：移除了不必要的统计报告文件
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import sys

from config import (
    SCAN_FILENAME, METADATA_FILENAME, OUTPUT_DIR, INPUT_DIR,
    CSV_FIELDS, BATCH_PROCESSING
)
from single_imagine import SingleImageSNREngine, analyze_single_scan


class BatchProcessor:
    """
    批量ProcessingMRI扫描
    自动Foundconverted_data中的扫描，运行分析，生成autoqa_results
    """
    
    def __init__(self, 
                 input_dir: str = INPUT_DIR,
                 output_dir: str = OUTPUT_DIR,
                 verbose: bool = True):
        """
        初始化批量Processing器
        
        Args:
            input_dir: Input directory（converted_data）
            output_dir: Output directory（autoqa_results）
            verbose: YesNo显示详细日志
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.engine = SingleImageSNREngine(verbose=verbose)
        
        # 统计信息
        self.stats = {
            'total_scans': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
        
        # 结果存储
        self.results = {}
        self.errors = []
        
        self.log(f"Batch processor initialized")
        self.log(f"Input directory: {self.input_dir.absolute()}")
        self.log(f"Output directory: {self.output_dir.absolute()}")
    
    def process_all(self) -> Dict[str, Any]:
        """
        Processing所有扫描
        
        Returns:
            批量Processing统计信息
        """
        self.stats['start_time'] = datetime.now()
        self.log(f"Starting batch processing: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        try:
            # 1. Found所有扫描
            scans = self.discover_scans()
            
            if not scans:
                self.log("No scans found。Please check input directory structure。")
                return self.stats
            
            self.stats['total_scans'] = len(scans)
            self.log(f"Found {len(scans)} scans")
            
            # 2. Processing每scans
            for i, scan_info in enumerate(scans, 1):
                self._process_single_scan(scan_info, i, len(scans))
            
            # 3. 生成汇总报告 - 简化版
            # 移除了：batch_summary.json, processing_statistics.txt等
            
        except Exception as e:
            self.log(f"Error during batch processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            self.stats['duration_seconds'] = duration
            
            self.log(f"Batch processing completed")
            self.log(f"Total duration: {duration:.1f}seconds")
            self._print_final_stats()
        
        return self.stats
    
    def discover_scans(self) -> List[Dict[str, Any]]:
        """
        Found所有可用的扫描
        
        Returns:
            扫描信息列表，每个包含：
            - scan_dir: 扫描目录Path
            - nifti_path: scan.nii.gzPath
            - metadata_path: metadata.jsonPath
            - patient_id: Patient ID
            - scan_name: Scan Name
        """
        scans = []
        
        if not self.input_dir.exists():
            self.log(f"Error: Input directory不存在: {self.input_dir}")
            return scans
        self.log(f"正在扫描目录: {self.input_dir.absolute()}")
         # 使用递归查找所有符合条件的扫描
         # 查找所有 scan.nii.gz 文件
        nifti_files = list(self.input_dir.rglob(SCAN_FILENAME))
    
        self.log(f"找到 {len(nifti_files)} 个 NIfTI 文件")
    
        for nifti_path in nifti_files:
            try:
                # 获取扫描目录（nifti文件所在目录）
                scan_dir = nifti_path.parent
            
                 # 检查对应的Metadata FileYesNo存在
                metadata_path = scan_dir / METADATA_FILENAME
                if not metadata_path.exists():
                    self.log(f"跳过 {scan_dir}: 缺少 {METADATA_FILENAME}")
                    self.stats['skipped'] += 1
                    continue
            
                 # 提取Patient ID和Scan Name
                 # 尝试从目录结构中推断
                relative_path = scan_dir.relative_to(self.input_dir)
                path_parts = list(relative_path.parts)
            
                if len(path_parts) >= 2:
                  # 假设结构: patient_id/scan_name/
                    patient_id = path_parts[0]
                    scan_name = path_parts[1]
                elif len(path_parts) == 1:
                     # 只有一层: scan_name/
                    patient_id = "unknown_patient"
                    scan_name = path_parts[0]
                else:
                    # 直接在根目录
                    patient_id = "unknown_patient"
                    scan_name = scan_dir.name
            
                     # 尝试从Metadata File中获取更准确的信息
                try:
                    import json
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                # 优先使用Metadata中的信息
                    meta_patient_id = metadata.get('patient_id')
                    if meta_patient_id and meta_patient_id != 'unknown':
                        patient_id = meta_patient_id
                
                    meta_scan_name = metadata.get('scan_name')
                    if meta_scan_name and meta_scan_name != 'unknown':
                        scan_name = meta_scan_name
                    
                except Exception as e:
                    self.log(f"读取 {metadata_path} MetadataFailed: {e}")
                    # 继续使用目录推断的信息
            
                   # 添加到Scan List
                scans.append({
                    'scan_dir': scan_dir,
                    'nifti_path': nifti_path,
                    'metadata_path': metadata_path,
                    'patient_id': patient_id,
                    'scan_name': scan_name,
                    'scan_id': f"{patient_id}/{scan_name}"
                })
            
                self.log(f"找到扫描: {patient_id}/{scan_name} ({scan_dir})")
            
            except Exception as e:
                self.log(f"Processing {nifti_path} 时出错: {e}")
                continue
    
        return scans


       
    def _process_single_scan(self, scan_info: Dict[str, Any], 
                           current: int, total: int):
        """
        Process single scan
        
        Args:
            scan_info: 扫描信息字典
            current: 当前扫描序号
            total: Total Scans
        """
        scan_id = scan_info['scan_id']
        patient_id = scan_info['patient_id']
        scan_name = scan_info['scan_name']
        
        print(f"\n[{current}/{total}] Processing: {scan_id}")
        try:
            nifti_rel = scan_info['nifti_path'].relative_to(self.input_dir)
            metadata_rel = scan_info['metadata_path'].relative_to(self.input_dir)
            print(f"  NIfTI: {nifti_rel}")
            print(f"  Metadata: {metadata_rel}")
        except:
            # 如果无法计算相对Path，显示绝对Path
            print(f"  NIfTI: {scan_info['nifti_path']}")
            print(f"  Metadata: {scan_info['metadata_path']}")
        
        
        try:
            # 运行分析
            result = self.engine.analyze_scan(
                scan_info['nifti_path'],
                scan_info['metadata_path'],
                self.output_dir
            )
            
            # 检查Analysis Status
            if result.get('analysis_status') == 'COMPLETED':
                self.results[scan_id] = result
                self.stats['successful'] += 1
                
                # 提取关键信息用于日志
                snr = result['snr_results']['recommended']['snr']
                rating = result['quality_assessment']['snr_rating']['level']
                
                print(f"  ✓ Completed: SNR={snr:.1f}, 评级={rating}")
            else:
                error_msg = result.get('error', 'UnknownError')
                self.errors.append({
                    'scan_id': scan_id,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                self.stats['failed'] += 1
                print(f"  ✗ Failed: {error_msg}")
                
        except Exception as e:
            error_msg = str(e)
            self.errors.append({
                'scan_id': scan_id,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            self.stats['failed'] += 1
            print(f"  ✗ Exception: {error_msg}")
            
            if BATCH_PROCESSING['skip_on_error']:
                print(f"  Skipping this scan, continuing to next...")
            else:
                raise
    
    def export_to_csv(self):
        """导出结果到CSV文件"""
        try:
            csv_path = self.output_dir / "quality_summary.csv"
            self.log(f"正在导出质量汇总CSV: {csv_path}")
            # 修复：在循环前初始化变量
            total_rows = 0
            completed_rows = 0
            basic_rows = 0
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
                writer.writeheader()
                
                for scan_id, result in self.results.items():
                    status = result.get('analysis_status', 'UNKNOWN')
                    if status == 'COMPLETED':
                        row = self._result_to_csv_row(result, scan_id)
                        if row:
                            writer.writerow(row)
                            completed_rows += 1
                        else:
                            # 如果转换失败，使用基本行
                            basic_row = self._create_basic_csv_row(result, scan_id)
                            writer.writerow(basic_row)
                            basic_rows += 1
                            if self.verbose:
                                self.log(f"  ⚠️ {scan_id}: 完整转换失败，使用基本行")   
                    else:
                        # 对于非COMPLETED状态，使用基本行
                        basic_row = self._create_basic_csv_row(result, scan_id)
                        writer.writerow(basic_row)
                        basic_rows += 1
                        if self.verbose:
                            self.log(f"  ⚠️ {scan_id}: 状态={status}，使用基本行")
                
                    total_rows += 1
            self.log(f"CSV export completed: {csv_path}")
            
        except Exception as e:
            self.log(f"CSV export failed: {e}")
    
    def _result_to_csv_row(self, result: Dict[str, Any], scan_id: str) -> Optional[Dict]:
        """转换单个结果为CSV行"""
        try:
            acquisition = result['acquisition']
            snr_results = result['snr_results']['recommended']
            quality = result['quality_assessment']
            
            # 提取CNR信息
            cnr_analysis = quality.get('cnr_analysis', {})
            if 'best_cnr' in cnr_analysis:
                best_cnr = cnr_analysis['best_cnr']
                cnr_value = best_cnr.get('cnr_value', 0.0)
                cnr_rating = best_cnr.get('cnr_rating', '')
                cnr_tissue_pair = best_cnr.get('description', '')
            else:
                cnr_value = 0.0
                cnr_rating = ''
                cnr_tissue_pair = ''
            
            # 提取Noise Uniformity
            uniformity = quality.get('noise_uniformity', {})
            noise_cv = uniformity.get('cv', 0.0)
            noise_rating = uniformity.get('uniformity_rating', '')
            
            # 提取背景Quality
            bg_quality = quality.get('background_quality', {})
            
            # 提取Signal ROI
            roi_info = result.get('roi_info', {}).get('signal', {})
            roi_stats = roi_info.get('statistics', {})
            
            # 构建CSV行
            row = {
                'patient_id': acquisition.get('patient_id', ''),
                'scan_name': acquisition.get('scan_name', ''),
                'anatomical_region': acquisition.get('anatomical_region', ''),
                'sequence_type': acquisition.get('sequence_type', ''),
                'field_strength_t': acquisition.get('field_strength', '').replace('t', ''),
                'parallel_imaging': 'Yes' if acquisition.get('parallel_imaging', False) else 'No',
                'acceleration_factor': acquisition.get('acceleration_factor', 1.0),
                'snr_value': snr_results.get('snr', 0.0),
                'snr_db': snr_results.get('snr_db', 0.0),
                'snr_rating': quality['snr_rating'].get('level', ''),
                'cnr_value': cnr_value,
                'cnr_rating': cnr_rating,
                'cnr_tissue_pair': cnr_tissue_pair,
                'noise_uniformity_cv': noise_cv,
                'noise_uniformity_rating': noise_rating,
                'signal_mean': snr_results.get('signal_mean', 0.0),
                'noise_std': result['snr_results']['traditional'].get('noise_std', 0.0),
                'g_factor_applied': result.get('snr_results', {}).get('parallel_correction', {}).get('correction', {}).get('g_factor', 1.0),
                'background_pixels': bg_quality.get('pixel_count', 0),
                'roi_mean': roi_stats.get('mean', 0.0),
                'roi_std': roi_stats.get('std', 0.0),
                'algorithm_confidence': 'HIGH',  # 可根据需要从结果中提取
                'image_quality': quality['snr_rating'].get('level', ''),
                'validation_status': result['validation_info'].get('status', ''),
                'analysis_date': result['analysis_info'].get('date', ''),
                'analysis_status': result['analysis_status']
            }
            
            return row
            
        except Exception as e:
            self.log(f"Failed to convert result to CSV row {scan_id}: {e}")
            return None
    
    def _create_basic_csv_row(self, result: Dict[str, Any], scan_id: str) -> Dict:
        """创建基本CSV行（用于失败或部分成功的扫描）"""
        return {
            'patient_id': result.get('acquisition', {}).get('patient_id', ''),
            'scan_name': result.get('acquisition', {}).get('scan_name', ''),
            'analysis_status': result.get('analysis_status', 'UNKNOWN'),
            'error': result.get('error', ''),
            'scan_id': scan_id
        }
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        print("\n" + "=" * 70)
        print("Batch processing completed摘要")
        print("=" * 70)
        print(f"Total Scans: {self.stats['total_scans']}")
        print(f"Successfully Analyzed: {self.stats['successful']} (绿色)")
        print(f"Analysis Failed: {self.stats['failed']} (红色)")
        print(f"Skipped Scans: {self.stats['skipped']} (黄色)")
        
        if self.stats['total_scans'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_scans']) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        if 'duration_seconds' in self.stats:
            print(f"Total duration: {self.stats['duration_seconds']:.1f}seconds")
            if self.stats['successful'] > 0:
                avg_time = self.stats['duration_seconds'] / self.stats['successful']
                print(f"平均每扫描: {avg_time:.1f}seconds")
        
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print("Generated files:")
        print(f"  • quality_summary.csv - CSV格式Quality数据")
        if self.errors:
            print(f"  • processing_errors.json - Error报告")
        print(f"  • [patient_id]/[scan_name]/ - Detailed results for each scan")
        print("=" * 70)
    
    def log(self, message: str):
        """日志记录"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")


# 便捷函数
def run_batch_processing(input_dir: str = INPUT_DIR,
                        output_dir: str = OUTPUT_DIR,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    运行批量Processing的便捷函数
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        verbose: YesNo显示详细日志
        
    Returns:
        Processing统计信息
    """
    processor = BatchProcessor(input_dir, output_dir, verbose)
    return processor.process_all()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MRIQuality分析批量Processing器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # 使用Default目录
  python batch_processor.py
  
  # 指定自定义目录
  python batch_processor.py --input my_data --output my_results
  
  # 安静模式（最小输出）
  python batch_processor.py --quiet
  
  # 单扫描Test模式
  python batch_processor.py --test
        """
    )
    
    parser.add_argument('--input', '-i', default=INPUT_DIR,
                       help=f'Input directory (default:: {INPUT_DIR}）')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR,
                       help=f'Output directory (default:: {OUTPUT_DIR}）')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode, reduce output')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test模式，只Processing第一scans')
    
    args = parser.parse_args()
    
    print("MRIQuality分析批量Processing器 v1.0")
    print("=" * 60)
    
    if args.test:
        # Test模式：只Processing第一个找到的扫描
        print("运行Test模式...")
        processor = BatchProcessor(args.input, args.output, not args.quiet)
        scans = processor.discover_scans()
        
        if scans:
            print(f"找到 {len(scans)} scans，TestProcessing第一个...")
            processor._process_single_scan(scans[0], 1, 1)
        else:
            print("未找到任何扫描进行Test")
    else:
        # 正常批量Processing
        stats = run_batch_processing(
            input_dir=args.input,
            output_dir=args.output,
            verbose=not args.quiet
        )
        
        # 返回退出码（如果有Failed则返回1）
        if stats['failed'] > 0:
            sys.exit(1)