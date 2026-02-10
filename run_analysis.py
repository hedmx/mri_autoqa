#!/usr/bin/env python3
"""
MRIQuality分析系统 - 主运行脚本
单图像法版本，兼容双图像法输出格式
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 导入自定义模块
from batch_processor import run_batch_processing
from single_imagine import analyze_single_scan
from config import INPUT_DIR, OUTPUT_DIR


def print_banner():
    """打印程序横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      MRI Quality Analysis System - Single Image Edition v1.0                         ║
║      MRI Quality Analysis System - Single Image Edition         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_system_info():
    """打印System Information"""
    import platform
    import numpy as np
    import nibabel as nib
    
    print("System Information:")
    print("=" * 60)
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"NumPy Version: {np.__version__}")
    print(f"NiBabel Version: {nib.__version__}")
    print(f"Input directory: {Path(INPUT_DIR).absolute()}")
    print(f"Output directory: {Path(OUTPUT_DIR).absolute()}")
    print("=" * 60)
    print()


def validate_environment():
    """验证运行环境"""
    try:
        import numpy as np
        import nibabel as nib
        import matplotlib
        import json
        import csv
        
        print("✓ 环境检查通过:")
        print(f"  - NumPy: {np.__version__}")
        print(f"  - NiBabel: {nib.__version__}")
        print(f"  - Matplotlib: {matplotlib.__version__}")
        
        # 检查Single Image V3 Engine
        try:
            from single_image_v3_engine import SingleImageSNREngineFixedV3
            print("  - Single Image V3 Engine: 可用")
        except ImportError:
            try:
                v3_file = Path("single_image_v3_engine.py")
                if v3_file.exists():
                    print("  - Single Image V3 Engine: Loaded from file")
                else:
                    print("  ⚠ Single Image V3 Engine: Not found, will use simplified version")
            except:
                print("  ⚠ Single Image V3 Engine: Not found, will use simplified version")
        
        return True
        
    except ImportError as e:
        print(f"✗ Environment check failed: Missing dependency - {e}")
        print("\nPlease install the following dependencies:")
        print("  pip install numpy nibabel matplotlib")
        return False


def run_single_scan_mode(args):
    """运行单扫描模式"""
    print("\n" + "=" * 70)
    print("Single Scan Analysis Mode")
    print("=" * 70)
    
    nifti_path = Path(args.nifti)
    metadata_path = Path(args.metadata)
    
    # 验证文件存在
    if not nifti_path.exists():
        print(f"Error: NIfTI file does not exist - {nifti_path}")
        return False
    
    if not metadata_path.exists():
        print(f"Error: Metadata File不存在 - {metadata_path}")
        return False
    
    # 创建Output directory
    output_dir = Path(args.output) if args.output else Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scan File: {nifti_path.name}")
    print(f"Metadata File: {metadata_path.name}")
    print(f"Output directory: {output_dir.absolute()}")
    print("-" * 70)
    
    try:
        # 运行分析
        result = analyze_single_scan(
            str(nifti_path),
            str(metadata_path),
            str(output_dir)
        )
        
        # 检查Analysis Status
        if result.get('analysis_status') == 'COMPLETED':
            snr = result['snr_results']['recommended']['snr']
            rating = result['quality_assessment']['snr_rating']['level']
            
            print(f"\n✓ Analysis completed!")
            print(f"  SNR Value: {snr:.1f}")
            print(f"  SNR Rating: {rating}")
            
            # 显示CNR Results（如果有）
            cnr_analysis = result['quality_assessment'].get('cnr_analysis', {})
            if 'best_cnr' in cnr_analysis:
                cnr = cnr_analysis['best_cnr']['cnr_value']
                cnr_desc = cnr_analysis['best_cnr']['description']
                print(f"  CNR值: {cnr:.2f} ({cnr_desc})")
            
            # 显示Noise Uniformity
            uniformity = result['quality_assessment'].get('noise_uniformity', {})
            if uniformity:
                cv = uniformity.get('cv', 0)
                rating = uniformity.get('uniformity_rating', '')
                print(f"  Noise Uniformity: CV={cv:.3f} ({rating})")
            
            # 显示Parallel correction信息（如果应用了）
            if 'parallel_correction' in result['snr_results']:
                correction = result['snr_results']['parallel_correction']
                diff = correction.get('snr_difference_percent', 0)
                g_factor = correction['correction'].get('g_factor', 1.0)
                print(f"  Parallel correction: {diff:+.1f}% (g-factor={g_factor:.2f})")
            
            print(f"\nResult files:")
            print(f"  JSON report: {output_dir}/qa_report.json")
            print(f"  Visualization: {output_dir}/visualization.png")
            print(f"  Text summary: {output_dir}/quality_summary.txt")
            
            return True
            
        else:
            error_msg = result.get('error', 'UnknownError')
            print(f"\n✗ Analysis Failed: {error_msg}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_batch_mode(args):
    """运行批量模式"""
    print("\n" + "=" * 70)
    print("Batch Analysis Mode")
    print("=" * 70)
    
    input_dir = Path(args.input) if args.input else Path(INPUT_DIR)
    output_dir = Path(args.output) if args.output else Path(OUTPUT_DIR)
    
    if not input_dir.exists():
        print(f"Error: Input directory不存在 - {input_dir}")
        print(f"Please ensure DICOM converter has been run and converted_data directory exists")
        return False
    
    print(f"Input directory: {input_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print("-" * 70)
    
    # 运行批量Processing
    stats = run_batch_processing(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        verbose=not args.quiet
    )
    
    # 根据Processing结果返回退出码
    if stats.get('failed', 0) > 0:
        return False
    else:
        return True


def list_scans(args):
    """列出所有可用的扫描"""
    input_dir = Path(args.input) if args.input else Path(INPUT_DIR)
    
    if not input_dir.exists():
        print(f"Error: Input directory不存在 - {input_dir}")
        return False
    
    print(f"\nScan List - {input_dir.absolute()}")
    print("=" * 70)
    
    from batch_processor import BatchProcessor
    processor = BatchProcessor(str(input_dir), verbose=False)
    scans = processor.discover_scans()
    
    if not scans:
        print("No scans found")
        return True
    
    print(f"Found {len(scans)} scans:\n")
    
    # 按Patient分组
    patients = {}
    for scan in scans:
        patient_id = scan['patient_id']
        if patient_id not in patients:
            patients[patient_id] = []
        patients[patient_id].append(scan)
    
    for patient_id, patient_scans in patients.items():
        print(f"Patient: {patient_id}")
        print("-" * 40)
        
        for scan in patient_scans:
            scan_name = scan['scan_name']
            nifti_path = scan['nifti_path'].relative_to(input_dir)
            
            # 尝试加载Metadata获取更多信息
            try:
                import json
                with open(scan['metadata_path'], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                sequence = metadata.get('SequenceAnalysis', {}).get('primary', 'unknown')
                anatomy = metadata.get('AnatomicalRegion', {}).get('standardized', 'unknown')
                parallel = metadata.get('ParallelImaging', {}).get('is_parallel', False)
                
                parallel_info = ""
                if parallel:
                    accel = metadata['ParallelImaging'].get('acceleration_factor', 1.0)
                    parallel_info = f", Parallel Imaging(R={accel:.1f})"
                
                print(f"  {scan_name}: {sequence.upper()} - {anatomy}{parallel_info}")
                
            except:
                print(f"  {scan_name}: {nifti_path}")
        
        print()
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MRIQuality分析系统 - 单图像法版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # 显示System Information
  python run_analysis.py --info
  
  # List all scans
  python run_analysis.py --list
  
  # Batch process all scans
  python run_analysis.py --batch
  
  # 批量Processing（安静模式）
  python run_analysis.py --batch --quiet
  
  # Process single scan
  python run_analysis.py --single --nifti converted_data/p001/T1_1/scan.nii.gz --metadata converted_data/p001/T1_1/metadata.json
  
  # Process single scan并指定Output directory
  python run_analysis.py --single --nifti scan.nii.gz --metadata metadata.json --output my_results

Input directory结构:
  converted_data/
  ├── p001/
  │   ├── T1_1/
  │   │   ├── scan.nii.gz
  │   │   └── metadata.json
  │   └── T2_2/
  │       ├── scan.nii.gz
  │       └── metadata.json
  └── p002/...

Output directory structure:
  autoqa_results/
  ├── p001/
  │   ├── T1_1/
  │   │   ├── qa_report.json
  │   │   ├── visualization.png
  │   │   └── quality_summary.txt
  │   └── T2_2/...
  ├── batch_summary.json
  ├── quality_summary.csv
  └── processing_statistics.txt
        """
    )
    
    # 模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--batch', '-b', action='store_true',
                          help='批量Processing模式（Processing所有扫描）')
    mode_group.add_argument('--single', '-s', action='store_true',
                          help='单扫描Processing模式')
    mode_group.add_argument('--list', '-l', action='store_true',
                          help='列出所有可用扫描')
    mode_group.add_argument('--info', '-i', action='store_true',
                          help='显示System Information')
    
    # 单扫描模式参数
    parser.add_argument('--nifti', '-n', 
                       help='Single scan mode: NIfTI file path')
    parser.add_argument('--metadata', '-m',
                       help='单扫描模式：Metadata JSON文件Path')
    
    # 通用参数
    parser.add_argument('--input', default=INPUT_DIR,
                       help=f'Input directory (default:: {INPUT_DIR}）')
    parser.add_argument('--output', default=OUTPUT_DIR,
                       help=f'Output directory (default:: {OUTPUT_DIR}）')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode, reduce output')
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 显示System Information
    if args.info:
        print_system_info()
        if validate_environment():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # 验证环境
    if not validate_environment():
        sys.exit(1)
    
    # 根据模式执行
    success = False
    
    try:
        if args.list:
            success = list_scans(args)
            
        elif args.single:
            if not args.nifti or not args.metadata:
                print("Error: 单扫描模式需要 --nifti 和 --metadata 参数")
                parser.print_help()
                sys.exit(1)
            success = run_single_scan_mode(args)
            
        elif args.batch:
            success = run_batch_mode(args)
            
        else:
            # Default模式：交互式选择或Show help
            print("Please select operation mode:")
            print("  1. Batch process all scans (--batch)")
            print("  2. Process single scan (--single)")
            print("  3. List all scans (--list)")
            print("  4. Show help")
            print()
            
            choice = input("Please select (1-4): ").strip()
            
            if choice == '1':
                args.batch = True
                success = run_batch_mode(args)
            elif choice == '2':
                args.single = True
                # 交互式输入文件Path
                args.nifti = input("Please enter NIfTI file path: ").strip()
                args.metadata = input("请输入Metadata FilePath: ").strip()
                success = run_single_scan_mode(args)
            elif choice == '3':
                args.list = True
                success = list_scans(args)
            else:
                parser.print_help()
                success = True
        
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nProgram execution error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # 退出程序
    if success:
        print("\n✓ Program executed successfully")
        sys.exit(0)
    else:
        print("\n✗ Program execution failed")
        sys.exit(1)


if __name__ == "__main__":
    main()