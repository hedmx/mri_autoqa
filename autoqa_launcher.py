# autoqa_launcher.py
#!/usr/bin/env python3
"""
MRI AutoQA Launcher - One-command automated batch quality analysis
一键启动自动化批量质量分析，自动输出完整报告
修复版：添加扫描参数清单导出功能
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import shutil
import json
import pandas as pd
from typing import Dict, Any, Optional, List
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

class AutoQALauncher:
    """
    自动化MRI质量分析启动器
    单个命令完成：数据验证 → 批量处理 → 报告生成 → 可视化
    重新设计报告输出系统，带时间戳的分层报告
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.results_dir = None
        self.report_timestamp = None
        self.report_id = None
        self.scan_count = 0
        
        # 默认路径配置
        self.paths = {
            'input_data': PROJECT_ROOT / "converted_data",
            'output_results': PROJECT_ROOT / "autoqa_results",
            'batch_report': PROJECT_ROOT / "batch_reports",
            'visualizations': PROJECT_ROOT / "visualizations",
            'logs': PROJECT_ROOT / "logs"
        }
        
        # 模块可用性标志
        self.modules_available = {
            'skimage': False,
            'visualization': False,
            'batch_visualization': False
        }
        
        self.log("=" * 70)
        self.log("MRI AutoQA Launcher - Automated Batch Quality Analysis")
        self.log("重新设计报告输出系统 v2.0")
        self.log("=" * 70)
        
    def run_full_pipeline(self, 
                         input_dir: str = None,
                         output_dir: str = None,
                         skip_visualization: bool = False,
                         force_clean: bool = False) -> bool:
        """
        运行完整分析流水线
        
        Args:
            input_dir: 输入数据目录（如未指定使用默认）
            output_dir: 输出结果目录（如未指定使用默认）
            skip_visualization: 是否跳过可视化生成
            force_clean: 是否清理旧结果
            
        Returns:
            成功状态
        """
        self.start_time = datetime.now()
        self.report_timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.report_id = f"AUTOQA-{self.report_timestamp}"
        
        try:
            # 1. 验证和准备环境
            self._validate_environment()
            
            # 2. 设置输入输出目录
            self._setup_directories(input_dir, output_dir, force_clean)
            
            # 3. 检查输入数据
            self.scan_count = self._check_input_data()
            if self.scan_count == 0:
                self.log("❌ 未找到任何扫描数据，请检查输入目录")
                return False
            
            self.log(f"✅ 找到 {self.scan_count} 个待分析扫描")
            
            # 4. 运行批量分析
            if not self._run_batch_analysis():
                return False
            
            # 5. 生成带时间戳的分层报告
            if not self._generate_timestamped_reports():
                return False
            
            # 6. 生成可视化报告（可选）
            if not skip_visualization:
                if not self._generate_visualizations():
                    self.log("⚠️ 可视化生成失败，继续其他步骤")
            
            # 7. 清理临时文件
            self._cleanup_temp_files()
            
            # 8. 创建最新报告符号链接
            self._create_latest_symlink()
            
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            
            self._print_success_summary(duration, self.scan_count)
            return True
            
        except Exception as e:
            self.log(f"❌ 流水线执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_environment(self):
        """验证Python环境和依赖"""
        self.log("🔍 验证环境...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            raise EnvironmentError(f"需要Python 3.8+，当前版本: {sys.version}")
        
        self.log(f"   Python版本: {sys.version}")
        
        # 检查必要模块（核心功能必需）
        required_modules = [
            'numpy', 'nibabel', 'matplotlib', 'pandas', 
            'scipy', 'psutil'
        ]
        
        # 检查可选模块
        optional_modules = [
            ('skimage', 'scikit-image', '高级图像处理'),
            ('sklearn', 'scikit-learn', '机器学习分析'),
            ('seaborn', 'seaborn', '可视化美化')
        ]
        
        missing_required = []
        missing_optional = []
        
        # 检查核心必需模块
        for module in required_modules:
            try:
                __import__(module)
                self.log(f"   ✓ {module} (必需)")
            except ImportError:
                missing_required.append(module)
                self.log(f"   ✗ {module} (必需，缺失)")
        
        # 检查可选模块
        for module_name, pip_name, description in optional_modules:
            try:
                __import__(module_name)
                self.log(f"   ✓ {module_name} (可选)")
                # 记录skimage是否可用
                if module_name == 'skimage':
                    self.modules_available['skimage'] = True
            except ImportError:
                missing_optional.append((module_name, pip_name, description))
                self.log(f"   ⚠ {module_name} (可选，缺失)")
        
        # 如果有缺失的必需模块，报错
        if missing_required:
            self.log(f"\n❌ 缺失核心必需模块: {', '.join(missing_required)}")
            self.log("请使用以下命令安装:")
            self.log(f"   pip install {' '.join(missing_required)}")
            raise ImportError(f"缺失核心模块: {missing_required}")
        
        # 如果有缺失的可选模块，警告但不中断
        if missing_optional:
            self.log(f"\n⚠️  缺失可选模块:")
            for module_name, pip_name, description in missing_optional:
                self.log(f"   - {module_name}: {description}")
            
            self.log("\n  这些模块用于增强功能，不影响核心分析:")
            self.log("  如需安装: pip install " + " ".join([name for _, name, _ in missing_optional]))
        
        # 检查项目模块
        required_project_modules = [
            'single_imagine.py',
            'batch_processor.py', 
            'run_analysis.py',
            'config.py'
        ]
        
        # 可视化模块是可选的
        optional_project_modules = [
            ('visualization.py', '可视化生成器'),
            ('batch_visualization.py', '批量可视化'),
            ('visualization_config.py', '可视化配置')
        ]
        
        for module in required_project_modules:
            module_path = PROJECT_ROOT / module
            if module_path.exists():
                self.log(f"   ✓ {module} (项目文件)")
            else:
                self.log(f"   ✗ {module} (项目文件，缺失)")
                raise FileNotFoundError(f"项目文件缺失: {module}")
        
        # 检查可视化模块
        viz_modules_exist = []
        for module, description in optional_project_modules:
            module_path = PROJECT_ROOT / module
            if module_path.exists():
                self.log(f"   ✓ {module} ({description})")
                viz_modules_exist.append(True)
                
                # 记录可视化模块可用性 - 使用精确匹配
                if module == 'visualization.py':
                    self.modules_available['visualization'] = True
                elif module == 'batch_visualization.py':
                    self.modules_available['batch_visualization'] = True
            else:
                self.log(f"   ⚠ {module} ({description}，缺失)")
                viz_modules_exist.append(False)
        
        # 如果所有可视化模块都存在，标记为可用
        if all(viz_modules_exist):
            self.log("   ✓ 所有可视化模块可用")
        else:
            self.log("   ⚠ 部分可视化模块缺失，某些功能可能受限")
        
        self.log("✅ 环境验证通过")
    
    def _setup_directories(self, input_dir: str, output_dir: str, force_clean: bool):
        """设置输入输出目录"""
        self.log("📁 设置目录...")
        
        # 1. 处理输入目录
        if input_dir:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise FileNotFoundError(f"输入目录不存在: {input_path}")
            self.paths['input_data'] = input_path
        
        input_data_path = self.paths['input_data']
        if not input_data_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_data_path}")
        
        self.log(f"   输入目录: {input_data_path.absolute()}")
        
        # 2. 处理输出目录
        if output_dir:
            output_path = Path(output_dir)
            self.paths['output_results'] = output_path
        
        # 确保results_dir被设置
        self.results_dir = self.paths['output_results']
        
        # 3. 创建所有输出目录
        self.log(f"   创建输出目录结构...")
        for key, path in self.paths.items():
            if key != 'input_data':
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    if self.verbose:
                        self.log(f"     - {key}: {path.absolute()}")
                except Exception as e:
                    self.log(f"     ⚠️ 创建目录失败 {key}: {e}")
        
        # 4. 强制清理（如果指定）
        if force_clean and self.results_dir.exists():
            self.log(f"⚠️ 强制清理模式: 删除旧结果目录")
            try:
                # 先删除目录内容
                for item in self.results_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                self.log(f"   清理完成: {self.results_dir}")
            except Exception as e:
                self.log(f"   清理失败: {e}")
        
        # 5. 最终确认
        self.log(f"✅ 目录设置完成")
        self.log(f"   输入目录: {self.paths['input_data'].absolute()}")
        self.log(f"   输出目录: {self.results_dir.absolute()}")
        
        # 确保results_dir存在
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"   创建输出目录: {self.results_dir.absolute()}")
    
    def _check_input_data(self) -> int:
        """检查输入数据，返回扫描数量"""
        input_dir = self.paths['input_data']
    
        self.log("🔍 检查输入数据...")
        self.log(f"   输入目录: {input_dir.absolute()}")
    
        # 查找所有NIfTI文件
        nifti_files = list(input_dir.rglob("*.nii.gz")) + list(input_dir.rglob("*.nii"))
    
        if not nifti_files:
            self.log("   未找到NIfTI文件")
            return 0
    
        scan_count = 0
        patients = {}
    
        for nifti_path in nifti_files:
            # 提取患者ID和扫描名称
            try:
                rel_path = nifti_path.relative_to(input_dir)
                parts = rel_path.parts
            
                if len(parts) >= 2:
                    patient_id = parts[0]
                    scan_name = parts[1] if len(parts) > 1 else nifti_path.stem
                
                    if patient_id not in patients:
                        patients[patient_id] = set()
                    patients[patient_id].add(scan_name)
                
                    scan_count += 1
            except Exception as e:
                # 如果无法解析路径，只计数
                self.log(f"   警告: 无法解析路径 {nifti_path}: {e}")
                scan_count += 1
    
        # 显示统计信息
        self.log(f"   找到 {scan_count} 个NIfTI文件")
        if patients:
            self.log(f"   涉及 {len(patients)} 个患者")
            for patient_id, scans in list(patients.items())[:5]:
                self.log(f"     - {patient_id}: {len(scans)} 个扫描")
            if len(patients) > 5:
                self.log(f"     ... 和 {len(patients) - 5} 个更多患者")
        else:
            self.log("   警告: 无法按患者组织文件")
    
        return scan_count
    
    def _run_batch_analysis(self) -> bool:
        """运行批量分析"""
        self.log("\n🚀 开始批量质量分析...")
    
        try:
            # 导入批量处理器
            sys.path.append(str(PROJECT_ROOT))
            from batch_processor import run_batch_processing
        
            # 运行批量处理
            stats = run_batch_processing(
                input_dir=str(self.paths['input_data']),
                output_dir=str(self.results_dir),
                verbose=self.verbose
            )
        
            # 检查结果
            if stats.get('successful', 0) == 0 and stats.get('total_scans', 0) > 0:
                self.log("❌ 批量分析失败：没有成功处理的扫描")
                return False
        
            self.log(f"✅ 批量分析完成")
            self.log(f"   总计扫描: {stats.get('total_scans', 0)}")
            self.log(f"   成功分析: {stats.get('successful', 0)}")
            self.log(f"   分析失败: {stats.get('failed', 0)}")
            self.log(f"   跳过扫描: {stats.get('skipped', 0)}")
            self.log(f"   处理时间: {stats.get('duration_seconds', 0):.1f}秒")
        
            # 保存统计信息
            stats_file = self.results_dir / "analysis_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
            return True
        
        except Exception as e:
            self.log(f"❌ 批量分析失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_timestamped_reports(self) -> bool:
        """
        生成带时间戳的分层报告系统
        主入口函数，调用各个子报告生成函数
        """
        self.log(f"\n📊 生成带时间戳的分层报告系统...")
        self.log(f"   报告ID: {self.report_id}")
        self.log(f"   时间戳: {self.report_timestamp}")
        
        try:
            # 创建主报告目录
            report_dir = self.results_dir / f"batch_report_{self.report_timestamp}"
            report_dir.mkdir(exist_ok=True)
            
            # 创建子目录结构
            subdirs = [
                "00_executive_summary",
                "01_detailed_data", 
                "02_visualizations",
                "03_quality_analysis",
                "04_technical_appendix"
            ]
            
            for subdir in subdirs:
                (report_dir / subdir).mkdir(exist_ok=True)
            
            # 1. 提取所有数据
            self.log("   1. 提取扫描数据...")
            all_results = self._extract_all_scan_data()
            if not all_results:
                self.log("❌ 没有有效的结果数据")
                return False
            
            # 创建DataFrame
            df = pd.DataFrame(all_results)
            
            # 2. 生成各种报告
            self.log("   2. 生成执行摘要...")
            self._generate_executive_summary(df, report_dir)
            
            self.log("   3. 生成详细数据文件...")
            self._generate_detailed_data_files(df, report_dir)
            
            self.log("   4. 生成统计报告...")
            self._generate_statistical_reports(df, report_dir)
            
            self.log("   5. 生成质量分析报告...")
            self._generate_quality_analysis(df, report_dir)
            
            self.log("   6. 生成技术附录...")
            self._generate_technical_appendix(df, report_dir)
            
            # 7. 创建报告索引文件
            self._create_report_index(report_dir, df)
            
            self.log(f"✅ 分层报告生成完成")
            self.log(f"   报告目录: {report_dir.absolute()}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ 报告生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_all_scan_data(self) -> List[Dict]:
        """提取所有扫描数据"""
        result_files = list(self.results_dir.rglob("**/qa_report.json"))
        
        if not result_files:
            self.log("❌ 未找到分析结果文件")
            return []
        
        self.log(f"   找到 {len(result_files)} 个分析结果")
        
        all_results = []
        
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 提取扫描信息
                scan_info = self._extract_scan_info_v2(result_data, result_file)
                if scan_info:
                    all_results.append(scan_info)
                    
            except Exception as e:
                self.log(f"   警告: 无法处理 {result_file}: {e}")
        
        return all_results
    
    def _extract_scan_info_v2(self, result_data: Dict[str, Any], result_file: Path) -> Optional[Dict]:
        """
        提取扫描信息 - 修复版，与JSON结构一致
        """
        try:
            # 基础信息
            scan_info = {
                # 报告标识
                'scan_id': result_data.get('analysis_info', {}).get('scan_id', ''),
                'analysis_date': result_data.get('analysis_info', {}).get('date', ''),
                'algorithm_version': result_data.get('analysis_info', {}).get('algorithm_version', ''),
                'analysis_status': result_data.get('analysis_status', 'UNKNOWN'),
                
                # 临床信息
                'patient_id': result_data.get('acquisition', {}).get('patient_id', ''),
                'scan_name': result_data.get('acquisition', {}).get('scan_name', ''),
                'anatomical_region': result_data.get('acquisition', {}).get('anatomical_region', ''),
                'sequence_type': result_data.get('acquisition', {}).get('sequence_type', ''),
                'field_strength': result_data.get('acquisition', {}).get('field_strength', ''),
                'acquisition_mode': result_data.get('acquisition', {}).get('acquisition_mode', ''),
                'acceleration_factor': result_data.get('acquisition', {}).get('acceleration_factor', 1.0),
                'parallel_imaging': result_data.get('acquisition', {}).get('parallel_imaging', False),
                
                # 验证信息
                'validation_status': result_data.get('validation_info', {}).get('status', ''),
            }
            
            # 如果是失败扫描，添加错误信息
            if scan_info['analysis_status'] != 'COMPLETED':
                scan_info['error_message'] = result_data.get('error', '')
                scan_info['error_type'] = result_data.get('error_type', '')
                return scan_info
            
            # 成功扫描，提取详细结果
            
            # 1. SNR相关
            snr_results = result_data.get('snr_results', {})
            traditional = snr_results.get('traditional', {})
            recommended = snr_results.get('recommended', {})
            rayleigh = snr_results.get('rayleigh_correction', {})
            
            scan_info.update({
                'snr_raw': traditional.get('snr', 0.0),
                'snr_corrected': recommended.get('snr', 0.0),
                'snr_improvement_percent': rayleigh.get('improvement_percent', 0.0),
                'snr_rating': result_data.get('quality_assessment', {}).get('snr_rating', {}).get('level', ''),
                'correction_factor': rayleigh.get('correction_factor', 1.0),
            })
            
            # 2. CNR相关
            cnr_analysis = result_data.get('quality_assessment', {}).get('cnr_analysis', {})
            if cnr_analysis and 'best_cnr' in cnr_analysis:
                best_cnr = cnr_analysis['best_cnr']
                scan_info.update({
                    'cnr_value': best_cnr.get('cnr_value', 0.0),
                    'cnr_rating': best_cnr.get('cnr_rating', ''),
                    'cnr_tissue_pair': best_cnr.get('description', ''),
                })
            else:
                scan_info.update({
                    'cnr_value': 0.0,
                    'cnr_rating': 'N/A',
                    'cnr_tissue_pair': '',
                })
            
            # 3. 质量评分
            quality_assessment = result_data.get('quality_assessment', {})
            quality_scores = quality_assessment.get('quality_scores', {})
            if quality_scores:
                dimensions = quality_scores.get('dimensions', {})
                scan_info.update({
                    'quality_score_total': quality_scores.get('total_score', 0.0),
                    'quality_snr': dimensions.get('snr_quality', {}).get('score', 0.0),
                    'quality_cnr': dimensions.get('cnr_quality', {}).get('score', 0.0),
                    'quality_noise': dimensions.get('noise_quality', {}).get('score', 0.0),
                    'quality_artifact': dimensions.get('artifact_free', {}).get('score', 0.0),
                })
            
            # 4. 置信度评估
            overall_confidence = quality_assessment.get('overall_confidence', {})
            scan_info.update({
                'confidence_score': overall_confidence.get('score', 0.0),
                'algorithm_confidence': overall_confidence.get('level', 'UNKNOWN'),
            })
            
            # 5. ROI信息
            roi_info = result_data.get('roi_info', {})
            signal_roi = roi_info.get('signal', {})
            background_roi = roi_info.get('background', {})
            
            if signal_roi:
                signal_stats = signal_roi.get('statistics', {})
                scan_info.update({
                    'signal_mean': signal_stats.get('mean', 0.0),
                    'signal_std': signal_stats.get('std', 0.0),
                    'signal_cv': signal_stats.get('std', 0.0) / signal_stats.get('mean', 1.0) 
                                if signal_stats.get('mean', 0) > 0 else 0.0,
                })
            
            if background_roi:
                bg_stats = background_roi.get('statistics', {})
                scan_info.update({
                    'background_mean': bg_stats.get('mean', 0.0),
                    'background_std': bg_stats.get('std', 0.0),
                    'noise_uniformity_cv': result_data.get('quality_assessment', {})
                                       .get('noise_uniformity', {}).get('cv', 0.0),
                })
            
            return scan_info
            
        except Exception as e:
            self.log(f"   提取扫描信息失败 {result_file}: {e}")
            return None
    
    def _generate_executive_summary(self, df: pd.DataFrame, report_dir: Path):
        """生成执行摘要"""
        summary_dir = report_dir / "00_executive_summary"
        
        # 计算统计信息
        completed_df = df[df['analysis_status'] == 'COMPLETED'].copy()
        failed_df = df[df['analysis_status'] != 'COMPLETED'].copy()
        
        total_scans = len(df)
        successful_scans = len(completed_df)
        failed_scans = len(failed_df)
        success_rate = successful_scans / total_scans * 100 if total_scans > 0 else 0
        
        # 计算核心指标统计
        if len(completed_df) > 0:
            snr_stats = {
                'mean': completed_df['snr_corrected'].mean(),
                'median': completed_df['snr_corrected'].median(),
                'std': completed_df['snr_corrected'].std(),
                'min': completed_df['snr_corrected'].min(),
                'max': completed_df['snr_corrected'].max(),
                'q1': completed_df['snr_corrected'].quantile(0.25),
                'q3': completed_df['snr_corrected'].quantile(0.75),
            }
            
            cnr_stats = {
                'mean': completed_df['cnr_value'].mean() if 'cnr_value' in completed_df.columns else 0,
                'median': completed_df['cnr_value'].median() if 'cnr_value' in completed_df.columns else 0,
            }
            
            quality_stats = {
                'mean': completed_df['quality_score_total'].mean() if 'quality_score_total' in completed_df.columns else 0,
                'median': completed_df['quality_score_total'].median() if 'quality_score_total' in completed_df.columns else 0,
            }
            
            confidence_stats = {
                'mean': completed_df['confidence_score'].mean() if 'confidence_score' in completed_df.columns else 0,
                'median': completed_df['confidence_score'].median() if 'confidence_score' in completed_df.columns else 0,
            }
        else:
            snr_stats = cnr_stats = quality_stats = confidence_stats = {}
        
        # 生成Markdown报告
        md_file = summary_dir / f"executive_summary_{self.report_timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            self._write_executive_summary_content(f, df, completed_df, failed_df, 
                                                 total_scans, successful_scans, failed_scans, 
                                                 success_rate, snr_stats, cnr_stats, 
                                                 quality_stats, confidence_stats)
        
        # 生成文本摘要
        txt_file = summary_dir / f"quick_summary_{self.report_timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            self._write_quick_summary(f, total_scans, successful_scans, success_rate, snr_stats)
    
    def _write_executive_summary_content(self, f, df, completed_df, failed_df, 
                                        total_scans, successful_scans, failed_scans,
                                        success_rate, snr_stats, cnr_stats,
                                        quality_stats, confidence_stats):
        """写入执行摘要内容"""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        f.write(f"""# MRI AutoQA 批量分析执行摘要
## 报告ID: {self.report_id}

### 📋 报告信息
| 项目 | 内容 |
|------|------|
| **报告生成时间** | {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} |
| **分析时长** | {duration:.0f}秒 ({duration/60:.1f}分钟) |
| **扫描总数** | {total_scans} |
| **成功分析** | {successful_scans} ({success_rate:.1f}%) |
| **分析失败** | {failed_scans} |
| **软件版本** | MRI_AutoQA_v2.0 |
| **算法版本** | single_image_v3_rayleigh |

### 🎯 核心指标概览
| 指标 | 平均值 | 中位数 | 标准差 | 最小值 | 最大值 |
|------|--------|--------|--------|--------|--------|
| **校正SNR** | {snr_stats.get('mean', 0):.1f} | {snr_stats.get('median', 0):.1f} | {snr_stats.get('std', 0):.1f} | {snr_stats.get('min', 0):.1f} | {snr_stats.get('max', 0):.1f} |
| **CNR值** | {cnr_stats.get('mean', 0):.2f} | {cnr_stats.get('median', 0):.2f} | - | - | - |
| **质量总分** | {quality_stats.get('mean', 0):.3f} | {quality_stats.get('median', 0):.3f} | - | - | - |
| **置信度** | {confidence_stats.get('mean', 0):.3f} | {confidence_stats.get('median', 0):.3f} | - | - | - |

### 📈 质量评级分布
""")
        
        # SNR评级分布
        if 'snr_rating' in completed_df.columns:
            rating_dist = completed_df['snr_rating'].value_counts().sort_index()
            f.write("#### SNR评级分布\n")
            f.write("| 评级 | 扫描数 | 百分比 |\n")
            f.write("|------|--------|--------|\n")
            for rating, count in rating_dist.items():
                percentage = count / successful_scans * 100
                f.write(f"| {rating} | {count} | {percentage:.1f}% |\n")
            f.write("\n")
        
        # 置信度分布
        if 'algorithm_confidence' in completed_df.columns:
            conf_dist = completed_df['algorithm_confidence'].value_counts().sort_index()
            f.write("#### 算法置信度分布\n")
            f.write("| 置信度等级 | 扫描数 | 百分比 |\n")
            f.write("|------------|--------|--------|\n")
            for level, count in conf_dist.items():
                percentage = count / successful_scans * 100
                f.write(f"| {level} | {count} | {percentage:.1f}% |\n")
            f.write("\n")
        
        # 问题扫描
        if failed_scans > 0:
            f.write("### ⚠️ 问题扫描\n")
            f.write("| 扫描ID | 错误类型 | 错误信息 |\n")
            f.write("|--------|----------|----------|\n")
            for _, row in failed_df.iterrows():
                scan_id = f"{row.get('patient_id', '')}/{row.get('scan_name', '')}"
                error_type = row.get('error_type', 'UNKNOWN')
                error_msg = str(row.get('error_message', ''))[:50]
                f.write(f"| {scan_id} | {error_type} | {error_msg}... |\n")
        
        # 建议行动
        f.write("\n### 🚀 建议行动\n")
        f.write("1. **检查低质量扫描**：查看质量分<0.7的扫描\n")
        f.write("2. **审查失败扫描**：分析失败原因并重新处理\n")
        f.write("3. **优化采集参数**：关注低SNR扫描的采集设置\n")
        f.write("4. **定期质量监控**：建立质量基准并持续跟踪\n")
        
        f.write(f"\n---\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    def _write_quick_summary(self, f, total_scans, successful_scans, success_rate, snr_stats):
        """写入快速摘要"""
        f.write(f"MRI AutoQA 快速摘要\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"报告ID: {self.report_id}\n")
        f.write(f"\n核心统计:\n")
        f.write(f"  扫描总数: {total_scans}\n")
        f.write(f"  成功分析: {successful_scans} ({success_rate:.1f}%)\n")
        if snr_stats:
            f.write(f"  平均SNR: {snr_stats.get('mean', 0):.1f}\n")
            f.write(f"  SNR范围: {snr_stats.get('min', 0):.1f} - {snr_stats.get('max', 0):.1f}\n")
    
    def _generate_detailed_data_files(self, df: pd.DataFrame, report_dir: Path):
        """生成详细数据文件"""
        data_dir = report_dir / "01_detailed_data"
        
        # 1. 完整数据CSV
        csv_file = data_dir / f"detailed_results_{self.report_timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        self.log(f"   ✓ 详细结果CSV: {csv_file.name}")
        
        # 2. 成功扫描数据
        completed_df = df[df['analysis_status'] == 'COMPLETED'].copy()
        if len(completed_df) > 0:
            completed_csv = data_dir / f"successful_scans_{self.report_timestamp}.csv"
            completed_df.to_csv(completed_csv, index=False, encoding='utf-8')
            self.log(f"   ✓ 成功扫描CSV: {completed_csv.name}")
        
        # 3. 失败扫描数据
        failed_df = df[df['analysis_status'] != 'COMPLETED'].copy()
        if len(failed_df) > 0:
            failed_csv = data_dir / f"failed_scans_{self.report_timestamp}.csv"
            failed_df.to_csv(failed_csv, index=False, encoding='utf-8')
            self.log(f"   ✓ 失败扫描CSV: {failed_csv.name}")
        
        # 4. 扫描参数清单
        #self._generate_scan_parameters_summary(data_dir, df)
        
        # 5. 字段说明文档
        fields_doc = data_dir / f"data_fields_{self.report_timestamp}.md"
        with open(fields_doc, 'w', encoding='utf-8') as f:
            f.write("# 数据字段说明\n\n")
            f.write("| 字段名 | 说明 | 类型 | 示例 |\n")
            f.write("|--------|------|------|------|\n")
            fields = [
                ('scan_id', '扫描标识符', '字符串', 'p001/T1_1'),
                ('patient_id', '患者ID', '字符串', '01620360'),
                ('scan_name', '扫描名称', '字符串', 'T1_tse_sag_320'),
                ('anatomical_region', '解剖区域', '字符串', 'lumbar'),
                ('sequence_type', '序列类型', '字符串', 't1'),
                ('field_strength', '磁场强度', '字符串', '1.5t'),
                ('snr_corrected', '校正后SNR', '浮点数', '24.93'),
                ('snr_rating', 'SNR评级', '字符串', 'GOOD'),
                ('cnr_value', 'CNR值', '浮点数', '5.79'),
                ('quality_score_total', '质量总分', '浮点数', '0.88'),
                ('confidence_score', '置信度分数', '浮点数', '0.64'),
                ('algorithm_confidence', '置信度等级', '字符串', 'MEDIUM'),
                ('analysis_status', '分析状态', '字符串', 'COMPLETED'),
            ]
            for field_name, description, field_type, example in fields:
                f.write(f"| {field_name} | {description} | {field_type} | {example} |\n")
        
        self.log(f"   ✓ 字段说明文档")
    
   

    def _generate_statistical_reports(self, df: pd.DataFrame, report_dir: Path):
        """生成统计报告"""
        stats_dir = report_dir / "03_quality_analysis"
        completed_df = df[df['analysis_status'] == 'COMPLETED'].copy()
        
        if len(completed_df) == 0:
            return
        
        # 1. 基础统计报告
        stats_file = stats_dir / f"statistical_report_{self.report_timestamp}.md"
        with open(stats_file, 'w', encoding='utf-8') as f:
            self._write_statistical_report(f, completed_df)
        
        # 2. 按解剖区域统计
        if 'anatomical_region' in completed_df.columns:
            anatomy_stats = self._calculate_anatomy_statistics(completed_df)
            anatomy_file = stats_dir / f"anatomy_statistics_{self.report_timestamp}.csv"
            anatomy_stats.to_csv(anatomy_file, encoding='utf-8')
        
        # 3. 按序列类型统计
        if 'sequence_type' in completed_df.columns:
            sequence_stats = self._calculate_sequence_statistics(completed_df)
            sequence_file = stats_dir / f"sequence_statistics_{self.report_timestamp}.csv"
            sequence_stats.to_csv(sequence_file, encoding='utf-8')
    
    def _write_statistical_report(self, f, completed_df):
        """写入统计报告"""
        total_scans = len(completed_df)
        
        f.write(f"""# MRI质量统计报告
## 报告ID: {self.report_id}
## 统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 一、总体统计
- **统计扫描数**: {total_scans}
- **统计时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 二、关键指标统计
""")
        
        # SNR统计
        if 'snr_corrected' in completed_df.columns:
            f.write("#### 1. SNR统计\n")
            f.write("```\n")
            f.write(f"  平均值: {completed_df['snr_corrected'].mean():.2f}\n")
            f.write(f"  中位数: {completed_df['snr_corrected'].median():.2f}\n")
            f.write(f"  标准差: {completed_df['snr_corrected'].std():.2f}\n")
            f.write(f"  最小值: {completed_df['snr_corrected'].min():.2f}\n")
            f.write(f"  最大值: {completed_df['snr_corrected'].max():.2f}\n")
            f.write(f"  25分位数: {completed_df['snr_corrected'].quantile(0.25):.2f}\n")
            f.write(f"  75分位数: {completed_df['snr_corrected'].quantile(0.75):.2f}\n")
            f.write("```\n\n")
        
        # CNR统计
        if 'cnr_value' in completed_df.columns:
            f.write("#### 2. CNR统计\n")
            f.write("```\n")
            f.write(f"  平均值: {completed_df['cnr_value'].mean():.2f}\n")
            f.write(f"  中位数: {completed_df['cnr_value'].median():.2f}\n")
            f.write(f"  标准差: {completed_df['cnr_value'].std():.2f}\n")
            f.write("```\n\n")
        
        # 质量分统计
        if 'quality_score_total' in completed_df.columns:
            f.write("#### 3. 质量分统计\n")
            f.write("```\n")
            f.write(f"  平均值: {completed_df['quality_score_total'].mean():.3f}\n")
            f.write(f"  中位数: {completed_df['quality_score_total'].median():.3f}\n")
            f.write(f"  标准差: {completed_df['quality_score_total'].std():.3f}\n")
            f.write("```\n\n")
        
        # 置信度统计
        if 'confidence_score' in completed_df.columns:
            f.write("#### 4. 置信度统计\n")
            f.write("```\n")
            f.write(f"  平均值: {completed_df['confidence_score'].mean():.3f}\n")
            f.write(f"  中位数: {completed_df['confidence_score'].median():.3f}\n")
            f.write(f"  标准差: {completed_df['confidence_score'].std():.3f}\n")
            f.write("```\n")
    
    def _calculate_anatomy_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算解剖区域统计"""
        if 'anatomical_region' not in df.columns:
            return pd.DataFrame()
        
        stats_list = []
        
        for region in df['anatomical_region'].unique():
            region_df = df[df['anatomical_region'] == region]
            if len(region_df) == 0:
                continue
            
            stats = {
                'anatomical_region': region,
                'scan_count': len(region_df),
                'snr_mean': region_df['snr_corrected'].mean() if 'snr_corrected' in region_df.columns else 0,
                'snr_median': region_df['snr_corrected'].median() if 'snr_corrected' in region_df.columns else 0,
                'cnr_mean': region_df['cnr_value'].mean() if 'cnr_value' in region_df.columns else 0,
                'quality_mean': region_df['quality_score_total'].mean() if 'quality_score_total' in region_df.columns else 0,
                'confidence_mean': region_df['confidence_score'].mean() if 'confidence_score' in region_df.columns else 0,
            }
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def _calculate_sequence_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算序列类型统计"""
        if 'sequence_type' not in df.columns:
            return pd.DataFrame()
        
        stats_list = []
        
        for seq_type in df['sequence_type'].unique():
            seq_df = df[df['sequence_type'] == seq_type]
            if len(seq_df) == 0:
                continue
            
            stats = {
                'sequence_type': seq_type,
                'scan_count': len(seq_df),
                'snr_mean': seq_df['snr_corrected'].mean() if 'snr_corrected' in seq_df.columns else 0,
                'snr_median': seq_df['snr_corrected'].median() if 'snr_corrected' in seq_df.columns else 0,
                'cnr_mean': seq_df['cnr_value'].mean() if 'cnr_value' in seq_df.columns else 0,
                'quality_mean': seq_df['quality_score_total'].mean() if 'quality_score_total' in seq_df.columns else 0,
                'confidence_mean': seq_df['confidence_score'].mean() if 'confidence_score' in seq_df.columns else 0,
            }
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def _generate_quality_analysis(self, df: pd.DataFrame, report_dir: Path):
        """生成质量分析报告"""
        analysis_dir = report_dir / "03_quality_analysis"
        completed_df = df[df['analysis_status'] == 'COMPLETED'].copy()
        
        if len(completed_df) == 0:
            return
        
        # 1. 识别问题扫描
        problem_scans = self._identify_problem_scans(completed_df)
        if len(problem_scans) > 0:
            problem_file = analysis_dir / f"problem_scans_{self.report_timestamp}.csv"
            problem_scans.to_csv(problem_file, index=False, encoding='utf-8')
            
            # 生成问题扫描报告
            problem_report = analysis_dir / f"problem_analysis_{self.report_timestamp}.md"
            with open(problem_report, 'w', encoding='utf-8') as f:
                self._write_problem_analysis(f, problem_scans)
        
        # 2. 识别低置信度扫描
        if 'confidence_score' in completed_df.columns:
            low_confidence = completed_df[completed_df['confidence_score'] < 0.6].copy()
            if len(low_confidence) > 0:
                low_conf_file = analysis_dir / f"low_confidence_scans_{self.report_timestamp}.csv"
                low_confidence.to_csv(low_conf_file, index=False, encoding='utf-8')
    
    def _identify_problem_scans(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别问题扫描"""
        problem_filters = []
        
        # 低SNR扫描
        if 'snr_corrected' in df.columns:
            low_snr = df['snr_corrected'] < 15
            problem_filters.append(low_snr)
        
        # 低质量分扫描
        if 'quality_score_total' in df.columns:
            low_quality = df['quality_score_total'] < 0.7
            problem_filters.append(low_quality)
        
        # 低CNR扫描
        if 'cnr_value' in df.columns:
            low_cnr = df['cnr_value'] < 3.0
            problem_filters.append(low_cnr)
        
        # 组合所有筛选条件
        if problem_filters:
            problem_mask = problem_filters[0]
            for filter_mask in problem_filters[1:]:
                problem_mask = problem_mask | filter_mask
            
            problem_df = df[problem_mask].copy()
            
            # 添加问题类型列
            problem_types = []
            for _, row in problem_df.iterrows():
                types = []
                if row.get('snr_corrected', 100) < 15:
                    types.append('低SNR')
                if row.get('quality_score_total', 1) < 0.7:
                    types.append('低质量分')
                if row.get('cnr_value', 10) < 3.0:
                    types.append('低CNR')
                problem_types.append('、'.join(types))
            
            problem_df['problem_type'] = problem_types
            return problem_df
        
        return pd.DataFrame()
    
    def _write_problem_analysis(self, f, problem_df):
        """写入问题分析报告"""
        f.write(f"""# 问题扫描分析报告
## 报告ID: {self.report_id}
## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 概述
- **总问题扫描数**: {len(problem_df)}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 问题扫描详情
| 扫描ID | 患者ID | 问题类型 | SNR | CNR | 质量分 | 置信度 | 建议 |
|--------|--------|----------|-----|-----|--------|--------|------|
""")
        
        for _, row in problem_df.iterrows():
            scan_id = f"{row.get('patient_id', '')}/{row.get('scan_name', '')}"
            patient_id = row.get('patient_id', '')
            problem_type = row.get('problem_type', '未知')
            snr = row.get('snr_corrected', 0)
            cnr = row.get('cnr_value', 0)
            quality = row.get('quality_score_total', 0)
            confidence = row.get('confidence_score', 0)
            
            # 根据问题类型给出建议
            if '低SNR' in problem_type:
                suggestion = '检查采集参数，考虑重新采集'
            elif '低质量分' in problem_type:
                suggestion = '全面检查图像质量'
            elif '低CNR' in problem_type:
                suggestion = '优化序列参数'
            else:
                suggestion = '需要进一步分析'
            
            f.write(f"| {scan_id} | {patient_id} | {problem_type} | {snr:.1f} | {cnr:.2f} | {quality:.3f} | {confidence:.3f} | {suggestion} |\n")
        
        f.write("\n### 改进建议\n")
        f.write("1. **低SNR扫描**：检查采集时间、线圈位置、序列参数\n")
        f.write("2. **低质量分扫描**：全面评估图像质量，检查伪影\n")
        f.write("3. **低CNR扫描**：优化序列对比度参数\n")
        f.write("4. **定期复查**：建立质量监控机制\n")
    
    def _generate_technical_appendix(self, df: pd.DataFrame, report_dir: Path):
        """生成技术附录"""
        tech_dir = report_dir / "04_technical_appendix"
        
        # 1. 处理统计
        stats_file = tech_dir / f"processing_stats_{self.report_timestamp}.json"
        stats = {
            'report_info': {
                'report_id': self.report_id,
                'timestamp': self.report_timestamp,
                'generated_at': datetime.now().isoformat(),
                'total_scans': len(df),
                'successful_scans': len(df[df['analysis_status'] == 'COMPLETED']),
                'failed_scans': len(df[df['analysis_status'] != 'COMPLETED']),
            },
            'system_info': {
                'python_version': sys.version,
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__ if hasattr(np, '__version__') else 'unknown',
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 2. 字段映射文档
        mapping_file = tech_dir / f"field_mapping_{self.report_timestamp}.md"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            f.write("# 字段映射文档\n\n")
            f.write("| JSON字段 | 报告字段 | 说明 |\n")
            f.write("|----------|----------|------|\n")
            mappings = [
                ('analysis_info.scan_id', 'scan_id', '扫描标识符'),
                ('acquisition.patient_id', 'patient_id', '患者ID'),
                ('acquisition.scan_name', 'scan_name', '扫描名称'),
                ('acquisition.anatomical_region', 'anatomical_region', '解剖区域'),
                ('acquisition.sequence_type', 'sequence_type', '序列类型'),
                ('acquisition.field_strength', 'field_strength', '磁场强度'),
                ('snr_results.recommended.snr', 'snr_corrected', '校正后SNR'),
                ('quality_assessment.snr_rating.level', 'snr_rating', 'SNR评级'),
                ('quality_assessment.cnr_analysis.best_cnr.cnr_value', 'cnr_value', 'CNR值'),
                ('quality_assessment.quality_scores.total_score', 'quality_score_total', '质量总分'),
                ('quality_assessment.overall_confidence.score', 'confidence_score', '置信度分数'),
                ('quality_assessment.overall_confidence.level', 'algorithm_confidence', '置信度等级'),
            ]
            for json_field, report_field, description in mappings:
                f.write(f"| `{json_field}` | `{report_field}` | {description} |\n")
    
    def _create_report_index(self, report_dir: Path, df: pd.DataFrame):
        """创建报告索引文件"""
        index_file = report_dir / f"REPORT_INDEX_{self.report_timestamp}.md"
        
        completed_df = df[df['analysis_status'] == 'COMPLETED'].copy()
        total_scans = len(df)
        successful_scans = len(completed_df)
        success_rate = successful_scans / total_scans * 100 if total_scans > 0 else 0
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(f"""# MRI AutoQA 批量分析报告索引
## 报告ID: {self.report_id}

### 📊 快速统计
- **报告生成时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **扫描总数**: {total_scans}
- **成功分析**: {successful_scans} ({success_rate:.1f}%)
- **分析失败**: {total_scans - successful_scans}
""")
            f.write("\n### 📊 扫描参数清单说明\n")
            f.write("- **文件**: `01_detailed_data/scan_parameters_summary_*.csv`\n")
            f.write("- **内容**: 所有扫描的完整参数清单\n")
            f.write("- **scan_id格式**: 与质量分析结果一致（优先使用处理后的ID格式）\n")
            f.write("- **包含**: 已分析和未分析的扫描参数\n")
            f.write("- **关联字段**: `original_scan_id` 字段保持原始DICOM ID格式\n")
            if len(completed_df) > 0 and 'snr_corrected' in completed_df.columns:
                avg_snr = completed_df['snr_corrected'].mean()
                f.write(f"- **平均校正SNR**: {avg_snr:.1f}\n")
            
            f.write("\n### 📁 文件结构\n")
            f.write("```\n")
            self._print_directory_tree(report_dir, f, max_depth=2)
            f.write("```\n\n")
            
            f.write("### 📄 文件说明\n")
            f.write("| 文件/目录 | 说明 |\n")
            f.write("|-----------|------|\n")
            f.write("| `00_executive_summary/` | 执行摘要，适合管理者阅读 |\n")
            f.write("| `01_detailed_data/` | 详细数据文件，供数据分析 |\n")
            f.write("| `02_visualizations/` | 可视化图表（如生成） |\n")
            f.write("| `03_quality_analysis/` | 质量分析和统计报告 |\n")
            f.write("| `04_technical_appendix/` | 技术附录和元数据 |\n")
            f.write(f"| `REPORT_INDEX_{self.report_timestamp}.md` | 本索引文件 |\n")
            
            f.write("\n### 🚀 使用指南\n")
            f.write("1. **快速浏览**：查看 `00_executive_summary/executive_summary_*.md`\n")
            f.write("2. **数据分析**：使用 `01_detailed_data/detailed_results_*.csv`\n")
            f.write("3. **扫描参数**：查看 `01_detailed_data/scan_parameters_summary_*.csv`\n")
            f.write("4. **问题排查**：查看 `03_quality_analysis/problem_scans_*.csv`\n")
            f.write("5. **技术参考**：查阅 `04_technical_appendix/` 中的文件\n")
            
            f.write("\n### 📧 报告信息\n")
            f.write(f"- **唯一标识**: {self.report_id}\n")
            f.write(f"- **时间戳**: {self.report_timestamp}\n")
            f.write(f"- **数据版本**: single_image_v3_rayleigh\n")
            f.write(f"- **生成工具**: MRI_AutoQA_v2.0\n")
            
            f.write(f"\n---\n*索引生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    def _generate_visualizations(self) -> bool:
        """生成可视化报告 - 修复版"""
        self.log("\n🎨 检查并生成可视化报告...")
        
        # 检查skimage是否可用
        if not self.modules_available.get('skimage', False):
            self.log("⚠️  skimage模块未安装，跳过可视化生成")
            self.log("   如需可视化功能，请安装: pip install scikit-image")
            return True
        
        # 检查可视化模块是否可用
        if not self.modules_available.get('visualization', False):
            self.log("⚠️  可视化模块文件缺失，跳过可视化生成")
            self.log("   确保 visualization.py 和 visualization_config.py 存在")
            return True
        
        try:
            # 尝试导入可视化模块
            from visualization import create_visualization_for_scan
            
            # 1. 查找所有成功的分析结果
            result_files = list(self.results_dir.rglob("**/qa_report.json"))
            
            if not result_files:
                self.log("   未找到分析结果文件")
                return True
            
            completed_results = []
            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    if result_data.get('analysis_status') == 'COMPLETED':
                        completed_results.append((result_file, result_data))
                except:
                    continue
            
            if not completed_results:
                self.log("   没有成功的分析结果")
                return True
            
            self.log(f"   找到 {len(completed_results)} 个成功分析结果")
            
            # 2. 检查哪些扫描已经生成了可视化
            need_viz = []
            already_have_viz = []
            
            for result_file, result_data in completed_results:
                result_dir = result_file.parent
                
                # 检查是否已存在可视化文件
                viz_files = list(result_dir.glob("*.png"))
                if viz_files:
                    already_have_viz.append((result_dir, viz_files))
                else:
                    need_viz.append((result_file, result_data, result_dir))
            
            # 3. 显示统计信息
            self.log(f"   可视化状态统计:")
            self.log(f"     • 已生成可视化: {len(already_have_viz)} 个扫描")
            self.log(f"     • 需生成可视化: {len(need_viz)} 个扫描")
            
            # 显示部分已存在的可视化文件
            if already_have_viz and self.verbose:
                self.log(f"   已存在的可视化文件（前5个）:")
                for result_dir, viz_files in already_have_viz[:5]:
                    self.log(f"     - {result_dir.name}: {[f.name for f in viz_files]}")
                if len(already_have_viz) > 5:
                    self.log(f"     ... 和 {len(already_have_viz) - 5} 个更多扫描")
            
            # 4. 处理单扫描可视化
            if not need_viz:
                self.log("   所有扫描已具备单扫描可视化文件")
                # 继续执行批量可视化
            else:
                self.log(f"\n   开始为 {len(need_viz)} 个扫描生成单扫描可视化...")
                
                success_count = 0
                failed_count = 0
                
                for i, (result_file, result_data, result_dir) in enumerate(need_viz, 1):
                    try:
                        # 进度显示
                        if self.verbose and i % 10 == 0:
                            self.log(f"    处理中: {i}/{len(need_viz)} ({i/len(need_viz)*100:.1f}%)")
                        
                        # 查找对应的NIfTI文件
                        nifti_file = result_dir / "scan.nii.gz"
                        if not nifti_file.exists():
                            # 尝试从原始路径查找
                            scan_id = result_data.get('analysis_info', {}).get('scan_id', '')
                            if '/' in scan_id:
                                patient_id, scan_name = scan_id.split('/', 1)
                                nifti_file = self.paths['input_data'] / patient_id / scan_name / "scan.nii.gz"
                        
                        if not nifti_file.exists():
                            self.log(f"    警告: 无法找到NIfTI文件，跳过 {result_dir.name}")
                            failed_count += 1
                            continue
                        
                        # 加载图像数据
                        import nibabel as nib
                        img = nib.load(nifti_file).get_fdata()
                        
                        # 提取中间切片
                        if len(img.shape) == 3:
                            mid_slice = img.shape[2] // 2
                            slice_img = img[:, :, mid_slice]
                        else:
                            slice_img = img
                        
                        # 检查图像数据有效性
                        if slice_img is None or slice_img.size == 0:
                            self.log(f"    警告: 图像数据无效，跳过 {result_dir.name}")
                            failed_count += 1
                            continue
                        
                        # 生成可视化
                        success = create_visualization_for_scan(
                            result_data,
                            slice_img,
                            str(result_dir),
                            "visualization.png"
                        )
                        
                        if success:
                            success_count += 1
                            if self.verbose:
                                self.log(f"    ✓ 生成可视化: {result_dir.name}")
                        else:
                            failed_count += 1
                            if self.verbose:
                                self.log(f"    ✗ 生成失败: {result_dir.name}")
                                
                    except Exception as e:
                        failed_count += 1
                        if self.verbose:
                            self.log(f"    ✗ 异常失败 {result_dir.name}: {str(e)[:50]}...")
                
                # 5. 显示单扫描可视化生成结果
                self.log(f"\n   单扫描可视化生成完成:")
                self.log(f"     • 成功生成: {success_count}")
                self.log(f"     • 生成失败: {failed_count}")
                self.log(f"     • 已存在: {len(already_have_viz)}")
            
            # 6. 生成批量可视化报告
            self.log(f"\n📊 开始批量可视化报告生成...")
            
            # 直接检查文件是否存在（绕过 modules_available 检查）
            import sys
            from pathlib import Path
            batch_viz_path = Path(__file__).parent / "batch_visualization.py"
            
            if not batch_viz_path.exists():
                self.log(f"   ⚠ 文件不存在: {batch_viz_path}")
                self.log("   请确保 batch_visualization.py 在项目根目录")
                return True
            
            self.log(f"   ✓ 找到文件: {batch_viz_path}")
            
            try:
                # 确保项目目录在 Python 路径中
                if str(Path(__file__).parent) not in sys.path:
                    sys.path.insert(0, str(Path(__file__).parent))
                
                from batch_visualization import visualize_batch_results
                self.log("   ✓ 批量可视化模块导入成功")
                
                # 查找最新的报告目录
                report_dirs = list(self.results_dir.glob("batch_report_*"))
                if not report_dirs:
                    self.log("   ⚠ 未找到报告目录，跳过批量可视化")
                    return True
                
                latest_report_dir = max(report_dirs, key=lambda x: x.stat().st_mtime)
                self.log(f"   使用最新报告目录: {latest_report_dir.name}")
                
                data_dir = latest_report_dir / "01_detailed_data"
                csv_files = list(data_dir.glob("detailed_results_*.csv"))
                
                if not csv_files:
                    self.log(f"   ⚠ 未找到CSV文件: {data_dir}")
                    return True
                
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                self.log(f"   使用数据文件: {latest_csv.name}")
                
                # 创建批量可视化目录
                viz_dir = latest_report_dir / "02_visualizations"
                viz_dir.mkdir(exist_ok=True)
                
                # 生成批量可视化
                self.log(f"   正在生成批量可视化报告...")
                viz_results = visualize_batch_results(
                    str(latest_csv),
                    str(viz_dir)
                )
                
                if viz_results:
                    if isinstance(viz_results, dict):
                        viz_success = sum(viz_results.values())
                        viz_total = len(viz_results)
                        self.log(f"   ✓ 批量可视化完成: {viz_success}/{viz_total} 个图表生成成功")
                    else:
                        self.log(f"   ✓ 批量可视化完成")
                
                self.log("✅ 批量可视化报告生成完成")
                return True
                
            except ImportError as e:
                self.log(f"⚠️  无法导入批量可视化模块: {e}")
                return True
            except Exception as e:
                self.log(f"❌ 批量可视化生成失败: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        except ImportError as e:
            self.log(f"⚠️  无法导入可视化模块: {e}")
            self.log("   跳过可视化生成，继续其他步骤")
            return True
        except Exception as e:
            self.log(f"❌ 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        if self.verbose:
            self.log("\n🧹 清理临时文件...")
        
        try:
            # 查找所有同时存在两个可视化文件的目录
            duplicate_count = 0
            for result_dir in self.results_dir.rglob("**/qa_report.json"):
                result_dir = result_dir.parent
                
                viz1 = result_dir / "visualization.png"
                viz2 = result_dir / "quality_report.png"
                
                if viz1.exists() and viz2.exists():
                    # 删除 quality_report.png，保留 visualization.png
                    viz2.unlink()
                    duplicate_count += 1
            
            if duplicate_count > 0:
                self.log(f"   清理 {duplicate_count} 个重复的可视化文件")
            
        except Exception as e:
            if self.verbose:
                self.log(f"   清理临时文件时出错: {e}")
    
    def _create_latest_symlink(self):
        """创建指向最新报告的符号链接"""
        try:
            latest_link = self.results_dir / "latest_report"
            
            # 移除旧的符号链接（如果存在）
            if latest_link.exists():
                if latest_link.is_symlink():
                    latest_link.unlink()
                else:
                    # 如果是目录，可能需要特殊处理
                    shutil.rmtree(latest_link)
            
            # 创建新的符号链接
            target_dir = f"batch_report_{self.report_timestamp}"
            latest_link.symlink_to(target_dir)
            
            self.log(f"✅ 创建最新报告符号链接: {latest_link} -> {target_dir}")
            
        except Exception as e:
            self.log(f"⚠️ 创建符号链接失败: {e}")
    
    def _print_directory_tree(self, path: Path, file, prefix: str = "", max_depth: int = 3, depth: int = 0):
        """打印目录树"""
        if depth > max_depth:
            file.write(f"{prefix}...\n")
            return
        
        # 获取目录内容
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except:
            return
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            
            if item.is_dir():
                file.write(f"{prefix}{connector}{item.name}/\n")
                next_prefix = prefix + ("    " if is_last else "│   ")
                self._print_directory_tree(item, file, next_prefix, max_depth, depth + 1)
            else:
                file.write(f"{prefix}{connector}{item.name}\n")
    
    def _print_success_summary(self, duration: float, scan_count: int):
        """打印成功摘要"""
        self.log("\n" + "=" * 70)
        self.log("🎉 自动化批量分析完成！")
        self.log("=" * 70)
        
        self.log(f"📊 分析统计:")
        self.log(f"   • 处理扫描: {scan_count}")
        self.log(f"   • 总耗时: {duration:.1f}秒")
        if scan_count > 0:
            self.log(f"   • 平均每扫描: {duration/scan_count:.1f}秒")
        
        self.log(f"\n📁 生成报告:")
        self.log(f"   • 报告ID: {self.report_id}")
        self.log(f"   • 时间戳: {self.report_timestamp}")
        self.log(f"   • 报告目录: batch_report_{self.report_timestamp}/")
        self.log(f"        - 00_executive_summary/ (执行摘要)")
        self.log(f"        - 01_detailed_data/ (详细数据)")
        self.log(f"        - 03_quality_analysis/ (质量分析)")
        self.log(f"        - 04_technical_appendix/ (技术附录)")
        
        # 创建符号链接路径
        latest_link = self.results_dir / "latest_report"
        self.log(f"   • 最新报告链接: {latest_link}")
        
        self.log(f"\n🚀 使用指南:")
        self.log(f"   1. 查看摘要: latest_report/00_executive_summary/")
        self.log(f"   2. 分析数据: latest_report/01_detailed_data/")
        self.log(f"   3. 查看参数: latest_report/01_detailed_data/scan_parameters_summary_*.csv")
        self.log(f"   4. 排查问题: latest_report/03_quality_analysis/")
        self.log(f"   5. 查看索引: latest_report/REPORT_INDEX_*.md")
        
        self.log(f"\n📧 报告位置: {self.results_dir.absolute()}")
        self.log("=" * 70)
    
    def log(self, message: str):
        """日志记录"""
        if self.verbose:
            print(message)
        
        # 同时写入日志文件
        log_dir = self.paths['logs']
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"autoqa_{self.report_timestamp}.log"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MRI AutoQA Launcher - One-command automated batch quality analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认目录运行完整分析
  python autoqa_launcher.py
  
  # 指定输入输出目录
  python autoqa_launcher.py --input /path/to/converted_data --output /path/to/results
  
  # 跳过可视化生成（加快速度）
  python autoqa_launcher.py --skip-vis
  
  # 强制清理旧结果并重新分析
  python autoqa_launcher.py --force-clean
  
  # 安静模式（仅错误信息）
  python autoqa_launcher.py --quiet
  
  # 组合选项
  python autoqa_launcher.py --input my_data --output my_results --skip-vis --force-clean

Output Directory Structure:
  autoqa_results/
  ├── batch_report_YYYYMMDD_HHMMSS/      # 带时间戳的报告目录
  │   ├── 00_executive_summary/          # 执行摘要
  │   ├── 01_detailed_data/              # 详细数据（包含scan_parameters_summary）
  │   ├── 02_visualizations/             # 可视化图表
  │   ├── 03_quality_analysis/           # 质量分析
  │   ├── 04_technical_appendix/         # 技术附录
  │   └── REPORT_INDEX_*.md              # 报告索引
  ├── latest_report -> batch_report_...  # 符号链接
  └── patient_*/                         # 原始分析结果
        """
    )
    
    parser.add_argument('--input', '-i', 
                       help='Input directory containing converted NIfTI files (default: converted_data/)')
    parser.add_argument('--output', '-o',
                       help='Output directory for analysis results (default: autoqa_results/)')
    parser.add_argument('--skip-vis', '-s', action='store_true',
                       help='Skip visualization generation for faster processing')
    parser.add_argument('--force-clean', '-f', action='store_true',
                       help='Force clean output directory before analysis')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode, only show errors and final summary')
    
    args = parser.parse_args()
    
    # 创建启动器
    launcher = AutoQALauncher(verbose=not args.quiet)
    
    # 运行完整流水线
    success = launcher.run_full_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        skip_visualization=args.skip_vis,
        force_clean=args.force_clean
    )
    
    # 退出代码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()