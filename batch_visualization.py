#!/usr/bin/env python3
"""
Simple Batch Visualization for MRI AutoQA
English version - Minimal and robust
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib to avoid font issues
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SimpleBatchVisualizer:
    """
    Simple batch visualizer for MRI quality analysis
    Minimal dependencies, robust error handling
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.colors = {
            'blue': '#2E86AB',
            'green': '#28A745', 
            'orange': '#F18F01',
            'red': '#DC3545',
            'purple': '#A23B72',
            'gray': '#6C757D'
        }
    
    def create_simple_report(self, csv_path: str, output_dir: str) -> Dict[str, bool]:
        """
        Create simple batch visualization report
        
        Args:
            csv_path: Path to CSV file with results
            output_dir: Output directory for plots
            
        Returns:
            Dictionary of plot creation status
        """
        results = {}
        
        try:
            if self.verbose:
                print(f"Creating visualization report...")
                print(f"  CSV file: {csv_path}")
            
            # 1. Load and validate data
            df = self._load_and_validate_data(csv_path)
            if df is None or len(df) == 0:
                print("No valid data to visualize")
                return results
            
            if self.verbose:
                print(f"  Valid data: {len(df)} scans")
            
            # 2. Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if self.verbose:
                print(f"  Output directory: {output_path}")
            
            # 3. Create simple plots
            results['snr_plot'] = self._create_simple_snr_plot(df, output_path)
            results['cnr_plot'] = self._create_simple_cnr_plot(df, output_path)
            results['quality_plot'] = self._create_simple_quality_plot(df, output_path)
            results['confidence_plot'] = self._create_simple_confidence_plot(df, output_path)
            results['summary_table'] = self._create_simple_summary(df, output_path)
            
            # 4. Create combined report
            if any(results.values()):
                self._create_combined_report(df, output_path, results)
            
            if self.verbose:
                success_count = sum(results.values())
                total_count = len(results)
                print(f"✓ Report created: {success_count}/{total_count} plots successful")
            
            return results
            
        except Exception as e:
            print(f"Error creating visualization report: {e}")
            import traceback
            traceback.print_exc()
            return results
    
    def _load_and_validate_data(self, csv_path: str) -> Optional[pd.DataFrame]:
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(csv_path)
            
            if len(df) == 0:
                print("Empty CSV file")
                return None
            
            # Filter only completed scans
            if 'analysis_status' in df.columns:
                df = df[df['analysis_status'] == 'COMPLETED'].copy()
            
            if len(df) == 0:
                print("No completed scans found")
                return None
            
            # Clean numeric columns
            numeric_cols = ['snr_corrected', 'cnr_value', 'quality_score_total', 'confidence_score']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN in key columns
            key_cols = [col for col in numeric_cols if col in df.columns]
            if key_cols:
                df = df.dropna(subset=key_cols)
            
            if len(df) == 0:
                print("No valid numeric data after cleaning")
                return None
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _create_simple_snr_plot(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Create simple SNR visualization"""
        try:
            if 'snr_corrected' not in df.columns:
                return False
            
            snr_data = df['snr_corrected'].dropna()
            if len(snr_data) < 1:
                return False
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(snr_data, bins=min(10, len(snr_data)), 
                    color=self.colors['blue'], alpha=0.7, edgecolor='black')
            ax1.axvline(snr_data.mean(), color='red', linestyle='-', 
                       linewidth=2, label=f'Mean: {snr_data.mean():.1f}')
            ax1.axvline(snr_data.median(), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {snr_data.median():.1f}')
            ax1.set_xlabel('SNR (Corrected)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('SNR Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(snr_data, patch_artist=True,
                       boxprops=dict(facecolor=self.colors['blue'], alpha=0.7))
            ax2.set_ylabel('SNR Value')
            ax2.set_title('SNR Box Plot')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_path / 'snr_statistics.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creating SNR plot: {e}")
            return False
    
    def _create_simple_cnr_plot(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Create simple CNR visualization"""
        try:
            if 'cnr_value' not in df.columns:
                return False
            
            cnr_data = df['cnr_value'].dropna()
            if len(cnr_data) < 1:
                return False
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Simple histogram with statistics
            n, bins, patches = ax.hist(cnr_data, bins=min(8, len(cnr_data)),
                                      color=self.colors['green'], alpha=0.7, edgecolor='black')
            
            # Add statistics lines
            ax.axvline(cnr_data.mean(), color='red', linestyle='-',
                      linewidth=2, label=f'Mean: {cnr_data.mean():.2f}')
            ax.axvline(cnr_data.median(), color='blue', linestyle='--',
                      linewidth=2, label=f'Median: {cnr_data.median():.2f}')
            
            ax.set_xlabel('CNR Value')
            ax.set_ylabel('Frequency')
            ax.set_title('CNR Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text box with statistics
            stats_text = (f'Statistics:\n'
                         f'N = {len(cnr_data)}\n'
                         f'Mean = {cnr_data.mean():.2f}\n'
                         f'Std = {cnr_data.std():.2f}\n'
                         f'Min = {cnr_data.min():.2f}\n'
                         f'Max = {cnr_data.max():.2f}')
            
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(output_path / 'cnr_statistics.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creating CNR plot: {e}")
            return False
    
    def _create_simple_quality_plot(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Create simple quality score visualization"""
        try:
            if 'quality_score_total' not in df.columns:
                return False
            
            quality_data = df['quality_score_total'].dropna()
            if len(quality_data) < 1:
                return False
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram with quality thresholds
            ax1.hist(quality_data, bins=min(8, len(quality_data)),
                    color=self.colors['orange'], alpha=0.7, edgecolor='black')
            
            # Quality thresholds
            ax1.axvline(0.9, color='green', linestyle='--', alpha=0.7, label='Excellent (≥0.9)')
            ax1.axvline(0.7, color='blue', linestyle='--', alpha=0.7, label='Good (≥0.7)')
            ax1.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Fair (≥0.5)')
            
            ax1.set_xlabel('Quality Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Quality Score Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Quality vs SNR scatter
            if 'snr_corrected' in df.columns:
                scatter = ax2.scatter(df['quality_score_total'], df['snr_corrected'],
                                     c=df.get('confidence_score', 0.7),
                                     cmap='viridis', alpha=0.7, s=60)
                ax2.set_xlabel('Quality Score')
                ax2.set_ylabel('SNR (Corrected)')
                ax2.set_title('Quality vs SNR')
                ax2.grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax2, label='Confidence Score')
            else:
                # Simple box plot if no SNR data
                ax2.boxplot(quality_data, patch_artist=True,
                           boxprops=dict(facecolor=self.colors['orange'], alpha=0.7))
                ax2.set_ylabel('Quality Score')
                ax2.set_title('Quality Score Box Plot')
                ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_path / 'quality_statistics.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creating quality plot: {e}")
            return False
    
    def _create_simple_confidence_plot(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Create simple confidence score visualization"""
        try:
            if 'confidence_score' not in df.columns:
                return False
            
            confidence_data = df['confidence_score'].dropna()
            if len(confidence_data) < 1:
                return False
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram with confidence levels
            n, bins, patches = ax1.hist(confidence_data, bins=min(8, len(confidence_data)),
                                       color=self.colors['purple'], alpha=0.7, edgecolor='black')
            
            # Confidence thresholds
            ax1.axvline(0.8, color='green', linestyle='--', alpha=0.7, label='High (≥0.8)')
            ax1.axvline(0.6, color='orange', linestyle='--', alpha=0.7, label='Medium (≥0.6)')
            ax1.axvline(0.4, color='red', linestyle='--', alpha=0.7, label='Low (<0.6)')
            
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Confidence Score Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Confidence level pie chart
            confidence_levels = []
            for score in confidence_data:
                if score >= 0.8:
                    confidence_levels.append('High')
                elif score >= 0.6:
                    confidence_levels.append('Medium')
                else:
                    confidence_levels.append('Low')
            
            level_counts = pd.Series(confidence_levels).value_counts()
            colors_pie = [self.colors['green'], self.colors['orange'], self.colors['red']]
            
            ax2.pie(level_counts.values, labels=level_counts.index,
                   colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Confidence Level Distribution')
            
            plt.tight_layout()
            plt.savefig(output_path / 'confidence_statistics.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creating confidence plot: {e}")
            return False
    
    def _create_simple_summary(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Create simple summary table - Horizontal layout"""
        try:
            if len(df) < 1:
                return False
        
            # 使用更宽的图形以适应多列
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('off')
        
            # 准备横排数据 - 5列：指标 + 4个数值
            summary_data = []
        
            # 表头
            summary_data.append(['Metric', 'Value 1', 'Value 2', 'Value 3', 'Value 4'])
        
            # Basic info - 只占一列
            summary_data.append(['📊 Basic Info', f'Scans: {len(df)}', '', '', ''])
        
            # SNR summary - 4个数值平铺
            if 'snr_corrected' in df.columns:
                summary_data.append(['🎯 SNR', 
                                   f'Mean: {df["snr_corrected"].mean():.1f}',
                                   f'Std: {df["snr_corrected"].std():.1f}',
                                   f'Min: {df["snr_corrected"].min():.1f}',
                                   f'Max: {df["snr_corrected"].max():.1f}'])
            else:
                summary_data.append(['🎯 SNR', 'N/A', 'N/A', 'N/A', 'N/A'])
        
            # CNR summary
            if 'cnr_value' in df.columns:
                summary_data.append(['🎯 CNR',
                                   f'Mean: {df["cnr_value"].mean():.2f}',
                                   f'Std: {df["cnr_value"].std():.2f}',
                                   f'Min: {df["cnr_value"].min():.2f}',
                                   f'Max: {df["cnr_value"].max():.2f}'])
            else:
                summary_data.append(['🎯 CNR', 'N/A', 'N/A', 'N/A', 'N/A'])
        
            # Quality summary
            if 'quality_score_total' in df.columns:
                quality_mean = df['quality_score_total'].mean()
                excellent = len(df[df['quality_score_total'] >= 0.9])
                good = len(df[(df['quality_score_total'] >= 0.7) & (df['quality_score_total'] < 0.9)])
                fair = len(df[(df['quality_score_total'] >= 0.5) & (df['quality_score_total'] < 0.7)])
                poor = len(df[df['quality_score_total'] < 0.5])
            
                summary_data.append(['📈 Quality',
                                   f'Mean: {quality_mean:.3f}',
                                   f'Excellent: {excellent}',
                                   f'Good: {good}',
                                   f'Fair: {fair}'])
            else:
                summary_data.append(['📈 Quality', 'N/A', 'N/A', 'N/A', 'N/A'])
        
            # Confidence summary
            if 'confidence_score' in df.columns:
                conf_mean = df['confidence_score'].mean()
                high = len(df[df['confidence_score'] >= 0.8])
                medium = len(df[(df['confidence_score'] >= 0.6) & (df['confidence_score'] < 0.8)])
                low = len(df[df['confidence_score'] < 0.6])
            
                summary_data.append(['🔬 Confidence',
                                   f'Mean: {conf_mean:.3f}',
                                   f'High: {high}',
                                   f'Medium: {medium}',
                                   f'Low: {low}'])
            else:
                summary_data.append(['🔬 Confidence', 'N/A', 'N/A', 'N/A', 'N/A'])
        
            # 创建5列表格
            table = ax.table(cellText=summary_data, loc='center',
                            cellLoc='center', colWidths=[0.18, 0.205, 0.205, 0.205, 0.205])
        
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
        
            # 设置表格样式
            for i in range(len(summary_data)):
                # 第一列（指标名称）
                table[(i, 0)].set_facecolor(self.colors['gray'])
                table[(i, 0)].set_text_props(color='white', fontweight='bold', fontsize=11)
            
                # 数值列
                for j in range(1, 5):
                    table[(i, j)].set_facecolor('white')
            
                # 表头行特殊样式
                if i == 0:
                    for j in range(5):
                        table[(i, j)].set_facecolor(self.colors['blue'])
                        table[(i, j)].set_text_props(color='white', fontweight='bold')
        
            ax.set_title('MRI Batch Analysis Summary - Key Metrics', 
                        fontsize=14, fontweight='bold', y=1.05)
        
            plt.tight_layout()
            plt.savefig(output_path / 'summary_table.png', dpi=150, bbox_inches='tight')
            plt.close()
        
            return True
        
    
            
        except Exception as e:
            print(f"Error creating summary table: {e}")
            return False
    
    def _create_combined_report(self, df: pd.DataFrame, output_path: Path, results: Dict[str, bool]):
        """Create a combined PDF report"""
        try:
            if len(df) < 1:
                return
            
            # Create a simple combined figure with key metrics
            fig = plt.figure(figsize=(15, 10))
            
            # Only include successful plots
            successful_plots = [name for name, success in results.items() if success]
            
            if len(successful_plots) >= 3:
                # Create a 2x2 grid
                gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                
                # Plot 1: SNR (top-left)
                ax1 = fig.add_subplot(gs[0, 0])
                if 'snr_corrected' in df.columns:
                    ax1.hist(df['snr_corrected'].dropna(), bins=min(8, len(df)),
                            color=self.colors['blue'], alpha=0.7)
                    ax1.set_xlabel('SNR')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('SNR Distribution')
                    ax1.grid(True, alpha=0.3)
                
                # Plot 2: Quality (top-right)
                ax2 = fig.add_subplot(gs[0, 1])
                if 'quality_score_total' in df.columns:
                    ax2.hist(df['quality_score_total'].dropna(), bins=min(8, len(df)),
                            color=self.colors['orange'], alpha=0.7)
                    ax2.set_xlabel('Quality Score')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Quality Score Distribution')
                    ax2.grid(True, alpha=0.3)
                
                # Plot 3: CNR (bottom-left)
                ax3 = fig.add_subplot(gs[1, 0])
                if 'cnr_value' in df.columns:
                    ax3.hist(df['cnr_value'].dropna(), bins=min(8, len(df)),
                            color=self.colors['green'], alpha=0.7)
                    ax3.set_xlabel('CNR')
                    ax3.set_ylabel('Frequency')
                    ax3.set_title('CNR Distribution')
                    ax3.grid(True, alpha=0.3)
                
                # Plot 4: Confidence (bottom-right)
                ax4 = fig.add_subplot(gs[1, 1])
                if 'confidence_score' in df.columns:
                    ax4.hist(df['confidence_score'].dropna(), bins=min(8, len(df)),
                            color=self.colors['purple'], alpha=0.7)
                    ax4.set_xlabel('Confidence Score')
                    ax4.set_ylabel('Frequency')
                    ax4.set_title('Confidence Distribution')
                    ax4.grid(True, alpha=0.3)
                
                # Add overall title
                fig.suptitle(f'MRI Batch Quality Analysis Report\nTotal Scans: {len(df)}', 
                           fontsize=16, fontweight='bold', y=0.98)
                
                plt.tight_layout()
                plt.savefig(output_path / 'combined_report.pdf', 
                          dpi=150, bbox_inches='tight', format='pdf')
                plt.close()
                
                if self.verbose:
                    print(f"✓ Combined PDF report created")
            
        except Exception as e:
            print(f"Error creating combined report: {e}")


# Convenience function for backward compatibility
def visualize_batch_results(csv_path: str, output_dir: str) -> Dict[str, bool]:
    """
    Convenience function for batch visualization
    
    Args:
        csv_path: Path to CSV file
        output_dir: Output directory
        
    Returns:
        Dictionary of plot creation status
    """
    visualizer = SimpleBatchVisualizer(verbose=True)
    return visualizer.create_simple_report(csv_path, output_dir)


if __name__ == "__main__":
    # Test the visualizer
    print("Testing Simple Batch Visualizer...")
    
    # Create test data
    test_data = {
        'scan_id': ['p001/T1_1', 'p001/T1_2', 'p002/T1_1', 'p002/T1_2'],
        'analysis_status': ['COMPLETED'] * 4,
        'snr_corrected': [24.93, 24.85, 65.49, 48.38],
        'cnr_value': [5.79, 4.49, 2.44, 3.00],
        'quality_score_total': [0.881, 0.865, 0.873, 0.884],
        'confidence_score': [0.640, 0.724, 0.699, 0.696],
    }
    
    test_df = pd.DataFrame(test_data)
    test_csv = 'test_batch_data.csv'
    test_df.to_csv(test_csv, index=False)
    
    # Test visualization
    visualizer = SimpleBatchVisualizer(verbose=True)
    results = visualizer.create_simple_report(test_csv, './test_output')
    
    print(f"\nTest results:")
    for plot_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {plot_name}: {status}")
    
    # Cleanup
    import os
    if os.path.exists(test_csv):
        os.remove(test_csv)
    
    print("\n✅ Test completed!")