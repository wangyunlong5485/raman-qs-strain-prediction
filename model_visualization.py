"""
Model Performance Visualization Script
Generates various model performance charts
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import json
import warnings
import os
import platform
warnings.filterwarnings('ignore')

# Set Chinese font support
try:
    import matplotlib.font_manager as fm
    
    # Windows font paths
    if platform.system() == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simsun.ttc',
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                fm.fontManager.addfont(fp)
                prop = fm.FontProperties(fname=fp)
                plt.rcParams['font.family'] = prop.get_name()
                print(f"[INFO] Loaded font: {fp}")
                break
    
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"[WARNING] Font setup issue: {e}")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model_and_data(model_path='best_model_resnet.pth', data_path='qs_strain_data.csv'):
    """Load model and data"""
    print("[INFO] Loading model and data...")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Load training log
    log_path = 'training_logs/resnet_training_log_20260403_234627.json'
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            training_log = json.load(f)
    except:
        training_log = None
        print(f"[WARNING] Cannot load training log: {log_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    return checkpoint, training_log, df


def plot_fold_comparison(training_log, save_path='visualization/fold_comparison.png'):
    """Plot fold performance comparison"""
    print("[INFO] Generating fold comparison chart...")
    
    if training_log is None:
        print("[WARNING] Cannot generate, missing training log")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    folds = list(range(1, 11))
    fold_results = training_log['results']
    
    metrics = {
        'R²': ([f['r2'] for f in fold_results], 'R² Score', axes[0, 0]),
        'MSE': ([f['mse'] for f in fold_results], 'MSE', axes[0, 1]),
        'RMSE': ([f['rmse'] for f in fold_results], 'RMSE', axes[1, 0]),
        'MAE': ([f['mae'] for f in fold_results], 'MAE', axes[1, 1])
    }
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (metric_name, (values, ylabel, ax)) in enumerate(metrics.items()):
        color = colors[idx]
        
        # Bar chart
        bars = ax.bar(folds, values, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {mean_val:.4f}')
        
        # Mark best and worst
        best_idx = np.argmax(values) if metric_name == 'R²' else np.argmin(values)
        worst_idx = np.argmin(values) if metric_name == 'R²' else np.argmax(values)
        
        bars[best_idx].set_color('#2ecc71')
        bars[best_idx].set_alpha(1.0)
        bars[worst_idx].set_color('#e74c3c')
        bars[worst_idx].set_alpha(1.0)
        
        # Value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} by Fold', fontsize=12, fontweight='bold')
        ax.set_xticks(folds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_correlation_heatmap(training_log, save_path='visualization/correlation_heatmap.png'):
    """Plot performance metrics correlation heatmap"""
    print("[INFO] Generating correlation heatmap...")
    
    if training_log is None:
        print("[WARNING] Cannot generate, missing training log")
        return
    
    fold_results = training_log['results']
    metrics_df = pd.DataFrame({
        'Fold': [f['fold'] for f in fold_results],
        'MSE': [f['mse'] for f in fold_results],
        'RMSE': [f['rmse'] for f in fold_results],
        'MAE': [f['mae'] for f in fold_results],
        'R²': [f['r2'] for f in fold_results]
    })
    
    corr_matrix = metrics_df[['MSE', 'RMSE', 'MAE', 'R²']].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.4f', 
                cmap='RdYlBu_r', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": .8}, ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Performance Metrics Correlation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_training_summary(training_log, save_path='visualization/training_summary.png'):
    """Plot training summary dashboard"""
    print("[INFO] Generating training summary dashboard...")
    
    if training_log is None:
        print("[WARNING] Cannot generate, missing training log")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    fig.suptitle('ResNet Model Training Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Overall metrics
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    fold_results = training_log['results']
    overall_r2 = np.mean([f['r2'] for f in fold_results])
    overall_mse = np.mean([f['mse'] for f in fold_results])
    overall_rmse = np.mean([f['rmse'] for f in fold_results])
    overall_mae = np.mean([f['mae'] for f in fold_results])
    
    metrics_text = f"""
    Overall Performance:
    R² = {overall_r2:.4f}  |  MSE = {overall_mse:.4f}
    RMSE = {overall_rmse:.4f}  |  MAE = {overall_mae:.4f}
    CV Folds: {training_log['n_splits']}
    """
    ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # R² distribution
    ax2 = fig.add_subplot(gs[1, 0])
    r2_values = [f['r2'] for f in fold_results]
    ax2.hist(r2_values, bins=10, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(r2_values), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(r2_values):.4f}')
    ax2.set_xlabel('R²', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax2.set_title('R² Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # MSE distribution
    ax3 = fig.add_subplot(gs[1, 1])
    mse_values = [f['mse'] for f in fold_results]
    ax3.hist(mse_values, bins=10, color='red', alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(mse_values), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(mse_values):.4f}')
    ax3.set_xlabel('MSE', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax3.set_title('MSE Distribution', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Fold R² bar chart
    ax4 = fig.add_subplot(gs[1, 2])
    folds = list(range(1, 11))
    colors = ['#2ecc71' if r2 > np.mean(r2_values) else '#e74c3c' for r2 in r2_values]
    ax4.bar(folds, r2_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=np.mean(r2_values), color='blue', linestyle='--', linewidth=2, label='Average')
    ax4.set_xlabel('Fold', fontsize=10, fontweight='bold')
    ax4.set_ylabel('R²', fontsize=10, fontweight='bold')
    ax4.set_title('R² by Fold', fontsize=11, fontweight='bold')
    ax4.set_xticks(folds)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # MAE distribution
    ax5 = fig.add_subplot(gs[2, 0])
    mae_values = [f['mae'] for f in fold_results]
    ax5.hist(mae_values, bins=10, color='orange', alpha=0.7, edgecolor='black')
    ax5.axvline(x=np.mean(mae_values), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(mae_values):.4f}')
    ax5.set_xlabel('MAE', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax5.set_title('MAE Distribution', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # RMSE distribution
    ax6 = fig.add_subplot(gs[2, 1])
    rmse_values = [f['rmse'] for f in fold_results]
    ax6.hist(rmse_values, bins=10, color='purple', alpha=0.7, edgecolor='black')
    ax6.axvline(x=np.mean(rmse_values), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rmse_values):.4f}')
    ax6.set_xlabel('RMSE', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax6.set_title('RMSE Distribution', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Performance radar chart
    ax7 = fig.add_subplot(gs[2, 2], projection='polar')
    
    metrics_normalized = {
        'R²': np.mean(r2_values),
        '1-MSE': 1 - np.mean(mse_values) / max(mse_values),
        '1-RMSE': 1 - np.mean(rmse_values) / max(rmse_values),
        '1-MAE': 1 - np.mean(mae_values) / max(mae_values)
    }
    
    categories = list(metrics_normalized.keys())
    values = list(metrics_normalized.values())
    values += values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax7.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax7.fill(angles, values, alpha=0.25, color='#3498db')
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(categories, fontsize=9)
    ax7.set_ylim(0, 1)
    ax7.set_title('Performance Radar', fontsize=11, fontweight='bold', pad=20)
    ax7.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_performance_boxplot(training_log, save_path='visualization/performance_boxplot.png'):
    """Plot performance metrics boxplot"""
    print("[INFO] Generating performance boxplot...")
    
    if training_log is None:
        print("[WARNING] Cannot generate, missing training log")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    fold_results = training_log['results']
    
    metrics_data = {
        'R²': [f['r2'] for f in fold_results],
        'MSE': [f['mse'] for f in fold_results],
        'RMSE': [f['rmse'] for f in fold_results],
        'MAE': [f['mae'] for f in fold_results]
    }
    
    colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    for idx, ((metric_name, data), color) in enumerate(zip(metrics_data.items(), colors)):
        ax = axes[idx // 2, idx % 2]
        
        # Boxplot
        bp = ax.boxplot([data], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        
        # Scatter points
        x = np.random.normal(1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=50, c='navy', zorder=3)
        
        # Mean line
        mean_val = np.mean(data)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.4f}')
        
        # Statistics
        median_val = np.median(data)
        std_val = np.std(data)
        
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution\nMedian: {median_val:.4f}, Std: {std_val:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks([1])
        ax.set_xticklabels([metric_name])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('10-Fold Cross-Validation Performance Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_performance_trend(training_log, save_path='visualization/performance_trend.png'):
    """Plot performance trend across folds"""
    print("[INFO] Generating performance trend chart...")
    
    if training_log is None:
        print("[WARNING] Cannot generate, missing training log")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    folds = list(range(1, 11))
    fold_results = training_log['results']
    
    # Plot trend lines
    ax.plot(folds, [f['r2'] for f in fold_results], 'o-', linewidth=2, 
            markersize=8, label='R²', color='#2ecc71')
    ax.plot(folds, [f['mse'] for f in fold_results], 's-', linewidth=2, 
            markersize=8, label='MSE', color='#e74c3c')
    ax.plot(folds, [f['rmse'] for f in fold_results], '^-', linewidth=2, 
            markersize=8, label='RMSE', color='#9b59b6')
    ax.plot(folds, [f['mae'] for f in fold_results], 'd-', linewidth=2, 
            markersize=8, label='MAE', color='#f39c12')
    
    # Mean lines
    for metric, color in zip(['r2', 'mse', 'rmse', 'mae'], 
                             ['#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']):
        mean_val = np.mean([f[metric] for f in fold_results])
        ax.axhline(y=mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics Trend Across Folds', fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_metric_comparison_bar(training_log, save_path='visualization/metric_comparison.png'):
    """Plot multi-metric comparison bar chart"""
    print("[INFO] Generating multi-metric comparison chart...")
    
    if training_log is None:
        print("[WARNING] Cannot generate, missing training log")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    folds = list(range(1, 11))
    fold_results = training_log['results']
    
    r2_values = [f['r2'] for f in fold_results]
    mse_values = [f['mse'] for f in fold_results]
    rmse_values = [f['rmse'] for f in fold_results]
    mae_values = [f['mae'] for f in fold_results]
    
    x = np.arange(len(folds))
    width = 0.2
    
    # Grouped bar chart
    bars1 = ax.bar(x - 1.5*width, r2_values, width, label='R²', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, mse_values, width, label='MSE', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, rmse_values, width, label='RMSE', color='#9b59b6', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, mae_values, width, label='MAE', color='#f39c12', alpha=0.8)
    
    # Value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics Comparison by Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def main():
    """Main function: generate all visualization charts"""
    print("="*60)
    print("Model Performance Visualization Generation")
    print("="*60)
    
    os.makedirs('visualization', exist_ok=True)
    
    # Load data
    checkpoint, training_log, df = load_model_and_data()
    
    # Generate visualizations
    plot_fold_comparison(training_log)
    plot_correlation_heatmap(training_log)
    plot_training_summary(training_log)
    plot_performance_boxplot(training_log)
    plot_performance_trend(training_log)
    plot_metric_comparison_bar(training_log)
    
    print("\n" + "="*60)
    print("All visualization charts generated successfully!")
    print("="*60)
    print("\nGenerated charts:")
    print("  1. visualization/fold_comparison.png - Fold performance comparison")
    print("  2. visualization/correlation_heatmap.png - Metrics correlation heatmap")
    print("  3. visualization/training_summary.png - Training summary dashboard")
    print("  4. visualization/performance_boxplot.png - Performance metrics boxplot")
    print("  5. visualization/performance_trend.png - Performance trend chart")
    print("  6. visualization/metric_comparison.png - Multi-metric comparison")


if __name__ == "__main__":
    main()
