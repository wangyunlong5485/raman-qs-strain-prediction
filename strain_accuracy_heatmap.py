"""
菌株预测准确度热力图生成脚本
使用最佳模型对每个菌株进行预测，生成预测值vs真实值的热力图
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    import matplotlib.font_manager as fm
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"[INFO] 加载字体: {font_path}")
        chinese_font = prop
    else:
        chinese_font = None
        print("[WARNING] 字体文件不存在")
except Exception as e:
    print(f"[WARNING] 字体设置问题: {e}")
    chinese_font = None

plt.style.use('seaborn-v0_8-darkgrid')


class ResNetBlock(nn.Module):
    """ResNet残差块"""
    def __init__(self, in_channels, filters, kernel_size, dropout_rate):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(filters)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 残差连接的维度匹配
        self.shortcut = nn.Sequential()
        if in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, filters, 1),
                nn.BatchNorm1d(filters)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ResNet1D(nn.Module):
    """1D ResNet模型"""
    def __init__(self, input_length, params):
        super(ResNet1D, self).__init__()
        
        kernel_size = params['kernel_size']
        dropout_rate = params['dropout_rate']
        num_blocks = params['num_blocks']
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 残差块
        self.blocks = nn.ModuleList()
        in_channels = 64
        
        for i in range(num_blocks):
            filters = 64 * (2 ** (i // 2))
            self.blocks.append(ResNetBlock(in_channels, filters, kernel_size, dropout_rate))
            in_channels = filters
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(filters, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch, length, 2) -> (batch, 2, length)
        x = x.transpose(1, 2)
        
        x = self.init_conv(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc_layers(x)
        return x


def load_all_spectra():
    """加载所有光谱数据 - 与训练时一致"""
    base_path = '给云龙师弟拉曼光谱数据'
    
    # 加载QS值映射
    df = pd.read_csv('qs_strain_data.csv')
    strain_qs_map = {}
    for _, row in df.iterrows():
        strain_name = str(row['菌株名']).strip().lower()
        if strain_name.endswith('.0'):
            strain_name = strain_name[:-2]
        strain_qs_map[strain_name] = float(row['QS值'])
    
    X_list = []
    y_list = []
    strain_names_list = []
    
    # 遍历所有类别
    for category in ['qs强', 'qs中', 'qs低', 'qs无']:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            continue
        
        for strain_name in os.listdir(category_path):
            strain_path = os.path.join(category_path, strain_name)
            if not os.path.isdir(strain_path):
                continue
            
            key = str(strain_name).lower()
            if key not in strain_qs_map:
                continue
            
            qs_value = strain_qs_map[key]
            
            # 加载该菌株的所有光谱文件
            txt_files = glob.glob(os.path.join(strain_path, '*.txt'))
            for txt_file in txt_files:
                try:
                    data = np.loadtxt(txt_file, encoding='utf-8')
                    
                    # 处理数据格式 - 与训练代码一致
                    if data.ndim >= 2 and data.shape[1] >= 2:
                        # 多列数据：直接使用前两列（波数和强度）
                        if len(data) > 2000:
                            data = data[:2000]
                        elif len(data) < 2000:
                            data = np.pad(data, ((0, 2000 - len(data)), (0, 0)), 'constant')
                        X_sample = data[:, :2]  # 取前两列
                    elif data.ndim == 1:
                        # 单列数据：需要添加波数列
                        intensity = data.flatten()
                        if len(intensity) > 2000:
                            intensity = intensity[:2000]
                        elif len(intensity) < 2000:
                            intensity = np.pad(intensity, (0, 2000 - len(intensity)), 'constant')
                        wave_numbers = np.arange(len(intensity))
                        X_sample = np.column_stack([intensity, wave_numbers])
                    else:
                        continue
                    
                    X_list.append(X_sample)
                    y_list.append(qs_value)
                    strain_names_list.append(strain_name)
                    
                except Exception as e:
                    print(f"[WARNING] 无法加载 {txt_file}: {e}")
    
    return np.array(X_list), np.array(y_list), strain_names_list


def normalize_data(X, y, scaler_y=None):
    """标准化数据 - 与训练时一致：每个样本单独标准化"""
    scaler_X = StandardScaler()
    
    X_normalized = np.zeros_like(X)
    for i in range(len(X)):
        X_normalized[i, :, 0] = scaler_X.fit_transform(X[i, :, 0].reshape(-1, 1)).flatten()
        X_normalized[i, :, 1] = scaler_X.fit_transform(X[i, :, 1].reshape(-1, 1)).flatten()
    
    if scaler_y is None:
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    return X_normalized, y_scaled, scaler_y


def predict_with_best_model(checkpoint):
    """使用最佳模型对所有菌株进行预测"""
    print("[INFO] 使用最佳模型进行预测...")
    
    params = checkpoint['params']
    model_state = checkpoint['model_state_dict']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    # 加载所有光谱数据
    X, y, strain_names = load_all_spectra()
    print(f"[INFO] 加载了 {len(X)} 个光谱样本，对应 {len(set(strain_names))} 个菌株")
    
    # 使用保存的 scaler_y
    scaler_y = checkpoint.get('scaler_y')
    
    # 标准化 - 与训练时一致
    X_normalized, y_scaled, _ = normalize_data(X, y, scaler_y)
    
    # 加载模型
    model = ResNet1D(X.shape[1], params).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    # 预测 - 分批处理
    batch_size = 256
    all_predictions = []
    n_samples = len(X_normalized)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"[INFO] 开始预测，共 {n_batches} 批...")
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_X = X_normalized[start_idx:end_idx]
            
            X_tensor = torch.FloatTensor(batch_X).to(device)
            batch_pred = model(X_tensor).cpu().numpy().squeeze()
            if batch_pred.ndim == 0:
                batch_pred = np.array([batch_pred])
            all_predictions.append(batch_pred)
            
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{n_batches} 批")
    
    predictions_scaled = np.concatenate(all_predictions)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).squeeze()
    
    # 按菌株汇总结果
    strain_results = {}
    for i, strain in enumerate(strain_names):
        if strain not in strain_results:
            strain_results[strain] = {
                'true_qs': y[i],
                'predictions': [],
                'strain_name': strain
            }
        strain_results[strain]['predictions'].append(predictions[i])
    
    # 计算每个菌株的平均预测值和准确度
    results = []
    all_true = []
    all_pred = []
    
    for strain, data in strain_results.items():
        true_val = data['true_qs']
        pred_val = np.mean(data['predictions'])
        
        # 计算准确度: 1 - |预测-真实|/真实值
        if true_val != 0:
            accuracy = max(0, min(100, (1 - abs(pred_val - true_val) / true_val) * 100))
        else:
            accuracy = 100.0 if abs(pred_val) < 0.01 else 0.0
        
        results.append({
            'strain': strain,
            'true_qs': true_val,
            'predicted_qs': pred_val,
            'accuracy': accuracy,
            'error': abs(pred_val - true_val),
            'n_spectra': len(data['predictions'])
        })
        
        all_true.append(true_val)
        all_pred.append(pred_val)
    
    print(f"[INFO] 成功预测 {len(results)} 个菌株")
    
    # 调试：查看高QS值菌株的预测情况
    print("\n[DEBUG] 高QS值菌株 (QS > 0.5) 的预测情况:")
    for r in results:
        if r['true_qs'] > 0.5:
            print(f"  {r['strain']}: 真实={r['true_qs']:.3f}, 预测={r['predicted_qs']:.3f}, 样本数={r['n_spectra']}")
    
    return pd.DataFrame(results), np.array(all_true), np.array(all_pred)


def plot_prediction_heatmap(results_df, save_path='visualization/strain_accuracy_heatmap.png'):
    """绘制菌株预测准确度热力图 - 显示真实值、预测值和准确度"""
    print("[INFO] 生成菌株预测热力图...")
    
    if results_df is None or len(results_df) == 0:
        print("[ERROR] 没有数据可绘制!")
        return None
    
    # 按准确度排序
    results_df = results_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    n_strains = len(results_df)
    
    # 创建3列数据: 真实值、预测值、准确度
    data_matrix = np.zeros((n_strains, 3))
    data_matrix[:, 0] = results_df['true_qs'].values
    data_matrix[:, 1] = results_df['predicted_qs'].values
    data_matrix[:, 2] = results_df['accuracy'].values
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(14, max(10, n_strains * 0.35)), 
                             gridspec_kw={'width_ratios': [1, 1, 0.8]})
    
    # 列标签
    col_labels = ['真实 QS值', '预测 QS值', '准确度(%)']
    
    # 颜色映射
    cmap_qs = 'YlOrRd'  # QS值使用暖色
    cmap_acc = 'RdYlGn'  # 准确度使用红-黄-绿
    
    # 找到QS值的最大值用于归一化
    max_qs = max(results_df['true_qs'].max(), results_df['predicted_qs'].max())
    
    for idx, ax in enumerate(axes):
        if idx < 2:  # QS值列
            values = data_matrix[:, idx]
            im = ax.imshow(values.reshape(-1, 1), cmap=cmap_qs, aspect='auto', 
                          vmin=0, vmax=max_qs)
            
            # 添加数值
            for i in range(n_strains):
                val = values[i]
                color = "white" if val / max_qs < 0.5 else "black"
                ax.text(0, i, f'{val:.3f}', ha="center", va="center", 
                       color=color, fontsize=9, fontweight='bold')
        else:  # 准确度列
            values = data_matrix[:, idx]
            im = ax.imshow(values.reshape(-1, 1), cmap=cmap_acc, aspect='auto', 
                          vmin=0, vmax=100)
            
            # 添加数值
            for i in range(n_strains):
                val = values[i]
                color = "white" if val < 50 else "black"
                ax.text(0, i, f'{val:.1f}%', ha="center", va="center", 
                       color=color, fontsize=9, fontweight='bold')
        
        # 设置坐标轴
        ax.set_xticks([0])
        if chinese_font:
            ax.set_xticklabels([col_labels[idx]], fontsize=11, fontweight='bold', fontproperties=chinese_font)
        else:
            ax.set_xticklabels(['True QS' if idx==0 else 'Predicted QS' if idx==1 else 'Accuracy (%)'], 
                              fontsize=11, fontweight='bold')
        ax.set_yticks(range(n_strains))
        
        if idx == 0:
            ax.set_yticklabels(results_df['strain'].values, fontsize=9)
        else:
            ax.set_yticklabels([])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        if idx == 2:
            if chinese_font:
                cbar.set_label('准确度 (%)', rotation=270, labelpad=15, fontsize=10, fontproperties=chinese_font)
            else:
                cbar.set_label('Accuracy (%)', rotation=270, labelpad=15, fontsize=10)
        else:
            if chinese_font:
                cbar.set_label('QS值', rotation=270, labelpad=15, fontsize=10, fontproperties=chinese_font)
            else:
                cbar.set_label('QS Value', rotation=270, labelpad=15, fontsize=10)
    
    # 总标题
    mean_acc = results_df['accuracy'].mean()
    if chinese_font:
        fig.suptitle(f'菌株QS值预测结果对比 (最佳模型)\n平均准确度: {mean_acc:.1f}%', 
                     fontsize=14, fontweight='bold', y=0.98, fontproperties=chinese_font)
    else:
        fig.suptitle(f'Strain QS Prediction Results (Best Model)\nMean Accuracy: {mean_acc:.1f}%', 
                     fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {save_path}")
    plt.close()
    
    return {
        'mean_accuracy': results_df['accuracy'].mean(),
        'std_accuracy': results_df['accuracy'].std(),
        'min_accuracy': results_df['accuracy'].min(),
        'max_accuracy': results_df['accuracy'].max(),
        'best_strain': results_df.iloc[0]['strain'],
        'worst_strain': results_df.iloc[-1]['strain'],
        'mean_error': results_df['error'].mean()
    }


def plot_prediction_scatter(y_true, y_pred, save_path='visualization/prediction_scatter.png'):
    """绘制预测值vs真实值散点图"""
    print("[INFO] 生成预测散点图...")
    
    # 计算R²
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点
    scatter = ax.scatter(y_true, y_pred, c='steelblue', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    
    # 绘制完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 添加趋势线
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.8, 
            label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax.set_xlabel('True QS Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted QS Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Predicted vs True QS Values (Strain-level)\nR² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {save_path}")
    plt.close()


def plot_error_distribution(results_df, save_path='visualization/error_distribution.png'):
    """绘制误差分布图"""
    print("[INFO] 生成误差分布图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    errors = results_df['error'].values
    accuracies = results_df['accuracy'].values
    
    # 左图：误差直方图
    axes[0].hist(errors, bins=15, color='coral', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(errors):.4f}')
    axes[0].axvline(x=np.median(errors), color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {np.median(errors):.4f}')
    axes[0].set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Strains', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 右图：准确度直方图
    axes[1].hist(accuracies, bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=np.mean(accuracies), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(accuracies):.1f}%')
    axes[1].axvline(x=np.median(accuracies), color='blue', linestyle='--', linewidth=2, 
                    label=f'Median: {np.median(accuracies):.1f}%')
    axes[1].set_xlabel('Prediction Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Strains', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Prediction Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("="*60)
    print("菌株预测准确度热力图生成 (最佳模型)")
    print("="*60)
    
    os.makedirs('visualization', exist_ok=True)
    
    # 加载模型检查点
    print("[INFO] 加载模型...")
    checkpoint = torch.load('best_model_resnet.pth', map_location='cpu', weights_only=False)
    
    # 使用最佳模型进行预测
    results_df, y_true, y_pred = predict_with_best_model(checkpoint)
    
    if results_df is None:
        print("[ERROR] 无法生成图表，预测失败")
        return
    
    # 生成热力图
    stats = plot_prediction_heatmap(results_df)
    
    # 生成散点图
    plot_prediction_scatter(y_true, y_pred)
    
    # 生成误差分布图
    plot_error_distribution(results_df)
    
    if stats:
        print("\n" + "="*60)
        print("统计摘要:")
        print("="*60)
        print(f"平均准确度: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}%")
        print(f"最高准确度: {stats['max_accuracy']:.2f}% (菌株: {stats['best_strain']})")
        print(f"最低准确度: {stats['min_accuracy']:.2f}% (菌株: {stats['worst_strain']})")
        print(f"平均绝对误差: {stats['mean_error']:.4f}")
        
        # 计算R²
        r2 = r2_score(y_true, y_pred)
        print(f"R2: {r2:.4f}")
    
    print("\n" + "="*60)
    print("生成的图表:")
    print("  1. visualization/strain_accuracy_heatmap.png - 菌株预测准确度热力图")
    print("  2. visualization/prediction_scatter.png - 预测值vs真实值散点图")
    print("  3. visualization/error_distribution.png - 误差分布图")


if __name__ == "__main__":
    main()
