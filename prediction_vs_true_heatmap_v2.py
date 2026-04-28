"""
预测值 vs 真实值热力图 - 修正版
使用与训练时相同的数据加载方式
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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
        chinese_font = prop
    else:
        chinese_font = None
except:
    chinese_font = None

plt.style.use('seaborn-v0_8-darkgrid')


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dropout_rate):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(filters)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if in_channels != filters:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, filters, 1), nn.BatchNorm1d(filters))

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
    def __init__(self, input_length, params):
        super(ResNet1D, self).__init__()
        kernel_size = params['kernel_size']
        dropout_rate = params['dropout_rate']
        num_blocks = params['num_blocks']
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.blocks = nn.ModuleList()
        in_channels = 64
        for i in range(num_blocks):
            filters = 64 * (2 ** (i // 2))
            self.blocks.append(ResNetBlock(in_channels, filters, kernel_size, dropout_rate))
            in_channels = filters
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(filters, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc_layers(x)
        return x


def load_all_spectra(checkpoint):
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
                    
                    # 处理数据格式
                    if data.ndim >= 2 and data.shape[1] >= 2:
                        # 已经是多列数据
                        if len(data) > 2000:
                            data = data[:2000]
                        elif len(data) < 2000:
                            data = np.pad(data, ((0, 2000 - len(data)), (0, 0)), 'constant')
                        X_list.append(data[:, :2])  # 取前两列
                    elif data.ndim == 1:
                        # 单列数据，需要添加波数列
                        if len(data) > 2000:
                            data = data[:2000]
                        elif len(data) < 2000:
                            data = np.pad(data, (0, 2000 - len(data)), 'constant')
                        wave_numbers = np.arange(len(data))
                        combined = np.column_stack([data, wave_numbers])
                        X_list.append(combined)
                    
                    y_list.append(qs_value)
                except Exception as e:
                    print(f"[WARNING] 无法加载 {txt_file}: {e}")
    
    return np.array(X_list), np.array(y_list)


def normalize_data(X, y, scaler_y=None):
    """标准化数据 - 与训练时一致"""
    scaler_X = StandardScaler()
    
    X_normalized = np.zeros_like(X)
    for i in range(len(X)):
        X_normalized[i, :, 0] = scaler_X.fit_transform(X[i, :, 0].reshape(-1, 1)).flatten()
        X_normalized[i, :, 1] = scaler_X.fit_transform(X[i, :, 1].reshape(-1, 1)).flatten()
    
    if scaler_y is None:
        scaler_y = StandardScaler()
        y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        y_normalized = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    return X_normalized, y_normalized, scaler_y


def get_predictions(checkpoint, batch_size=256):
    """获取预测结果 - 分批处理"""
    params = checkpoint['params']
    model_state = checkpoint['model_state_dict']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    print("[INFO] 加载光谱数据...")
    X, y = load_all_spectra(checkpoint)
    print(f"[INFO] 加载了 {len(X)} 个光谱样本")
    
    # 使用保存的 scaler_y
    scaler_y = checkpoint.get('scaler_y')
    
    # 标准化
    X_normalized, y_scaled, _ = normalize_data(X, y, scaler_y)
    
    # 预测 - 分批处理
    model = ResNet1D(X.shape[1], params).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
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
            all_predictions.append(batch_pred)
            
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{n_batches} 批")
    
    predictions_scaled = np.concatenate(all_predictions)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).squeeze()
    
    return pd.DataFrame({
        'true_qs': y,
        'predicted_qs': predictions
    })


def plot_heatmap(results_df, n_bins=10, save_path='visualization/prediction_vs_true_heatmap_v2.png'):
    """绘制预测值 vs 真实值热力图"""
    print("[INFO] 生成热力图...")
    
    y_true = results_df['true_qs'].values
    y_pred = results_df['predicted_qs'].values
    
    # 计算统计指标
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_true, y_pred)  # 使用正确的决定系数R²
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 创建分箱
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    bins = np.linspace(min_val * 0.95, max_val * 1.05, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 创建热力图矩阵
    heatmap_matrix = np.zeros((n_bins, n_bins))
    for i in range(len(y_true)):
        true_bin = np.digitize(y_true[i], bins) - 1
        pred_bin = np.digitize(y_pred[i], bins) - 1
        true_bin = max(0, min(n_bins - 1, true_bin))
        pred_bin = max(0, min(n_bins - 1, pred_bin))
        heatmap_matrix[n_bins - 1 - pred_bin, true_bin] += 1
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    im = ax.imshow(heatmap_matrix, cmap=cmap, aspect='auto')
    
    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    
    bin_labels = [f'{b:.3f}' for b in bin_centers]
    if chinese_font:
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9, fontproperties=chinese_font)
        ax.set_yticklabels(bin_labels[::-1], fontsize=9, fontproperties=chinese_font)
        ax.set_xlabel('真实 QS 值', fontsize=12, fontproperties=chinese_font)
        ax.set_ylabel('预测 QS 值', fontsize=12, fontproperties=chinese_font)
        ax.set_title(f'预测值 vs 真实值热力图 (最佳模型)\nR² = {r2:.4f}', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    else:
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(bin_labels[::-1], fontsize=9)
        ax.set_xlabel('True QS Value', fontsize=12)
        ax.set_ylabel('Predicted QS Value', fontsize=12)
        ax.set_title(f'Predicted vs True QS Value Heatmap (Best Model)\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    if chinese_font:
        cbar.set_label('样本数量', rotation=270, labelpad=20, fontsize=11, fontproperties=chinese_font)
    else:
        cbar.set_label('Sample Count', rotation=270, labelpad=20, fontsize=11)
    
    # 对角线 - 完美预测线 (从左下到右上)
    # 由于矩阵行被反转，对角线应该从 [0, n_bins-1] 到 [n_bins-1, 0]
    ax.plot([0, n_bins-1], [n_bins-1, 0], 'g--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    # 数值标注
    for i in range(n_bins):
        for j in range(n_bins):
            count = int(heatmap_matrix[i, j])
            if count > 0:
                text_color = 'white' if count > heatmap_matrix.max() * 0.5 else 'black'
                ax.text(j, i, str(count), ha='center', va='center', color=text_color, fontsize=8)
    
    # 统计信息
    stats_text = f'R² = {r2:.4f}\nMSE = {mse:.6f}\nMAE = {mae:.6f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("预测值 vs 真实值热力图生成 (修正版)")
    print("="*60)
    
    os.makedirs('visualization', exist_ok=True)
    
    checkpoint = torch.load('best_model_resnet.pth', map_location='cpu', weights_only=False)
    results_df = get_predictions(checkpoint)
    
    print(f"[INFO] 成功预测 {len(results_df)} 个样本")
    
    plot_heatmap(results_df, n_bins=10)
    
    print("\n" + "="*60)
    print("生成的图表: visualization/prediction_vs_true_heatmap_v2.png")


if __name__ == "__main__":
    main()
