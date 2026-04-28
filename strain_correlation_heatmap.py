"""
菌株间相关性热力图生成脚本
横纵轴都是菌株名，单元格表示预测准确度相关性
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
        x = x.transpose(1, 2)
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc_layers(x)
        return x


def load_strain_data():
    """加载菌株光谱数据 - 与训练时一致"""
    print("[INFO] 加载菌株光谱数据...")
    
    base_path = '给云龙师弟拉曼光谱数据'
    strain_data = {}
    
    for category in ['qs低', 'qs中', 'qs强', 'qs无']:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            continue
            
        for strain_name in os.listdir(category_path):
            strain_path = os.path.join(category_path, strain_name)
            if not os.path.isdir(strain_path):
                continue
            
            txt_files = glob.glob(os.path.join(strain_path, '*.txt'))
            
            if len(txt_files) > 0:
                spectra = []
                for txt_file in txt_files:
                    try:
                        data = np.loadtxt(txt_file)
                        # 处理数据格式 - 与训练时一致
                        if data.ndim >= 2 and data.shape[1] >= 2:
                            # 已经是多列数据（波数+强度）
                            if len(data) > 2000:
                                data = data[:2000]
                            elif len(data) < 2000:
                                data = np.pad(data, ((0, 2000 - len(data)), (0, 0)), 'constant')
                            spectra.append(data[:, :2])  # 取前两列
                        elif data.ndim == 1:
                            # 单列数据（仅强度），需要添加波数列
                            if len(data) > 2000:
                                data = data[:2000]
                            elif len(data) < 2000:
                                data = np.pad(data, (0, 2000 - len(data)), 'constant')
                            wave_numbers = np.arange(len(data))
                            combined = np.column_stack([data, wave_numbers])
                            spectra.append(combined)
                    except Exception as e:
                        print(f"[WARNING] 无法加载 {txt_file}: {e}")
                
                if spectra:
                    strain_data[strain_name] = np.array(spectra)
    
    print(f"[INFO] 成功加载 {len(strain_data)} 个菌株")
    return strain_data


def predict_with_best_model(df, strain_data, checkpoint):
    """使用最佳模型对所有菌株进行预测"""
    print("[INFO] 使用最佳模型进行预测...")
    
    params = checkpoint['params']
    model_state = checkpoint['model_state_dict']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    # 加载QS值映射（小写匹配）
    strain_qs_map = {}
    for _, row in df.iterrows():
        strain_name = str(row['菌株名']).strip().lower()
        if strain_name.endswith('.0'):
            strain_name = strain_name[:-2]
        strain_qs_map[strain_name] = float(row['QS值'])
    
    X_list = []
    y_list = []
    strain_names = []
    
    for strain_name in strain_data.keys():
        key = str(strain_name).lower()
        if key not in strain_qs_map:
            continue
        
        qs_value = strain_qs_map[key]
        spectra = strain_data[strain_name]
        
        if len(spectra) == 0:
            continue
        
        # 计算平均光谱 - 保持2列格式
        avg_spectrum = np.mean(spectra, axis=0)
        
        if len(avg_spectrum) > 2000:
            avg_spectrum = avg_spectrum[:2000]
        elif len(avg_spectrum) < 2000:
            avg_spectrum = np.pad(avg_spectrum, ((0, 2000 - len(avg_spectrum)), (0, 0)), 'constant')
        
        X_list.append(avg_spectrum[:, :2])
        y_list.append(qs_value)
        strain_names.append(strain_name)
    
    if len(X_list) == 0:
        print("[ERROR] 没有匹配的菌株数据!")
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"[INFO] 数据形状: X={X.shape}, y={y.shape}")
    print(f"[INFO] 菌株数量: {len(strain_names)}")
    
    # 使用保存的scaler_y
    scaler_y = checkpoint.get('scaler_y')
    
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
    
    model = ResNet1D(X.shape[1], params).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_normalized).to(device)
        predictions_scaled = model(X_tensor).cpu().numpy().squeeze()
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).squeeze()
    
    results = []
    for i, strain in enumerate(strain_names):
        true_val = y[i]
        pred_val = predictions[i] if predictions.ndim > 0 else predictions
        
        # 计算准确度: 1 - |预测-真实|/真实值，转换为百分比
        if true_val != 0:
            accuracy = max(0, min(100, (1 - abs(pred_val - true_val) / true_val) * 100))
        else:
            accuracy = 100.0 if abs(pred_val) < 0.01 else 0.0
        
        results.append({
            'strain': strain,
            'true_qs': true_val,
            'predicted_qs': pred_val,
            'accuracy': accuracy,
            'error': abs(pred_val - true_val)
        })
    
    return pd.DataFrame(results), y, predictions


def plot_strain_correlation_heatmap(results_df, save_path='visualization/correlation_heatmap.png'):
    """
    绘制菌株间相关性热力图
    横纵轴都是菌株名，单元格表示预测准确度相关性
    """
    print("[INFO] 生成菌株间相关性热力图...")
    
    if results_df is None or len(results_df) == 0:
        print("[ERROR] 没有数据可绘制!")
        return None
    
    # 获取菌株列表
    strains = results_df['strain'].values
    n_strains = len(strains)
    
    # 创建相关性矩阵
    # 这里我们创建一个矩阵，其中每个单元格(i,j)表示菌株i和菌株j的预测准确度相关性
    # 由于每个菌株只有一个准确度值，我们使用一种简化的方式：
    # 矩阵对角线是该菌株自身的准确度
    # 非对角线是两个菌株准确度的调和平均或几何平均
    
    correlation_matrix = np.zeros((n_strains, n_strains))
    accuracies = results_df['accuracy'].values
    
    for i in range(n_strains):
        for j in range(n_strains):
            if i == j:
                # 对角线：菌株自身的准确度
                correlation_matrix[i, j] = accuracies[i]
            else:
                # 非对角线：两个菌株准确度的几何平均
                # 这表示两个菌株预测质量的综合相关性
                correlation_matrix[i, j] = np.sqrt(accuracies[i] * accuracies[j])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(max(14, n_strains * 0.5), max(12, n_strains * 0.4)))
    
    # 绘制热力图
    cmap = sns.diverging_palette(10, 133, s=85, l=55, n=100, as_cmap=True)  # 红到绿
    im = ax.imshow(correlation_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    
    # 设置坐标轴 - 纵轴反转，使对角线从左上到右下
    ax.set_xticks(range(n_strains))
    ax.set_yticks(range(n_strains))
    
    # 纵轴菌株顺序反转，这样矩阵的(0,0)在左下角，对角线从左上到右下
    strains_reversed = strains[::-1]
    
    if chinese_font:
        ax.set_xticklabels(strains, rotation=45, ha='right', fontsize=8, fontproperties=chinese_font)
        ax.set_yticklabels(strains_reversed, fontsize=8, fontproperties=chinese_font)
    else:
        ax.set_xticklabels(strains, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(strains_reversed, fontsize=8)
    
    # 同时反转矩阵的行顺序，保持数据对应关系
    correlation_matrix = correlation_matrix[::-1, :]
    
    # 添加数值标注
    for i in range(n_strains):
        for j in range(n_strains):
            value = correlation_matrix[i, j]
            # 根据数值选择文字颜色
            if value < 50:
                text_color = 'white'
            else:
                text_color = 'black'
            
            # 对角线显示准确度，非对角线显示相关性
            if i == j:
                text = f'{value:.1f}'
                fontweight = 'bold'
            else:
                text = f'{value:.1f}'
                fontweight = 'normal'
            
            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=6, fontweight=fontweight)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    if chinese_font:
        cbar.set_label('预测准确度相关性 (%)', rotation=270, labelpad=20, fontsize=11, fontproperties=chinese_font)
    else:
        cbar.set_label('Prediction Accuracy Correlation (%)', rotation=270, labelpad=20, fontsize=11)
    
    # 标题
    mean_acc = results_df['accuracy'].mean()
    if chinese_font:
        ax.set_title(f'菌株间预测准确度相关性热力图 (最佳模型)\n平均准确度: {mean_acc:.1f}%', 
                    fontsize=14, fontweight='bold', pad=15, fontproperties=chinese_font)
        ax.set_xlabel('菌株', fontsize=11, fontproperties=chinese_font)
        ax.set_ylabel('菌株', fontsize=11, fontproperties=chinese_font)
    else:
        ax.set_title(f'Strain Prediction Accuracy Correlation Heatmap (Best Model)\nMean Accuracy: {mean_acc:.1f}%', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Strain', fontsize=11)
        ax.set_ylabel('Strain', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {save_path}")
    plt.close()
    
    return correlation_matrix


def main():
    """主函数"""
    print("="*60)
    print("菌株间相关性热力图生成 (最佳模型)")
    print("="*60)
    
    os.makedirs('visualization', exist_ok=True)
    
    # 加载模型检查点
    print("[INFO] 加载模型...")
    checkpoint = torch.load('best_model_resnet.pth', map_location='cpu', weights_only=False)
    
    # 加载QS值数据
    df = pd.read_csv('qs_strain_data.csv')
    print(f"[INFO] 加载了 {len(df)} 个菌株的QS值")
    
    # 加载光谱数据
    strain_data = load_strain_data()
    
    # 使用最佳模型进行预测
    results_df, y_true, y_pred = predict_with_best_model(df, strain_data, checkpoint)
    
    if results_df is None:
        print("[ERROR] 无法生成图表，预测失败")
        return
    
    # 生成菌株间相关性热力图
    correlation_matrix = plot_strain_correlation_heatmap(results_df)
    
    print("\n" + "="*60)
    print("生成的图表:")
    print("  1. visualization/correlation_heatmap.png - 菌株间预测准确度相关性热力图")


if __name__ == "__main__":
    main()
