"""
特征重要性分析脚本 - 解释ResNet1D模型的光谱特征重要性
使用Integrated Gradients方法分析模型关注的波长区域
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
        chinese_font = prop
        print(f"[INFO] 加载字体: {font_path}")
    else:
        chinese_font = None
except Exception as e:
    print(f"[WARNING] 字体设置问题: {e}")
    chinese_font = None

plt.style.use('seaborn-v0_8-whitegrid')


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
    """加载菌株光谱数据"""
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
            if txt_files:
                spectra = []
                for txt_file in txt_files:
                    try:
                        data = np.loadtxt(txt_file)
                        if data.ndim >= 2 and data.shape[1] >= 2:
                            if len(data) > 2000:
                                data = data[:2000]
                            elif len(data) < 2000:
                                data = np.pad(data, ((0, 2000 - len(data)), (0, 0)), 'constant')
                            spectra.append(data[:, :2])
                        elif data.ndim == 1:
                            if len(data) > 2000:
                                data = data[:2000]
                            elif len(data) < 2000:
                                data = np.pad(data, (0, 2000 - len(data)), 'constant')
                            wave_numbers = np.arange(len(data))
                            combined = np.column_stack([data, wave_numbers])
                            spectra.append(combined)
                    except:
                        pass
                if spectra:
                    strain_data[strain_name] = np.array(spectra)
    
    print(f"[INFO] 成功加载 {len(strain_data)} 个菌株")
    return strain_data


def compute_spectral_gradients(model, X_sample, device):
    """
    使用梯度计算光谱重要性（基于梯度的特征归因方法）
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_sample).unsqueeze(0).to(device)
    X_tensor.requires_grad = True
    
    # 前向传播
    output = model(X_tensor)
    
    # 反向传播获取梯度
    output.backward()
    
    # 获取梯度作为重要性分数
    gradients = X_tensor.grad.cpu().numpy()[0]  # shape: (2000, 2)
    
    # 计算每个波长点的梯度幅值（两个通道的平方和开根号）
    importance = np.sqrt(gradients[:, 0]**2 + gradients[:, 1]**2)
    
    return importance


def compute_integrated_gradients(model, X_sample, baseline, device, n_steps=50):
    """
    计算Integrated Gradients（更稳定的特征重要性方法）
    """
    model.eval()
    
    # 创建插值路径
    alphas = np.linspace(0, 1, n_steps + 1)
    integrated_grads = np.zeros_like(X_sample)
    
    X_tensor = torch.FloatTensor(X_sample).unsqueeze(0).to(device)
    baseline_tensor = torch.FloatTensor(baseline).unsqueeze(0).to(device)
    
    for alpha in alphas[:-1]:
        # 插值点
        interpolated = baseline_tensor + alpha * (X_tensor - baseline_tensor)
        interpolated.requires_grad = True
        
        # 前向传播
        output = model(interpolated)
        
        # 反向传播
        output.backward()
        
        # 累加梯度
        integrated_grads += interpolated.grad.cpu().numpy()[0]
    
    # 平均并乘以输入差值
    integrated_grads = integrated_grads / n_steps * (X_sample - baseline)
    
    # 计算每个波长点的重要性
    importance = np.sqrt(integrated_grads[:, 0]**2 + integrated_grads[:, 1]**2)
    
    return importance


def load_spectra_by_strain(samples_per_strain=10):
    """按菌株加载光谱数据，每个菌株抽取指定数量的样本
    
    数据格式：
    - 第1列：波数 (cm^-1)，范围约 380-1800
    - 第2列：拉曼强度
    
    模型输入格式：
    - 通道0：拉曼强度（标准化后）
    - 通道1：波数索引（0-1999）
    """
    base_path = '给云龙师弟拉曼光谱数据'
    strain_spectra = {}  # 按菌株存储光谱
    strain_wavenumbers = {}  # 按菌株存储波数轴
    
    total_files_found = 0
    total_loaded = 0
    errors = []
    
    for category in ['qs低', 'qs中', 'qs强', 'qs无']:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            print(f"[WARNING] 目录不存在: {category_path}")
            continue
            
        strain_dirs = [d for d in os.listdir(category_path) 
                      if os.path.isdir(os.path.join(category_path, d))]
        print(f"[INFO] 类别 {category}: 发现 {len(strain_dirs)} 个菌株目录")
        
        for strain_name in strain_dirs:
            strain_path = os.path.join(category_path, strain_name)
            
            if strain_name not in strain_spectra:
                strain_spectra[strain_name] = []
                strain_wavenumbers[strain_name] = []
            
            txt_files = glob.glob(os.path.join(strain_path, '*.txt'))
            total_files_found += len(txt_files)
            
            for txt_file in txt_files:
                try:
                    data = np.loadtxt(txt_file, encoding='utf-8')
                    if data.ndim >= 2 and data.shape[1] >= 2:
                        if len(data) > 2000:
                            data = data[:2000]
                        elif len(data) < 2000:
                            data = np.pad(data, ((0, 2000 - len(data)), (0, 0)), 'constant')
                        
                        # 验证波数范围是否合理 (正常拉曼光谱在 300-4000 cm^-1)
                        wavenumbers = data[:, 0]
                        if wavenumbers.min() < 100 or wavenumbers.max() > 20000:
                            errors.append(f"{strain_name}: 波数范围异常 [{wavenumbers.min():.1f}, {wavenumbers.max():.1f}]")
                            continue
                        
                        # 创建模型输入格式：通道0=强度，通道1=波数索引
                        wave_indices = np.arange(len(data))
                        model_input = np.column_stack([data[:, 1], wave_indices])  # [强度, 索引]
                        
                        strain_spectra[strain_name].append(model_input)
                        strain_wavenumbers[strain_name].append(wavenumbers)
                        total_loaded += 1
                    else:
                        errors.append(f"{strain_name}: 数据格式错误，维度={data.ndim}")
                except Exception as e:
                    errors.append(f"{strain_name}: 加载失败 - {str(e)[:50]}")
    
    print(f"[INFO] 总共发现 {total_files_found} 个数据文件，成功加载 {total_loaded} 个")
    
    if len(errors) > 0:
        print(f"[WARNING] 加载错误 ({len(errors)} 个):")
        for err in errors[:5]:  # 只显示前5个错误
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors) - 5} 个错误")
    
    # 从每个菌株中随机抽取指定数量的样本
    selected_spectra = []
    selected_wavenumbers = []
    strain_info = []  # 记录每个样本对应的菌株
    strain_sample_counts = {}  # 记录每个菌株的样本数
    
    for strain_name in sorted(strain_spectra.keys()):
        spectra_list = strain_spectra[strain_name]
        wavenum_list = strain_wavenumbers[strain_name]
        
        if len(spectra_list) == 0:
            print(f"[WARNING] 菌株 {strain_name} 没有成功加载的光谱数据")
            continue
        
        # 随机抽取，如果样本不足则取全部
        n_to_sample = min(samples_per_strain, len(spectra_list))
        indices = np.random.choice(len(spectra_list), n_to_sample, replace=False)
        
        for idx in indices:
            selected_spectra.append(spectra_list[idx])
            selected_wavenumbers.append(wavenum_list[idx])
            strain_info.append(strain_name)
        
        strain_sample_counts[strain_name] = n_to_sample
    
    # 检查样本数量
    expected_samples = len(strain_spectra) * samples_per_strain
    actual_samples = len(selected_spectra)
    
    print(f"\n{'='*60}")
    print(f"[SUMMARY] 数据加载统计")
    print(f"{'='*60}")
    print(f"  菌株总数: {len(strain_spectra)}")
    print(f"  期望样本数: {expected_samples} ({len(strain_spectra)} 菌株 x {samples_per_strain} 样本)")
    print(f"  实际样本数: {actual_samples}")
    print(f"  完成率: {actual_samples/expected_samples*100:.1f}%" if expected_samples > 0 else "  完成率: N/A")
    
    # 找出样本不足的菌株
    insufficient_strains = []
    for strain_name in sorted(strain_spectra.keys()):
        available = len(strain_spectra[strain_name])
        selected = strain_sample_counts.get(strain_name, 0)
        if selected < samples_per_strain:
            insufficient_strains.append((strain_name, available, selected))
    
    if insufficient_strains:
        print(f"\n[WARNING] 以下 {len(insufficient_strains)} 个菌株样本不足 {samples_per_strain} 个:")
        print(f"  菌株名          可用样本    已选择    状态")
        print(f"  {'-'*50}")
        for strain_name, available, selected in insufficient_strains:
            status = "样本不足" if available < samples_per_strain else "异常"
            print(f"  {strain_name:<15} {available:>8} {selected:>8}  {status}")
    else:
        print(f"\n[OK] 所有菌株均成功抽取 {samples_per_strain} 个样本")
    
    if actual_samples == 0:
        raise ValueError("没有成功加载任何光谱数据！请检查数据路径和文件格式。")
    
    return np.array(selected_spectra), np.array(selected_wavenumbers), strain_info


def analyze_spectral_importance(samples_per_strain=10):
    """分析光谱特征重要性"""
    print("="*60)
    print("光谱特征重要性分析 (Integrated Gradients方法)")
    print("="*60)
    
    # 加载模型
    checkpoint = torch.load('best_model_resnet.pth', map_location='cpu', weights_only=False)
    params = checkpoint['params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ResNet1D(2000, params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载数据 - 每个菌株抽取指定数量样本
    print("[INFO] 加载光谱数据...")
    spectra, wavenumbers_array, strain_info = load_spectra_by_strain(samples_per_strain=samples_per_strain)
    print(f"[INFO] 加载了 {len(spectra)} 个样本，来自 {len(set(strain_info))} 个菌株")
    
    # 获取波数轴
    wavenumbers = wavenumbers_array[0]  # 所有样本的波数轴相同
    print(f"[INFO] 波数范围: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm^-1")
    
    # 标准化数据
    X_normalized = np.zeros_like(spectra)
    for i in range(len(spectra)):
        scaler = StandardScaler()
        X_normalized[i, :, 0] = scaler.fit_transform(spectra[i, :, 0].reshape(-1, 1)).flatten()
        X_normalized[i, :, 1] = scaler.fit_transform(spectra[i, :, 1].reshape(-1, 1)).flatten()
    
    # 创建基线（全零或平均光谱）
    baseline = np.zeros_like(X_normalized[0])
    
    print(f"[INFO] 分析 {len(X_normalized)} 个样本的特征重要性...")
    
    # 计算每个样本的特征重要性
    all_importance = []
    for i, x in enumerate(X_normalized):
        importance = compute_integrated_gradients(
            model, x, baseline, device, n_steps=30
        )
        all_importance.append(importance)
        if (i + 1) % 50 == 0:
            print(f"  已完成 {i+1}/{len(X_normalized)}")
    
    all_importance = np.array(all_importance)
    
    # 计算平均重要性和标准差
    mean_importance = np.mean(all_importance, axis=0)
    std_importance = np.std(all_importance, axis=0)
    
    # 获取波长轴（从第一个样本推断）
    sample_wavelengths = wavenumbers
    
    # 保存结果
    os.makedirs('visualization', exist_ok=True)
    
    # 绘制特征重要性图
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 上图：平均特征重要性
    ax1 = axes[0]
    x = np.arange(len(mean_importance))
    
    # 绘制重要性曲线
    ax1.fill_between(x, mean_importance - std_importance, mean_importance + std_importance, 
                      alpha=0.3, color='blue', label='±1 Std Dev')
    ax1.plot(x, mean_importance, 'b-', linewidth=2, label='Mean Importance')
    
    # 标记重要区域（重要性>阈值）
    threshold = np.percentile(mean_importance, 90)
    important_regions = x[mean_importance > threshold]
    
    for region in important_regions[:20]:  # 最多标记20个点
        ax1.axvline(x=region, color='red', alpha=0.3, linestyle='--')
    
    ax1.set_xlabel('Wavenumber Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Importance for QS Prediction\n(Integrated Gradients Method)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 下图：按QS类别分组的重要性
    ax2 = axes[1]
    
    # 加载QS值数据
    df = pd.read_csv('qs_strain_data.csv')
    strain_qs_map = {}
    for _, row in df.iterrows():
        strain_name = str(row['菌株名']).strip().lower()
        if strain_name.endswith('.0'):
            strain_name = strain_name[:-2]
        strain_qs_map[strain_name] = float(row['QS值'])
    
    # 获取每个样本的QS值
    sample_qs = []
    for s in strain_info:
        key = str(s).lower()
        sample_qs.append(strain_qs_map.get(key, 0))
    sample_qs = np.array(sample_qs)
    
    # 按QS值分组
    qs_low_mask = sample_qs < 0.2
    qs_mid_mask = (sample_qs >= 0.2) & (sample_qs < 0.5)
    qs_high_mask = sample_qs >= 0.5
    
    if np.any(qs_low_mask):
        ax2.plot(x, np.mean(all_importance[qs_low_mask], axis=0), 
                'g-', linewidth=2, label='QS < 0.2 (Low)', alpha=0.8)
    if np.any(qs_mid_mask):
        ax2.plot(x, np.mean(all_importance[qs_mid_mask], axis=0), 
                'b-', linewidth=2, label='0.2 ≤ QS < 0.5 (Medium)', alpha=0.8)
    if np.any(qs_high_mask):
        ax2.plot(x, np.mean(all_importance[qs_high_mask], axis=0), 
                'r-', linewidth=2, label='QS ≥ 0.5 (High)', alpha=0.8)
    
    ax2.set_xlabel('Wavenumber Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Importance by QS Category', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization/ig_spectral_importance.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: visualization/ig_spectral_importance.png")
    plt.close()
    
    # 绘制Top重要区域的热力图
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 选择Top 50最重要的波长点
    top_k = 50
    top_indices = np.argsort(mean_importance)[-top_k:]
    
    # 绘制这些区域的重要性
    importance_map = np.zeros((10, len(mean_importance)))
    for i in range(10):
        importance_map[i, :] = mean_importance
    
    # 使用viridis配色方案，高值用亮黄色，低值用深蓝色，对比度更清晰
    im = ax.imshow(importance_map, aspect='auto', cmap='viridis', interpolation='bilinear')
    ax.set_xlabel('Wavenumber Index', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_title('Spectral Feature Importance Heatmap (Top Regions Highlighted)', 
                 fontsize=14, fontweight='bold')
    
    # 标记Top区域
    for idx in top_indices:
        ax.axvline(x=idx, color='white', alpha=0.7, linewidth=1.5)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Importance', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('visualization/ig_importance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: visualization/ig_importance_heatmap.png")
    plt.close()
    
    # 保存重要性数据
    results = {
        'wavenumber_indices': list(range(len(mean_importance))),
        'mean_importance': mean_importance.tolist(),
        'std_importance': std_importance.tolist(),
        'top_50_indices': top_indices.tolist(),
        'top_50_importance': mean_importance[top_indices].tolist()
    }
    
    with open('visualization/ig_importance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] 已保存: visualization/ig_importance_results.json")
    
    # 打印Top 10重要区域
    print("\n[Top 10 重要波长区域]")
    print("-" * 40)
    top_10 = np.argsort(mean_importance)[-10:][::-1]
    for i, idx in enumerate(top_10, 1):
        print(f"{i:2d}. Index {idx:4d}: Importance = {mean_importance[idx]:.6f}")
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)


if __name__ == "__main__":
    analyze_spectral_importance(samples_per_strain=10)
