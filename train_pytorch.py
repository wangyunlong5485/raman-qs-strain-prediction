"""
拉曼光谱QS值预测 - PyTorch版本
支持1D CNN和ResNet两种架构
针对RTX 4070 Ti (8GB)优化

使用方法:
  # 快速训练(CNN): python train_pytorch.py --model cnn --quick
  # 快速训练(ResNet): python train_pytorch.py --model resnet --quick
  # 完整优化: python train_pytorch.py --optimize
  # 对比两个模型: python train_pytorch.py --model both --quick
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import warnings
import argparse
import gc
import json
from datetime import datetime
from scipy import stats
from tqdm import tqdm
warnings.filterwarnings('ignore')
import optuna

# 设置中文字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 检测GPU
print("\n" + "="*70)
print("GPU Detection:")
print("="*70)
if torch.cuda.is_available():
    print(f"[OK] GPU detected: {torch.cuda.device_count()} device(s)")
    for i in range(torch.cuda.device_count()):
        print(f"  - {torch.cuda.get_device_name(i)}")
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    device = torch.device('cuda:0')
    print(f"[OK] Using device: {device}")
else:
    print("[WARN] No GPU detected, using CPU")
    device = torch.device('cpu')


class RamanDataset(Dataset):
    """拉曼光谱数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RamanDataLoader:
    """加载和处理拉曼光谱数据"""

    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.strain_qs_map = {}
        self.load_qs_values()

    def load_qs_values(self):
        """从CSV文件加载QS值"""
        df = pd.read_csv(self.csv_path)

        for idx, row in df.iterrows():
            strain_name = str(row['菌株名']).strip().lower()
            if strain_name.endswith('.0'):
                strain_name = strain_name[:-2]
            qs_value = float(row['QS值'])
            self.strain_qs_map[strain_name] = qs_value

        print(f"加载了 {len(self.strain_qs_map)} 个菌株的QS值")

    def load_raman_spectrum(self, txt_path):
        """加载单个拉曼光谱TXT文件"""
        try:
            data = np.loadtxt(txt_path, encoding='utf-8')
            if data.ndim >= 2 and data.shape[1] >= 2:
                return data
            elif data.ndim == 1:
                return data.reshape(-1, 1)
            return None
        except Exception as e:
            return None

    def load_all_data(self):
        """加载所有菌株的拉曼光谱数据"""
        X = []
        y = []
        strain_names = []

        qs_categories = ['qs强', 'qs中', 'qs低', 'qs无']

        for qs_cat in qs_categories:
            qs_dir = os.path.join(self.data_dir, qs_cat)
            if not os.path.exists(qs_dir):
                continue

            for strain_name in os.listdir(qs_dir):
                strain_dir = os.path.join(qs_dir, strain_name)
                if not os.path.isdir(strain_dir):
                    continue

                key = str(strain_name).lower()
                if key not in self.strain_qs_map:
                    continue

                qs_value = self.strain_qs_map[key]

                txt_files = [f for f in os.listdir(strain_dir) if f.endswith('.txt')]

                for txt_file in txt_files:
                    txt_path = os.path.join(strain_dir, txt_file)
                    spectrum = self.load_raman_spectrum(txt_path)

                    if spectrum is not None and len(spectrum) > 0:
                        X.append(spectrum)
                        y.append(qs_value)
                        strain_names.append(strain_name)

        return np.array(X), np.array(y), np.array(strain_names)


class CNN1D(nn.Module):
    """1D CNN模型"""
    def __init__(self, input_length, params):
        super(CNN1D, self).__init__()
        
        num_blocks = params['num_blocks']
        filters_base = params['filters_base']
        kernel_size = params['kernel_size']
        dropout_rate = params['dropout_rate']
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        in_channels = 2
        
        for i in range(num_blocks):
            filters = filters_base * (2 ** i)
            
            self.convs.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Conv1d(filters, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate)
            ])
            in_channels = filters
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(filters_base * (2 ** (num_blocks - 1)), 128),
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
        
        for conv in self.convs:
            x = conv(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc_layers(x)
        return x


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


def normalize_data(X, y):
    """标准化数据"""
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_normalized = np.zeros_like(X)
    for i in range(len(X)):
        X_normalized[i, :, 0] = scaler_X.fit_transform(X[i, :, 0].reshape(-1, 1)).flatten()
        X_normalized[i, :, 1] = scaler_X.fit_transform(X[i, :, 1].reshape(-1, 1)).flatten()

    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X_normalized, y_normalized, scaler_y


def train_with_cross_validation(X, y, model_type='cnn', params=None, n_splits=10, device='cuda', save_best_model=True):
    """十折交叉验证训练 - 使用tqdm实时进度条"""

    if params is None:
        params = {
            'num_blocks': 3,
            'filters_base': 32,
            'kernel_size': 7,
            'dropout_rate': 0.2077735584403887,
            'learning_rate': 0.0005581372793160988,
            'batch_size': 128,
            'epochs': 200
        }

    # 标准化数据
    X_normalized, y_normalized, scaler_y = normalize_data(X, y)

    # 十折交叉验证
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    all_predictions = []
    all_true_values = []
    all_histories = []

    print(f"\n{'='*60}")
    print(f"开始 {n_splits} 折交叉验证 - 模型: {model_type.upper()}")
    print(f"Batch Size: {params['batch_size']}, Epochs: {params['epochs']}")
    print(f"{'='*60}")

    # 创建总进度条
    pbar = tqdm(range(n_splits), desc=f"{model_type.upper()} 训练进度", unit="fold")

    for fold in pbar:
        train_idx, test_idx = list(kfold.split(X_normalized))[fold]

        X_train, X_test = X_normalized[train_idx], X_normalized[test_idx]
        y_train, y_test = y_normalized[train_idx], y_normalized[test_idx]

        # 构建模型
        if model_type == 'cnn':
            model = CNN1D(X_train.shape[1], params)
        else:
            model = ResNet1D(X_train.shape[1], params)

        model = model.to(device)

        # 创建数据加载器
        train_dataset = RamanDataset(X_train, y_train)
        test_dataset = RamanDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

        # 训练 - 使用tqdm
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 30

        # Epoch进度条
        epoch_pbar = tqdm(range(params['epochs']), desc=f"  Fold {fold+1}/{n_splits}",
                         leave=False, unit="epoch")

        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(test_loader)

            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # 更新进度条显示
            epoch_pbar.set_postfix({
                'Train': f'{train_loss:.6f}',
                'Val': f'{val_loss:.6f}',
                'Best': f'{best_val_loss:.6f}'
            })

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                epoch_pbar.set_postfix({
                    'Train': f'{train_loss:.6f}',
                    'Val': f'{val_loss:.6f}',
                    'Status': 'Early Stop'
                })
                break

        epoch_pbar.close()

        # 加载最佳模型
        model.load_state_dict(best_model_state)

        # 预测
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        y_pred_normalized = np.array(all_preds)
        y_test_normalized = np.array(all_targets)

        # 反标准化
        y_test_original = scaler_y.inverse_transform(y_test_normalized.reshape(-1, 1)).flatten()
        y_pred_original = scaler_y.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()

        # 计算指标
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)

        fold_results.append({
            'fold': fold + 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

        all_predictions.extend(y_pred_original)
        all_true_values.extend(y_test_original)
        all_histories.append(history)

        # 更新总进度条
        pbar.set_postfix({
            'MSE': f'{mse:.6f}',
            'R²': f'{r2:.6f}',
            'Epochs': len(history['train_loss'])
        })

        # 释放内存
        del model, train_dataset, test_dataset, train_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    pbar.close()

    # 汇总结果
    results_df = pd.DataFrame(fold_results)
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} - 交叉验证汇总结果")
    print(f"{'='*60}")
    print(f"平均 MSE: {results_df['mse'].mean():.6f} ± {results_df['mse'].std():.6f}")
    print(f"平均 RMSE: {results_df['rmse'].mean():.6f} ± {results_df['rmse'].std():.6f}")
    print(f"平均 MAE: {results_df['mae'].mean():.6f} ± {results_df['mae'].std():.6f}")
    print(f"平均 R²: {results_df['r2'].mean():.6f} ± {results_df['r2'].std():.6f}")

    # 保存最佳模型权重
    if save_best_model:
        # 找到MSE最小的fold
        best_fold_idx = results_df['mse'].idxmin()
        print(f"\n[INFO] 最佳模型来自 Fold {best_fold_idx + 1} (MSE: {results_df['mse'][best_fold_idx]:.6f})")

        # 重新训练最佳fold的模型并保存
        train_idx, test_idx = list(kfold.split(X_normalized))[best_fold_idx]
        X_train, X_test = X_normalized[train_idx], X_normalized[test_idx]
        y_train, y_test = y_normalized[train_idx], y_normalized[test_idx]

        # 构建模型
        if model_type == 'cnn':
            model = CNN1D(X_train.shape[1], params)
        else:
            model = ResNet1D(X_train.shape[1], params)

        model = model.to(device)

        # 创建数据加载器
        train_dataset = RamanDataset(X_train, y_train)
        test_dataset = RamanDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

        # 训练最佳模型
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 30

        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(test_loader)

            scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                break

        # 加载最佳权重并保存
        model.load_state_dict(best_model_state)
        model_save_path = f'best_model_{model_type}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'params': params,
            'scaler_y': scaler_y,
            'input_shape': X.shape[1:],
            'best_val_loss': best_val_loss,
            'best_fold': best_fold_idx + 1
        }, model_save_path)
        print(f"[OK] 最佳模型权重已保存到: {model_save_path}")

    # 生成可视化图表
    plot_training_results(all_histories, all_true_values, all_predictions, results_df, model_type, n_splits)

    return results_df, np.mean(results_df['mse'])


def plot_training_results(all_histories, all_true_values, all_predictions, results_df, model_type, n_splits):
    """生成训练过程和结果的可视化图表"""
    
    # 创建输出目录
    os.makedirs('training_logs', exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 训练损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    for i, history in enumerate(all_histories):
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], alpha=0.3, linewidth=1, label=f'Fold {i+1}')
        ax1.plot(epochs, history['val_loss'], alpha=0.3, linewidth=1, linestyle='--')

    max_epochs = max([len(h['train_loss']) for h in all_histories])
    avg_train_loss = []
    avg_val_loss = []
    for epoch in range(1, max_epochs + 1):
        train_losses = [h['train_loss'][epoch-1] for h in all_histories if epoch <= len(h['train_loss'])]
        val_losses = [h['val_loss'][epoch-1] for h in all_histories if epoch <= len(h['val_loss'])]
        avg_train_loss.append(np.mean(train_losses))
        avg_val_loss.append(np.mean(val_losses))

    ax1.plot(range(1, max_epochs+1), avg_train_loss, 'b-', linewidth=3, label='Avg Train Loss')
    ax1.plot(range(1, max_epochs+1), avg_val_loss, 'r-', linewidth=3, label='Avg Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_type.upper()} Training Loss Curve (10-Fold Avg)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 预测值 vs 真实值散点图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(all_true_values, all_predictions, alpha=0.5, s=10, edgecolors='none')

    min_val = min(min(all_true_values), min(all_predictions))
    max_val = max(max(all_true_values), max(all_predictions))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')

    r2 = r2_score(all_true_values, all_predictions)
    ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes,
             fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('真实QS值', fontsize=12)
    ax2.set_ylabel('预测QS值', fontsize=12)
    ax2.set_title('预测值 vs 真实值', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 残差图
    ax3 = fig.add_subplot(gs[0, 2])
    residuals = np.array(all_predictions) - np.array(all_true_values)
    ax3.scatter(all_true_values, residuals, alpha=0.5, s=10, edgecolors='none')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('真实QS值', fontsize=12)
    ax3.set_ylabel('残差 (预测-真实)', fontsize=12)
    ax3.set_title('残差分布', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 残差直方图
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7, density=True)

    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'正态分布\nμ={mu:.4f}, σ={sigma:.4f}')

    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零残差')
    ax4.set_xlabel('残差', fontsize=12)
    ax4.set_ylabel('密度', fontsize=12)
    ax4.set_title('残差分布直方图', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. 各fold性能对比
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(results_df))
    width = 0.2

    ax5.bar(x - 1.5*width, results_df['mse'], width, label='MSE', alpha=0.8)
    ax5.bar(x - 0.5*width, results_df['rmse'], width, label='RMSE', alpha=0.8)
    ax5.bar(x + 0.5*width, results_df['mae'], width, label='MAE', alpha=0.8)
    ax5.set_xlabel('Fold', fontsize=12)
    ax5.set_ylabel('值', fontsize=12)
    ax5.set_title('各Fold性能指标', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'Fold {i+1}' for i in range(len(results_df))], rotation=45)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. R² 柱状图
    ax6 = fig.add_subplot(gs[1, 2])
    colors = ['green' if r2 > results_df['r2'].median() else 'orange' for r2 in results_df['r2']]
    ax6.bar(x, results_df['r2'], color=colors, alpha=0.8, edgecolor='black')
    ax6.axhline(y=results_df['r2'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'平均值: {results_df["r2"].mean():.4f}')
    ax6.set_xlabel('Fold', fontsize=12)
    ax6.set_ylabel('R²', fontsize=12)
    ax6.set_title('各Fold R² 得分', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'Fold {i+1}' for i in range(len(results_df))], rotation=45)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. 性能指标箱线图
    ax7 = fig.add_subplot(gs[2, 0])
    data_to_plot = [results_df['mse'], results_df['rmse'], results_df['mae'], results_df['r2']]
    bp = ax7.boxplot(data_to_plot, labels=['MSE', 'RMSE', 'MAE', 'R²'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax7.set_ylabel('值', fontsize=12)
    ax7.set_title('性能指标分布 (10折)', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. 误差随QS值分布
    ax8 = fig.add_subplot(gs[2, 1])
    bins = np.linspace(min(all_true_values), max(all_true_values), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mse_by_bin = []
    for i in range(len(bins) - 1):
        mask = (np.array(all_true_values) >= bins[i]) & (np.array(all_true_values) < bins[i+1])
        if mask.sum() > 0:
            mse_by_bin.append(mean_squared_error(
                np.array(all_true_values)[mask],
                np.array(all_predictions)[mask]
            ))
        else:
            mse_by_bin.append(0)

    ax8.plot(bin_centers, mse_by_bin, 'o-', linewidth=2, markersize=8)
    ax8.set_xlabel('QS值区间', fontsize=12)
    ax8.set_ylabel('平均MSE', fontsize=12)
    ax8.set_title('不同QS值区间的预测误差', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9. 训练过程对比
    ax9 = fig.add_subplot(gs[2, 2])
    for i, history in enumerate(all_histories):
        epochs = range(1, len(history['val_loss']) + 1)
        ax9.plot(epochs, history['val_loss'], alpha=0.5, linewidth=1, label=f'Fold {i+1}')
    
    ax9.set_xlabel('Epoch', fontsize=12)
    ax9.set_ylabel('验证Loss', fontsize=12)
    ax9.set_title('各Fold验证损失曲线对比', fontsize=14, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # 保存大图
    plt.savefig(f'training_logs/{model_type}_training_results_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    print(f"\n[OK] Training results saved: training_logs/{model_type}_training_results_{timestamp}.png")

    # 保存训练日志
    log_data = {
        'timestamp': timestamp,
        'model_type': model_type,
        'n_splits': n_splits,
        'results': results_df.to_dict('records'),
        'statistics': {
            'mean_mse': float(results_df['mse'].mean()),
            'std_mse': float(results_df['mse'].std()),
            'mean_rmse': float(results_df['rmse'].mean()),
            'std_rmse': float(results_df['rmse'].std()),
            'mean_mae': float(results_df['mae'].mean()),
            'std_mae': float(results_df['mae'].std()),
            'mean_r2': float(results_df['r2'].mean()),
            'std_r2': float(results_df['r2'].std()),
        },
        'overall_metrics': {
            'r2': float(r2_score(all_true_values, all_predictions)),
            'mse': float(mean_squared_error(all_true_values, all_predictions)),
            'rmse': float(np.sqrt(mean_squared_error(all_true_values, all_predictions))),
            'mae': float(mean_absolute_error(all_true_values, all_predictions))
        }
    }

    with open(f'training_logs/{model_type}_training_log_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Training log saved: training_logs/{model_type}_training_log_{timestamp}.json")

    plt.close()


def objective(trial, X, y, model_type, device):
    """Optuna优化目标函数"""
    params = {
        'num_blocks': trial.suggest_int('num_blocks', 2, 4),
        'filters_base': trial.suggest_categorical('filters_base', [32, 64, 128]),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
        'epochs': 200
    }

    _, mse = train_with_cross_validation(X, y, model_type=model_type, params=params, n_splits=3, device=device)

    torch.cuda.empty_cache()
    gc.collect()

    return mse


def run_optimization(X, y, model_type, n_trials, device):
    """运行Optuna贝叶斯优化"""
    print(f"\n{'='*60}")
    print(f"开始Optuna超参数优化 - 模型: {model_type}")
    print(f"{'='*60}")
    print(f"优化目标: 最小化MSE (3折交叉验证)")
    print(f"试验次数: {n_trials}")

    study = optuna.create_study(direction='minimize')

    study.optimize(
        lambda trial: objective(trial, X, y, model_type, device),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )

    print(f"\n{'='*60}")
    print("Optuna优化完成")
    print(f"{'='*60}")
    print(f"最佳MSE: {study.best_value:.6f}")
    print(f"\n最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存最佳参数
    best_params_file = f'best_params_{model_type}.json'
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2, ensure_ascii=False)
    print(f"\n最佳参数已保存到: {best_params_file}")

    # 绘制优化历史
    try:
        plt.figure(figsize=(10, 6))
        plt.plot([trial.value for trial in study.trials], 'o-')
        plt.xlabel('Trial')
        plt.ylabel('MSE')
        plt.title(f'Optuna Optimization History - {model_type}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'optuna_history_{model_type}.png', dpi=300, bbox_inches='tight')
        print(f"优化历史图已保存到: optuna_history_{model_type}.png")
    except:
        pass

    return study.best_params, study.best_value


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='拉曼光谱QS值预测模型训练(PyTorch版)')
    parser.add_argument('--quick', action='store_true', help='快速训练模式(默认参数,不使用Optuna)')
    parser.add_argument('--optimize', action='store_true', help='Optuna超参数优化模式')
    parser.add_argument('--model', type=str, choices=['cnn', 'resnet', 'both'], default='both',
                       help='选择模型架构: cnn, resnet 或 both')
    parser.add_argument('--ntrials', type=int, default=50, help='Optuna优化试验次数')

    args = parser.parse_args()

    print("="*70)
    print("拉曼光谱QS值预测 - 十折交叉验证训练 (PyTorch版)")
    print("="*70)

    # 加载数据
    print("\n1. 加载数据...")
    loader = RamanDataLoader(
        data_dir=r'给云龙师弟拉曼光谱数据',
        csv_path=r'qs_strain_data.csv'
    )
    X, y, strain_names = loader.load_all_data()
    print(f"加载了 {len(X)} 个拉曼光谱样本")
    print(f"数据形状: {X.shape}")

    # 根据参数选择模式
    if args.optimize:
        # Optuna优化模式
        if args.model in ['cnn', 'both']:
            print(f"\n{'='*60}")
            print(f"开始优化1D CNN模型...")
            print(f"{'='*60}")
            best_params_cnn, best_mse_cnn = run_optimization(X, y, 'cnn', args.ntrials, device)

        if args.model in ['resnet', 'both']:
            print(f"\n{'='*60}")
            print(f"开始优化ResNet模型...")
            print(f"{'='*60}")
            best_params_resnet, best_mse_resnet = run_optimization(X, y, 'resnet', args.ntrials, device)

        # 对比优化结果
        if args.model == 'both':
            print(f"\n{'='*60}")
            print("优化结果对比")
            print(f"{'='*60}")
            print(f"1D CNN - 最佳MSE: {best_mse_cnn:.6f}")
            print(f"ResNet - 最佳MSE: {best_mse_resnet:.6f}")

            if best_mse_cnn < best_mse_resnet:
                print(f"\n推荐使用1D CNN模型")
            else:
                print(f"\n推荐使用ResNet模型")

    else:
        # 快速训练模式(默认参数)
        if args.model in ['cnn', 'both']:
            print("\n2. 训练1D CNN模型...")
            cnn_results, cnn_mse = train_with_cross_validation(X, y, model_type='cnn', device=device, save_best_model=True)
            cnn_results.to_csv('cnn_results.csv', index=False, encoding='utf-8-sig')

        if args.model in ['resnet', 'both']:
            print("\n3. 训练ResNet模型...")
            resnet_results, resnet_mse = train_with_cross_validation(X, y, model_type='resnet', device=device, save_best_model=True)
            resnet_results.to_csv('resnet_results.csv', index=False, encoding='utf-8-sig')

        # 对比结果
        if args.model == 'both':
            print(f"\n{'='*60}")
            print("模型对比")
            print(f"{'='*60}")
            print(f"1D CNN - 平均MSE: {cnn_mse:.6f}")
            print(f"ResNet - 平均MSE: {resnet_mse:.6f}")

            if cnn_mse < resnet_mse:
                print(f"\n推荐使用1D CNN模型")
            else:
                print(f"\n推荐使用ResNet模型")

            print(f"\n提示: 使用Optuna进行贝叶斯优化可以进一步提升性能!")
            print(f"运行: python train_pytorch.py --optimize")


if __name__ == '__main__':
    main()
