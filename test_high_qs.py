"""测试高QS值菌株的预测"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler

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
            nn.Conv1d(2, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
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

# 加载模型
checkpoint = torch.load('best_model_resnet.pth', map_location='cpu', weights_only=False)
params = checkpoint['params']
model_state = checkpoint['model_state_dict']
scaler_y = checkpoint.get('scaler_y')

device = torch.device('cpu')
model = ResNet1D(2000, params).to(device)
model.load_state_dict(model_state)
model.eval()

# 测试高QS值菌株
base_path = '给云龙师弟拉曼光谱数据'
test_strains = [
    ('qs强', '35', 1.379),
    ('qs强', 'HD10', 1.342),
    ('qs中', 'HD3', 0.438),
    ('qs低', '1-2-35', 0.197),
]

print("测试不同QS值菌株的预测:")
print("="*60)

for category, strain_name, true_qs in test_strains:
    strain_path = os.path.join(base_path, category, strain_name)
    if not os.path.exists(strain_path):
        strain_path = os.path.join(base_path, category, strain_name.lower())
    
    txt_files = glob.glob(os.path.join(strain_path, '*.txt'))[:10]
    
    if not txt_files:
        print(f"{strain_name}: 未找到光谱文件")
        continue
    
    predictions = []
    for txt_file in txt_files:
        data = np.loadtxt(txt_file, encoding='utf-8')
        
        # 处理数据格式 - 与训练代码一致
        if data.ndim >= 2 and data.shape[1] >= 2:
            # 多列数据：直接使用前两列
            if len(data) > 2000:
                data = data[:2000]
            elif len(data) < 2000:
                data = np.pad(data, ((0, 2000 - len(data)), (0, 0)), 'constant')
            X_sample = data[:, :2]
        else:
            # 单列数据
            intensity = data.flatten()[:2000]
            if len(intensity) < 2000:
                intensity = np.pad(intensity, (0, 2000 - len(intensity)), 'constant')
            wave_numbers = np.arange(len(intensity))
            X_sample = np.column_stack([intensity, wave_numbers])
        
        # 标准化
        scaler_X = StandardScaler()
        X_norm = np.zeros_like(X_sample)
        X_norm[:, 0] = scaler_X.fit_transform(X_sample[:, 0].reshape(-1, 1)).flatten()
        X_norm[:, 1] = scaler_X.fit_transform(X_sample[:, 1].reshape(-1, 1)).flatten()
        
        # 预测
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).unsqueeze(0).to(device)
            pred_scaled = model(X_tensor).cpu().numpy().squeeze()
            pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred)
    
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    print(f"{strain_name} (真实QS={true_qs:.3f}): 预测={mean_pred:.3f}±{std_pred:.3f}, 样本数={len(predictions)}")

print("="*60)
