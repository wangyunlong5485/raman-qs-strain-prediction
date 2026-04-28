# 拉曼光谱QS值预测 - PyTorch深度学习模型

## 项目概述

使用深度学习从单细胞拉曼光谱预测细菌群体感应(QS)值

### 数据信息
- **训练样本**: 19,293个拉曼光谱文件
- **Input形状**: (2000, 2) - [波长, 强度]
- **Output**: QS值(连续值)
- **菌株数**: 57个
- **GPU**: RTX 4070 Laptop GPU (8GB)

## 文件说明

### 核心文件
- `train_pytorch.py` - PyTorch统一训练脚本(支持快速训练、Optuna优化、CNN/ResNet)
- `parse_excel_v4.py` - 解析Excel文件,生成QS菌株数据CSV
- `qs_strain_data.csv` - 解析后的菌株-QS值对应表
- `requirements.txt` - Python依赖包

## 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 准备数据
```bash
python parse_excel_v4.py
```

### 3. 训练模型

#### 快速训练(推荐先用这个)
```bash
# 训练CNN和ResNet对比
python train_pytorch.py --quick

# 只训练CNN
python train_pytorch.py --quick --model cnn

# 只训练ResNet
python train_pytorch.py --quick --model resnet
```

#### 贝叶斯优化(完整训练)
```bash
# 优化CNN和ResNet
python train_pytorch.py --optimize

# 只优化CNN (更快)
python train_pytorch.py --optimize --model cnn

# 指定试验次数
python train_pytorch.py --optimize --ntrials 50
```

## 模型架构

### 1D CNN (推荐)
```
Input(2000,2) → Conv1D×2 → MaxPool → Conv1D×2 → MaxPool
→ Conv1D×2 → GlobalAvgPool → Dense×2 → Output(1)

优点:
- 适合处理序列数据(光谱信号)
- 能够捕获局部特征模式(峰、谷)
- 参数较少,训练效率高
```

### ResNet
```
优点:
- 深层网络,可学习更复杂特征
- 残差连接避免梯度消失
- 适合大规模数据

适用: 如果1D CNN表现不佳
```

## 训练策略

### 十折交叉验证
- 将数据分成10份
- 每次用9份训练,1份测试
- 重复10次,得到稳健性能评估

### 贝叶斯优化 (Optuna)
- 高效搜索超参数空间
- 优化目标: 最小化验证集MSE

### 训练技巧
- **早停法** (Early Stopping): 防止过拟合
- **学习率衰减** (ReduceLROnPlateau): 动态调整学习率
- **GPU加速**: PyTorch自动利用CUDA 12.x加速
- **正则化**: Dropout, Batch Normalization

## 性能优化

### RTX 4070 Laptop GPU (8GB显存)优化
- ✓ PyTorch CUDA加速 - 完美支持RTX 40系列
- ✓ 大batch size (64-256) - 充分利用8GB显存
- ✓ 自动内存管理 - 每fold后释放GPU内存

### 预期训练时间 (RTX 4070 Laptop GPU 8GB)
- **快速训练** (10折交叉验证): 30-45分钟
- **贝叶斯优化** (50次试验): 2-3小时

## 评估指标
- **MSE** (均方误差) - 主指标
- **RMSE** (均方根误差) - 可解释性强
- **MAE** (平均绝对误差) - 对异常值鲁棒
- **R²** (决定系数) - 模型解释能力

## 结果输出

### 快速训练
- `cnn_results.csv` - CNN十折交叉验证结果
- `resnet_results.csv` - ResNet十折交叉验证结果
- `training_logs/` - 详细训练日志和可视化图表
  - `{model}_training_results_{timestamp}.png` - 九合一训练结果图表
  - `{model}_training_log_{timestamp}.json` - 完整训练日志JSON

### Optuna优化
- `best_params_cnn.json` - CNN最佳超参数
- `best_params_resnet.json` - ResNet最佳超参数
- `optuna_history_*.png` - 优化历史图

## 训练日志和可视化

生成的图表包含9个子图:

1. **训练损失曲线** - 10折平均的训练/验证损失
2. **MAE训练曲线** - 平均绝对误差变化趋势
3. **预测vs真实值散点图** - 模型预测准确性可视化
4. **残差图** - 预测误差分布分析
5. **残差直方图** - 残差统计分布(含正态拟合)
6. **各Fold性能对比** - 每折的MSE/RMSE/MAE
7. **R²柱状图** - 每折的决定系数
8. **性能指标箱线图** - 10折性能分布统计
9. **误差随QS值分布** - 不同QS值区间的预测误差

JSON日志包含:
- 每折详细指标
- 总体统计(均值±标准差)
- 模型架构参数
- 时间戳

## 推荐方案

**首选**: 1D CNN + 贝叶斯优化
- CNN适合光谱数据特征提取
- 模型简单高效,易于训练
- 19K样本量足够训练
- 贝叶斯优化自动调参

**备选**: 如果CNN效果不佳 → ResNet
