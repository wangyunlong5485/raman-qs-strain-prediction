# 快速开始指南

## 推荐使用流程

### 第一步: 准备数据
```bash
python parse_excel_v4.py
```

### 第二步: 快速对比训练两个模型 (30-45分钟)
```bash
python train_pytorch.py --quick --model both
```
这将同时训练1D CNN和ResNet两个模型，使用默认参数进行十折交叉验证，对比选择效果更好的模型。

### 第三步: 贝叶斯优化选定的模型 (2-3小时)
```bash
# 如果CNN更好
python train_pytorch.py --optimize --model cnn --ntrials 50

# 如果ResNet更好
python train_pytorch.py --optimize --model resnet --ntrials 50
```
使用Optuna自动搜索最优超参数。

## 命令参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--quick` | 快速训练模式(默认参数) | 否 |
| `--optimize` | Optuna优化模式 | 否 |
| `--model` | 模型类型: cnn/resnet/both | both |
| `--ntrials` | Optuna试验次数 | 50 |

## 常见问题

### Q: 为什么要先用 `--quick --model both` 模式?
A: 快速对比CNN和ResNet两个模型的效果，选择性能更好的一个进行后续优化。只需30-45分钟。

### Q: RTX 4070 Laptop GPU 预计训练时间?
A:
- 快速模式(10折): 30-45分钟
- Optuna优化(50 trials): 2-3小时

### Q: 如何选择CNN还是ResNet?
A:
1. 运行 `python train_pytorch.py --quick --model both`
2. 对比两者的平均MSE
3. 选择MSE更低的模型进行Optuna优化
4. CNN通常更快且效果足够好

### Q: Optuna优化需要多久?
A: 每个trial大约3-5分钟(使用3折交叉验证),50次试验约2-3小时。

## 输出文件说明

### 训练结果
- `cnn_results.csv` - CNN每折的MSE/RMSE/MAE/R²
- `resnet_results.csv` - ResNet每折的MSE/RMSE/MAE/R²

### 优化结果
- `best_params_cnn.json` - CNN最优超参数
- `best_params_resnet.json` - ResNet最优超参数
- `optuna_history_cnn.png` - 优化过程可视化
- `optuna_history_resnet.png` - 优化过程可视化
