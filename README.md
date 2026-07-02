# Deep Learning for Tool Wear Prediction

本项目面向 NASA Milling 刀具磨损数据，主要包含两个实验方向：

1. 基于 1D DeepLabV3-ResNet50 的多通道时序信号磨损状态分类。
2. 基于 Stacked-BiLSTM + Attention 的少样本 VB 磨损值回归预测复现实验。

当前主网络是 `1D DeepLabV3-ResNet50`，用于将铣削过程中的多通道传感器信号映射为刀具磨损类别。

## 数据集

默认数据文件：

```text
3. Milling/mill.mat
```

使用的 6 个信号通道：

```text
smcAC
smcDC
vib_table
vib_spindle
AE_table
AE_spindle
```

标签来自 `VB` 刀具后刀面磨损值。当前分类实验采用二分类设置：

```text
class 0: lower wear
class 1: high wear
```

默认使用 `VB` 的 2/3 分位数作为二分类阈值，并在训练、验证数据集中保持同一个阈值。

## 主网络结构

主模型定义在：

```text
src/deeplabv3_model.py
src/resnet_backbone.py
```

整体结构：

```text
6-channel 1D signal
    -> ResNet50_1D Backbone
    -> ASPP1D
    -> DeepLabHead1D
    -> Global Average Pooling
    -> Binary Classification Logits
```

### ResNet50_1D Backbone

`src/resnet_backbone.py` 将标准 ResNet50 改造成一维卷积网络：

- `Conv2d` 改为 `Conv1d`
- 输入通道数为 6
- Bottleneck block 数量为 `[3, 4, 6, 3]`
- Layer3 和 Layer4 使用 dilation 替代部分下采样，以保留更长的时序分辨率

Backbone 输出两个特征：

```text
layer3 feature: 用于 aux classifier
layer4 feature: 用于 ASPP 主分支
```

### ASPP1D

`ASPP1D` 是 DeepLabV3 的一维版本，包含：

- `1x1 Conv1d`
- 多个不同 dilation rate 的 `3x1 Conv1d`
- 全局平均池化分支
- 拼接后通过 `1x1 Conv1d` 投影

当前 dilation rates：

```text
12, 24, 36
```

### 分类头

模型在 `classification=True` 时，会对时序维度做自适应平均池化：

```text
[batch, num_classes, length] -> [batch, num_classes]
```

因此当前任务不是逐点分割，而是样本级磨损状态分类。

## 训练流程

训练入口：

```text
train.py
```

主要设置：

```text
batch_size = 16
seq_length = 8192
epochs = 30
num_classes = 2
optimizer = AdamW
learning rate = 5e-4
weight_decay = 1e-4
scheduler = CosineAnnealingLR
loss = CrossEntropyLoss + aux loss
```

训练前会做以下处理：

- 固定随机种子
- 分层划分训练集和验证集
- 只使用训练集计算通道均值和标准差，避免验证集数据泄漏
- 训练集使用随机裁剪 `RandomCrop1D`
- 验证集使用确定性中心裁剪 `CenterCrop1D`
- 使用类别权重缓解类别不均衡
- 按验证集 `Macro F1` 保存最佳模型

运行训练：

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' train.py
```

## 测试结果

### 1D DeepLabV3-ResNet50 分类结果

实验记录保存在：

```text
training_results.txt
```

数据划分：

```text
训练集: 116
验证集: 30
Binary VB threshold: 0.400000
```

验证集结果：

```text
Accuracy: 0.9333
Macro F1: 0.9282
```

混淆矩阵：

```text
Classes: [0 1]

[[18  1]
 [ 1 10]]
```

含义：

- 低磨损类 `0`：19 个样本，18 个预测正确
- 高磨损类 `1`：11 个样本，10 个预测正确
- 总计 30 个验证样本，28 个预测正确

### Stacked-BiLSTM + Attention 回归结果

复现实验代码位于：

```text
stacked_bilstm_reproduction/
```

当前更接近论文的默认协议：

```text
sample_mode: segment_sequence
impute_vb: true
n_segments: 16
segment_window: 8
segment_step: 4
train_ratio: 0.30
val_ratio: 0.20
```

#### 较严谨划分：random_run

该设置按 run 分组随机划分，避免同一个 run 的不同窗口同时进入训练集和测试集。

运行命令：

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py --data-root '3. Milling' --split-mode random_run
```

测试集结果：

```text
n_runs: 167
n_sequences: 501
train/val/test: 150 / 99 / 252

MAE: 0.1086
RMSE: 0.1651
R2: 0.5884
MAPE: 38.38%
```

#### 窗口级随机划分：random

该设置更接近一些论文中常见的随机窗口实验，指标更高，但可能存在同一 run 的窗口被分到不同集合的风险，因此结果偏乐观。

运行命令：

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py --data-root '3. Milling' --split-mode random
```

测试集结果：

```text
MAE: 0.0633
RMSE: 0.0985
R2: 0.8644
MAPE: 24.18%
```

### 结果说明

分类实验中的 1D DeepLabV3-ResNet50 在当前二分类验证集上表现较好，`Accuracy` 达到 `0.9333`，`Macro F1` 达到 `0.9282`。

回归实验中，`random_run` 更能反映模型对未见 run 的泛化能力；`random` 则更接近窗口级随机划分，结果更高，但需要在论文或报告中说明其划分方式。

## Stacked-BiLSTM 少样本复现实验

该目录实现了：

- VB 缺失值插值
- 原始信号分段
- 每个信号片段提取统计特征
- Stacked-BiLSTM + Attention
- 连续 VB 回归预测
- MAE、RMSE、R2、MAPE 指标

推荐运行：

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py --data-root '3. Milling'
```

批量少样本比例实验：

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\run_few_sample_experiments.py --data-root '3. Milling'
```

## 项目结构

```text
.
├── 3. Milling/
│   └── mill.mat
├── src/
│   ├── __init__.py
│   ├── deeplabv3_model.py
│   └── resnet_backbone.py
├── stacked_bilstm_reproduction/
│   ├── README.md
│   ├── dataset.py
│   ├── feature_extraction.py
│   ├── metrics.py
│   ├── model.py
│   ├── run_few_sample_experiments.py
│   └── train_stacked_bilstm.py
├── my_dataset.py
├── transforms.py
├── train.py
├── training_results.txt
└── README.md
```

## 说明

本项目当前更适合用于两类实验：

1. 将 1D DeepLabV3-ResNet50 用于刀具磨损状态分类。
2. 将 Stacked-BiLSTM + Attention 用于少样本连续 VB 回归预测。

如果用于论文实验，建议明确区分分类任务和回归任务，并在结果表中分别报告分类指标和回归指标。
