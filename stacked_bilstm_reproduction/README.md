# Stacked-BiLSTM Few-Sample Tool-Wear Reproduction

本文件夹包含论文的独立复制式实现：

```
Tool-wear-prediction-with-few-samples-based-on-stacked-BiLSTM-an_2026_Measur.pdf
```

PDF文本图层无法在此工作区中可靠提取，因此本代码遵循论文标题和本地NASA铣削数据集所暗示的方法：

- 原始多通道铣信号
- 每次切割批次手工制作的统计特征
- 少样本训练分段
- 堆叠双向LSTM
- 时间注意力
- 连续VB工具磨损预测
- MAE、RMSE、R2和MAPE评估

## 文件



- `feature_extraction.py`： 加载并提取每次运行的统计特征。`mill.mat`
- `dataset.py`：构建序列窗口和少量样本的列车/验证/测试拆分。
- `model.py`：堆叠BiLSTM并关注VB回归。
- `metrics.py`：回归指标。
- `train_stacked_bilstm.py`：主要培训和评估入口。
- `run_few_sample_experiments.py`： 在一个命令中运行多个少样本比例。

## 快速入门



摘自项目根源：

```
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py
```



运行多个少数样本比例：

```
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\run_few_sample_experiments.py
```



输出写在 。`stacked_bilstm_reproduction/outputs/`

## 默认实验



- 数据集：`3. Milling/mill.mat`
- 通道：， ， ，`smcAC``smcDC``vib_table``vib_spindle``AE_table``AE_spindle`
- 目标：连续`VB`
- 序列窗口：同一机壳内连续5次切割
- 少样本列车比：默认为0.30
- 模型：二层BiLSTM + 注意 + 回归头
