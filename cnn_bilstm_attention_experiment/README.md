# CNN-BiLSTM-Attention Experiment

This folder tests a simpler VB regression model:

```text
1D CNN local feature extractor
+ stacked BiLSTM
+ temporal attention
+ regression head
```

Data protocol:

- NASA Milling `mill.mat`
- Case 11 as E1 and Case 2 as E2
- DWT denoising
- cubic spline VB interpolation over augmented segments
- LOWESS-style feature smoothing
- 12 current features + 12 vibration features
- random 80% / 20% train-test split
- metrics: MAE, RMSE, R2, MAPE

Run from the project root:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' cnn_bilstm_attention_experiment\train_cnn_bilstm_attention.py
```

Outputs:

```text
cnn_bilstm_attention_experiment/results.txt
cnn_bilstm_attention_experiment/summary.json
cnn_bilstm_attention_experiment/case_*/training_log.csv
cnn_bilstm_attention_experiment/case_*/test_predictions.csv
```

## Latest Result

```text
Case 11 / E1:
MAE: 0.022723 mm (22.72 um)
RMSE: 0.029660 mm (29.66 um)
R2: 0.985618
MAPE: 18.36%

Case 2 / E2:
MAE: 0.010465 mm (10.47 um)
RMSE: 0.013582 mm (13.58 um)
R2: 0.982067
MAPE: 5.49%
```
