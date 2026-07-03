# PRes-MHSA-SBiLSTM Experiment

This folder tests a paper-like VB regression model:

```text
PResNet-style parallel residual branches
+ multi-head self-attention
+ stacked BiLSTM
+ temporal attention regression head
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
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' pres_mhsa_sbilstm_experiment\train_pres_mhsa_sbilstm.py
```

Outputs:

```text
pres_mhsa_sbilstm_experiment/results.txt
pres_mhsa_sbilstm_experiment/summary.json
pres_mhsa_sbilstm_experiment/case_*/training_log.csv
pres_mhsa_sbilstm_experiment/case_*/test_predictions.csv
```

## Latest Result

```text
Case 11 / E1:
MAE: 0.073472 mm (73.47 um)
RMSE: 0.117173 mm (117.17 um)
R2: 0.775545
MAPE: 51.35%

Case 2 / E2:
MAE: 0.020720 mm (20.72 um)
RMSE: 0.024198 mm (24.20 um)
R2: 0.943080
MAPE: 11.12%
```
