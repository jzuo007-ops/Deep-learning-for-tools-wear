# Stacked-BiLSTM Few-Sample Tool-Wear Reproduction

This folder contains a reproduction-style implementation for:

`Tool-wear-prediction-with-few-samples-based-on-stacked-BiLSTM-an_2026_Measur.pdf`

The PDF text layer can now be extracted in this workspace. This folder keeps two reproduction tracks:

1. A simpler Stacked-BiLSTM + temporal attention baseline.
2. A closer paper-like PMS experiment for Case 11 / Case 2.

The baseline protocol uses:

- missing VB interpolation within each machining case
- raw signal segmentation inside each cutting run
- handcrafted time-domain features per signal segment
- stacked BiLSTM + temporal attention
- continuous VB regression
- MAE, RMSE, R2 and MAPE evaluation

The paper-like PMS protocol adds:

- Case 11 / Case 2 only
- cubic spline VB interpolation over augmented segments
- 12 current features + 12 vibration features
- DWT soft-threshold denoising
- LOWESS-style local weighted feature smoothing
- PRes-MHSA-SBiLSTM style regression

## Files

- `feature_extraction.py`: loads `mill.mat`, cleans abnormal signal values, extracts features, and can interpolate missing VB labels.
- `dataset.py`: builds either cross-run sequences or within-run segment sequences.
- `model.py`: stacked BiLSTM with temporal attention for VB regression.
- `metrics.py`: regression metrics.
- `train_stacked_bilstm.py`: single experiment entry point.
- `run_few_sample_experiments.py`: few-sample train-ratio sweep.
- `paper_like_pms_experiment.py`: closer implementation of the paper's Case 11 / Case 2 PMS experiment.
- `paper_like_results.txt`: latest paper-like PMS metrics recorded as a text file.

## Recommended Run

From the project root:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py --data-root '3. Milling'
```

Default closer-to-paper protocol:

- `--sample-mode segment_sequence`
- `--impute-vb`
- `--n-segments 16`
- `--segment-window 8`
- `--segment-step 4`
- `--split-mode random_run`
- `--train-ratio 0.30`
- `--epochs 120`

Outputs are written to:

```text
stacked_bilstm_reproduction/outputs/single_run/
```

Important output files:

- `summary.json`: final metrics and experiment settings.
- `training_log.csv`: epoch-by-epoch validation metrics.
- `test_predictions.csv`: true and predicted VB values for test samples.
- `test_attention.npy`: attention weights.

## Few-Sample Sweep

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\run_few_sample_experiments.py --data-root '3. Milling'
```

The summary table is saved to:

```text
stacked_bilstm_reproduction/outputs/few_sample_sweep/few_sample_summary.csv
```

## Paper-Like PMS Experiment

Run:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\paper_like_pms_experiment.py --data-root '3. Milling'
```

Current settings:

- E1: Case 11, 20 valid VB runs, expanded to 100 segments.
- E2: Case 2, 13 valid VB runs, expanded to 104 segments.
- Train/test split: 80% / 20% random split.
- Optimizer: Adam.
- Initial learning rate: 0.012.
- Learning-rate decay factor: 0.892.
- Maximum iterations: 1500.
- Batch size: 15.

Latest result:

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

Known remaining gaps from the paper:

- EMD is not applied because no EMD package is installed in the current environment.
- The paper does not disclose the random seed, exact channel choice, full network dimensions, or LOWESS parameters.
- Small-sample 80/20 random splits are sensitive to the selected seed.

## Alternative Protocols

Use the older cross-run feature sequence:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py --data-root '3. Milling' --sample-mode run_sequence
```

Disable VB interpolation:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py --data-root '3. Milling' --no-impute-vb
```

Use sample-level random split. This often reports better numbers, but can leak windows from the same run across train/test when segment windows are used:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py --data-root '3. Milling' --split-mode random
```
