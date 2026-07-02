# Stacked-BiLSTM Few-Sample Tool-Wear Reproduction

This folder contains a reproduction-style implementation for:

`Tool-wear-prediction-with-few-samples-based-on-stacked-BiLSTM-an_2026_Measur.pdf`

The PDF text layer could not be extracted reliably in this workspace, so the code is built around the paper title and the local NASA Milling data. The current closer protocol uses:

- missing VB interpolation within each machining case
- raw signal segmentation inside each cutting run
- handcrafted time-domain features per signal segment
- stacked BiLSTM + temporal attention
- continuous VB regression
- MAE, RMSE, R2 and MAPE evaluation

## Files

- `feature_extraction.py`: loads `mill.mat`, cleans abnormal signal values, extracts features, and can interpolate missing VB labels.
- `dataset.py`: builds either cross-run sequences or within-run segment sequences.
- `model.py`: stacked BiLSTM with temporal attention for VB regression.
- `metrics.py`: regression metrics.
- `train_stacked_bilstm.py`: single experiment entry point.
- `run_few_sample_experiments.py`: few-sample train-ratio sweep.

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
