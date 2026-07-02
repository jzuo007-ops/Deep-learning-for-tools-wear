# Stacked-BiLSTM Few-Sample Tool-Wear Reproduction

This folder contains an independent reproduction-style implementation for the paper:

`Tool-wear-prediction-with-few-samples-based-on-stacked-BiLSTM-an_2026_Measur.pdf`

The PDF text layer could not be extracted reliably in this workspace, so this code follows the method implied by the paper title and the local NASA Milling dataset:

- raw multi-channel milling signals
- handcrafted statistical features per cutting run
- few-sample training splits
- stacked bidirectional LSTM
- temporal attention
- continuous VB tool-wear prediction
- MAE, RMSE, R2 and MAPE evaluation

## Files

- `feature_extraction.py`: loads `mill.mat` and extracts per-run statistical features.
- `dataset.py`: builds sequence windows and few-sample train/validation/test splits.
- `model.py`: stacked BiLSTM with attention for VB regression.
- `metrics.py`: regression metrics.
- `train_stacked_bilstm.py`: main training and evaluation entry point.
- `run_few_sample_experiments.py`: runs several few-sample ratios in one command.

## Quick Start

From the project root:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\train_stacked_bilstm.py
```

Run multiple few-sample ratios:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' stacked_bilstm_reproduction\run_few_sample_experiments.py
```

Outputs are written under `stacked_bilstm_reproduction/outputs/`.

## Default Experiment

- Dataset: `3. Milling/mill.mat`
- Channels: `smcAC`, `smcDC`, `vib_table`, `vib_spindle`, `AE_table`, `AE_spindle`
- Target: continuous `VB`
- Sequence window: 5 consecutive cutting runs within the same case
- Few-sample train ratio: 0.30 by default
- Model: 2-layer BiLSTM + attention + regression head

