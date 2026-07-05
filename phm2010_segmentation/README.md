# PHM 2010 1D Process-State Segmentation

This folder contains the first version of a PHM 2010 process-state segmentation pipeline.

The goal is not to predict tool wear directly. The goal is to train a 1D DeepLabV3 model to identify useful cutting states in each complete machining waveform, then pass stable-cutting segments to a VB regression model such as CNN-BiLSTM-Attention.

## Classes

The first version uses rule-based pseudo labels with three classes:

```text
0: non_cutting
1: transition
2: stable_cutting
```

The pseudo-label rule computes an activity score from force, vibration, and acoustic emission channels:

```text
force_energy = RMS(Fx, Fy, Fz)
vib_energy   = RMS(Vx, Vy, Vz)
ae_energy    = abs(AE)
score        = normalized force + normalized vibration + normalized AE
```

The longest high-activity region is treated as the machining region. Its beginning and ending parts are labeled as `transition`, and the middle part is labeled as `stable_cutting`.

## Files

- `pseudo_label.py`: rule-based pseudo-label generation.
- `build_label_cache.py`: generates reusable `.npz` labels once before training.
- `label_cache.py`: cache path, save, load, and config-check helpers.
- `dataset.py`: PHM 2010 segmentation dataset with random crops.
- `metrics.py`: point accuracy, mIoU, macro F1, and per-class metrics.
- `train_deeplabv3_segmentation.py`: 1D DeepLabV3 training and six-tool cross validation.
- `plot_segmentation_result.py`: draws complete waveforms with colored pseudo-label or prediction regions.

## Debug Run

Build labels once before training. The default cache path is ignored by Git:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\build_label_cache.py --tools all
```

For a quick local check, build one cut per tool:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\build_label_cache.py --tools all --max-cuts-per-tool 1 --overwrite
```

Then use a tiny subset to verify the training pipeline:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_deeplabv3_segmentation.py --fold c1 --epochs 0 --max-cuts-per-tool 1 --crop-length 2048 --batch-size 1 --cpu
```

The training script now requires cached labels by default. If the cache is missing, it stops and asks you to run `build_label_cache.py`. This prevents every training run from repeatedly pseudo-labeling the same CSV files.

## Remote Training

Run one fold:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\build_label_cache.py --tools all
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_deeplabv3_segmentation.py --fold c1 --epochs 30 --batch-size 4 --crop-length 8192
```

Run six folds:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\build_label_cache.py --tools all
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_deeplabv3_segmentation.py --fold all --epochs 30 --batch-size 4 --crop-length 8192 --save-checkpoint
```

For each fold, one tool is used as test data, the next tool is used as validation data, and the remaining four tools are used for training. Example:

```text
fold c1:
train: c3, c4, c5, c6
val:   c2
test:  c1
```

## Visualization

Plot rule-based pseudo labels on a complete cut:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\plot_segmentation_result.py --cut-file 'PHM 2010\c1\c_1_253.csv'
```

If a cached label exists, the plotting script reads it; otherwise it generates labels only for that one figure.

Plot model predictions after training:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\plot_segmentation_result.py --cut-file 'PHM 2010\c1\c_1_253.csv' --checkpoint phm2010_segmentation\outputs\fold_c1\best_model.pth
```

Outputs are saved under:

```text
phm2010_segmentation/outputs/
```

The output figure uses colored time regions:

```text
gray:   non_cutting
yellow: transition
green:  stable_cutting
```
