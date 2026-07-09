# PHM 2010 1D Process-State Segmentation

This folder contains the first version of a PHM 2010 process-state segmentation pipeline.

The goal is not to predict tool wear directly. The goal is to train a 1D process-state segmentation model to identify useful cutting states in each complete machining waveform, then pass stable-cutting segments to a VB regression model such as CNN-BiLSTM-Attention.

## Research Positioning

The rule-based labeling algorithm is not the final contribution by itself. It is used as weak supervision to create initial process-state labels when PHM2010 does not provide point-wise machining-state annotations.

The segmentation model is meaningful only if it goes beyond simply reproducing the rule. Its role is to learn a deployable process-state recognizer that can run on short signal windows, generalize across tools and cutting conditions, and provide stable-cutting masks for downstream VB regression.

Therefore, the key paper claim should not be:

```text
rule labels -> segmentation model -> model copies the rule
```

The key paper claim should be:

```text
rule-based weak labels
-> train a 1D temporal segmentation model
-> extract stable-cutting segments
-> reduce non-stationary signal interference
-> improve VB wear prediction
```

The segmentation mIoU/F1 results are necessary, but they are not sufficient as the main proof. The decisive experiment should compare VB regression performance before and after process-state segmentation:

```text
complete waveform -> VB regression
rule-selected stable cutting -> VB regression
model-selected stable cutting -> VB regression
```

If the model-selected stable-cutting pipeline improves MAE, RMSE, R2, or MAPE over the complete-waveform baseline, then the segmentation module has practical value even though its first labels came from rules.

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

The updated pseudo-label rule uses hysteresis-style cutting-region detection instead of only taking the longest high-activity fragment. It first finds high-confidence cutting activity, expands through lower-confidence but still active signal, fills short gaps inside the same operation, then labels only the beginning and ending portions as `transition`. This avoids marking the early part of a continuous machining waveform as `non_cutting` just because the activity score has local valleys.

## Files

- `pseudo_label.py`: rule-based pseudo-label generation.
- `build_label_cache.py`: generates reusable `.npz` labels once before training.
- `label_cache.py`: cache path, save, load, and config-check helpers.
- `dataset.py`: PHM 2010 segmentation dataset with random crops.
- `metrics.py`: point accuracy, present-class mIoU, sample mIoU, macro F1, and per-class metrics.
- `train_process_state_segmentation.py`: process-state segmentation training and six-tool cross validation.
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
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_process_state_segmentation.py --fold c1 --epochs 0 --max-cuts-per-tool 1 --crop-length 2048 --batch-size 1 --cpu
```

The training script now requires cached labels by default. If the cache is missing, it stops and asks you to run `build_label_cache.py`. This prevents every training run from repeatedly pseudo-labeling the same CSV files.

If you change pseudo-label parameters such as `--inactive-threshold`, `--max-gap-ratio`, or `--transition-ratio`, rebuild the label cache with `--overwrite` before training. The cache stores a parameter fingerprint, so stale labels are rejected by default.

## Remote Training

Default editor run configuration:

- `fold = all`
- `task = binary`
- `eval_mode = boundary`
- `exclude_samples_csv = phm2010_segmentation/config/non_cutting_exclude_samples.csv`

This means running `train_process_state_segmentation.py` directly from an editor uses the current paper-oriented setting by default: transition/stable-cutting segmentation, persistent edge non-cutting sample exclusion, and boundary-aware validation/test crops.

Run one fold:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\build_label_cache.py --tools all
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_process_state_segmentation.py --fold c1 --epochs 30 --batch-size 4 --crop-length 8192
```

Switch the segmentation backbone:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_process_state_segmentation.py --model deeplabv3_1d --fold c1
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_process_state_segmentation.py --model unet_1d --fold c1
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_process_state_segmentation.py --model tcn_seg --fold c1
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_process_state_segmentation.py --model bilstm_seg --fold c1
```

Run six folds:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\build_label_cache.py --tools all
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_process_state_segmentation.py --fold all --epochs 30 --batch-size 4 --crop-length 8192 --save-checkpoint
```

Run binary segmentation after excluding persistent edge `non_cutting` candidates:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\build_label_cache.py --tools all --overwrite
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\train_process_state_segmentation.py --task binary --exclude-samples-csv phm2010_segmentation\config\non_cutting_exclude_samples.csv --fold all --epochs 30 --batch-size 4 --crop-length 8192 --save-checkpoint
```

Binary classes:

```text
0: transition
1: stable_cutting
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

Randomly audit multiple pseudo-labeled cuts:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\plot_random_label_samples.py --samples 24
```

Use a fixed seed to reproduce the same random set:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\plot_random_label_samples.py --samples 24 --seed 42
```

Each run creates a new folder under `phm2010_segmentation/outputs/random_label_samples/` with individual plots, JSON metadata, `random_label_samples_summary.csv`, and a contact-sheet overview image. This script always uses the current pseudo-label rule directly, rather than reading label cache files.

Find and collect cuts that contain `non_cutting` labels:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\find_non_cutting_samples.py --run-name current_non_cutting_scan
```

For a fast scan using only existing cached labels:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\find_non_cutting_samples.py --cache-only --no-plots --run-name cache_non_cutting_scan
```

The script writes results under `phm2010_segmentation/outputs/non_cutting_samples/<run-name>/`, including `non_cutting_samples.csv`, per-sample JSON files, optional waveform plots, and a contact sheet. Add `--copy-csv` only when you really want to duplicate the raw CSV files into that output folder.

To quickly stop after finding a few candidates:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\find_non_cutting_samples.py --tools c1 --stop-after-matches 5 --run-name c1_non_cutting_examples
```

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

## Label Rule Update Log

2026-07-05:

- Problem observed: the earlier longest-high-activity rule marked a large early section of `c1/c_1_001.csv` as `non_cutting`, even though the waveform visually remained in a continuous cutting state.
- Change made: replaced longest-fragment selection with hysteresis-style continuous cutting-region detection. The rule now uses `active_threshold` for confident activity, `inactive_threshold` for lower-confidence continuation, fills short gaps with `max_gap_ratio`, and only marks the first/last `transition_ratio` of the detected operation as transition.
- Debug result on `PHM 2010/c1/c_1_001.csv`: `active_start=0`, `active_end=127399`, `transition_len=6370`, with 12,740 transition points and 114,659 stable-cutting points.
- Debug command passed: `train_process_state_segmentation.py --fold c1 --epochs 0 --max-cuts-per-tool 1 --crop-length 2048 --batch-size 1 --cpu`.

## Experiment Result Log

2026-07-08 pseudo-label rule v3 correction:

- Visual check found that samples such as `c1/c_1_156.csv`, `c5/c_5_148.csv`, `c5/c_5_152.csv`, `c2/c_2_022.csv`, and `c2/c_2_023.csv` were false `non_cutting` detections. Their waveforms still show continuous cutting, but the old rule selected only the early high-activity region and treated later lower-energy cutting as stopped.
- Change made: active-region detection now keeps all sustained cutting regions that contain enough high-activity points, then spans from the first sustained region to the last sustained region. This prevents low-energy valleys inside one machining pass from being labeled as `non_cutting`, while still ignoring short tail spikes after a real stop.
- Check result on old candidates: the five false-positive samples above changed to `non_cutting = 0%`; persistent tail-stop samples such as `c1/c_1_315.csv` and `c1/c_1_309.csv` still keep their non-cutting tails.
- Added `label_rule_version = 3` to the label-cache fingerprint. Rebuild cached labels with `--overwrite` before any new training run.
- Updated `phm2010_segmentation/config/non_cutting_exclude_samples.csv` from 14 old candidates to 9 persistent edge `non_cutting` samples.

2026-07-08 binary segmentation after excluding non-cutting candidates:

- Remote scan result: `phm2010_segmentation/outputs/non_cutting_samples/current_non_cutting_scan/non_cutting_samples.csv` found 14 candidates with `non_cutting` labels among 1,890 PHM2010 cuts.
- Visual interpretation: several high-ratio candidates, such as `c1/c_1_156.csv`, were pseudo-label failures rather than reliable non-cutting samples under the old rule.
- Decision: do not train a three-class model with this sparse `non_cutting` class for now.
- Added tracked exclusion list: `phm2010_segmentation/config/non_cutting_exclude_samples.csv`.
- Added binary training mode: `--task binary` maps original `transition` to class 0 and `stable_cutting` to class 1, while removing cuts listed in `--exclude-samples-csv`.
- Debug command passed: `train_process_state_segmentation.py --task binary --exclude-samples-csv phm2010_segmentation\config\non_cutting_exclude_samples.csv --model tcn_seg --fold c1 --epochs 0 --max-cuts-per-tool 1 --crop-length 2048 --batch-size 1 --cpu`.

2026-07-08 metric calculation correction:

- Problem found: the first metric implementation always averaged IoU over all three classes. When validation/test crops contained only `stable_cutting`, a perfect prediction was reported as `(0 + 0 + 1) / 3 = 0.3333`.
- Change made: the main `mean_iou` and `macro_f1` now average only classes that appear in the ground-truth labels of the evaluated data. The old fixed-three-class values are still exported as `mean_iou_all_classes` and `macro_f1_all_classes` for diagnosis.
- Added `sample_mean_iou` and `sample_macro_f1`, which compute valid-class metrics for each sample and then average across samples.
- Best-checkpoint selection now uses validation `sample_mean_iou`, matching the per-sample valid-class evaluation idea.
- Debug result with one center crop from fold `c1`: `mean_iou = 1.0000`, `sample_mean_iou = 1.0000`, while the old diagnostic value remains `mean_iou_all_classes = 0.3333`. This confirms the earlier `0.3333` result came from the averaging denominator, not necessarily from segmentation failure.
- Important: this does not solve the evaluation-design problem. If center crops contain only `stable_cutting`, high mIoU still only proves that the model handles stable-cutting windows. Full-waveform or boundary-aware evaluation is still required for a meaningful three-class result.

2026-07-06 process-state segmentation outputs:

- Result files inspected: `phm2010_segmentation/outputs/cross_validation_summary.json` and `fold_c1` to `fold_c6` training logs.
- Six-fold validation metrics: point accuracy = 1.0000, mean IoU = 0.3333, macro F1 = 0.3333.
- Six-fold test metrics: point accuracy = 1.0000, mean IoU = 0.3333, macro F1 = 0.3333.
- Per-class test F1 average: `non_cutting = 0.0000`, `transition = 0.0000`, `stable_cutting = 1.0000`.
- Interpretation: this historical result used the old fixed-three-class mIoU denominator, so the `0.3333` value is not the correct present-class mIoU. However, it still should not be treated as a valid three-class segmentation result. The current validation/test pipeline uses fixed center crops, and those windows are almost entirely `stable_cutting`. The model can therefore obtain perfect point accuracy by predicting only class 2. The next version should evaluate full waveforms or sample windows that deliberately include entry and exit transition/non-cutting regions.

2026-07-06 pseudo-label visualization audit:

- Visualized 24 labeled cuts: `c1` to `c6`, each with cuts `001`, `105`, `210`, and `315`.
- Output directory: `phm2010_segmentation/outputs/label_samples/`.
- Contact sheet: `phm2010_segmentation/outputs/label_samples/label_samples_contact_sheet.jpg`.
- Summary table: `phm2010_segmentation/outputs/label_samples/label_samples_summary.csv`.
- Average label ratio over the 24 inspected cuts: `non_cutting = 0.00%`, `transition = 10.00%`, `stable_cutting = 90.00%`.
- Interpretation: the current pseudo-label rule identifies almost every complete cut as active from `active_start = 0` to `active_end = n_points`. Therefore it does not provide meaningful `non_cutting` samples. The `transition` class is mostly a fixed 5% entry and 5% exit band, not a truly learned boundary from signal changes. This confirms that the weak point is the label definition, not only the segmentation network.

2026-07-06 pseudo-label rule correction:

- Problem found from `c1_315_labels.png`: after about 218k samples, the waveform and activity score clearly drop into a stopped/non-cutting state, but the old rule still labeled it as active because `max_gap_ratio = 0.20` allowed a very long low-activity interval to be filled.
- Change made: active-region detection now selects the continuous low-threshold region that overlaps the main high-confidence cutting segment. The allowed gap fill was tightened to `max_gap_ratio = 0.03` and `max_gap_points = 8192`.
- New `c1/c_1_315.csv` result: `active_start = 0`, `active_end = 218305`, `n_points = 252492`, so the final 34,187 points are now labeled `non_cutting`.
- Re-visualized 24 cuts under `phm2010_segmentation/outputs/label_samples_v2/`.
- V2 contact sheet: `phm2010_segmentation/outputs/label_samples_v2/label_samples_v2_contact_sheet.jpg`.
- V2 summary table: `phm2010_segmentation/outputs/label_samples_v2/label_samples_v2_summary.csv`.
- V2 average label ratio over the same 24 inspected cuts: `non_cutting = 0.56%`, `transition = 9.94%`, `stable_cutting = 89.49%`. Among the inspected samples, only `c1_315` contains a clear non-cutting tail.
- Important: because the pseudo-label parameters changed, rebuild the full label cache before any new training run:

```powershell
& 'D:\AppInsDir\Anaconda3\envs\pytorch-py3.12\python.exe' phm2010_segmentation\build_label_cache.py --tools all --overwrite
```
