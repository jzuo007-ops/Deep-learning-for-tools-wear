import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class ToolWear1DDataset(Dataset):
    """
    NASA Milling 1D multi-channel tool-wear classification dataset.
    Current training uses binary labels:
    class 0 = lower wear, class 1 = high wear.
    """

    def __init__(
        self,
        data_root,
        mat_file="mill.mat",
        transforms=None,
        indices=None,
        label_thresholds=None,
        channel_keys=None,
        label_mode="quantile",
        impute_missing_vb=False,
        window_size=None,
        window_stride=None,
        include_tail_window=True,
    ):
        super(ToolWear1DDataset, self).__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.label_thresholds = label_thresholds or (0.4,)
        self.channel_keys = channel_keys or [
            "smcAC",
            "smcDC",
            "vib_table",
            "vib_spindle",
            "AE_table",
            "AE_spindle",
        ]
        self.label_mode = label_mode
        self.impute_missing_vb = impute_missing_vb
        self.window_size = window_size
        self.window_stride = window_stride or window_size
        self.include_tail_window = include_tail_window

        mat_path = os.path.join(self.data_root, mat_file)
        records = loadmat(mat_path)["mill"][0]
        self.records = []
        self.record_vbs = []
        measured_vbs = []
        for record in records:
            vb_value = self._read_scalar(record, "VB")
            if np.isfinite(vb_value):
                measured_vbs.append(vb_value)
            if self.impute_missing_vb or np.isfinite(vb_value):
                self.records.append(record)
                self.record_vbs.append(vb_value)

        self.measured_vbs = measured_vbs
        if self.impute_missing_vb:
            self.record_vbs = self._interpolate_missing_vbs(self.records, self.record_vbs)
        self.valid_vbs = [vb for vb in self.record_vbs if np.isfinite(vb)]

        if indices is None:
            self.indices = list(range(len(self.records)))
        else:
            self.indices = list(indices)

        self.binary_threshold = self._resolve_binary_threshold()
        self.record_labels = self._build_record_labels()
        self.samples = self._build_samples()
        self.labels = np.asarray([self.record_labels[record_idx] for record_idx, _ in self.samples], dtype=np.int64)

    @staticmethod
    def _read_scalar(record, name, default=np.nan):
        if name not in record.dtype.names:
            return default
        value = np.asarray(record[name]).squeeze()
        if value.size == 0:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _interpolate_missing_vbs(records, vbs):
        filled = np.asarray(vbs, dtype=np.float64)
        cases = np.asarray([ToolWear1DDataset._read_scalar(record, "case", idx) for idx, record in enumerate(records)])
        runs = np.asarray([ToolWear1DDataset._read_scalar(record, "run", idx) for idx, record in enumerate(records)])

        for case_id in np.unique(cases):
            case_indices = np.where(cases == case_id)[0]
            order = case_indices[np.argsort(runs[case_indices])]
            case_runs = runs[order]
            case_vbs = filled[order]
            known = np.isfinite(case_vbs)

            if known.sum() == 0:
                continue
            if known.sum() == 1:
                filled[order] = case_vbs[known][0]
            else:
                filled[order] = np.interp(case_runs, case_runs[known], case_vbs[known])

        return filled.tolist()

    @staticmethod
    def _clean_signal(values):
        values = np.asarray(values).reshape(-1).astype(np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            return np.zeros_like(values, dtype=np.float32)

        normal_values = values[finite & (np.abs(values) < 1e6)]
        if normal_values.size == 0:
            normal_values = np.sign(values[finite]) * np.log1p(np.abs(values[finite]))

        lower, upper = np.percentile(normal_values, [1.0, 99.0])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            median = np.median(normal_values)
            mad = np.median(np.abs(normal_values - median))
            scale = max(mad * 6.0, 1e-6)
            lower, upper = median - scale, median + scale

        fill_value = float(np.median(normal_values))
        values = np.where(np.abs(values) < 1e6, values, np.nan)
        values = np.clip(values, lower, upper)
        values = np.nan_to_num(values, nan=fill_value, posinf=upper, neginf=lower)
        return values.astype(np.float32)

    def _resolve_binary_threshold(self):
        if self.label_mode == "quantile" and len(self.valid_vbs) > 0:
            return float(np.quantile(self.valid_vbs, 2 / 3))

        thresholds = np.asarray(self.label_thresholds, dtype=np.float32).reshape(-1)
        if thresholds.size == 0:
            raise ValueError("label_thresholds must contain at least one threshold.")
        return float(thresholds[-1])

    def _build_record_labels(self):
        labels = []
        for vb in self.record_vbs:
            label = 1 if np.isfinite(vb) and vb >= self.binary_threshold else 0
            labels.append(label)
        return np.asarray(labels, dtype=np.int64)

    def _get_signal_length(self, record):
        lengths = [np.asarray(record[key]).reshape(-1).size for key in self.channel_keys]
        return int(min(lengths))

    def _build_samples(self):
        samples = []
        if self.window_size is None:
            return [(record_idx, None) for record_idx in self.indices]

        for record_idx in self.indices:
            signal_length = self._get_signal_length(self.records[record_idx])
            if signal_length <= self.window_size:
                samples.append((record_idx, 0))
                continue

            starts = list(range(0, signal_length - self.window_size + 1, self.window_stride))
            tail_start = signal_length - self.window_size
            if self.include_tail_window and starts[-1] != tail_start:
                starts.append(tail_start)
            samples.extend((record_idx, start) for start in starts)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record_idx, start_idx = self.samples[index]
        record = self.records[record_idx]

        channels = []
        for key in self.channel_keys:
            arr = self._clean_signal(record[key])
            if self.window_size is not None:
                end_idx = start_idx + self.window_size
                arr = arr[start_idx:end_idx]
                if arr.shape[0] < self.window_size:
                    arr = np.pad(arr, (0, self.window_size - arr.shape[0]), mode="constant")
            channels.append(arr)

        signal = np.stack(channels, axis=0)
        label = int(self.labels[index])

        signal = torch.from_numpy(signal).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transforms is not None:
            signal, label = self.transforms(signal, label)
        else:
            default_length = self.window_size or 2048
            if signal.shape[1] >= default_length:
                signal = signal[:, :default_length]
            else:
                signal = torch.nn.functional.pad(
                    signal,
                    (0, default_length - signal.shape[1]),
                    "constant",
                    0,
                )

        return signal, label

    @staticmethod
    def collate_fn(batch):
        signals, labels = list(zip(*batch))
        signals = torch.stack(signals, dim=0)
        labels = torch.stack(labels, dim=0)
        return signals, labels
