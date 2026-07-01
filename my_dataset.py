import os
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class ToolWear1DDataset(Dataset):
    """
    面向 NASA Milling 数据的 1D 多通道分类数据集。
    每个样本是一次刀具运行过程的多通道信号，标签由 VB 磨损值分成 3 类。
    """

    def __init__(self, data_root, mat_file="mill.mat", transforms=None, indices=None,
                 label_thresholds=None, channel_keys=None, label_mode="quantile"):
        super(ToolWear1DDataset, self).__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.label_thresholds = label_thresholds or (0.2, 0.4)
        self.channel_keys = channel_keys or [
            "smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"
        ]
        self.label_mode = label_mode

        mat_path = os.path.join(self.data_root, mat_file)
        records = loadmat(mat_path)["mill"][0]
        self.records = []
        self.valid_vbs = []
        for record in records:
            vb = np.asarray(record["VB"]).squeeze()
            vb_value = float(vb) if np.size(vb) > 0 else np.nan
            if np.isfinite(vb_value):
                self.records.append(record)
                self.valid_vbs.append(vb_value)

        if indices is None:
            self.indices = list(range(len(self.records)))
        else:
            self.indices = list(indices)

        self.labels = self._build_labels()

    def _build_labels(self):
        labels = []
        if self.label_mode == "quantile" and len(self.valid_vbs) > 0:
            q1, q2 = np.quantile(self.valid_vbs, [1 / 3, 2 / 3])
            thresholds = [q1, q2]
        else:
            thresholds = self.label_thresholds

        for idx in self.indices:
            record = self.records[idx]
            vb = float(np.asarray(record["VB"]).squeeze())
            if np.isnan(vb):
                label = 0
            elif vb < thresholds[1]:
                label = 0
            else:
                label = 1
            labels.append(label)
        return np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record_idx = self.indices[index]
        record = self.records[record_idx]

        channels = []
        for key in self.channel_keys:
            arr = np.asarray(record[key]).reshape(-1).astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            channels.append(arr)

        signal = np.stack(channels, axis=0)
        label = int(self.labels[index])

        signal = torch.from_numpy(signal).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transforms is not None:
            signal, label = self.transforms(signal, label)
        else:
            signal = signal[:, :2048] if signal.shape[1] >= 2048 else torch.nn.functional.pad(signal, (0, 2048 - signal.shape[1]), 'constant', 0)

        return signal, label

    @staticmethod
    def collate_fn(batch):
        signals, labels = list(zip(*batch))
        signals = torch.stack(signals, dim=0)
        labels = torch.stack(labels, dim=0)
        return signals, labels