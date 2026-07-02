import random

import torch


class Compose1D(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal, target):
        for transform in self.transforms:
            signal, target = transform(signal, target)
        return signal, target


class RandomCrop1D(object):
    """
    Random crop for 1D time-series training samples.
    Short signals are padded on the right.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, signal, target):
        seq_len = signal.shape[-1]

        if seq_len == self.size:
            return signal, target
        if seq_len < self.size:
            pad_len = self.size - seq_len
            signal = torch.nn.functional.pad(signal, (0, pad_len), "constant", 0)
            return signal, target

        start_idx = random.randint(0, seq_len - self.size)
        signal = signal[:, start_idx:start_idx + self.size]

        if isinstance(target, torch.Tensor) and target.ndim == 1 and target.numel() == seq_len:
            target = target[start_idx:start_idx + self.size]

        return signal, target


class CenterCrop1D(object):
    """
    Deterministic crop for validation/testing.
    Short signals are padded on the right.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, signal, target):
        seq_len = signal.shape[-1]

        if seq_len == self.size:
            return signal, target
        if seq_len < self.size:
            pad_len = self.size - seq_len
            signal = torch.nn.functional.pad(signal, (0, pad_len), "constant", 0)
            return signal, target

        start_idx = (seq_len - self.size) // 2
        signal = signal[:, start_idx:start_idx + self.size]

        if isinstance(target, torch.Tensor) and target.ndim == 1 and target.numel() == seq_len:
            target = target[start_idx:start_idx + self.size]

        return signal, target


class Normalize1D(object):
    """
    Channel-wise z-score normalization for 1D signals.
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1)

    def __call__(self, signal, target):
        signal = (signal - self.mean) / self.std
        return signal, target
