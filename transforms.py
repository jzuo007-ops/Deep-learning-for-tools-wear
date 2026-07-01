import random
import torch


class Compose1D(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal, target):
        for t in self.transforms:
            signal, target = t(signal, target)
        return signal, target


class RandomCrop1D(object):
    """
    在一维时间序列上进行随机长度裁剪，保证输入网络的数据长度固定为 L。
    这里兼容分类标签（标量）和分割标签（长度为 L 的向量）。
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, signal, target):
        seq_len = signal.shape[-1]

        if seq_len == self.size:
            return signal, target
        elif seq_len < self.size:
            pad_len = self.size - seq_len
            signal = torch.nn.functional.pad(signal, (0, pad_len), 'constant', 0)
            return signal, target

        start_idx = random.randint(0, seq_len - self.size)
        signal = signal[:, start_idx:start_idx + self.size]

        if isinstance(target, torch.Tensor) and target.ndim == 1 and target.numel() == seq_len:
            target = target[start_idx:start_idx + self.size]

        return signal, target


class Normalize1D(object):
    """
    对各个通道进行 Z-Score 标准化。
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1)
        self.std = torch.tensor(std).view(-1, 1)

    def __call__(self, signal, target):
        signal = (signal - self.mean) / self.std
        return signal, target