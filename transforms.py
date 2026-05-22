import random
import torch


class Compose1D(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal, mask):
        for t in self.transforms:
            signal, mask = t(signal, mask)
        return signal, mask


class RandomCrop1D(object):
    """
    在一维时间序列上进行随机长度裁剪，保证输入网络的数据长度固定为 L
    """

    def __init__(self, size):
        self.size = size  # 期望的序列长度 L

    def __call__(self, signal, mask):
        seq_len = signal.shape[-1]

        if seq_len == self.size:
            return signal, mask
        elif seq_len < self.size:
            # 如果原始序列比期望短，进行 padding (补零)
            pad_len = self.size - seq_len
            signal = torch.nn.functional.pad(signal, (0, pad_len), 'constant', 0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), 'constant', 255)  # 255 通常作为 ignore_index
            return signal, mask

        # 如果序列比期望长，随机找一个起始点裁剪
        start_idx = random.randint(0, seq_len - self.size)
        signal = signal[:, start_idx: start_idx + self.size]
        mask = mask[start_idx: start_idx + self.size]

        return signal, mask


class Normalize1D(object):
    """
    对各个通道进行 Z-Score 标准化
    注意: 工业数据不同通道(如电压、振动)量纲差异极大，必须标准化
    """

    def __init__(self, mean, std):
        # mean 和 std 应该是长度为通道数 (6,) 的 tuple 或 list
        self.mean = torch.tensor(mean).view(-1, 1)
        self.std = torch.tensor(std).view(-1, 1)

    def __call__(self, signal, mask):
        signal = (signal - self.mean) / self.std
        return signal, mask