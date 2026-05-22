import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ToolWear1DDataset(Dataset):
    """
    一维多通道信号数据集 (例如: 刀具磨损监测)
    """

    def __init__(self, data_root, txt_list, transforms=None):
        super(ToolWear1DDataset, self).__init__()
        self.data_root = data_root
        # txt_list 中保存了所有需要读取的文件名（不含扩展名），如 "sample_001"
        with open(txt_list, "r") as f:
            self.file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.transforms = transforms

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]

        # 假设数据和标签分别存放在 data 和 mask 文件夹中
        data_path = os.path.join(self.data_root, "data", f"{file_name}.npy")
        mask_path = os.path.join(self.data_root, "mask", f"{file_name}.npy")

        # 读取 numpy 数组
        # signal shape 期望为: (6, L), mask shape 期望为: (L,)
        signal = np.load(data_path)
        mask = np.load(mask_path)

        # 转换为 tensor
        signal = torch.from_numpy(signal).float()
        mask = torch.from_numpy(mask).long()

        if self.transforms is not None:
            signal, mask = self.transforms(signal, mask)

        return signal, mask

    @staticmethod
    def collate_fn(batch):
        signals, masks = list(zip(*batch))
        signals = torch.stack(signals, dim=0)
        masks = torch.stack(masks, dim=0)
        return signals, masks