import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.deeplabv3_model import DeepLabV3_1D
from my_dataset import ToolWear1DDataset
import transforms as T


def get_transform(train=True, seq_length=2048):
    # 根据你的全部数据集统计出 6 个通道的均值和标准差
    channel_means = [0.0] * 6
    channel_stds = [1.0] * 6

    base_transforms = [T.Normalize1D(mean=channel_means, std=channel_stds)]

    if train:
        base_transforms.insert(0, T.RandomCrop1D(size=seq_length))
    else:
        # 验证集通常可以使用固定的起始点裁剪，或者处理整段数据
        base_transforms.insert(0, T.RandomCrop1D(size=seq_length))

    return T.Compose1D(base_transforms)


def criterion(inputs, target, ignore_index=255):
    losses = {}
    for name, x in inputs.items():
        # x shape: (N, C, L), target shape: (N, L)
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index)
        if name == "out":
            losses[name] = loss
        elif name == "aux":
            # 辅助分支损失权重通常设为 0.4
            losses[name] = loss * 0.4

    return sum(losses.values())


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================= 1. 数据集准备 =================
    data_root = "./data_folder"  # 替换为你的真实路径
    batch_size = 16

    train_dataset = ToolWear1DDataset(data_root, txt_list="train.txt", transforms=get_transform(train=True))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=train_dataset.collate_fn)

    # ================= 2. 模型与优化器 =================
    num_classes = 3  # 根据你的任务设定 (如: 正常、轻度磨损、严重磨损)
    model = DeepLabV3_1D(in_channels=6, num_classes=num_classes, aux_loss=True)
    model.to(device)

    # 使用 SGD 或 AdamW
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # 使用多项式学习率衰减 (分割常用)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / 50) ** 0.9)  # 假设跑 50 个 epoch

    # ================= 3. 训练循环 =================
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (signals, masks) in enumerate(train_loader):
            signals, masks = signals.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(signals)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        lr_scheduler.step()
        print(f"Epoch {epoch + 1} finished. Average Loss: {running_loss / len(train_loader):.4f}")

        # 保存权重
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"deeplabv3_1d_epoch_{epoch + 1}.pth")


if __name__ == '__main__':
    main()