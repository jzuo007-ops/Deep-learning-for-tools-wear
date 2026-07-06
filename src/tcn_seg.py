import torch
import torch.nn as nn


class Chomp1D(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.1,
        causal=False,
    ):
        super().__init__()
        if causal:
            padding = (kernel_size - 1) * dilation
            chomp = padding
        else:
            padding = ((kernel_size - 1) * dilation) // 2
            chomp = 0

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            Chomp1D(chomp),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            Chomp1D(chomp),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        residual = self.downsample(x)
        if out.shape[-1] != residual.shape[-1]:
            out = out[..., : residual.shape[-1]]
        return self.relu(out + residual)


class TCNSeg(nn.Module):
    """
    Dilated temporal convolutional network for 1D point-wise segmentation.
    """

    def __init__(
        self,
        in_channels=7,
        num_classes=3,
        channels=(64, 64, 128, 128, 256),
        kernel_size=3,
        dropout=0.1,
        causal=False,
        aux_loss=False,
    ):
        super().__init__()
        self.aux_loss = aux_loss
        blocks = []
        current_channels = in_channels
        for layer_index, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock1D(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_index,
                    dropout=dropout,
                    causal=causal,
                )
            )
            current_channels = out_channels

        self.network = nn.ModuleList(blocks)
        self.classifier = nn.Conv1d(current_channels, num_classes, kernel_size=1)
        if aux_loss:
            aux_index = max(0, len(channels) // 2 - 1)
            self.aux_index = aux_index
            self.aux_classifier = nn.Conv1d(channels[aux_index], num_classes, kernel_size=1)

    def forward(self, x):
        result = {}
        aux = None
        for index, block in enumerate(self.network):
            x = block(x)
            if self.aux_loss and index == self.aux_index:
                aux = self.aux_classifier(x)

        result["out"] = self.classifier(x)
        if self.aux_loss and aux is not None:
            result["aux"] = aux
        return result
