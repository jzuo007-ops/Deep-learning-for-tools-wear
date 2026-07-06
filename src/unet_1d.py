import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = ConvBlock1D(in_channels, out_channels, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock1D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock1D(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet1D(nn.Module):
    """
    1D U-Net for point-wise process-state segmentation.

    Input shape:
        [batch, in_channels, sequence_length]

    Output:
        {"out": logits}, where logits has shape [batch, num_classes, sequence_length].
    """

    def __init__(
        self,
        in_channels=7,
        num_classes=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        dropout=0.1,
        aux_loss=False,
    ):
        super().__init__()
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        if len(channels) < 3:
            raise ValueError("UNet1D requires at least three channel levels.")

        self.aux_loss = aux_loss
        self.stem = ConvBlock1D(in_channels, channels[0], dropout=dropout)
        self.down_blocks = nn.ModuleList(
            DownBlock1D(channels[i], channels[i + 1], dropout=dropout)
            for i in range(len(channels) - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpBlock1D(channels[i], channels[i - 1], channels[i - 1], dropout=dropout)
            for i in range(len(channels) - 1, 0, -1)
        )
        self.classifier = nn.Conv1d(channels[0], num_classes, kernel_size=1)
        if aux_loss:
            self.aux_classifier = nn.Conv1d(channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        input_length = x.shape[-1]
        skips = [self.stem(x)]
        for down in self.down_blocks:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = up(x, skip)

        out = self.classifier(x)
        if out.shape[-1] != input_length:
            out = F.interpolate(out, size=input_length, mode="linear", align_corners=False)

        result = {"out": out}
        if self.aux_loss:
            aux = self.aux_classifier(skips[-1])
            aux = F.interpolate(aux, size=input_length, mode="linear", align_corners=False)
            result["aux"] = aux
        return result
