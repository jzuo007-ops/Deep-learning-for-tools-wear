import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMSeg(nn.Module):
    """
    BiLSTM segmentation head for multi-sensor time-series process states.

    The optional convolutional stem downsamples long crops before the LSTM and
    the logits are interpolated back to the original point resolution.
    """

    def __init__(
        self,
        in_channels=7,
        num_classes=3,
        stem_channels=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
        downsample_factor=4,
        aux_loss=False,
    ):
        super().__init__()
        if downsample_factor not in (1, 2, 4, 8):
            raise ValueError("downsample_factor must be one of 1, 2, 4, or 8.")

        self.aux_loss = aux_loss
        stem_layers = [
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        ]
        current_factor = 1
        while current_factor < downsample_factor:
            stem_layers.extend(
                [
                    nn.Conv1d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm1d(stem_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_factor *= 2
        self.stem = nn.Sequential(*stem_layers)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=stem_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv1d(hidden_size * 2, num_classes, kernel_size=1)
        if aux_loss:
            self.aux_classifier = nn.Conv1d(stem_channels, num_classes, kernel_size=1)

    def forward(self, x):
        input_length = x.shape[-1]
        features = self.stem(x)
        sequence = features.transpose(1, 2).contiguous()
        sequence, _ = self.lstm(sequence)
        sequence = self.norm(sequence)
        sequence = self.dropout(sequence)
        features_out = sequence.transpose(1, 2).contiguous()

        out = self.classifier(features_out)
        out = F.interpolate(out, size=input_length, mode="linear", align_corners=False)
        result = {"out": out}
        if self.aux_loss:
            aux = self.aux_classifier(features)
            aux = F.interpolate(aux, size=input_length, mode="linear", align_corners=False)
            result["aux"] = aux
        return result
