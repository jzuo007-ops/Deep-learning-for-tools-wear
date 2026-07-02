import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBackbone1D(nn.Module):
    """
    LSTM-based 1D backbone for multi-channel tool-wear signals.

    Input:
        x: Tensor with shape [batch, in_channels, sequence_length]

    Output:
        aux_features:  Tensor with shape [batch, aux_channels, reduced_length]
        main_features: Tensor with shape [batch, out_channels, reduced_length]

    The output channel sizes default to 1024 and 2048 so this backbone can
    replace the existing ResNet50_1D backbone in DeepLabV3-style heads.
    """

    def __init__(
        self,
        in_channels=6,
        stem_channels=128,
        aux_hidden=256,
        main_hidden=512,
        aux_channels=1024,
        out_channels=2048,
        num_layers=1,
        dropout=0.2,
        downsample_factor=8,
    ):
        super(LSTMBackbone1D, self).__init__()
        if downsample_factor < 4:
            raise ValueError("downsample_factor must be at least 4.")

        self.downsample_factor = downsample_factor
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.aux_lstm = nn.LSTM(
            input_size=stem_channels,
            hidden_size=aux_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.main_lstm = nn.LSTM(
            input_size=aux_hidden * 2,
            hidden_size=main_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.aux_norm = nn.LayerNorm(aux_hidden * 2)
        self.main_norm = nn.LayerNorm(main_hidden * 2)
        self.aux_project = nn.Sequential(
            nn.Conv1d(aux_hidden * 2, aux_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(aux_channels),
            nn.ReLU(inplace=True),
        )
        self.main_project = nn.Sequential(
            nn.Conv1d(main_hidden * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _extra_downsample(self, x):
        current_factor = 4
        while current_factor < self.downsample_factor:
            x = F.avg_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)
            current_factor *= 2
        return x

    @staticmethod
    def _to_sequence(x):
        return x.transpose(1, 2).contiguous()

    @staticmethod
    def _to_feature_map(x):
        return x.transpose(1, 2).contiguous()

    def forward(self, x):
        x = self.stem(x)
        x = self._extra_downsample(x)

        sequence = self._to_sequence(x)
        aux_sequence, _ = self.aux_lstm(sequence)
        aux_sequence = self.aux_norm(aux_sequence)

        main_sequence, _ = self.main_lstm(aux_sequence)
        main_sequence = self.main_norm(main_sequence)

        aux_features = self.aux_project(self._to_feature_map(aux_sequence))
        main_features = self.main_project(self._to_feature_map(main_sequence))
        return aux_features, main_features


def lstm_backbone_1d(in_channels=6, **kwargs):
    return LSTMBackbone1D(in_channels=in_channels, **kwargs)
