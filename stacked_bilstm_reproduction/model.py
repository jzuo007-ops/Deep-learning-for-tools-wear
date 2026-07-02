import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, sequence_outputs: torch.Tensor):
        energy = torch.tanh(self.proj(sequence_outputs))
        weights = torch.softmax(self.score(energy), dim=1)
        context = torch.sum(weights * sequence_outputs, dim=1)
        return context, weights.squeeze(-1)


class StackedBiLSTMAttentionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention_dim: int = 64,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.attention = TemporalAttention(hidden_dim * 2, attention_dim=attention_dim)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        sequence_outputs, _ = self.encoder(x)
        context, attention_weights = self.attention(sequence_outputs)
        prediction = self.regressor(context)
        return prediction, attention_weights

