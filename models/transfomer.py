import math

import torch
from torch.nn.functional import pad
from torch import nn, zeros, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self, in_channels: int = 3, num_classes: int = 1, dropout: float = 0.2,
    ):
        self.channel_pad = int(in_channels % 2)
        in_channels += self.channel_pad

        super().__init__()
        self.encoder = nn.Sequential(
            PositionalEncoding(in_channels),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=in_channels, nhead=2), num_layers=1,
            ),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(4096 * in_channels, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        padding = 2 * len(x.shape) * [0]
        padding[2] = self.channel_pad
        x = pad(x, padding)
        x = x.permute(2, 0, 1)

        x = self.encoder(x)
        x = x.permute(1, 2, 0).flatten(1)
        x = self.classifier(x)
        return x
