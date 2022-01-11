from typing import Any, Callable

from torch import nn, zeros, Tensor


class RNN(nn.Module):
    def __init__(
        self,
        rnn: Callable,
        in_channels: int = 3,
        num_classes: int = 1,
        width: int = 512,
        rnn_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.width = width
        self.rnn_layers = rnn_layers
        self.rnn = rnn(in_channels, width, rnn_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(width, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.rnn(x)[1]
        x = x[0] if isinstance(x, tuple) else x
        return self.classifier(x)


def rnn(**kwargs: Any) -> RNN:
    return RNN(nn.RNN, **kwargs)


def lstm(**kwargs: Any) -> RNN:
    return RNN(nn.LSTM, **kwargs)


def gru(**kwargs: Any) -> RNN:
    return RNN(nn.GRU, **kwargs)
