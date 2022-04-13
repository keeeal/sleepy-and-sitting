from torch import nn, Tensor


class DixonNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        width: int = 100,
        kernel_size: int = 16,
        pool_size: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Conv1d(width, width, kernel_size),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width, num_classes)

    def get_final_layer_weights(self) -> Tensor:
        return self.fc.weight

    def get_features(self, x: Tensor) -> Tensor:
        return self.features(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.get_features(x)
        x = self.avgpool(x)
        x = x.squeeze(2)
        x = self.drop(x)
        x = self.fc(x)
        return x
