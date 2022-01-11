from torch import nn, Tensor
from torch.nn.functional import avg_pool1d


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
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(width, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = avg_pool1d(x, kernel_size=x.shape[2])  # global avg pool
        x = x.squeeze(2)
        x = self.classifier(x)
        return x
