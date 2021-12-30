from torch import nn
from torch.nn.functional import avg_pool1d


class DixonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=100, kernel_size=16),
            nn.ReLU(),
            nn.Conv1d(100, 100, 16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(100, 100, 16),
            nn.ReLU(),
            nn.Conv1d(100, 100, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(100, 1),)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = avg_pool1d(x, kernel_size=x.shape[2])  # global avg pool
        x = x.squeeze(2)
        x = self.classifier(x)
        return x
