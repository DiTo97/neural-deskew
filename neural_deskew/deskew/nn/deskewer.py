import torch
import torch.nn as nn


class Deskewer(nn.Module):
    def __init__(
        self,
        num_channels: int,
        hidden_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super(Deskewer, self).__init__()

        self.conv1d = nn.Conv1d(
            num_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1
        )

        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=3)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x