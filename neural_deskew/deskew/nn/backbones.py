import torch
import torch.nn as nn


class Encoder(nn.Module):
    """A base encoder model for angle confidence vectors"""

    def __init__(
        self,
        num_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super(Encoder, self).__init__()

        self.conv1d = nn.Conv1d(
            num_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1
        )

        self.batch_norm1d = nn.BatchNorm1d(hidden_dim)
        
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=3)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.batch_norm1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)
        x = self.relu(x)

        return x
