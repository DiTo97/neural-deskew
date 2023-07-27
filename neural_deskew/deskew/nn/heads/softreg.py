import torch
import torch.nn as nn


class Softreg(nn.Module):
    """A simple head for soft regression"""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super(Softreg, self).__init__()

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
