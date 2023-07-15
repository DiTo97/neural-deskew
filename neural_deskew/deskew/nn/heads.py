import torch
import torch.nn as nn


class PIVEN(nn.Module):
    """The PIVEN head for regression with prediction intervals

    E. Simhayev, K. Gilad, and R. Lior, 2020
    PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """

    def __init__(self, hidden_dim: int) -> None:
        super(PIVEN, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, 2)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        interval = self.linear1(x)

        coeff = self.linear2(x)
        coeff = torch.sigmoid(coeff)

        return torch.cat((interval, coeff), dim=1)


class Regression(nn.Module):
    """A base head for regression"""

    def __init__(self, hidden_dim: int) -> None:
        super(Regression, self).__init__()

        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Classification(nn.Module):
    """A base head for classification"""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super(Classification, self).__init__()

        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probas = torch.softmax(logits, dim=1)

        return probas
