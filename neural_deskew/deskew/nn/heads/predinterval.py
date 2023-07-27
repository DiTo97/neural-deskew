import torch
import torch.nn as nn
from torch.nn import functional as F


def predinterval_to_point_estimate(outputs: torch.Tensor) -> torch.Tensor:
    U = outputs[:, 0]
    L = outputs[:, 1]
    v = outputs[:, 2]

    point_estimate = v * U + (1 - v) * L
    return point_estimate


def predinterval_to_confidence(
    outputs: torch.Tensor, alpha: float = 0.05
) -> torch.Tensor:
    U = outputs[:, 0]
    L = outputs[:, 1]

    distance = U - L

    confidence = F.sigmoid(-distance)
    confidence = confidence * (1 - alpha)

    return confidence


class PIVEN(nn.Module):
    """The PIVEN head for regression with prediction intervals
    
    Notes
    -----
    The head introduced before equation (3) in [1]_

    References
    ----------
    .. [1] E. Simhayev, K. Gilad, and R. Lior, 2020
       PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """

    def __init__(self, hidden_dim: int) -> None:
        super(PIVEN, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, 2)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        interval = self.linear1(x)

        v = self.linear2(x)
        v = F.sigmoid(v)

        return torch.cat((interval, v), dim=1)
