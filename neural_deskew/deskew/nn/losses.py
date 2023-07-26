import torch
import torch.nn as nn
from torch.nn import functional as F

from .functional import pred_interval_to_point_estimate


def mpiw(
    U: torch.Tensor, L: torch.Tensor, k: torch.Tensor, eps: float = 1e-6
) -> float:
    """The mean prediction interval width (MPIW) metric

    equation (4) from

    E. Simhayev, K. Gilad, and R. Lior, 2020
    PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """
    nume = torch.sum(torch.abs(U - L) * k)
    deno = torch.sum(k) + eps

    metric = nume / deno
    return metric


def picp(k: torch.Tensor) -> float:
    """The prediction interval coverage probability (PICP) metric
    
    equation (1) from

    E. Simhayev, K. Gilad, and R. Lior, 2020
    PIVEN: A DNN for Prediction Intervals with Specific Value Prediction"""
    metric = torch.mean(k)
    return metric


class PIVEN(nn.Module):
    """The PIVEN loss for regression with prediction intervals

    E. Simhayev, K. Gilad, and R. Lior, 2020
    PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """

    def __init__(
        self, 
        lambda_ : float = 15.0, 
        soft: float = 160.0, 
        alpha: float = 0.05, 
        beta: float = 0.5, 
        eps: float = 1e-6
    ) -> None:
        super().__init__()

        self.lambda_ = lambda_
        self.soft = soft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        U = outputs[:, 0]  # U(x)
        L = outputs[:, 1]  # L(x)
        v = outputs[:, 2]  # v(x)
        y = targets[:, 0]  # y(x)

        n = outputs.size(0)
        n = torch.tensor(n)

        k_soft_upper = torch.sigmoid(self.soft * (U - y))
        k_soft_lower = torch.sigmoid(self.soft * (y - L))
        
        k_hard_upper = torch.clamp(torch.sign(U - y), 0.)
        k_hard_lower = torch.clamp(torch.sign(y - L), 0.)

        k_soft = k_soft_upper * k_soft_lower
        k_hard = k_hard_upper * k_hard_lower

        mpiw_capt = mpiw(U, L, k_hard, self.eps)
        picp_soft = picp(k_soft)

        penalty = torch.clamp(1 - self.alpha - picp_soft, 0.)
        penalty = torch.square(penalty)

        loss = mpiw_capt + torch.sqrt(n) * self.lambda_ * penalty  # PI loss

        estimates = pred_interval_to_point_estimate(outputs)
        estimates = torch.reshape(estimates, (-1, 1))

        valloss = F.mse_loss(targets, estimates)  # equation (5)
        cumloss = self.beta * loss + (1 - self.beta) * valloss  # equation (6)

        return cumloss
