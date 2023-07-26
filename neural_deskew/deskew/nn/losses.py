import torch
import torch.nn as nn


class PIVEN(nn.Module):
    """The PIVEN loss for regression with prediction intervals

    E. Simhayev, K. Gilad, and R. Lior, 2020
    PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """

    def __init__(self, lambda_in=15.0, soften=160.0, alpha=0.05, beta=0.5, eps: float = 1e-6) -> None:
        super(PIVEN, self).__init__()

        self.lambda_in = lambda_in
        self.soften = soften
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        U = outputs[:, 0]  # U(x)
        L = outputs[:, 1]  # L(x)
        v = outputs[:, 2]  # v(x)
        y = targets[:, 0]  # y(x)

        batch_size = float(y.size(0))

        alpha_ = torch.tensor(self.alpha)
        lambda_ = torch.tensor(self.lambda_in)

        # k_soft uses sigmoid
        k_soft = torch.sigmoid((U - y) * self.soften) * torch.sigmoid(
            (y - L) * self.soften
        )

        # k_hard uses sign step function
        k_hard = torch.maximum(torch.sign(U - y), 0.0) * torch.maximum(
            torch.sign(y - L), 0.0
        )

        # MPIW_capt from equation 4
        MPIW_capt = torch.sum(torch.abs(U - L) * k_hard) / (torch.sum(k_hard) + self.eps)

        # equation 1 where k is k_soft
        PICP_soft = torch.mean(k_soft)

        # pi loss from section 4.2
        pi_loss = MPIW_capt + lambda_ * torch.sqrt(batch_size) * torch.square(
            torch.maximum(0.0, 1.0 - alpha_ - PICP_soft)
        )

        y_piven = v * U + (1 - v) * L  # equation 3
        y_piven = torch.reshape(y_piven, (-1, 1))

        v_loss = nn.functional.mse_loss(targets, y_piven)  # equation 5
        piven_loss = self.beta * pi_loss + (1 - self.beta) * v_loss  # equation 6

        return piven_loss
