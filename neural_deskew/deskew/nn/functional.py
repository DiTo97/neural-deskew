import torch


def pred_interval_to_point_estimate(outputs: torch.Tensor) -> torch.Tensor:
    U = outputs[:, 0]
    L = outputs[:, 1]
    v = outputs[:, 2]

    point_estimate = v * U + (1 - v) * L
    return point_estimate


def pred_interval_to_confidence(outputs: torch.Tensor, coverage: float = 0.95) -> torch.Tensor:
    U = outputs[:, 0]
    L = outputs[:, 1]
    
    distance = U - L

    confidence = torch.sigmoid(-distance) 
    confidence = confidence * coverage
    
    return confidence
