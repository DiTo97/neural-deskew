import torch


def pred_interval_to_point_estimate(output: torch.Tensor) -> torch.Tensor:
    U = output[:, 0]
    L = output[:, 1]
    v = output[:, 2]

    point_estimate = v * U + (1 - v) * L
    return point_estimate
        
def pred_interval_to_confidence(output: torch.Tensor, coverage: float = 0.95) -> torch.Tensor:
    U = output[:, 0]
    L = output[:, 1]
    
    distance = U - L

    confidence = torch.sigmoid(-distance) 
    confidence = confidence * coverage
    
    return confidence
