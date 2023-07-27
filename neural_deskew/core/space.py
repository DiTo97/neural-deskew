from functools import lru_cache

import numpy as np
import numpy.typing as np_typing
from scipy.special import softmax


@lru_cache(maxsize=5)
def angle_space(stop: int = 360, step: int = 4, endpoint: bool = False) -> np_typing.NDArray[np.float32]:
    """A linear angle space with variable step size in deg"""
    assert step >= 1, "The min step size is one"

    num_angles = stop * step

    space = np.linspace(0, stop, num_angles, endpoint=endpoint)
    
    return space


def angle_cross_similarity(
    space: np_typing.NDArray[np.float32], p: float = 2.0, smoothing: float = 1.25
) -> np_typing.NDArray[np.float32]:
    """The angle cross similarity for soft regression
    
    Any angle is encoded by a soft similarity distribution over the space.

    Parameters
    ----------
    space
        The linear angle space in deg

    p
        The distance penalty (Minkowski-like)

    smoothing
        The origin angle label smoothing

    Returns
    -------
        The square cross similarity matrix

    Notes
    -----
    The encoding is inspired by the soft contrastive loss in [1]_.

    References
    ----------
    .. [1] Y. Yang, X. Liu, et al., SimPer: Simple Self-supervised Learning of Periodic Targets, 2023
    """
    D = np.abs(space[:, None] - space[None, :])  # distance matrix

    # It accounts for the angle periodicity
    D = np.where(D > 180, 360 - D, D)

    cross_similarity = -(D ** p) / smoothing
    cross_similarity = softmax(cross_similarity, axis=-1)
    
    return cross_similarity
