from functools import lru_cache

import numpy as np
import numpy.typing as np_typing


@lru_cache(maxsize=5)
def angle_space(num_angles: int) -> np_typing.NDArray[np.float32]:
    """A linear angle space with constant step size in rad"""
    space = np.linspace(
        -np.pi / 2, np.pi / 2, num_angles
    )  # step size = 180 / num angles (deg)

    return space
