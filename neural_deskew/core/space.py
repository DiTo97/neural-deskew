from functools import lru_cache

import numpy as np
import numpy.typing as np_typing


@lru_cache(maxsize=5)
def angle_space(num_angles: int) -> np_typing.NDArray[np.float32]:
    """A linear angle space with constant step size in rad"""
    assert num_angles % 2 == 0, "The angle space must be symmetric"

    tot_split = int(num_angles / 2)

    neg_space = np.linspace(-np.pi / 2, 0, tot_split, endpoint=False)
    pos_space = np.linspace(0, +np.pi / 2, tot_split)

    space = np.hstack((neg_space, pos_space))  # It ensures 0 is in the space

    return space
