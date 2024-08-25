from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as np_typing
from PIL import Image

import neural_deskew
from neural_deskew.core.space import angle_space
from neural_deskew.core.transforms import decode, encode


class abc_Deskewer(ABC):
    """A base class for document image deskew"""

    def __init__(
        self,
        *args: Any,
        angle_stop: int = 360,
        angle_step: int = 4,
        **kwargs: dict[str, Any],
    ) -> None:
        self.angle_space = angle_space(angle_stop, angle_step)
        self.noangle = 0

    @abstractmethod
    def __call__(self, array: neural_deskew.Color) -> np_typing.NDArray[np.float32]:
        """It computes skew angle probas over the angle space"""

    def detect_angle(self, array: neural_deskew.Color) -> float:
        probas = self(array)
        maxidx = np.argmax(probas)

        return self.angle_space[maxidx]

    def deskew(
        self,
        image: Image.Image,
        *args: Any,
        max_size: int = 1280,
        **kwargs: dict[str, Any],
    ) -> Image.Image:
        encoded = encode(image, max_size)

        angle = self.detect_angle(encoded)

        decoded = decode(image, angle, **kwargs)

        return decoded


def forward(
    model: abc_Deskewer, array: neural_deskew.Color
) -> np_typing.NDArray[np.float32]:
    return model(array)
