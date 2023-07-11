from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image

import neural_deskew
from neural_deskew.core.transforms import decode, encode


class Deskewer(ABC):
    """A base class for document image deskew"""

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def detect_angle(self, array: neural_deskew.Array) -> float:
        pass

    def deskew(
        self,
        image: Image.Image,
        *args: Any,
        max_size: int = 1280,
        **kwargs: dict[str, Any],
    ) -> Image.Image:
        encoded = encode(image, max_size)

        angle = self.detect_angle(encoded)
        angle = np.rad2deg(angle)

        decoded = decode(image, angle, **kwargs)

        return decoded
