import typing
from typing import Any

import cv2 as opencv
import numpy as np
import torch
from PIL import Image
from skimage.transform import rotate

import neural_deskew


Transform = typing.Callable[[Image.Image], torch.Tensor]


def resize_if_necessary(image: Image.Image, max_size: int) -> Image.Image:
    size = image.size

    if max(size) < max_size:
        return image

    if size[0] > size[1]:
        aspect = max_size / size[0]

        min_axis = aspect * size[1]
        min_axis = int(min_axis)

        resized_size = (max_size, min_axis)
    else:
        aspect = max_size / size[1]

        min_axis = aspect * size[0]
        min_axis = int(min_axis)

        resized_size = (min_axis, max_size)

    mode = Image.ANTIALIAS
    return image.resize(resized_size, mode)


def encode(image: Image.Image, max_size: int) -> neural_deskew.Color:
    image = resize_if_necessary(image, max_size)
    array = np.asarray(image)

    encoded = opencv.cvtColor(array, opencv.COLOR_BGR2GRAY)

    return encoded


def decode(image: Image.Image, angle: float, **kwargs: dict[str, Any]) -> Image.Image:
    array = np.asarray(image)
    array = rotate(array, angle, **kwargs) * 255

    decoded = array.astype(np.uint8)
    decoded = Image.fromarray(decoded)

    return decoded
