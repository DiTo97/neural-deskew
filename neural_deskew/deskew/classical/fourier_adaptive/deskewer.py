import cv2 as opencv
import numpy as np
import numpy.typing as np_typing
from scipy.special import softmax

import neural_deskew
from neural_deskew.core.deskewer import abc_Deskewer


def _ensure_square(array: neural_deskew.Color) -> neural_deskew.Color:
    """It ensures the array is square adding border padding"""
    max_axis = max(array.shape[:2])
    out_size = opencv.getOptimalDFTSize(max_axis)

    return opencv.copyMakeBorder(
        src=array,
        top=0,
        bottom=out_size - array.shape[0],
        left=0,
        right=out_size - array.shape[1],
        borderType=opencv.BORDER_CONSTANT,
        value=255,
    )


def _spectrum_magnitude(array: neural_deskew.Color) -> np_typing.NDArray[np.float32]:
    squared = _ensure_square(array)

    squared = opencv.adaptiveThreshold(
        ~squared, 255, opencv.ADAPTIVE_THRESH_GAUSSIAN_C, opencv.THRESH_BINARY, 15, -10
    )

    spectrum = np.fft.fft2(squared)
    spectrum = np.fft.fftshift(spectrum)

    S = np.abs(spectrum)
    return S


class Deskewer(abc_Deskewer):
    def __init__(self, num_angles: int) -> None:
        super().__init__(num_angles)

    def __call__(self, array: neural_deskew.Color) -> np_typing.NDArray[np.float32]:
        """It computes skew angle probas over the angle space"""
        S = _spectrum_magnitude(array)

        assert S.shape[0] == S.shape[1], "The spectrum magnitude must be square"

        num_rows = int(S.shape[0] / 2)
        num_cols = int(S.shape[1] / 2)

        profile = self.angle_space.copy()

        def forward(t):
            _f = np.vectorize(
                lambda x: S[
                    num_cols + int(+x * np.cos(t)), num_cols + int(-x * np.sin(t))
                ]
            )

            _l = _f(range(num_rows))

            encoding = np.sum(_l)
            return encoding

        forward = np.vectorize(forward)

        logits = forward(profile)
        probas = softmax(logits)

        return probas
