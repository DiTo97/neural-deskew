import numpy as np
import numpy.typing as np_typing
from scipy.special import softmax
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

import neural_deskew
from neural_deskew.core.deskewer import abc_Deskewer


class Deskewer(abc_Deskewer):
    def __init__(
        self,
        num_angles: int,
        scale: float = 0.1,
        sigma: float = 3.0,
        num_peaks: int = 50,
    ) -> None:
        super().__init__(num_angles)

        self.scale = scale
        self.sigma = sigma
        self.num_peaks = num_peaks

    def __call__(self, array: neural_deskew.Color) -> np_typing.NDArray[np.float32]:
        """It computes skew angle probas over the angle space"""
        edgemap = canny(array, sigma=self.sigma)

        linemap, angles, distances = hough_line(edgemap, self.angle_space)

        threshold = self.scale * np.max(linemap)

        _, angles_peaks, _ = hough_line_peaks(
            linemap, angles, distances, num_peaks=self.num_peaks, threshold=threshold
        )

        logits = np.zeros_like(self.angle_space, dtype=np.float32)

        if len(angles_peaks) == 0:
            probas = logits
            probas[self.noangle] = 1.0

            return probas

        matches = angles_peaks[:, None] == self.angle_space
        matches = np.nonzero(matches)[1]

        np.add.at(logits, matches, 1)
        probas = softmax(logits)

        return probas
