import pathlib
import random
from typing import Any

import albumentations
import cv2 as opencv
import numpy as np
import pandas as pd
from skimage.transform import rotate
from torch.utils.data import Dataset


class SkewDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        transform: albumentations.Compose | None = None,
        seed: int = 10,
        max_angle: float = 90,
    ) -> None:
        random.seed(seed)

        dataset_dir = pathlib.Path(dataset_dir).resolve()

        images_dir = dataset_dir / "images"
        annotations_dir = dataset_dir / "annotations"

        annotations = pd.read_csv(annotations_dir / f"{split}.csv")

        def join_transform(filename: str) -> str:
            filepath = images_dir / filename
            filepath = str(filepath)

            return filepath

        annotations = annotations["filename"].apply(join_transform)
        annotations = np.array(annotations)

        self.annotations = annotations
        self.transform = transform
        self.max_angle = max_angle

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[Any, float]:
        image_path = self.annotations[idx]

        angle = random.uniform(-self.max_angle, +self.max_angle)

        image = opencv.imread(image_path)
        image = rotate(image, -angle, cval=1.0)

        if self.transform is not None:
            image = self.transform(image=image)
            image = image["image"]

        # TODO: It should transform to probas
        # TODO: It should encode the angle over the space

        return image, angle
