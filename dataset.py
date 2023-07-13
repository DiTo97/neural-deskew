import pathlib
import random

import albumentations
import cv2 as opencv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.transform import rotate
from torch.utils.data import Dataset

from neural_deskew.core import transforms
from neural_deskew.core.space import angle_space


class DeskewDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        transform: transforms.Transform,
        num_angles: int,
        image_transform: albumentations.Compose | None = None,
        seed: int = 10,
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

        self.angle_space = angle_space(num_angles)
        self.annotations = annotations
        self.transform = transform
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.annotations[idx]

        angle_idx = random.choice(range(len(self.angle_space))
        
        angle = self.angle_space[angle_idx]
        angle = np.rad2deg(angle)

        array = opencv.imread(image_path)
        array = rotate(array, angle, cval=1.0)

        if self.image_transform is not None:
            array = self.image_transform(array)
            array = array["image"]

        array = array * 255
        array = array.astype(np.uint8)
        
        image = Image.fromarray(array).convert("RGB")

        angle_distr = torch.zeros(len(self.angle_space))
        angle_distr = torch.float32(angle_distr)
        
        angle_distr[angle_idx] = 1.0

        return self.transform(image), angle_distr
