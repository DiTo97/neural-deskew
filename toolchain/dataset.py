import pathlib
import random
from functools import lru_cache

import albumentations
import cv2 as opencv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.transform import rotate
from torch.utils.data import Dataset

import neural_deskew
from neural_deskew.core import transforms
from neural_deskew.core.space import angle_cross_similarity, angle_space


@lru_cache(maxsize=128)
def imread(path: str) -> neural_deskew.Color:
    return opencv.imread(path)


class DeskewDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        encoder: transforms.Transform,
        image_transform: albumentations.Compose | None = None,
        softreg: bool = False,
        seed: int = 10,
        **kwargs,
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
        self.encoder = encoder
        self.image_transform = image_transform
        self.softreg = softreg

        space_kwargs = kwargs.get("angle-space", {})
        cross_similarity_kwargs = kwargs.get("angle-cross-similarity", {})

        self.angle_space = angle_space(**space_kwargs)

        self.angle_cross_similarity = (
            None
            if not self.softreg
            else angle_cross_similarity(self.angle_space, **cross_similarity_kwargs)
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float | torch.Tensor]:
        angleidx = random.choice(range(len(self.angle_space)))

        angle = self.angle_space[angleidx]

        array = imread(self.annotations[idx])
        array = rotate(array, angle, cval=1.0)

        if self.image_transform is not None:
            array = self.image_transform(array)
            array = array["image"]

        array = array * 255
        array = array.astype(np.uint8)

        image = Image.fromarray(array).convert("RGB")

        encoding = self.encoder(image)

        truth = (
            angle
            if not self.softreg
            else torch.from_numpy(self.angle_cross_similarity[angleidx])
        )

        return encoding, truth
