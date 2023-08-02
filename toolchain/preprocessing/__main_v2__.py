"""A script for pre-processing a document image skew estimation collection"""
import argparse
import os
import pathlib
import random
import tempfile
from importlib.machinery import SourceFileLoader as importlib_FileLoader
from typing import Any

import albumentations
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import rotate
from tqdm.auto import tqdm

import wandb
from neural_deskew.core.space import angle_space


Transform = object


def image_transform_from_config(config: str | None) -> Transform | None:
    if config is None:
        return None

    transform_config = pathlib.Path(config).resolve()

    format_ = transform_config.suffix
    format_ = format_.replace(".", "")

    transform_config = str(transform_config)

    if not os.path.exists(transform_config):
        message = f"No augmentation pipeline config file at {transform_config}"
        raise ValueError(message)

    transform = albumentations.load(transform_config, format_)

    return transform


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(__name__)

    parser.add_argument(
        "--datasets",
        help="The document image skew estimation datasets",
        type=str,
        required=True,
        nargs="+",
        choices=["dise2021", "rdocuments"],
    )

    parser.add_argument(
        "--artifact-dir", help="The local artifact dir", type=str, required=True
    )

    parser.add_argument("--artifact", help="The name of the W&B artifact", type=str)

    parser.add_argument(
        "--stop-angle", help="The stop angle in degrees", type=int, default=360
    )

    parser.add_argument(
        "--step-angle", help="The number of steps per angle", type=int, default=4
    )

    parser.add_argument(
        "--image-transform-config",
        help="The image augmentation pipeline config file",
        type=str,
    )

    parser.add_argument(
        "--num-image-transforms",
        help="The number of image transfoms per skew angle per image",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--num-angle-transforms",
        help="The number of skew angle transforms per image",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--train-split", help="The train split", type=float, default=0.6
    )

    parser.add_argument(
        "--valid-split", help="The valid split", type=float, default=0.2
    )

    parser.add_argument("--test-split", help="The test split", type=float, default=0.2)
    parser.add_argument("--seed", help="The random seed", type=int, default=10)

    args, _ = parser.parse_known_args()
    args = vars(args)

    return args


def main(
    datasets: list[str],
    artifact_dir: str,
    artifact: str | None,
    stop_angle: int,
    step_angle: int,
    image_transform_config: str | None,
    num_image_transforms: int,
    num_angle_transforms: int,
    train_split: float,
    valid_split: float,
    test_split: float,
    seed: int,
):
    random.seed(seed)

    cumsplit = train_split + valid_split + test_split

    if cumsplit != 1.0:
        message = (
            f"The train, valid and test splits must sum up to 1.0 - "
            f"({train_split}, {valid_split}, {test_split})"
        )

        raise ValueError(message)

    image_transform = image_transform_from_config(image_transform_config)

    if not os.path.exists(artifact_dir):
        os.mkdir(artifact_dir)

    images_dir = os.path.join(artifact_dir, "images")
    annotations_dir = os.path.join(artifact_dir, "annotations")

    os.mkdir(images_dir)
    os.mkdir(annotations_dir)

    angle_transform = angle_space(stop_angle, step_angle)

    with tempfile.TemporaryDirectory() as output_dir:
        ROOT = pathlib.Path(__file__).resolve().parent

        for dataset in tqdm(datasets, desc="Pre-processing"):
            pyfileloader = importlib_FileLoader(dataset, str(ROOT / f"{dataset}.py"))
            pydataset = pyfileloader.load_module()

            pydataset.preprocess(output_dir)

        examples = []

        for dataset in tqdm(datasets, desc="Augmentation"):
            dataset_dir = os.path.join(output_dir, dataset)

            metadata = os.path.join(dataset_dir, "metadata.csv")
            metadata = pd.read_csv(metadata)

            for idx, row in metadata.iterrows():
                image = row["image"]
                angle = row["angle"]

    data_dir = "."  # FIXME

    for split in tqdm(os.listdir(data_dir)):
        split_dir = os.path.join(data_dir, split)
        split_dir = pathlib.Path(split_dir)

        for name in tqdm(os.listdir(split_dir)):
            path = os.path.join(split_dir, name)

            basename, skew = name.split("[")

            skew, exte = skew.split("]")

            skew = float(skew)
            exte = exte.replace(".", "")

            out_name = f"{basename}.{exte}"
            out_path = os.path.join(images_dir, out_name)

            if out_name in examples:
                continue

            image = Image.open(path)

            array = np.asarray(image)
            array = rotate(array, -skew, cval=1.0)

            array = array * 255
            array = array.astype(np.uint8)

            image = Image.fromarray(array)
            image.save(out_path)

            examples.append(out_name)

    random.seed(seed)
    random.shuffle(examples)

    num_examples = len(examples)

    splits = []

    for split in ["train", "valid", "test"]:
        fullvar = f"{split}_split"
        ratio = locals()[fullvar]

        num_split = int(num_examples * ratio)

        splits += [split] * num_split

    dataframe = pd.DataFrame.from_records(
        zip(examples, splits), columns=["filename", "split"]
    )

    for split in ["train", "valid", "test"]:
        annot = dataframe[dataframe["split"] == split]
        annot = annot.reset_index(drop=True)

        annot_path = os.path.join(annotations_dir, f"{split}.csv")

        annot.to_csv(annot_path, index=False)

    if artifact is not None:
        env = wandb.init()

        artifact = wandb.Artifact(artifact, type="dataset")

        artifact.add_dir(images_dir)
        artifact.add_dir(annotations_dir)

        env.log_artifact(artifact)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
