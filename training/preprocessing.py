"""A script for pre-processing a skew detection dataset"""
import argparse
import os
import pathlib
import random
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import rotate
from tqdm.auto import tqdm

import wandb


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(__name__)

    parser.add_argument(
        "--data-dir", help="The local dataset dir", type=str, required=True
    )

    parser.add_argument(
        "--artifact", help="The name of the W&B artifact", type=str, required=True
    )

    parser.add_argument(
        "--artifact-dir", help="The local artifact dir", type=str, required=True
    )

    parser.add_argument(
        "--train-split", help="The train split", type=float, default=0.6
    )

    parser.add_argument(
        "--valid-split", help="The valid split", type=float, default=0.2
    )

    parser.add_argument("--test-split", help="The test split", type=float, default=0.2)
    parser.add_argument("--seed", help="The random seed", type=int, default=10)

    args = parser.parse_args()
    args = vars(args)

    return args


def main(
    data_dir: str,
    artifact: str,
    artifact_dir: str,
    train_split: float = 0.6,
    valid_split: float = 0.2,
    test_split: float = 0.2,
    seed: int = 10,
):
    if not os.path.exists(data_dir):
        message = f"No local dataset dir: {data_dir}"
        raise ValueError(message)

    cumsplit = train_split + valid_split + test_split

    if cumsplit != 1.0:
        message = (
            f"The train, valid and test splits must sum up to 1.0 - "
            f"({train_split}, {valid_split}, {test_split})"
        )

        raise ValueError(message)

    if not os.path.exists(artifact_dir):
        os.mkdir(artifact_dir)

    images_dir = os.path.join(artifact_dir, "images")
    annotations_dir = os.path.join(artifact_dir, "annotations")

    os.mkdir(images_dir)
    os.mkdir(annotations_dir)

    examples = []

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

    env = wandb.init()

    artifact = wandb.Artifact(artifact, type="dataset")

    artifact.add_dir(images_dir)
    artifact.add_dir(annotations_dir)

    env.log_artifact(artifact)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
