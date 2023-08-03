"""A script for pre-processing a document image skew estimation collection"""
import argparse
import os
import pathlib
import random
import sys
import tempfile
import typing
from typing import Any

import albumentations
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import rotate
from tqdm.auto import tqdm

import wandb
from neural_deskew.core.space import angle_space


if typing.TYPE_CHECKING:
    import types


Transform = object


def srcfile_import(modpath: str, modname: str) -> "types.ModuleType":
    """It imports a Python module from its srcfile

    Parameters
    ----------
    modpath
        The srcfile absolute path
    modname
        The module name in the scope

    Returns
    -------
        The imported module

    Raises
    ------
    ImportError
        If the module cannot be imported from the srcfile
    """
    import importlib.util

    #
    spec = importlib.util.spec_from_file_location(modname, modpath)

    if spec is None:
        message = f"No spec for module at {modpath}"
        raise ImportError(message)

    module = importlib.util.module_from_spec(spec)

    # It adds the module to the global scope
    sys.modules[modname] = module

    if spec.loader is None:
        message = f"No spec loader for module at {modpath}"
        raise ImportError(message)

    spec.loader.exec_module(module)

    return module


def image_transform_from_config(config: str | None) -> Transform | None:
    """A serialized image augmentation pipeline from config file"""
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
    np.random.seed(seed)

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

    examples = []

    with tempfile.TemporaryDirectory() as output_dir:
        ROOT = pathlib.Path(__file__).resolve().parent

        for dataset in tqdm(datasets, desc="Pre-processing"):
            modname = dataset
            modpath = str(ROOT / f"{dataset}.py")

            pydataset = srcfile_import(modpath, modname)
            pydataset.preprocess(output_dir)

        for dataset in tqdm(datasets, desc="Augmentation"):
            dataset_dir = os.path.join(output_dir, dataset)

            metadata = os.path.join(dataset_dir, "metadata.csv")
            metadata = pd.read_csv(metadata)

            metadata = metadata.sample(frac=1.0)

            splits = []

            for split in ["train", "valid", "test"]:
                fullvar = f"{split}_split"
                ratio = locals()[fullvar]

                num_split = int(len(metadata) * ratio)

                splits += [split] * num_split

            metadata["split"] = splits

            for series in metadata.itertuples():
                image_name, image_exte = series.image.rspit(".", 1)

                image = Image.open(series.imagepath)

                array = np.asarray(image)
                array = rotate(array, -series.angle, cval=1.0)

                array = array * 255
                array = array.astype(np.uint8)

                out_image_name = f"{image_name}.{image_exte}"
                out_image_path = os.path.join(images_dir, out_image_name)

                image = Image.fromarray(array)
                image.save(out_image_path)

                input = (out_image_name, series.split, dataset)
                examples.append(input)

                transform = np.random.choice(
                    angle_transform, num_angle_transforms, replace=False
                )

                for idx, angle in enumerate(transform):
                    rotated = rotate(array, angle, cval=1.0)

                    rotated = rotated * 255
                    rotated = rotated.astype(np.uint8)

                    key = f"{idx:03d}_000"

                    out_image_name = f"{image_name}_{key}.{image_exte}"
                    out_image_path = os.path.join(images_dir, out_image_name)

                    rotated_image = Image.fromarray(rotated)
                    rotated_image.save(out_image_path)

                    input = (out_image_name, series.split, dataset)
                    examples.append(input)

                    for jdx in range(num_image_transforms):
                        if jdx == 0:
                            continue

                        transformed = image_transform(image=rotated)["image"]

                        transformed = transformed * 255
                        transformed = transformed.astype(np.uint8)

                        key = f"{idx:03d}_{jdx:03d}"

                        out_image_name = f"{image_name}_{key}.{image_exte}"
                        out_image_path = os.path.join(images_dir, out_image_name)

                        transformed_image = Image.fromarray(transformed)
                        transformed_image.save(out_image_path)

                        input = (out_image_name, series.split, dataset)
                        examples.append(input)

    dataframe = pd.DataFrame.from_records(
        examples, columns=["filename", "split", "dataset"]
    )

    for split in ["train", "valid", "test"]:
        annot = dataframe[dataframe["split"] == split]
        annot = annot.reset_index(drop=True)

        annot_path = os.path.join(annotations_dir, f"{split}.csv")

        annot.to_csv(annot_path, index=False)

    if artifact is not None:
        env = wandb.init(project="neural-deskew")

        if image_transform is not None:
            image_transform = albumentations.to_dict(image_transform)

        artifact_kwargs = {
            "description": "A document image skew estimation collection",
            "metadata": {
                "datasets": tuple(datasets),
                "stop-angle": stop_angle,
                "step-angle": step_angle,
                "image-transform": image_transform,
                "num-image-transforms": num_image_transforms,
                "num-angle-transforms": num_angle_transforms,
                "train-split": train_split,
                "valid-split": valid_split,
                "test-split": test_split,
            },
            "type": "dataset",
        }

        artifact = wandb.Artifact(artifact, **artifact_kwargs)

        artifact.add_dir(images_dir, "images")
        artifact.add_dir(annotations_dir, "annotations")

        env.log_artifact(artifact)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
