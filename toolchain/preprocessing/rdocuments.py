"""A script for pre-processing the Rdocuments dataset"""
import pathlib

import kaggle
import pandas as pd
import parse


_artifact = "vishnunkumar/rdocuments"
_format = "Image-{:d}-{angle:d}{image}.{ext}"


def preprocess(output_dir: str) -> str:
    output_dir = pathlib.Path(output_dir).resolve()

    _ = parse.compile(_format)  # FIXME: parser

    artifact_dir = output_dir / "rdocuments"

    kaggle.api.authenticate()

    if not artifact_dir.exists():
        artifact_dir.mkdir()

    kaggle.api.dataset_download_files(_artifact, path=artifact_dir, unzip=True)

    metadata = pd.read_csv(artifact_dir / "r-images.csv")

    images_dir = artifact_dir / "rdocuments"

    def join_transform(filename: str) -> str:
        filepath = images_dir / filename
        filepath = str(filepath)

        return filepath

    metadata = metadata.rename(columns={"id": "image"})

    metadata_artifact = artifact_dir / "metadata.csv"

    metadata["image"] = metadata["image"].apply(join_transform)

    metadata.to_csv(metadata_artifact, index=False)

    return str(artifact_dir)