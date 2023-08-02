"""A script for pre-processing the DISE 2021 dataset"""
import pathlib

import gdown
import pandas as pd


_artifact = "dise2021_45.zip"
_URL = "https://drive.google.com/uc?id=1a-a6aOqdsghjeHGLnCLsDs7NoJIus-Pw"


def preprocess(output_dir: str) -> str:
    output_dir = pathlib.Path(output_dir).resolve()

    artifact_dir = output_dir / _artifact.split(".")[0]
    artifact = output_dir / _artifact

    gdown.cached_download(_URL, artifact, postprocess=gdown.extractall)

    metadata_artifact = artifact_dir / "metadata.csv"

    if metadata_artifact.exists():
        return str(artifact_dir)

    metadata = []

    for split_dir in artifact_dir.iterdir():
        for image in split_dir.iterdir():
            imagename = image.stem

            angle = imagename.split("[")[-1]
            angle = angle.split("]")[0]
            angle = float(angle)

            metadata.append((image, angle))

    metadata = pd.DataFrame.from_records(metadata, columns=["image", "angle"])
    metadata.to_csv(metadata_artifact, index=False)

    return str(artifact_dir)
