"""A script for pre-processing the DISE 2021 dataset"""
import os
import pathlib

import gdown
import pandas as pd
import parse


_artifact = "dise2021_45.zip"
_format = "{image}[{angle:f}].{ext}"
_URL = "https://drive.google.com/uc?id=1a-a6aOqdsghjeHGLnCLsDs7NoJIus-Pw"


def preprocess(output_dir: str) -> str:
    output_dir = pathlib.Path(output_dir).resolve()

    parser = parse.compile(_format)

    src_artifact_dir = output_dir / _artifact.split(".")[0]
    artifact_dir = output_dir / _artifact.split(".")[0].split("_")[0]

    _ = output_dir / _artifact  # FIXME: artifact

    gdown.cached_download(_URL, src_artifact_dir, postprocess=gdown.extractall)

    src_artifact_dir.rename(artifact_dir)

    metadata = []

    for split_dir in artifact_dir.iterdir():
        split_iter = split_dir.glob("*.png")
        split = pd.DataFrame(split_iter, columns=["imagepath"])

        metadata.append(split)

    metadata = pd.concat(metadata)

    def parse_transform(row: pd.Series) -> pd.Series:
        imagename = row.imagepath.split("/")[-1]
        imageinfo = parser.parse(imagename).named

        image = f"{imageinfo['image']}.{imageinfo['ext']}"
        angle = float(imageinfo["angle"])

        absangle = abs(angle)

        transformed = {
            "image": image,
            "imagepath": row.imagepath,
            "angle": angle,
            "absangle": absangle,
        }

        transformed = pd.Series(transformed)

        return transformed

    metadata = metadata.apply(parse_transform, axis=1)
    metadata = metadata.groupby("image").agg({"absangle": min})

    for split_dir in artifact_dir.iterdir():
        for image in split_dir.iterdir():
            imagename = image.stem

            angle = imagename.split("[")[-1]
            angle = angle.split("]")[0]
            angle = float(angle)

            metadata.append((image, angle))

    metadata_artifact = artifact_dir / "metadata.csv"

    metadata = pd.DataFrame.from_records(metadata, columns=["image", "angle"])
    metadata.to_csv(metadata_artifact, index=False)

    return str(artifact_dir)
