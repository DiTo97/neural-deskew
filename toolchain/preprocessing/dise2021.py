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
    artifact_dir = output_dir / _artifact.split(".")[0].split("_")[0]

    if not artifact_dir.exists():
        src_artifact = str(output_dir / _artifact)
        src_artifact_dir = output_dir / _artifact.split(".")[0]

        gdown.cached_download(_URL, src_artifact, postprocess=gdown.extractall)

        src_artifact_dir.rename(artifact_dir)

    metadata = []

    for split_dir in artifact_dir.iterdir():
        split_iter = split_dir.glob("*.png")
        split = pd.DataFrame(split_iter, columns=["imagepath"])

        metadata.append(split)

    metadata = pd.concat(metadata)

    parser = parse.compile(_format)

    def parse_transform(series: pd.Series) -> pd.Series:
        imagename = series.imagepath.name
        imageinfo = parser.parse(imagename).named

        image = f"{imageinfo['image']}.{imageinfo['ext']}"
        angle = float(imageinfo["angle"])

        absangle = abs(angle)

        transformed = {
            "image": image,
            "imagepath": series.imagepath,
            "angle": angle,
            "absangle": absangle,
        }

        transformed = pd.Series(transformed)

        return transformed

    metadata = metadata.apply(parse_transform, axis=1)

    min_absangle_mask = metadata.groupby("image")["absangle"].transform("min")
    min_absangle_mask = metadata["absangle"] == min_absangle_mask

    metadata = metadata[min_absangle_mask]
    metadata = metadata.sort_values("absangle").drop_duplicates("image")
    metadata = metadata.drop("absangle", axis=1)

    metadata_artifact = artifact_dir / "metadata.csv"

    metadata = pd.DataFrame.from_records(metadata, columns=["image", "imagepath", "angle"])
    metadata.to_csv(metadata_artifact, index=False)

    return str(artifact_dir)
