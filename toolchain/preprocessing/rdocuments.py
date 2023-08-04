"""A script for pre-processing the Rdocuments dataset"""
import pathlib
from functools import partial

import kaggle
import pandas as pd
import parse


_artifact = "vishnunkumar/rdocuments"
_format = "Image-{:d}-{angle:d}{image}.{ext}"


def _transform(
    series: pd.Series, parser: parse.Parser, images_dir: pathlib.Path
) -> pd.Series:
    """A transform that parses metadata to a standard format"""
    imagename = series.id
    imageinfo = parser.parse(imagename).named

    imagepath = images_dir / imagename
    imagepath = str(imagepath)

    image = f"{imageinfo['image']}.{imageinfo['ext']}"
    angle = float(imageinfo["angle"])

    message = (
        f"The angle in {imagename} is not consistent. "
        f"Expected {series.angle}, got {angle}"
    )

    assert angle == series.angle, message

    absangle = abs(series.angle)

    transformed = {
        "image": image,
        "imagepath": imagepath,
        "angle": angle,
        "absangle": absangle,
    }

    transformed = pd.Series(transformed)

    return transformed


def preprocess(output_dir: str) -> str:
    output_dir = pathlib.Path(output_dir).resolve()

    artifact_dir = output_dir / "rdocuments"

    if not artifact_dir.exists():
        artifact_dir.mkdir()

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(_artifact, path=artifact_dir, unzip=True)

    metadata = pd.read_csv(artifact_dir / "r-images.csv")

    images_dir = artifact_dir / "rdocuments"

    parse_transform = partial(
        _transform, parser=parse.compile(_format), images_dir=images_dir
    )

    metadata = metadata.apply(parse_transform, axis=1)

    min_absangle_mask = metadata.groupby("image")["absangle"].transform("min")
    min_absangle_mask = metadata["absangle"] == min_absangle_mask

    metadata = metadata[min_absangle_mask]
    metadata = metadata.sort_values("absangle").drop_duplicates("image")
    metadata = metadata.drop("absangle", axis=1)

    metadata_artifact = artifact_dir / "metadata.csv"

    metadata.to_csv(metadata_artifact, index=False)

    return str(artifact_dir)
