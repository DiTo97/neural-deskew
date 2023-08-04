import pathlib

from setuptools import find_packages, setup


ROOT = pathlib.Path(__file__).resolve().parent


if __name__ == "__main__":
    setup(
        name="neural-deskew",
        version="0.0.1",
        url="https://github.com/DiTo97/neural-deskew",
        author="Federico Minutoli",
        author_email="fede97.minutoli@gmail.com",
        description="A lightweight neural network for document image skew estimation",
        long_description=(ROOT / "README.md").read_text(),
        long_description_content_type="text/markdown",
        python_requires=">=3.10",
        packages=find_packages(exclude=["toolchain"]),
        install_requires=(ROOT / "requirements.txt").read_text().splitlines(),
    )
