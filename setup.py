from setuptools import setup

setup(
    name="hierarXi",
    version="0.1",
    description="arXiv graph and vector database",
    package_dir={"": "src"},
    packages=["data", "train", "tree", "utils"]
)