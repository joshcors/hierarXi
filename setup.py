from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

exts = [
    Extension(
        "reorder",
        ["src/tree/reorder.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="hierarXi",
    version="0.1",
    description="arXiv graph and vector database",
    ext_modules=cythonize(exts, language_level="3"),
    package_dir={"": "src"},
    packages=["data", "train", "tree", "utils"]
)