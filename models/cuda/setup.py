"""Build the custom PIVOT CUDA extension."""

import os
from pathlib import Path

from setuptools import setup

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)


ROOT = Path(__file__).resolve().parent

if os.name == "nt":
    cxx_flags = ["/O2"]
else:
    cxx_flags = ["-O3"]


setup(
    name="pivot_cuda",

    ext_modules=[
        CUDAExtension(
            name="pivot_cuda",
            sources=[
                str(ROOT / "bindings.cpp"),
                str(ROOT / "spatial_relations.cu"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ],
            },
        )
    ],

    cmdclass={
        "build_ext": BuildExtension
    },
)