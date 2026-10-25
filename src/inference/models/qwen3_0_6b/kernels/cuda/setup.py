"""用当前环境中的 PyTorch/CUDA 工具链构建可选 native op。"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


MODEL_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parent
os.chdir(MODEL_PACKAGE_ROOT.parents[2])
NVIDIA_ROOT = Path(torch.__file__).resolve().parents[1] / "nvidia"
NVIDIA_INCLUDE_DIRS = sorted(str(path) for path in NVIDIA_ROOT.glob("*/include"))
NVIDIA_LIBRARY_DIRS = sorted(
    str(path)
    for package in NVIDIA_ROOT.iterdir()
    if package.is_dir()
    for path in (package / "lib", package / "lib64")
    if path.is_dir()
) if NVIDIA_ROOT.is_dir() else []

setup(
    name="qwen3-cuda-ops",
    ext_modules=[
        CUDAExtension(
            name="inference.models.qwen3_0_6b.kernels.cuda._C",
            sources=[
                str(SOURCE_ROOT / "csrc" / "ops.cpp"),
                str(SOURCE_ROOT / "csrc" / "paged_decode.cu"),
            ],
            include_dirs=NVIDIA_INCLUDE_DIRS,
            library_dirs=NVIDIA_LIBRARY_DIRS,
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
