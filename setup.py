
import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "tapp_torch"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    # Path to the tapp library built in third_party/tapp/build
    tapp_lib_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "third_party", "tapp", "build",
    )

    extra_link_args = [
        f"-L{tapp_lib_dir}",
        f"-Wl,-rpath,{tapp_lib_dir}",
        "-ltapp-reference",
        # -Lpath/to/shared/objects if necessary
    ]
    include_dirs = [
        "third_party/tapp/api/include",
        # path/to/include if necessary
    ]
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
            # define TORCH_TARGET_VERSION with min version 2.10 to expose only the
            # stable API subset from torch
            # Format: [MAJ 1 byte][MIN 1 byte][PATCH 1 byte][ABI TAG 5 bytes]
            # 2.10.0 = 0x020A000000000000
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            # NVCC also needs TORCH_TARGET_VERSION for stable ABI in CUDA code
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
            # USE_CUDA is currently needed for aoti_torch_get_current_cuda_stream
            # declaration in shim.h. This will be improved in a future release.
            "-DUSE_CUDA",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch>=2.10.0"],
    description="TAPP extension for PyTorch using Stable ABI",
    long_description=open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    ).read(),
    long_description_content_type="text/markdown",
    url="",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)

# git submodule update --init --recursive
#
# mkdir build && cd build
#
# Usage
# You can now configure the build in two ways:

# 1. Fetch from GitHub (default behavior):

# 2. Use a custom local path:

# 3. Use a git submodule in third_party:

# This approach follows CMake best practices by:

# Using a CACHE PATH variable for the custom path
# Validating that the provided path contains a CMakeLists.txt
# Providing informative status/error messages
# Falling back to FetchContent when no custom path is provided