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

    tapp_lib_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "third_party", "tapp", "build",
    )
    tapp_include_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "third_party", "tapp", "api", "include",
    )
    tapp_cutensor_lib_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "third_party", "tapp", "build", "cutensor_bindings",
    )
    tapp_cutensor_include_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "third_party", "tapp", "cutensor_bindings",
    )

    extra_link_args = []
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
        [ extra_compile_args[c].append("-g") for c in ["cxx", "nvcc"] ]
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    # Convert absolute paths to relative (required for editable installs)
    sources = [os.path.relpath(s, this_dir) for s in sources]
    cuda_sources = [os.path.relpath(s, this_dir) for s in cuda_sources]

    ext_modules = []

    # CPU extension — links against libtapp-reference only
    ext_modules.append(
        CppExtension(
            f"{library_name}._C",
            sources,
            include_dirs=[tapp_include_dir],
            extra_compile_args=extra_compile_args["cxx"],
            extra_link_args=extra_link_args+[
                f"-L{tapp_lib_dir}",
                f"-Wl,-rpath,{tapp_lib_dir}",
                "-ltapp-reference",
            ],
            py_limited_api=py_limited_api,
        )
    )

    # CUDA extension — links against libcutensor_binds only (NOT libtapp-reference)
    if use_cuda:
        if cuda_sources:
            ext_modules.append(
                CUDAExtension(
                    f"{library_name}._C_cuda",
                    cuda_sources,
                    include_dirs=[tapp_include_dir, tapp_cutensor_include_dir],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args+[
                        f"-L{tapp_cutensor_lib_dir}",
                        f"-Wl,-rpath,{tapp_cutensor_lib_dir}",
                        "-lcutensor_bindings",
                    ],
                    py_limited_api=py_limited_api,
                )
            )

    return ext_modules


setup(
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)