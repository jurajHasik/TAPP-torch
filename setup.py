import os
import glob
import subprocess
import sys
import torch

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "tapp_torch"
tapp_cutensor_lib_name = "tapp-cutensor"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


class CMakeBuildExt(BuildExtension):
    # See https://github.com/pypa/setuptools/blob/main/docs/deprecated/distutils/apiref.rst
    user_options = BuildExtension.user_options + [
        ('tapp-force-build', None, "Force (re)building of TAPP"),
        ('tapp-cutensor-bindings=', None, "Build cutensor bindings (requires cuTensor)"),
        ('tapp-tblis', None, "Enable TBLIS backend for CPU"),
        ('tapp-tblis-source-dir=', None, "Path to TBLIS source directory (optional)"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.debug_mode = os.environ.get("DEBUG", "0").lower() in ["1", "on", "true"]
        self.tapp_force_build = os.environ.get("TAPP_FORCE_BUILD", "0").lower() in ["1", "on", "true"]
        self.tapp_cutensor_bindings = (torch.cuda.is_available() and CUDA_HOME is not None) \
            or (os.environ.get("TAPP_CUTENSOR_BINDINGS", "0").lower() in ["1", "on", "true"])
        self.tapp_tblis = os.environ.get("TAPP_REFERENCE_USE_TBLIS", "0").lower() in ["1", "on", "true"]
        self.tapp_tblis_source_dir = os.environ.get("TAPP_REFERENCE_TBLIS_SOURCE_DIR", "../../tblis") # default submodule location

    def finalize_options(self):
        super().finalize_options()
        # Under pip install, no command line options are passed. Values from initialize_options are used
        # Command line options (if any) override environment variables
        if (isinstance(self.debug_mode, int) and self.debug_mode in [1,]) or \
            (isinstance(self.debug_mode, str) and self.debug_mode.lower() in ["1", "on", "true"]):
            self.debug_mode = True
        if (isinstance(self.tapp_force_build, int) and self.tapp_force_build in [1,]) \
            or (isinstance(self.tapp_force_build, str) and self.tapp_force_build.lower() in ["1", "on", "true"]):
            self.tapp_force_build = True
        if (isinstance(self.tapp_cutensor_bindings, int) and self.tapp_cutensor_bindings in [1,]) \
            or (isinstance(self.tapp_cutensor_bindings, str) and self.tapp_cutensor_bindings.lower() in ["1", "on", "true"]):
            self.tapp_cutensor_bindings = True
        elif isinstance(self.tapp_cutensor_bindings, str) and self.tapp_cutensor_bindings.lower() in ["0", "off", "false"]:
            self.tapp_cutensor_bindings = False
        if (isinstance(self.tapp_tblis, int) and self.tapp_tblis in [1,]) \
            or (isinstance(self.tapp_tblis, str) and self.tapp_tblis.lower() in ["1", "on", "true"]):
            self.tapp_tblis = True
        if isinstance(self.tapp_tblis, str):
            self.tapp_tblis_source_dir = self.tapp_tblis


    def run(self):
        # Default paths for tapp build dir
        tapp_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "third_party", "tapp"))
        tapp_build = os.path.join(tapp_src, "build")
        tapp_lib_cutensor = os.path.join(tapp_build, "cutensor_bindings", f"{tapp_cutensor_lib_name}.so")
        tapp_lib_default = os.path.join(tapp_build, "libtapp-reference.so")

        # get environment variable to decide which tapp lib to link against
        build_cutensor_bindings = "ON" if self.tapp_cutensor_bindings else "OFF"
        use_tblis = "ON" if self.tapp_tblis else "OFF"
        tapp_tblis_source_dir= self.tapp_tblis_source_dir

        print(f"TAPP_torch tapp_force_build {self.tapp_force_build}")
        print(f"TAPP_torch tapp_cutensor_bindings {self.tapp_cutensor_bindings}")
        print(f"TAPP_torch tapp_tblis {self.tapp_tblis}")
        print(f"TAPP_torch tapp_tblis_source_dir {self.tapp_tblis_source_dir}")
        print(f"TAPP_torch {tapp_lib_default} found {os.path.exists(tapp_lib_default)}")
        print(f"TAPP_torch {tapp_lib_cutensor} found {os.path.exists(tapp_lib_cutensor)}")
        
        # Only build if missing or forced
        if (not os.path.exists(tapp_lib_default) or \
            (build_cutensor_bindings in ["ON",] and not os.path.exists(tapp_lib_cutensor)) \
            or self.tapp_force_build):
            print(f"Building TAPP in {tapp_build} ...")
            os.makedirs(tapp_build, exist_ok=True)
            # Configure
            subprocess.check_call([
                "cmake",
                "-DCMAKE_BUILD_TYPE="+("Debug" if self.debug_mode else "Release"),
                f"-DTAPP_CUTENSOR={build_cutensor_bindings}",
                f"-DTAPP_REFERENCE_USE_TBLIS={use_tblis}",
                f"-DTAPP_REFERENCE_TBLIS_SOURCE_DIR={tapp_tblis_source_dir}",
                ".."
            ], cwd=tapp_build)
            # Build
            subprocess.check_call([
                "cmake", "--build", ".", "--parallel"
            ], cwd=tapp_build)
        else:
            print(f"TAPP already built: {tapp_build}")

        # Now continue with normal build_ext
        super().run()

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
                        f"-l{tapp_cutensor_lib_name}",
                    ],
                    py_limited_api=py_limited_api,
                )
            )

    return ext_modules


setup(
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": CMakeBuildExt}, #BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)