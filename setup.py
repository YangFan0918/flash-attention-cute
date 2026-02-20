import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="flash_attention_cute",
    packages=["flash_attention"],
    ext_modules=[
        CUDAExtension(
            name="flash_attention._C",
            sources=[
                "csrc/flash_api.cpp",
                "csrc/flash_fwd_launch.cu",
            ],
            include_dirs=[
                os.path.join(this_dir, "3rd/cutlass/include"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++20", "-O3"],
                "nvcc": [
                    "-std=c++20",
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-gencode", "arch=compute_86,code=sm_86",
                    "-DCUTE_SM80_ENABLED",
                ],
            },
            libraries=["cudart"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
