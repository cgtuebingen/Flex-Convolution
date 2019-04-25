from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra"]
extra_compile_args += ['--expt-relaxed-constexpr']

flags = []

setup(
    name='patchmatch_cuda',
    ext_modules=[
        CUDAExtension(
            'flexpool_cuda',
            sources=[
                'flexpool_cuda.cpp',
                'flexpool_cuda_kernel.cu',
            ],
            extra_compile_args={
                "cxx": flags,
                "nvcc": flags + ["--expt-relaxed-constexpr", "-O2",
                                 "--gpu-architecture=sm_61"],
            },),
    ],
    extra_compile_args=extra_compile_args,
    cmdclass={
        'build_ext': BuildExtension
    })
