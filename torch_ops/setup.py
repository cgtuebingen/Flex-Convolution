from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flexpool_cuda',
    ext_modules=[
        CUDAExtension('flexpool_cuda', [
            'flexpool_cuda.cpp',
            'flexpool_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
