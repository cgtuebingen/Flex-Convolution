
from torch.utils.cpp_extension import load
flexpool_cuda = load(
    'flexpool_cuda', ['flexpool_cuda.cpp', 'flexpool_cuda_kernel.cu'], verbose=True)
help(flexpool_cuda)
