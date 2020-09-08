from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='stn_cuda',
    ext_modules=[
        CUDAExtension('stn_cuda', [
            'stn_cuda.cpp',
            'stn_cuda_kernel.cu',
        ],
            libraries=[
                'glog'
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-G']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
