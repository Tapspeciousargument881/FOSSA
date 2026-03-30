from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cur_path = os.getcwd()

setup(
    name='power_exp_psf_cuda',
    ext_modules=[
        CUDAExtension('power_exp_psf_cuda', [
            f'{cur_path}/power_exp_psf_cuda.cpp',
            f'{cur_path}/power_exp_psf_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })