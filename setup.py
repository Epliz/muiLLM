import codecs
import os.path
from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['use_ninja'] = True
        super().__init__(*args, **kwargs)

setup(
    name="muillm",
    description="Library for fast LLM training/inferenceon AMD GPUs",
    author="The muiLLM team",
    license="Apache 2.0 License",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=(
        "llm, ml, AI, Machine Learning, NLP"
    ),
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('muillm_ext', [
            'csrc/module.cpp',
            'csrc/linear_kernels.cu',
            'csrc/int8_dequantization_kernel.cu',
            'csrc/int8_linear_kernels.cu',
            'csrc/gateup_kernels.cu',
            'csrc/int8_gateup_kernels.cu',
            'csrc/int8_gateup_dequantization_kernel.cu',
            'csrc/rmsnorm_kernels.cu',
            'csrc/rotary_kernels.cu',
            'csrc/causal_transformer_decoding.cu',
            'csrc/sync.cu',
            'csrc/sync_torch.cpp',
            # comms
            'csrc/comm_base.cpp',
            'csrc/comm.cpp',
            'csrc/comm_torch.cpp',
            'csrc/comm_p2p.cu',
            'csrc/comm_staged.cu',
            # parallel
            'csrc/parallel_linear_kernels.cu',
            'csrc/parallel_gateup_kernels.cu',

        ],
        extra_compile_args={
            'cxx': ['-g'],  # Add debug symbols for C++ code
            'hipcc': ['-g']  # Add debug symbols for HIP code
        }
    )
    ],
    cmdclass={
        'build_ext': NinjaBuildExtension
    },
    install_requires=["torch", "transformers==4.45.2"],
    version=get_version("muillm/__init__.py"))