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
            'csrc/gateup_kernels.cu',
            'csrc/rmsnorm_kernels.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=["torch", "transformers==4.39.2"],
    version=get_version("muillm/__init__.py"))