import codecs
import os.path
from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_ninja"] = True
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
    keywords=("llm, ml, AI, Machine Learning, NLP"),
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "muillm_ext",
            [
                "csrc/module.cpp",
                "csrc/linear/linear.cu",
                "csrc/linear/linear_fp16_kernels.cu",
                "csrc/linear/linear_bf16_kernels.cu",
                "csrc/int8_dequantization_kernel.cu",
                "csrc/int8_linear_kernels.cu",
                "csrc/ffn/gateup.cu",
                "csrc/ffn/gateup_fp16_kernels.cu",
                "csrc/ffn/gateup_bf16_kernels.cu",
                "csrc/moeffn/gateupmoe.cu",
                "csrc/moeffn/gateupmoe_fp16_kernels.cu",
                "csrc/moeffn/gateupmoe_bf16_kernels.cu",
                "csrc/moeffn/moelinear.cu",
                "csrc/moeffn/moelinear_fp16_kernels.cu",
                "csrc/moeffn/moelinear_bf16_kernels.cu",
                "csrc/topk/topk.cu",
                "csrc/topk/topk_fp16_kernels.cu",
                "csrc/topk/topk_bf16_kernels.cu",
                "csrc/int8_gateup_kernels.cu",
                "csrc/int8_gateup_dequantization_kernel.cu",
                "csrc/norm/l2norm.cu",
                "csrc/norm/l2norm_fp16_kernels.cu",
                "csrc/norm/l2norm_bf16_kernels.cu",
                "csrc/norm/qkl2norm.cu",
                "csrc/norm/qkl2norm_fp16_kernels.cu",
                "csrc/norm/qkl2norm_bf16_kernels.cu",
                "csrc/norm/rmsnorm.cu",
                "csrc/norm/rmsnorm_fp16_kernels.cu",
                "csrc/norm/rmsnorm_bf16_kernels.cu",
                "csrc/temperaturetuning/temperature_tuning.cu",
                "csrc/temperaturetuning/temperature_tuning_fp16_kernels.cu",
                "csrc/temperaturetuning/temperature_tuning_bf16_kernels.cu",
                "csrc/rope/rotary.cu",
                "csrc/rope/rotary_fp16_kernels.cu",
                "csrc/rope/rotary_bf16_kernels.cu",
                "csrc/kvcaches/static_kvcache.cpp",
                "csrc/kvcaches/static_kvcache_xx16_kernels.cu",
                "csrc/kvcaches/sliding_kvcache.cpp",
                "csrc/kvcaches/sliding_kvcache_xx16_kernels.cu",
                "csrc/attention/causal_transformer_decoding.cu",
                "csrc/attention/half_fused_decoding.cu",
                "csrc/attention/half_fused_decoding_fp16_kernels.cu",
                "csrc/attention/half_fused_decoding_bf16_kernels.cu",
                "csrc/attention/flash_decoding.cu",
                "csrc/attention/flash_decoding_fp16_kernels.cu",
                "csrc/attention/flash_decoding_bf16_kernels.cu",
                "csrc/sync.cu",
                "csrc/sync_torch.cpp",
                "csrc/reduce/reduce.cu",
                "csrc/reduce/reduce_fp16_kernels.cu",
                "csrc/reduce/reduce_bf16_kernels.cu",
                # comms
                "csrc/comm_base.cpp",
                "csrc/comm.cpp",
                "csrc/comm_torch.cpp",
                "csrc/comm_p2p.cu",
                "csrc/comm_staged.cu",
                # parallel
                "csrc/parallel_linear_kernels.cu",
                "csrc/parallel_gateup_kernels.cu",
                "csrc/parallel_gateupmoe_kernels.cu",
                # modules
                "csrc/modules/linear_module.cpp",
                "csrc/modules/kvcache.cpp",
                "csrc/modules/static_kvcache.cpp",
                "csrc/modules/dynamic_kvcache.cpp",
                "csrc/modules/hybrid_chunked_kvcache.cpp",
                "csrc/modules/rotary_module.cpp",
                # parallel modules
                "csrc/modules/parallel_linear_module.cpp",
                "csrc/modules/parallel_multilinear_module.cpp",
                "csrc/modules/parallel_gateup_module_interface.cpp",
                "csrc/modules/parallel_gateup_module.cpp",
                "csrc/modules/parallel_gateupmoe_module.cpp",
                "csrc/modules/parallel_attention_module.cpp",
                "csrc/modules/parallel_llama4_attention_module.cpp",
                "csrc/modules/parallel_decoder_module.cpp",
                "csrc/modules/parallel_llama4_decoder_module.cpp",
                "csrc/modules/parallel_decoder_stack.cpp",
                "csrc/modules/parallel_llama4_decoder_stack.cpp",
                # other
                "csrc/engine.cpp",
                "csrc/gpu_info.cpp",
            ],
            extra_compile_args={
                "cxx": ["-g"],  # Add debug symbols for C++ code
                "hipcc": ["-g"],  # Add debug symbols for HIP code
            },
        )
    ],
    cmdclass={"build_ext": NinjaBuildExtension},
    install_requires=["torch", "transformers==4.52.4", "accelerate"],
    version=get_version("muillm/__init__.py"),
)
