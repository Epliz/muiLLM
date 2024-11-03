muiLLM (Machine Ultra Instinct LLM) is an experimental Python library for fast inference on AMD MI GPUs.
We are on a journey to reach 1000+ tokens/s for inference at batch size 1 on MI300x (on Mistral 7b).

It works by replacing the implementation of HuggingFace Transformers layers to reach higher performance.

It has currently been tested on:
* AMD MI100 GPUs
* AMD MI300x GPUs

(MI250x GPUs probably work as well, but have not been tested.)

The library being experimental means that there is probably quite some bugs lurking in there, but speed results should be representative.

## Supported models

Currently, only the [Mistral 7b instruct v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model is supported.

## Optimizations

The following optimizations are already implemented:
* optimized linear layers with faster GEMV
* fused QKV
* fused MLP (Gate/Up + SiLU)
* fused residuals in linear layers
* fused RMSNorm in linear layers
* fused RMSNorm with write out in dynamic cache
* experimental support for int8 RTN
* basic semi-fused attention
* reduced CPU/GPU synchronizations due to attention mask checks
* reduced CPU/GPU synchronizations during sampling

Future optimizations (by order of likely implementation):
* further improvements to linear/fused MLP to reach higher memory bandwidth
* use Python less to be less CPU limited
* tensor parallelism
* better attention implementation (flash decoding)
* static cache support
* layer interleaving

## Performance numbers

The numbers are changing at every commit, try it out by yourself!

But if you can't, here is the approximate performance on a small prompt, generating 50 tokens:

* fp16 on MI300x: 17ms token-prefill, 354ms generation (146 tokens/s)
* int8 RTN on MI300x: 19ms token-prefill, ~250ms generation (~200 tokens/s)


## Installation

The library has to be installed from source.

Before doing so, Pytorch for ROCM has to be installed first.

Please refer to the [Pytorch website](https://pytorch.org/get-started/locally/) for how to install pytorch for ROCm.

To make the building process faster, make sure you have ninja installed as well:

```shell
pip install ninja
```

### Installing from source

First clone the repository:

```shell
git clone https://github.com/Epliz/muiLLM.git
```

go to the directory of the cloned repository:

```shell
cd muiLLM
```

And install the library (creating a virtual environment beforehand is recommended):

```shell
pip install --upgrade build
pip install wheel

python setup.py bdist_wheel && pip install ./dist/muillm-0.0.1-cp310-cp310-linux_x86_64.whl
```

Then you can run one of the examples

## Examples

Please find at [examples/mistral7b.py](examples/mistral7b.py) an example of how to use muiLLM on the HuggingFace Transformers Mistral 7b model.
