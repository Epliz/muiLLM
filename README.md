muiLLM (Machine Ultra Instinct LLM) is an experimental Python library for fast inference on AMD MI GPUs.
It replaces the implementation of HuggingFace Transformers models to reach higher performance.

It has currently been tested on:
* AMD MI100 GPUs
* AMD MI300x GPUs

(MI250x GPUs probably work as well, but have not been tested.)

The library being experimental means that there is probably quite some bugs lurking in there, but speed results should be representative.

## Supported models

Currently, only the [Mistral 7b instruct v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model is supported.

## Performance numbers


## Installation

The library has to be installed from source.

Before doing so, Pytorch for ROCM has to be installed first.

### Installing Pytorch

Please refer to the [Pytorch website](https://pytorch.org/get-started/locally/) for this part.

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