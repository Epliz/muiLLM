from typing import Tuple
import torch
import torch.nn as nn

from muillm.torch.dtensor import to_local_tensor


def _param_type(parameter_name: str) -> str:
    _, param_type = (
        parameter_name.rsplit(".", 1)
        if "." in parameter_name
        else (parameter_name, None)
    )
    return param_type


# Generic conversion function for tensor parallel plans that use DTensors
def _convert_from_dtensors(param: nn.Parameter, param_name: str) -> nn.Parameter:
    requires_grad = param.requires_grad
    return nn.Parameter(to_local_tensor(param), requires_grad=requires_grad)


# All conversion functions for the different tensor parallel plans
def _convert_colwise_parallel(param: nn.Parameter, param_name: str) -> nn.Parameter:
    # uses DTensors for column-wise tensor parallelism
    return _convert_from_dtensors(param, param_name)


def _convert_rowwise_parallel(param: nn.Parameter, param_name: str) -> nn.Parameter:
    # uses DTensors for row-wise tensor parallelism
    return _convert_from_dtensors(param, param_name)


def _convert_colwise_rep_parallel(param: nn.Parameter, param_name: str) -> nn.Parameter:
    # uses DTensors for column-wise replicated tensor parallelism
    return _convert_from_dtensors(param, param_name)


def _convert_rowwise_rep_parallel(param: nn.Parameter, param_name: str) -> nn.Parameter:
    # uses DTensors for row-wise replicated tensor parallelism
    return _convert_from_dtensors(param, param_name)


def _convert_local_colwise_parallel(
    param: nn.Parameter, param_name: str
) -> nn.Parameter:
    # column-wise tensor parallelism for HF means the output has sharded columns
    # which means the transformation parameters are sharded row-wise in muiLLM terms

    world_size = torch.distributed.get_world_size()

    requires_grad = param.requires_grad

    all_tensors = [torch.empty_like(param) for _ in range(world_size)]
    torch.distributed.all_gather(all_tensors, param)

    is_bias = _param_type(param_name) == "bias"
    if is_bias:
        # bias is sharded by columns, need to be concatenated
        gathered_tensor = torch.cat(
            all_tensors,
            dim=-1,
        )

    else:
        # weights are sharded by rows, se we need to concatenate them along dim -2
        gathered_tensor = torch.cat(
            all_tensors,
            dim=-2,
        )

    return nn.Parameter(gathered_tensor, requires_grad=requires_grad)


def _convert_local_rowwise_parallel(
    param: nn.Parameter, param_name: str
) -> nn.Parameter:
    # row-wise tensor parallelism for HF means that the transformation parameters are sharded column-wise

    world_size = torch.distributed.get_world_size()

    requires_grad = param.requires_grad

    is_bias = _param_type(param_name) == "bias"
    if is_bias:
        # bias is replicated across all ranks
        gathered_tensor = param.data
    else:
        # weights are sharded by columns, so we need to concatenate them along dim -1
        all_tensors = [torch.empty_like(param) for _ in range(world_size)]
        torch.distributed.all_gather(all_tensors, param.data)

        gathered_tensor = torch.cat(
            all_tensors,
            dim=-1,
        )

    return nn.Parameter(gathered_tensor, requires_grad=requires_grad)


def _convert_isolated_parallel(module: nn.Parameter, param_name: str) -> nn.Parameter:
    # parameters are all local, so no need to convert them
    return module


def _convert_gather_parallel(module: nn.Module, param_name: str) -> nn.Module:
    raise NotImplementedError(
        "Gather tensor parallelism conversion is not implemented yet."
    )


def _convert_local_packed_rowwise_parallel(
    param: nn.Parameter, param_name: str
) -> nn.Parameter:
    # row-wise tensor parallelism for HF means that the transformation parameters are sharded column-wise

    world_size = torch.distributed.get_world_size()

    requires_grad = param.requires_grad

    is_bias = _param_type(param_name) == "bias"
    if is_bias:
        # bias is replicated across all ranks
        gathered_tensor = param.data
    else:
        # weights are packed, so we need to insert a dimension of 2
        shape = list(param.shape)
        last_dim = shape[-1]
        unpacked_shape = shape[:-1] + [2, last_dim // 2]

        unpacked_tensor = param.view(*unpacked_shape)

        all_tensors = [torch.empty_like(unpacked_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(all_tensors, unpacked_tensor)

        # weights are sharded by columns, so we need to concatenate them along dim -1
        unpacked_gathered_tensor = torch.cat(
            all_tensors,
            dim=-1,
        )

        unpacked_gathered_shape = list(unpacked_gathered_tensor.shape)
        unpacked_gathered_shape[-2] = 1
        unpacked_gathered_shape[-1] *= 2  # merge the last two

        gathered_tensor = unpacked_gathered_tensor.view(
            *unpacked_gathered_shape,
        ).squeeze(-2)

    return nn.Parameter(gathered_tensor, requires_grad=requires_grad)


def _convert_sequence_parallel(module: nn.Parameter, param_name: str) -> nn.Parameter:
    # uses DTensors for sequence parallelism
    return _convert_from_dtensors(module, param_name)


def _convert_replicate_parallel(module: nn.Parameter, param_name: str) -> nn.Parameter:
    # uses DTensors for replicated tensor parallelism
    return _convert_from_dtensors(module, param_name)


def _convert_to_local_parameter(
    tp_plan, param: nn.Parameter, param_name: str
) -> nn.Parameter:

    # For each of the possible tensor parallel plans, we will convert the module to a local module
    if tp_plan == "colwise":
        return _convert_colwise_parallel(param, param_name)
    elif tp_plan == "rowwise":
        return _convert_rowwise_parallel(param, param_name)
    elif tp_plan == "colwise_rep":
        return _convert_colwise_rep_parallel(param, param_name)
    elif tp_plan == "rowwise_rep":
        return _convert_rowwise_rep_parallel(param, param_name)
    elif tp_plan == "local_colwise":
        return _convert_local_colwise_parallel(param, param_name)
    elif tp_plan == "local_rowwise":
        return _convert_local_rowwise_parallel(param, param_name)
    elif tp_plan == "local":
        return _convert_isolated_parallel(param, param_name)
    elif tp_plan == "gather":
        return _convert_gather_parallel(param, param_name)
    elif tp_plan == "local_packed_rowwise":
        return _convert_local_packed_rowwise_parallel(param, param_name)
    elif tp_plan == "sequence_parallel":
        return _convert_sequence_parallel(param, param_name)
    elif tp_plan == "replicate":
        return _convert_replicate_parallel(param, param_name)
    else:
        # If the tp_plan is not supported, raise an error
        raise ValueError(
            f"Module {param} has a tp_plan: {tp_plan}, which is not supported yet."
        )


def _to_local_module(model: nn.Module, module: nn.Module, name: str = "") -> nn.Module:
    tp_plan = None
    if hasattr(model, "_tp_plan"):
        tp_plan = model._tp_plan

    if tp_plan is None:
        # if the model does not have a tp_plan, we assume it is a local module
        return module

    # We have a tp_plan, so we need to convert the module to a local module
    from transformers.integrations.tensor_parallel import (
        _get_parameter_tp_plan,
    )

    # convert all paramters of the sub-modules of this module
    all_params = [
        (name, param) for name, param in module.named_parameters(recurse=True)
    ]
    for param_name, param in all_params:
        full_param_name = name + "." + param_name if name != "" else param_name

        module_name, param_type = (
            full_param_name.rsplit(".", 1)
            if "." in full_param_name
            else (full_param_name, None)
        )
        param_module = model.get_submodule(module_name)

        if not hasattr(param_module, "_hf_tp_plan"):
            # check if the module was actually partitioned
            continue

        # get the module tp plan
        param_tp_plan = _get_parameter_tp_plan(full_param_name, tp_plan)

        if param_tp_plan is None:
            # if the module does not have a tp_plan, we assume it is replicated
            param_tp_plan = "replicate"

        # Convert this parameter
        converted_param = _convert_to_local_parameter(
            param_tp_plan, param, full_param_name
        )

        setattr(param_module, param_type, converted_param)

    return module
