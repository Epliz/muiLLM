#ifndef __MUILLM_GENERATION_KERNELS_H__
#define __MUILLM_GENERATION_KERNELS_H__

#include <torch/extension.h>

#include <tuple>

// returns unfinished_sequences and input_ids
std::tuple<torch::Tensor, torch::Tensor> muillm_add_tokens_check_finished(
    torch::Tensor input_ids,
    torch::Tensor next_tokens,
    torch::Tensor unfinished_sequences,
    bool has_eos_stopping_criteria,
    int pad_token_id,
    torch::Tensor eos_token_ids,
    bool has_max_length_stopping_criteria,
    int max_length
);

#endif /* __MUILLM_GENERATION_KERNELS_H__ */