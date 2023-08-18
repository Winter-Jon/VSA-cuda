#ifndef VSATTN
#define VSATTN

#include "VSAttn_kernel.cuh"

#define T_CHECK(var) \
TORCH_CHECK(var.device().type() == torch::kCUDA, #var" must be a CUDA tensor"); \
TORCH_CHECK(var.is_contiguous(), #var" must to be contiguous"); 

torch::Tensor VSAttn_forward(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &sampling_matrix,
    const int ws,
    const float attn_scale
);

#endif