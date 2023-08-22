#ifndef VSATTN_KERNEL
#define VSATTN_KERNEL

#include <torch/extension.h>
#include <vector>

// #define T_CHECK(var) \
// TORCH_CHECK(var.device().type() == torch::kCUDA, #var" must be a CUDA tensor"); 
// TORCH_CHECK(var.is_contiguous(), #var" must to be contiguous"); 


torch::Tensor VSAttn_kernel_forward(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &sampling_matrix,
    const int ws,
    const double attn_scale
);


std::vector<torch::Tensor> VSAttn_kernel_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor sampling_matrix,
    torch::Tensor grad_output,
    const int ws,
    const double attn_scale
);



#endif