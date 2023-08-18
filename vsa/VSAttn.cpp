// #include <torch/extension.h>
#include "VSAttn_kernel.cuh"
#include "VSAttn.h"

torch::Tensor VSAttn_forward(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &sampling_matrix,
    const int ws,
    const float attn_scale
)
{
    // auto a = q.is_contiguous();
    // T_CHECK(q);T_CHECK(k);T_CHECK(v);T_CHECK(sampling_matrix);

    return VSAttn_kernel_forward(q,k,v,sampling_matrix,ws,attn_scale);
}

std::vector<torch::Tensor> VSAttn_backward(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &sampling_matrix,
    torch::Tensor &grad_output,
    const int ws,
    const float attn_scale
)
{
    // T_CHECK(q);T_CHECK(k);T_CHECK(v);T_CHECK(sampling_matrix);T_CHECK(grad_output);
    return VSAttn_kernel_backward(q,k,v,sampling_matrix,grad_output,ws,attn_scale);
}

// std::vector<at::Tensor>
// VSA_attn_backward(
//     const at::Tensor &value, 
//     const at::Tensor &spatial_shapes,
//     const at::Tensor &level_start_index,
//     const at::Tensor &sampling_loc,
//     const at::Tensor &attn_weight,
//     const at::Tensor &grad_output,
//     const int im2col_step)
// {
//     if (value.type().is_cuda())
//     {
// #ifdef WITH_CUDA
//         return ms_deform_attn_cuda_backward(
//             value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
// #else
//         AT_ERROR("Not compiled with GPU support");
// #endif
//     }
//     AT_ERROR("Not implemented on the CPU");
// }


PYBIND11_MODULE(VSAttn, m) {
	m.doc() = "Varied-Size Window Attention";
	m.def("forward", &VSAttn_forward, "Varied-Size Window Attention forward function");
    m.def("backward", &VSAttn_backward, "Varied-Size Window Attention forward function");
}