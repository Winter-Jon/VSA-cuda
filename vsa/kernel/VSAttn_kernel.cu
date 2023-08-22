#include "VSAttn_kernel.cuh"
#include "VSAttn_gpu_kernel.cuh"
// #include "VSAttn_kernel_cudnn.cuh"
#include <iostream>

// #define scalar_t float

torch::Tensor VSAttn_kernel_forward(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &sampling_matrix,
    const int ws,
    const double attn_scale
){

    auto batch_size=q.size(0), num_heads=q.size(1), per_head_dim=q.size(2), h=q.size(3), w=q.size(4);

    // TORCH_CHECK(h%ws==0,"h must be divisible by window-size")
    // TORCH_CHECK(w%ws==0,"w must be divisible by window-size")

    int h_num_windows = h / ws;
    int w_num_windows = w / ws;

    q = q.reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,3,5,1,4,6,2}).reshape({-1});
    k = k.permute({0,1,3,4,2}).reshape({-1});
    v = v.permute({0,1,3,4,2}).reshape({-1});
    sampling_matrix = sampling_matrix.reshape({batch_size, num_heads, h_num_windows, ws, w_num_windows, ws, 2}).permute({0,2,4,1,3,5,6}).reshape({-1});

    auto output = torch::zeros_like(q).reshape({batch_size*h_num_windows*w_num_windows,  ws*ws, num_heads, per_head_dim});

    // <--- C++ frontend impletations
    // sampling_matrix = sampling_matrix.flatten(0,1);

    // auto grid_options = torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true);
    // auto k_selected = torch::nn::functional::grid_sample(k.flatten(0,1),sampling_matrix,grid_options).reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,3,5,1,4,6,2}).flatten(4,5);
    // auto v_selected = torch::nn::functional::grid_sample(v.flatten(0,1),sampling_matrix,grid_options).reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,3,5,1,4,6,2}).flatten(4,5);

    // auto attn = torch::matmul(q, k_selected.transpose(-1,-2)) * attn_scale;
    // auto output = torch::matmul(attn, v_selected);

    // auto attn = torch::zeros_like(q).reshape({batch_size*h_num_windows*w_num_windows, num_heads, ws*ws, ws*ws});
    // auto output = torch::zeros_like(q).reshape({batch_size*h_num_windows*w_num_windows, num_heads, ws*ws, per_head_dim});
    // --->

    // q = q.reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,3,5,4,6,1,2}).reshape({-1});
    // k = k.permute({0,3,4,1,2}).reshape({-1});
    // v = v.permute({0,3,4,1,2}).reshape({-1});
    // sampling_matrix = sampling_matrix.reshape({batch_size, num_heads, h_num_windows, ws, w_num_windows, ws, 2}).permute({0,2,4,3,5,1,6}).reshape({-1});
    // auto output = torch::zeros_like(q).reshape({batch_size*h_num_windows*w_num_windows,  ws*ws, num_heads, per_head_dim});
    
    // k = k.reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,1,3,5,4,6,2}).reshape({-1});
    // v = v.reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,1,3,5,4,6,2}).reshape({-1});



    auto block = dim3(ws*ws, h_num_windows*w_num_windows*num_heads);
    auto thread = dim3(per_head_dim);


    // TORCH_CHECK(q1.dim() == 4, "");
    // TORCH_CHECK(k3.dim() == 4, "");
    // TORCH_CHECK(v3.dim() == 4, "");
    // TORCH_CHECK(output.dim() == 4, "");
    // auto a = sampling_matrix.dim();
    
    // auto qv1 = q.index({0,0,0,0}).data();
    // auto q_acc = q3.accessor<float,4>();
    // auto qv = q_acc[0][0][0][0];
    // auto q_ptr = q.data_ptr<float>();
    // auto b = q.is_contiguous();

    for(int i=0;i<batch_size;i++)
    {
        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "need float types", ([&] {
            VSAttn_gpu_kernel_forward<scalar_t><<<block,thread,0,stream>>>(
                stream,
                q.data_ptr<scalar_t>() + i * h_num_windows*w_num_windows*num_heads * ws*ws * per_head_dim,
                k.data_ptr<scalar_t>() + i * num_heads * h * w * per_head_dim,
                v.data_ptr<scalar_t>() + i * num_heads * h * w * per_head_dim,
                output.data_ptr<scalar_t>() + i * h_num_windows*w_num_windows*num_heads * ws*ws * per_head_dim,
                sampling_matrix.data_ptr<scalar_t>() + i * h_num_windows*w_num_windows*num_heads * ws*ws * 2,
                ws, per_head_dim, num_heads, h_num_windows, w_num_windows, batch_size,
                attn_scale
            );
        }));
    }

    // AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "need float types", ([&] {
    //     VSAttn_kernel_forward_cudnn<scalar_t>(
    //         q.data_ptr<scalar_t>(),
    //         k.data_ptr<scalar_t>(),
    //         v.data_ptr<scalar_t>(),
    //         attn.data_ptr<scalar_t>(),
    //         output.data_ptr<scalar_t>(),
    //         sampling_matrix.data_ptr<scalar_t>(),
    //         ws, per_head_dim, num_heads, h_num_windows, w_num_windows, batch_size,
    //         attn_scale
    //     );
    // }));

    // batch_size, num_heads,  h_num_windows, w_num_windows, ws, ws, per_head_dim
    // AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "need float types", ([&] {
    //     VSAttn_gpu_kernel_forward<scalar_t><<<block,thread,0,stream>>>(
    //         stream,
    //         q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    //         k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    //         v.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    //         attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    //         output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    //         sampling_matrix.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    //         ws, per_head_dim, num_heads, h_num_windows, w_num_windows, batch_size,
    //         attn_scale
    //     );
    // }));
    return output;
}

std::vector<torch::Tensor> VSAttn_kernel_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor sampling_matrix,
    torch::Tensor grad_output,
    const int ws,
    const double attn_scale
){
    auto output = std::vector<torch::Tensor>();
    return output;
}