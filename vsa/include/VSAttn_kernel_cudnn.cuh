#ifndef VSATTN_GPU_KERNEL_CUDNNN
#define VSATTN_GPU_KERNEL_CUDNNN

#include <cudnn.h>
// #include <cublas_v2.h>
#include <torch/extension.h>

#define checkCUDNN(stat)\
    if(stat != CUDNN_STATUS_SUCCESS)\
        printf("error code:%d",stat);


template <typename scalar_t> 
void VSAttn_kernel_forward_cudnn(
    const scalar_t *q,
    const scalar_t *k,
    const scalar_t *v,
    scalar_t *attn,
    scalar_t *output,
    const scalar_t *sampling_matrix,
    const int ws, const int dim, const int num_heads, const int h_num_windows, const int w_num_windows, const int batch_size,
    const double attn_scale
){}

template <>
void VSAttn_kernel_forward_cudnn<float>(
    const float *q,
    const float *k,
    const float *v,
    float *attn,
    float *output,
    const float *sampling_matrix,
    const int ws, const int dim, const int num_heads, const int h_num_windows, const int w_num_windows, const int batch_size,
    const double attn_scale
){
    cudnnHandle_t handle;

    checkCUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t q_desc;
    cudnnTensorDescriptor_t k_desc;
    cudnnTensorDescriptor_t v_desc;
    cudnnTensorDescriptor_t attn_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t sampling_matrix_desc;

    checkCUDNN(cudnnCreateTensorDescriptor(&q_desc));



}

// template <>
// void VSAttn_kernel_forward_cudnn<double>(
//     const double *q,
//     const double *k,
//     const double *v,
//     double *attn,
//     double *output,
//     const double *sampling_matrix,
//     const int ws, const int dim, const int num_heads, const int h_num_windows, const int w_num_windows, const int batch_size,
//     const double attn_scale
// ){
//     cublasStatus_t stat;
//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     const double beta = 1.0;

//     stat = cublasDgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,
//                                     ws*ws, ws*ws, dim,
//                                     &attn_scale,
//                                     q, ws*ws,
//                                     ws*ws*dim,
//                                     k, dim,
//                                     ws*ws*dim,
//                                     &beta,
//                                     attn, ws*ws,
//                                     ws*ws*ws*ws,
//                                     batch_size*w_num_windows*h_num_windows*num_heads);


// }

#endif