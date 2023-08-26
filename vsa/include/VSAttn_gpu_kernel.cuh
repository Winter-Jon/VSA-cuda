#ifndef VSATTN_GPU_KERNEL
#define VSATTN_GPU_KERNEL

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

#define TRY(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

#define LOCDIM(blockX,thread_loc,dim)\
    (block_loc+blockX) * dim + thread_loc

#define LOC(blockX,thread_loc)\
    (block_loc+blockX) * blockDim.x + thread_loc

#define LOCSAMP(h,w,dim) \
    (((b*height)+h)*width+w)*num_dims+dim
    // (((blockZ*height)+h)*width+w)*num_dims+dim
// template <typename scalar_t>
// __global__ void VSAttn_gpu_kernel_forward(
//     at::cuda::CUDAStream stream,
//     const scalar_t* q,
//     const scalar_t* k,
//     const scalar_t* v,
//     scalar_t* attn,
//     scalar_t* output,
//     const scalar_t* sampling_matrix,
//     const int ws, const int dim, const int num_heads, const int h_num_windows, const int w_num_windows, const int batch_size,
//     const float attn_scale
// );


// template <typename scalar_t>
// __global__ void VSAttn_gpu_kernel_accessor_forward(
//     at::cuda::CUDAStream stream,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> k,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> attn,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sampling_matrix,
//     const int ws, const int dim, const int num_heads, const int h_num_windows, const int w_num_windows, const int batch_size,
//     const float attn_scale
// );


// template <typename scalar_t> 
// __global__ void VSAttn_gpu_kernel_accessor_forward(
//     at::cuda::CUDAStream stream,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> k,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> attn,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sampling_matrix,
//     const int ws, const int dim, const int num_heads, const int h_num_windows, const int w_num_windows, const int batch_size,
//     const float attn_scale
// );

template <typename scalar_t> 
__device__ scalar_t bilinear(
    const scalar_t *data,
    const float _h,
    const float _w,
    const int dim,
    const int num_dims,
    // const int num_heads,
    const int ws,
    const int h_num_windows,
    const int w_num_windows
);

template <typename scalar_t> 
__global__ void VSAttn_gpu_kernel_forward(
    at::cuda::CUDAStream stream,
    const scalar_t *q,
    const scalar_t *k,
    const scalar_t *v,
    cudaTextureObject_t k_obj,
    cudaTextureObject_t v_obj,
    scalar_t *output,
    const float *sampling_matrix,

    const int ws, const int dim, const int num_heads, const int h_num_windows, const int w_num_windows, const int batch_size, const int b,
    const float attn_scale
){
    const int ws2 = ws*ws;
    // __shared__ scalar_t q_shared[16][64];
    // __shared__ scalar_t sampling_matrix_shared[16][49][2];
    // __shared__ scalar_t attn_shared[16][49];

    __shared__ scalar_t q_shared[64];
    __shared__ scalar_t sampling_matrix_shared[49][2];
    __shared__ scalar_t attn_shared[49];
    scalar_t attn_temp = 0.0;
    // __shared__ float q_shared[64][64];
    // __shared__ float k_shared[64][64];
    // __shared__ float v_shared[64][64];

    // q_shared[blockIdx.x][threadIdx.x] = q[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x];
    // k_shared[blockIdx.x][threadIdx.x] = k[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x];
    // v_shared[threadIdx.x][blockIdx.x] = v[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x]; 

    // const int cur_batch_size = blockIdx.z;
    // const int cur_num_window = blockIdx.y;

    const int block_loc = gridDim.x * blockIdx.y;
    const int blockX = blockIdx.x;
    const int thread_loc = threadIdx.x;
    int num_heads_1 = gridDim.y / (h_num_windows * w_num_windows);
    int head_num_1 = blockIdx.y % num_heads_1;
    // const int head_num = thread_loc / dim;
    // const int dim_num = thread_loc % dim;

    
    q_shared[thread_loc] = q[LOC(blockX,thread_loc)];

    // for(int i=0;i<num_heads*ws*ws*2;i+=blockDim.x)
    //     if(thread_loc + i * blockDim.x < num_heads*ws*ws*2)
    //         sampling_matrix_shared[(thread_loc+i*blockDim.x)/(ws*ws*2)][((thread_loc+i*blockDim.x)%(ws*ws*2))/2][(thread_loc+i*blockDim.x)%2] = sampling_matrix[block_loc+thread_loc+i*blockDim.x];
    
    // __syncthreads();


    if(thread_loc < ws2)
    {
        sampling_matrix_shared[thread_loc][0] = sampling_matrix[LOCDIM(thread_loc,0,2)];
        sampling_matrix_shared[thread_loc][1] = sampling_matrix[LOCDIM(thread_loc,1,2)];
    }   


    __syncthreads();
    if(thread_loc < ws2)
    {
        for(int i=0;i<dim;++i)
        {
            auto w_1 = (sampling_matrix_shared[thread_loc][0]+1)/2 * (ws*w_num_windows-1) + 0.5;
            auto h_1 = (sampling_matrix_shared[thread_loc][1]+1)/2 * (ws*h_num_windows-1) + 0.5;

             attn_temp += q_shared[i] * attn_scale * tex2DLayered<float>(k_obj, w_1,h_1,(head_num_1)*dim+i); 
            // attn_temp += q_shared[i] * attn_scale * bilinear(k,sampling_matrix_shared[thread_loc][1],sampling_matrix_shared[thread_loc][0],i,dim,ws,h_num_windows,w_num_windows);
            // attn_temp += q_shared[i] * attn_scale;
        }
        attn_shared[thread_loc] = attn_temp;
    }

    __syncthreads();

    scalar_t out = 0.0;
    for(int i=0;i<ws2;++i)
    {
        auto w_2 = (sampling_matrix_shared[i][0]+1)/2 * (ws*w_num_windows-1) + 0.5;
        auto h_2 = (sampling_matrix_shared[i][1]+1)/2 * (ws*h_num_windows-1) + 0.5;

         out += attn_shared[i] * tex2DLayered<float>(v_obj, w_2,h_2,(head_num_1)*dim+thread_loc); 
        // out += attn_shared[i] * bilinear(v,sampling_matrix_shared[i][1],sampling_matrix_shared[i][0],thread_loc,dim,ws,h_num_windows,w_num_windows);
        // out += attn_shared[i];
    }
    output[LOC(blockX,thread_loc)] = out;



    // if(thread_loc < ws2*num_heads)
    // {
    //     for(int i=0;i<dim;++i)
    //     {
    //         attn_temp += q_shared[thread_loc/ws2][i] * attn_scale 
    //             * bilinear(k,sampling_matrix_shared[thread_loc/ws2][thread_loc%ws][1],sampling_matrix_shared[thread_loc/ws2][thread_loc%ws][0],i,dim,num_heads,ws,h_num_windows,w_num_windows);
    //     }
    //     attn_shared[thread_loc/ws2][thread_loc] = attn_temp; 
    // }




    
}

// template <typename scalar_t>
// void VSAttn_gpu_kernel_backward(
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> q,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> k,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> v,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sampling_matrix,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output,
//     const int ws,
//     const float attn_scale
// ){}


// template <typename scalar_t> 
// __global__ void VSAttn_gpu_kernel_accessor_forward(
//     at::cuda::CUDAStream stream,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> q,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> k,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> v,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> attn,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sampling_matrix,
//     const int ws, const int dim, const int num_heads, const int h_num_windows, const int w_num_windows, const int batch_size,
//     const float attn_scale
// ){
//     const int ws2 = ws*ws;
//     // __shared__ float attn[64][64];
//     // __shared__ float q_shared[64][64];
//     // __shared__ float k_shared[64][64];
//     // __shared__ float v_shared[64][64];

//     // q_shared[blockIdx.x][threadIdx.x] = q[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x];
//     // k_shared[blockIdx.x][threadIdx.x] = k[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x];
//     // v_shared[threadIdx.x][blockIdx.x] = v[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x]; 

//     printf("%f",q[0][0][0][0]);
//     printf("%f",q[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x]);

//     for(int i=0;i<ws2;i++)
//         attn[blockIdx.z][blockIdx.y][i][threadIdx.x] += q[blockIdx.z][blockIdx.y][i][threadIdx.x] * k[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x];
    
//     __syncthreads();
//     float out = 0;
//     for(int i=0;i<dim;i++)
//     {
//         output[blockIdx.z][blockIdx.y][blockIdx.x][i] += attn[blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.x] * v[blockIdx.z][blockIdx.y][blockIdx.x][i];
//     }
    
// }

template <typename scalar_t> 
__device__ scalar_t bilinear(
    const scalar_t *data,
    const float _h,
    const float _w,
    const int dim,
    const int num_dims,
    // const int num_heads,
    const int ws,
    const int h_num_windows,
    const int w_num_windows
){
    scalar_t output = 0.0;

    // const int block_loc = gridDim.y * gridDim.x * blockIdx.z + gridDim.x * blockIdx.y;
    // const int blockX = blockIdx.x;
    // const int thread_loc = threadIdx.x;

    int height = ws * h_num_windows;
    int width = ws * w_num_windows;

    // [b,num_windows_h,num_windows_w,num_heads,ws2,2]
    // auto _h = sampling_matrix[LOCDIM(loc,1,2)];
    // auto _w = sampling_matrix[LOCDIM(loc,0,2)];

    auto h = (_h+1)/2 * (height-1);
    auto w = (_w+1)/2 * (width-1);

    int h_low = floor(h), w_low = floor(w);
    int h_high = h_low + 1, w_high = w_low + 1;

    scalar_t v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0;

    // [_h,_w,dim]

    // blockIdx girdDim

    int num_heads = gridDim.y / (h_num_windows * w_num_windows);
    int head_num = blockIdx.y % num_heads;
    int b = head_num;
    
    if (0 <= h_low && h_low <= height - 1 && 0 <= w_low && w_low <= height - 1)
        v1 = data[LOCSAMP(h_low,w_low,dim)];
    if (0 <= h_low && h_low <= height - 1 && 0 <= w_high && w_high <= height - 1)
        v2 = data[LOCSAMP(h_low,w_high,dim)];
    if (0 <= h_high && h_high <= height - 1 && 0 <= w_low && w_low <= height - 1)
        v3 = data[LOCSAMP(h_high,w_low,dim)];
    if (0 <= h_high && h_high <= height - 1 && 0 <= w_high && w_high <= height - 1)
        v4 = data[LOCSAMP(h_high,w_high,dim)];

    auto lh = h - h_low, lw = w - w_low;
    auto hh = 1 - lh, hw = 1 - lw;
    auto w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    output = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return output;
}

#endif