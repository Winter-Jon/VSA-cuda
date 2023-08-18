import time
import math
import os
import torch 
import torch.nn as nn

import nvtx

from VSA_pytorch import VSA_QK_torch
# from VSA_taichi import VSA_QK_taichi, VSA_Attn_taichi
from VSA_cuda import VSAttnFunction


b, num_heads, per_head_dim, h, w = 16,16,64,56,56
ws = 7

def generate_base_coords(h,w,ws):
    image_reference_w = torch.linspace(-1, 1, w, device="cuda")
    image_reference_h = torch.linspace(-1, 1, h, device="cuda")
    image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
    window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=ws)
    window_num_h, window_num_w = window_reference.shape[-2:]
    window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

    base_coords_h = torch.arange(ws, device="cuda") * 2 * ws / ws / (h-1)
    base_coords_h = (base_coords_h - base_coords_h.mean())
    base_coords_w = torch.arange(ws, device="cuda") * 2 * ws / ws / (w-1)
    base_coords_w = (base_coords_w - base_coords_w.mean())

    expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
    assert expanded_base_coords_h.shape[0] == window_num_h
    assert expanded_base_coords_h.shape[1] == ws
    expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
    assert expanded_base_coords_w.shape[0] == window_num_w
    assert expanded_base_coords_w.shape[1] == ws
    expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
    expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
    coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, ws, window_num_w, ws)
    base_coords = (window_reference+coords).cuda()
    coords = coords.cuda()
    return base_coords

def generate_sampling_matrix(q,ws,base_coords,sampling_offsets, sampling_scales):
    b, num_heads, per_head_dim, h, w = q.shape
    out_dim = num_heads * per_head_dim
    window_num_h, window_num_w = h // ws, w // ws

    coords = base_coords.repeat(b * num_heads, 1, 1, 1, 1, 1)
    num_predict_total = b * num_heads
    sampling_offsets = sampling_offsets.reshape(num_predict_total, 2, window_num_h, window_num_w)
    sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (w // ws)
    sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (h // ws)
    
    sampling_scales = sampling_scales      #B, heads*2, h // window_size, w // window_size
    sampling_scales = sampling_scales.reshape(num_predict_total, 2, window_num_h, window_num_w)
    
    coords = coords + coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :, None]
    sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(b, num_heads, window_num_h * ws, window_num_w * ws, 2)
    return sample_coords



def test_taichi():
    base_coords = generate_base_coords(h,w,ws)
    compute_time = 0

    
    q = torch.randn(b, num_heads, per_head_dim, h, w, device="cuda").contiguous()
    k = torch.randn_like(q, device="cuda").contiguous()
    v = torch.randn_like(q, device="cuda").contiguous()
    attn_scale = 1 / math.sqrt(per_head_dim)
    num_windows_h = h // ws
    num_windows_w = w // ws

    sampling_offset = torch.randn(b, num_heads*2, num_windows_h, num_windows_w, device="cuda") / h
    sampling_scale = torch.randn(b, num_heads*2, num_windows_h, num_windows_w, device="cuda") / 3

    sampling_matrix = generate_sampling_matrix(q, ws, base_coords, sampling_offset, sampling_scale).contiguous()
    attn = torch.empty(b, num_windows_h, num_windows_w, num_heads, ws*ws, ws*ws, device="cuda").contiguous()
    output = torch.zeros(b, num_windows_h, num_windows_w, num_heads, ws*ws, ws*ws, device="cuda").contiguous()
    
    for i in range(100):
        o = output.clone()
        start_time = time.time()
        VSA_QK_taichi(q,k,ws,attn_scale,sampling_matrix,attn)
        VSA_Attn_taichi(attn,v,ws,1.0,sampling_matrix,output)
        end_time = time.time()
        iter_compute_time = end_time - start_time
        compute_time += iter_compute_time
        
        if i % 10 == 0:
            print("iter:",i, " time:", iter_compute_time.__format__(".3e"))
    
    print(output.shape)
    print("time:", compute_time)


def test_torch():
    base_coords = generate_base_coords(h,w,ws)
    compute_time = 0

    
    q = torch.randn(b, num_heads, per_head_dim, h, w, device="cuda").contiguous()
    k = torch.randn_like(q, device="cuda").contiguous()
    v = torch.randn_like(q, device="cuda").contiguous()
    attn_scale = 1 / math.sqrt(per_head_dim)
    num_windows_h = h // ws
    num_windows_w = w // ws

    sampling_offset = torch.randn(b, num_heads*2, num_windows_h, num_windows_w, device="cuda") / h
    sampling_scale = torch.randn(b, num_heads*2, num_windows_h, num_windows_w, device="cuda") / 3

    sampling_matrix = generate_sampling_matrix(q, ws, base_coords, sampling_offset, sampling_scale).contiguous()

    for i in range(100):
        start_time = time.time()
        attn, output = VSA_QK_torch(q,k,v,ws,attn_scale,sampling_matrix)
        end_time = time.time()

        compute_time += end_time - start_time
        # if i % 10 == 0:
        #     print(i)
    print("torch time:", compute_time)


def test_cuda():
    base_coords = generate_base_coords(h,w,ws)
    cuda_compute_time = 0
    torch_compute_time = 0

    
    q = torch.randn(b, num_heads, per_head_dim, h, w, device="cuda").contiguous()
    k = torch.randn_like(q, device="cuda").contiguous()
    v = torch.randn_like(q, device="cuda").contiguous()
    attn_scale = 1 / math.sqrt(per_head_dim)
    num_windows_h = h // ws
    num_windows_w = w // ws

    sampling_offset = torch.randn(b, num_heads*2, num_windows_h, num_windows_w, device="cuda") / h
    sampling_scale = torch.randn(b, num_heads*2, num_windows_h, num_windows_w, device="cuda") / 3

    sampling_matrix = generate_sampling_matrix(q, ws, base_coords, sampling_offset, sampling_scale).contiguous()

    VSAttn = VSAttnFunction()


    for i in range(100):


        torch_start_time = time.time()
        attn, output = VSA_QK_torch(q,k,v,ws,attn_scale,sampling_matrix)
        torch.cuda.synchronize()
        torch_end_time = time.time()

        cuda_start_time = time.time()
        output_cuda = VSAttn.forward(q,k,v,sampling_matrix,ws,attn_scale)
        torch.cuda.synchronize()
        cuda_end_time = time.time()


        time.sleep(0.01)
        # a = output_cuda+1
        if i > 10:
            cuda_compute_time += cuda_end_time - cuda_start_time
            torch_compute_time += torch_end_time - torch_start_time
        # if i % 10 == 0:
        #     print(i)
    print("cuda time:", cuda_compute_time)
    print("torch time:", torch_compute_time)


    


def test_eq():
    base_coords = generate_base_coords(h,w,ws)
    compute_time = 0

    for i in range(10):
        q = torch.randn(b, num_heads, per_head_dim, h, w, device="cuda").contiguous()
        k = torch.randn_like(q, device="cuda").contiguous()
        v = torch.randn_like(q, device="cuda").contiguous()
        attn_scale = 1 / math.sqrt(per_head_dim)
        num_windows_h = h // ws
        num_windows_w = w // ws

        sampling_offset = torch.randn(b, num_heads*2, num_windows_h, num_windows_w, device="cuda") / h
        sampling_scale = torch.randn(b, num_heads*2, num_windows_h, num_windows_w, device="cuda") / 3

        sampling_matrix = generate_sampling_matrix(q, ws, base_coords, sampling_offset, sampling_scale).contiguous()

        attn_torch, output_torch = VSA_QK_torch(q,k,v,ws,attn_scale,sampling_matrix)

        # attn_taichi = torch.empty(b, num_windows_h, num_windows_w, num_heads, ws*ws, ws*ws, device="cuda").contiguous()
        # output_taichi = torch.empty(b, num_windows_h, num_windows_w, num_heads, ws*ws, per_head_dim, device="cuda").contiguous()
        # VSA_QK_taichi(q,k,ws,attn_scale,sampling_matrix,attn_taichi)
        # VSA_Attn_taichi(attn_taichi,v,ws,1.0,sampling_matrix,output_taichi)

        # output_attn = attn_taichi.view(-1) - attn_torch.view(-1)
        # output = output_taichi.view(-1) - output_torch.view(-1)

        VSAttn = VSAttnFunction()
        output_cuda = VSAttn.forward(q,k,v,sampling_matrix,ws,attn_scale)

        output = output_cuda- output_torch

        print(torch.sum(torch.abs(output)))


# test_taichi()
# test_torch()
test_cuda()
test_eq()