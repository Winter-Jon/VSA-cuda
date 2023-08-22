from typing import Any
import torch
import os
from torch.utils.cpp_extension import load

if not os.path.exists("build"):
    os.mkdir("build")

VSAttn = load(name="VSAttn", 
                     extra_include_paths=["vsa/include"],
                     build_directory="build",
                     sources=["vsa/VSAttn.cpp","vsa/kernel/VSAttn_kernel.cu"],verbose=True)


class VSAttnFunction(torch.autograd.Function):
    def forward(ctx, q, k, v, sampling_matrix, window_size, attn_scale):

        output = VSAttn.forward(q, k, v, sampling_matrix, window_size, attn_scale)
        # ctx.save_for_backward(output)
        return output

    def backward(ctx, *grad_outputs):
        return super().backward(ctx, *grad_outputs)
