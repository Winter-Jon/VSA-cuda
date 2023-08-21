import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda)
@ti.kernel
def VSA_QK_taichi(
    q: ti.types.ndarray(),
    k: ti.types.ndarray(),
    ws: ti.types.int32,
    attn_scale: ti.types.f32,
    sampling_matrix: ti.types.ndarray(),
    output: ti.types.ndarray()
):
    q_b, q_num_heads, q_per_head_dim, q_h, q_w = q.shape
    t_b, t_num_windows_h, t_num_windows_w, t_num_heads, t_q_ws2, t_sp_ws2 = output.shape
    
    for b, num_windows_h, num_windows_w, num_heads, q_ws2, sp_ws2 in output:
        for dim in ti.ndrange(q_per_head_dim):
            output[b, num_windows_h, num_windows_w, num_heads, q_ws2, sp_ws2] \
            += q[b, num_heads, dim, num_windows_h*ws+q_ws2//ws, num_windows_w*ws+q_ws2%ws] \
            *  bilinear(k, sampling_matrix, b, num_heads, dim, num_windows_h, num_windows_w, ws, sp_ws2) * attn_scale


@ti.kernel
def VSA_Attn_taichi(
    attn: ti.types.ndarray(),
    v: ti.types.ndarray(),
    ws: ti.types.int32,
    attn_scale: ti.types.f32,
    sampling_matrix: ti.types.ndarray(),
    output: ti.types.ndarray()
):
    a_b, a_num_windows_h, a_num_windows_w, a_num_heads, aa_ws2, dim_ws2 = attn.shape
    t_b, t_num_windows_h, t_num_windows_w, t_num_heads, t_q_ws2, t_per_head_dim = output.shape

    for b, num_windows_h, num_windows_w, num_heads, a_ws2, head_dim in output:
        for dim in ti.ndrange(dim_ws2):
            output[b, num_windows_h, num_windows_w, num_heads, a_ws2, head_dim] \
            += attn[b, num_windows_h, num_windows_w, num_heads, a_ws2, dim]\
            *  bilinear(v, sampling_matrix, b, num_heads, head_dim, num_windows_h, num_windows_w, ws, dim)


@ti.func
def bilinear(
    data: ti.types.ndarray(),
    sampling_loc: ti.types.ndarray(),
    b: ti.types.i32,
    num_heads: ti.types.i32,
    dim: ti.types.i32,
    num_windows_h: ti.types.i32,
    num_windows_w: ti.types.i32,
    ws: ti.types.i32,
    sp_ws2: ti.types.i32
):
    _,_,_,height, width = data.shape
    _h, _w = sampling_loc[b,num_heads,num_windows_h*ws+sp_ws2//ws,num_windows_w*ws+sp_ws2%ws,1], sampling_loc[b,num_heads,num_windows_h*ws+sp_ws2//ws,num_windows_w*ws+sp_ws2%ws,0]
    h, w = (_h+1)/2 * (height-1), (_w+1)/2 * (width-1)
    h_low, w_low = int(tm.floor(h)), int(tm.floor(w))
    h_high, w_high = h_low + 1, w_low + 1

    #zeros
    v1, v2, v3, v4 = 0.0,0.0,0.0,0.0

    if 0 <= h_low <= height-1 and 0 <= w_low <= height-1:
        v1 = data[b,num_heads,dim,h_low,w_low]
    if 0 <= h_low <= height-1 and 0 <= w_high <= height-1:
        v2 = data[b,num_heads,dim,h_low,w_high]
    if 0 <= h_high <= height-1 and 0 <= w_low <= height-1:
        v3 = data[b,num_heads,dim,h_high,w_low]
    if 0 <= h_high <= height-1 and 0 <= w_high <= height-1:
        v4 = data[b,num_heads,dim,h_high,w_high]

    lh, lw = h - h_low, w - w_low
    hh, hw = 1 - lh, 1 - lw 
    w1, w2, w3, w4 = hh * hw, hh * lw,  lh * hw, lh * lw
    output = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)

    return output
