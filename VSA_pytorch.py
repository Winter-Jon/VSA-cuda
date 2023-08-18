import torch
import torch.nn.functional as F

def VSA_QK_torch(q, k, v, ws, attn_scale, sampling_matrix):
    b, num_heads, per_head_dim, h, w = q.shape
    out_dim = num_heads * per_head_dim
    num_predict_total = b * num_heads
    window_num_h, window_num_w = h // ws, w // ws
    k_selected = F.grid_sample(
                    k.reshape(num_predict_total, out_dim // num_heads, h, w), 
                    grid=sampling_matrix.reshape(num_predict_total, h, w, 2), padding_mode='zeros', align_corners=True
                    ).reshape(b * num_heads, out_dim // num_heads, h, w)
    q = q.reshape(b, num_heads, out_dim // num_heads, window_num_h, ws, window_num_w, ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, num_heads, ws*ws, out_dim//num_heads)
    k = k_selected.reshape(b, num_heads, out_dim // num_heads, window_num_h, ws, window_num_w, ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, num_heads, ws*ws, out_dim//num_heads)
    dots = (q @ k.transpose(-2, -1)) * attn_scale

    v_selected = F.grid_sample(
                    v.reshape(num_predict_total, out_dim // num_heads, h, w), 
                    grid=sampling_matrix.reshape(num_predict_total, h, w, 2), padding_mode='zeros', align_corners=True
                    ).reshape(b*num_heads, out_dim // num_heads, h, w)
    v = v_selected.reshape(b, num_heads, out_dim // num_heads, window_num_h, ws, window_num_w, ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, num_heads, ws*ws, out_dim//num_heads)
    attn = dots.softmax(dim=-1)
    output = attn @ v
    return dots, output

def VSA_Attn(q, k, v, ws, attn_scale, sampling_offsets, sampling_scales, relative_position_bias, base_coords):
    b, num_heads, per_head_dim, h, w = q.shape
    out_dim = num_heads * per_head_dim
    window_num_h, window_num_w = h/ws, w/ws

    coords = base_coords.repeat(b * num_heads, 1, 1, 1, 1, 1)
    
    num_predict_total = b * num_heads
    sampling_offsets = sampling_offsets.reshape(num_predict_total, 2, window_num_h, window_num_w)
    sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (w // ws)
    sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (h // ws)
    
    sampling_scales = sampling_scales      #B, heads*2, h // window_size, w // window_size
    sampling_scales = sampling_scales.reshape(num_predict_total, 2, window_num_h, window_num_w)
    
    coords = coords + coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :, None]
    sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(num_predict_total, ws * window_num_h, ws * window_num_w, 2)


    k_selected = F.grid_sample(
                    k.reshape(num_predict_total, out_dim // num_heads, h, w), 
                    grid=sample_coords, padding_mode='zeros', align_corners=True
                    ).reshape(b * num_heads, out_dim // num_heads, h, w)
    v_selected = F.grid_sample(
                    v.reshape(num_predict_total, out_dim // num_heads, h, w), 
                    grid=sample_coords, padding_mode='zeros', align_corners=True
                    ).reshape(b*num_heads, out_dim // num_heads, h, w)

    q = q.reshape(b, num_heads, out_dim // num_heads, window_num_h, ws, window_num_w, ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, num_heads, ws*ws, out_dim//num_heads)
    k = k_selected.reshape(b, num_heads, out_dim // num_heads, window_num_h, ws, window_num_w, ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, num_heads, ws*ws, out_dim//num_heads)
    v = v_selected.reshape(b, num_heads, out_dim // num_heads, window_num_h, ws, window_num_w, ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, num_heads, ws*ws, out_dim//num_heads)
    
    dots = (q @ k.transpose(-2, -1)) * attn_scale

    if relative_position_bias != None:
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        # self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        dots += relative_position_bias.unsqueeze(0)

    attn = dots.softmax(dim=-1)
    out = attn @ v
    return out