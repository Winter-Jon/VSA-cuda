#include <torch/extension.h>
#include "VSAttn.h"
#include <iostream>

using namespace torch::indexing;

int main()
{
    auto options = torch::TensorOptions().device(torch::kCUDA, 0);
    auto q = torch::randn({16,16,64,28,28},options);
    auto k = torch::randn_like(q);
    auto v = torch::ones_like(q);
    auto s = torch::empty({1,1,1,1,1},options);
    auto batch_size=q.size(0), num_heads=q.size(1), per_head_dim=q.size(2), h=q.size(3), w=q.size(4);
    int ws = 7;
    int h_num_windows = h/ws, w_num_windows=w/ws;


    auto q_ = q.reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,1,3,5,4,6,2}).reshape({batch_size*num_heads*h_num_windows*w_num_windows,ws*ws,per_head_dim});
    auto k_ = k.reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,1,3,5,4,6,2}).flatten(-3,-2).transpose(-1,-2).reshape({batch_size*num_heads*h_num_windows*w_num_windows,per_head_dim,ws*ws});
    auto v_ = v.reshape({batch_size, num_heads, per_head_dim, h_num_windows, ws, w_num_windows, ws}).permute({0,1,3,5,4,6,2}).reshape({batch_size*num_heads*h_num_windows*w_num_windows,ws*ws,per_head_dim});

    auto k_select = torch::grid_sampler

    torch::Tensor res,res_attn;
    std::tie(res,res_attn) = VSAttn_forward(q,k,v,s,7,1.0);


    // std::cout<<q_.size(0)<<" ";
    // for(int i=0;i<3;i++)
    //     std::cout<<q_.size(i)<<" ";

    // std::cout<<k_.size(0)<<" ";
    // for(int i=0;i<3;i++)
    //     std::cout<<k_.size(i)<<" ";


    auto attn = torch::matmul(q_,k_);

    // std::cout<<attn.size(0)<<" ";

    // for(int i=0;i<3;i++)
    //     std::cout<<attn.size(i)<<" ";
    // std::cout<<std::endl;


    auto ans = torch::matmul(attn,v_).reshape({-1});
    auto ans_attn = attn.reshape({-1});


    std::cout<<res_attn.index({0});
    std::cout<<res_attn.index({0});
    std::cout<<ans_attn.index({0});


    // auto ind = torch::nonzero(res_attn-ans_attn);
    // if(ind.numel() >= 100)
    //     for(int i=0;i<98;i++)
    //         std::cout<<ind.index({i});

    std::cout<<"max(sub_attn):"<<torch::max(torch::abs(res_attn-ans_attn));
    std::cout<<"sum(sub_attn):"<<torch::sum(torch::abs(res_attn-ans_attn));
    std::cout<<"max(sub):"<<torch::max(torch::abs(res-ans));
    std::cout<<"sum(sub):"<<torch::sum(torch::abs(res-ans));

    return 0;
}