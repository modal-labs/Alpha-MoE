#include <pybind11/functional.h>
#include <torch/python.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

using fp8 = __nv_fp8_e4m3;

void fused_moe_w8a8_wgmma_up_down_acc(
        const fp8* x,
        const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const float* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num,
        int block_m,
        int block_n,
        int warp_n,
        int stages,
        int producer_threads,
        float scaling_factor
);
torch::Tensor fused_moe_launcher_up_down(
        torch::Tensor& x,
        torch::Tensor& x_scale,
        torch::Tensor& w,
        torch::Tensor& w_scale,
        torch::Tensor& w2,
        torch::Tensor& w2_scale,
        torch::Tensor& sorted_token_ids,
        torch::Tensor& expert_ids,
        torch::Tensor& num_tokens_post_padded,
        torch::Tensor& topk_weights,
        torch::Tensor& out,
        int top_k,
        int block_m,
        int block_n,
        int warp_n,
        int stages,
        int producer_threads,
        float scaling_factor
        )
{
    fused_moe_w8a8_wgmma_up_down_acc(static_cast<__nv_fp8_e4m3*>(x.data_ptr()),
            static_cast<float*>(x_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w.data_ptr()),
            static_cast<float*>(w_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w2.data_ptr()),
            static_cast<float*>(w2_scale.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(sorted_token_ids.data_ptr()),
            static_cast<int*>(expert_ids.data_ptr()),
            static_cast<int*>(num_tokens_post_padded.data_ptr()),
            static_cast<float*>(topk_weights.data_ptr()),
            top_k,
            x.size(0),
            x.size(1),
            w.size(1),
            sorted_token_ids.size(0),
            block_m,
            block_n,
            warp_n,
            stages,
            producer_threads,
            scaling_factor
                );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_moe_w8a8_up_down", &fused_moe_launcher_up_down);
}
