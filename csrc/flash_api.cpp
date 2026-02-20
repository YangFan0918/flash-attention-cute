#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "flash.h"

namespace flash {
void run_flash_attention_forward(ForwardParams& params, cudaStream_t stream);
}

at::Tensor flash_attn_forward(
    at::Tensor q, at::Tensor k, at::Tensor v,
    bool is_causal)
{
    // Input validation
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16, "q must be fp16 or bf16");
    TORCH_CHECK(q.stride(-1) == 1, "q must be contiguous in last dim");
    TORCH_CHECK(k.stride(-1) == 1, "k must be contiguous in last dim");
    TORCH_CHECK(v.stride(-1) == 1, "v must be contiguous in last dim");

    // Shape: (batch, seqlen, heads, head_dim)
    const int batch    = q.size(0);
    const int seqlen_q = q.size(1);
    const int heads    = q.size(2);
    const int head_dim = q.size(3);
    const int seqlen_k = k.size(1);
    const int heads_k  = k.size(2);

    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(heads % heads_k == 0, "heads must be divisible by heads_k (for GQA)");

    auto o = torch::empty_like(q);

    flash::ForwardParams params;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = o.data_ptr();

    params.batch    = batch;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.heads    = heads;
    params.heads_k  = heads_k;
    params.head_dim = head_dim;

    params.q_batch_stride = q.stride(0);
    params.q_row_stride   = q.stride(1);
    params.q_head_stride  = q.stride(2);
    params.k_batch_stride = k.stride(0);
    params.k_row_stride   = k.stride(1);
    params.k_head_stride  = k.stride(2);
    params.v_batch_stride = v.stride(0);
    params.v_row_stride   = v.stride(1);
    params.v_head_stride  = v.stride(2);
    params.o_batch_stride = o.stride(0);
    params.o_row_stride   = o.stride(1);
    params.o_head_stride  = o.stride(2);

    params.softmax_scale      = 1.0f / sqrtf(static_cast<float>(head_dim));
    params.softmax_scale_log2 = params.softmax_scale * float(M_LOG2E);
    params.is_causal = is_causal;
    params.is_bf16   = q.dtype() == at::kBFloat16;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    flash::run_flash_attention_forward(params, stream);

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "Flash Attention forward (CUDA)");
}
