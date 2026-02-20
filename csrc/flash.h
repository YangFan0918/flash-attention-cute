#pragma once
#include <cstdint>

namespace flash {

struct ForwardParams {
    // data pointers
    void *q_ptr = nullptr;
    void *k_ptr = nullptr;
    void *v_ptr = nullptr;
    void *o_ptr = nullptr;

    // sequence lengths
    int batch = 0;
    int seqlen_q = 0;
    int seqlen_k = 0;
    int heads = 0;
    int heads_k = 0;
    int head_dim = 0;

    // strides (in elements, not bytes)
    int64_t q_batch_stride = 0;
    int64_t q_head_stride = 0;
    int64_t q_row_stride = 0;

    int64_t k_batch_stride = 0;
    int64_t k_head_stride = 0;
    int64_t k_row_stride = 0;

    int64_t v_batch_stride = 0;
    int64_t v_head_stride = 0;
    int64_t v_row_stride = 0;

    int64_t o_batch_stride = 0;
    int64_t o_head_stride = 0;
    int64_t o_row_stride = 0;

    // softmax scale
    float softmax_scale = 0.0f;
    float softmax_scale_log2 = 0.0f;

    // flags
    bool is_causal = false;
    bool is_bf16 = false;
};

} // namespace flash
