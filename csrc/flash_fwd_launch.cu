#include "flash.h"
#include "flash_fwd_traits.h"
#include "flash_fwd_kernel.cuh"
#include "static_switch.h"

namespace flash {

template<typename Traits>
void run_flash_fwd(const ForwardParams& params, cudaStream_t stream) {
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int smem_size = Traits::kSmemSize;

    dim3 grid(cute::ceil_div(params.seqlen_q, kBlockM), params.heads, params.batch);
    dim3 block(Traits::kNThreads);

    auto kernel = flash_fwd_kernel<Traits>;

    if constexpr (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
}

void run_flash_attention_forward(ForwardParams& params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEAD_DIM_SWITCH(params.head_dim, [&] {
            run_flash_fwd<FlashFwdTraits<elem_type, kHeadDim>>(params, stream);
        });
    });
}

} // namespace flash
