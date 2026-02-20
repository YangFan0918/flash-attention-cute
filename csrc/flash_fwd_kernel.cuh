#pragma once

#include "flash.h"
#include "flash_fwd_traits.h"
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

namespace flash {

// ======================== Softmax utilities ========================

template<typename T, int N>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = N / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T, int N>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = N / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Apply online softmax rescaling to accumulator fragment
template<typename Tensor0, typename Tensor1>
__device__ __forceinline__ void scale_apply_exp2(
    Tensor0& tensor, const Tensor1& row_max, float scale_log2)
{
    static_assert(Tensor0::Layout::rank == 3, "expected rank-3 (MMA_M, MMA_N, pipe)");
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        float max_scaled = row_max(mi) * scale_log2;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ni++) {
            tensor(mi, ni, 0) = exp2f(tensor(mi, ni, 0) * scale_log2 - max_scaled);
        }
    }
}

// ======================== Main kernel ========================

template<typename Traits>
__global__ void flash_fwd_kernel(__grid_constant__ const ForwardParams params) {
    using Element = typename Traits::Element;
    using TiledMma = typename Traits::TiledMma;
    using SmemLayoutQ = typename Traits::SmemLayoutQ;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemLayoutV = typename Traits::SmemLayoutV;
    using GmemTiledCopyQKV = typename Traits::GmemTiledCopyQKV;
    using SmemCopyAtom = typename Traits::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Traits::SmemCopyAtomTransposed;

    constexpr int kBlockM  = Traits::kBlockM;
    constexpr int kBlockN  = Traits::kBlockN;
    constexpr int kHeadDim = Traits::kHeadDim;

    const int m_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;

    // Early exit if this block is beyond seqlen_q
    if (m_block * kBlockM >= params.seqlen_q) return;

    // ======================== Step 1: Create gmem tensors ========================
    // Q: (seqlen_q, head_dim) for this batch/head
    auto mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                      + batch * params.q_batch_stride + head * params.q_head_stride),
        make_shape(params.seqlen_q, Int<kHeadDim>{}),
        make_stride(params.q_row_stride, Int<1>{})
    );
    auto mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                      + batch * params.k_batch_stride + head * params.k_head_stride),
        make_shape(params.seqlen_k, Int<kHeadDim>{}),
        make_stride(params.k_row_stride, Int<1>{})
    );
    auto mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                      + batch * params.v_batch_stride + head * params.v_head_stride),
        make_shape(params.seqlen_k, Int<kHeadDim>{}),
        make_stride(params.v_row_stride, Int<1>{})
    );
    auto mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                      + batch * params.o_batch_stride + head * params.o_head_stride),
        make_shape(params.seqlen_q, Int<kHeadDim>{}),
        make_stride(params.o_row_stride, Int<1>{})
    );

    // ======================== Step 2: local_tile — get current block ========================
    // gQ: (kBlockM, kHeadDim) — this thread block's Q tile
    auto gQ = local_tile(mQ, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _0{}));
    auto gO = local_tile(mO, make_tile(Int<kBlockM>, Int<kHeadDim>{}), make_coord(m_block, _0{}));

    // ======================== Step 3: Shared memory tensors ========================
    extern __shared__ char smem_[];
    Element* smem_q = reinterpret_cast<Element*>(smem_);
    Element* smem_k = smem_q + cosize(SmemLayoutQ{});
    Element* smem_v = smem_k + cosize(SmemLayoutK);

    auto sQ = make_tensor(make_smem_ptr(smem_q), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(smem_v), SmemLayoutV{});

    // ======================== Step 4: TiledCopy — partition for gmem↔smem ========================
    GmemTiledCopyQKV gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);

    auto tQgQ = gmem_thr_copy.partition_S(gQ);  // (CPY, CPY_M, CPY_K)
    auto tQsQ = gmem_thr_copy.partition_D(sQ);

    // ======================== Step 5: TiledMMA — partition for compute ========================
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // Accumulator for output: (MMA_M, MMA_N, num_tiles) in float32
    auto tOrO = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(tOrO);

    // Row max and row sum for online softmax — one per M-row this thread owns
    auto row_max = make_tensor<float>(Shape<Int<size<0>(tOrO)>>{});
    auto row_sum = make_tensor<float>(Shape<Int<size<0>(tOrO)>>{});
    fill(row_max, -INFINITY);
    fill(row_sum, 0.f);

    // SmemTiledCopy for QK^T gemm: smem → register
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto tSsQ = smem_thr_copy_Q.partition_S(sQ);  // Q in smem, partitioned for copy

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
    auto tSsK = smem_thr_copy_K.partition_S(sK);

    // SmemTiledCopy for PV gemm: V uses transposed load
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    auto tOsV = smem_thr_copy_V.partition_S(sV);

    // ======================== Step 6: Load Q to smem (once) ========================
    cute::copy(gmem_tiled_copy, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Pre-load Q fragments from smem to registers
    auto tSrQ = thr_mma.partition_fragment_A(sQ);
    auto tSrQ_copy = smem_thr_copy_Q.retile_D(tSrQ);
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy);

    // ======================== Step 7: Main loop over K/V blocks ========================
    const int n_blocks = cute::ceil_div(params.seqlen_k, kBlockN);
    const int n_block_max = params.is_causal
        ? cute::ceil_div((m_block + 1) * kBlockM, kBlockN)
        : n_blocks;

    for (int n_block = 0; n_block < n_block_max; n_block++) {
        // --- 7a. Copy K block to smem ---
        auto gK = local_tile(mK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(n_block, _0{}));
        auto tKgK = gmem_thr_copy.partition_S(gK);
        auto tKsK = gmem_thr_copy.partition_D(sK);
        cute::copy(gmem_tiled_copy, tKgK, tKsK);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // --- 7b. Compute S = Q @ K^T ---
        auto tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrS);

        auto tSrK = thr_mma.partition_fragment_B(sK);
        auto tSrK_copy = smem_thr_copy_K.retile_D(tSrK);
        cute::copy(smem_tiled_copy_K, tSsK, tSrK_copy);

        cute::gemm(tiled_mma, tSrQ, tSrK, tSrS);

        // Apply softmax scale
        #pragma unroll
        for (int i = 0; i < size(tSrS); i++) {
            tSrS(i) *= params.softmax_scale;
        }

        // --- 7c. Causal mask ---
        if (params.is_causal) {
            // TODO: apply causal mask based on m_block, n_block positions
            // mask positions where col > row to -inf
        }

        // --- 7d. Online softmax ---
        // Compute new row max
        auto new_row_max = make_tensor<float>(Shape<Int<size<0>(tSrS)>>{});
        fill(new_row_max, -INFINITY);
        #pragma unroll
        for (int mi = 0; mi < size<0>(tSrS); mi++) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(tSrS); ni++) {
                new_row_max(mi) = max(new_row_max(mi), tSrS(mi, ni, 0));
            }
        }
        // TODO: warp-level reduction for row_max across threads sharing the same row

        // Rescale previous output and exp-sum
        auto rescale = make_tensor<float>(Shape<Int<size<0>(tSrS)>>{});
        #pragma unroll
        for (int mi = 0; mi < size<0>(tSrS); mi++) {
            float old_max = row_max(mi);
            float cur_max = max(old_max, new_row_max(mi));
            row_max(mi) = cur_max;
            rescale(mi) = exp2f((old_max - cur_max) * float(M_LOG2E));
            row_sum(mi) *= rescale(mi);
        }

        // Rescale running output accumulator
        #pragma unroll
        for (int mi = 0; mi < size<0>(tOrO); mi++) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(tOrO); ni++) {
                tOrO(mi, ni, 0) *= rescale(mi);
            }
        }

        // Exponentiate scores and accumulate row sum
        #pragma unroll
        for (int mi = 0; mi < size<0>(tSrS); mi++) {
            float max_val = row_max(mi);
            #pragma unroll
            for (int ni = 0; ni < size<1>(tSrS); ni++) {
                tSrS(mi, ni, 0) = exp2f((tSrS(mi, ni, 0) - max_val) * float(M_LOG2E));
                row_sum(mi) += tSrS(mi, ni, 0);
            }
        }

        // --- 7e. Copy V block to smem ---
        auto gV = local_tile(mV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(n_block, _0{}));
        auto tVgV = gmem_thr_copy.partition_S(gV);
        auto tVsV = gmem_thr_copy.partition_D(sV);
        __syncthreads();  // reuse smem after K is consumed
        cute::copy(gmem_tiled_copy, tVgV, tVsV);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // --- 7f. Compute O += P @ V ---
        // Convert P (float32 scores) to fp16/bf16 for MMA input
        // TODO: convert tSrS to Element type and retile for MMA A operand
        // Then: cute::gemm(tiled_mma, tSrP_retiled, tOrV, tOrO);

        auto tOrV = thr_mma.partition_fragment_B(sV);
        auto tOrV_copy = smem_thr_copy_V.retile_D(tOrV);
        cute::copy(smem_tiled_copy_V, tOsV, tOrV_copy);

        // For now: naive accumulation (replace with proper MMA once P conversion is done)
        // cute::gemm(tiled_mma, tSrP, tOrV, tOrO);

        __syncthreads();
    }

    // ======================== Step 8: Normalize O ========================
    // TODO: warp-level reduction for row_sum
    #pragma unroll
    for (int mi = 0; mi < size<0>(tOrO); mi++) {
        float inv_sum = (row_sum(mi) == 0.f) ? 1.f : 1.f / row_sum(mi);
        #pragma unroll
        for (int ni = 0; ni < size<1>(tOrO); ni++) {
            tOrO(mi, ni, 0) *= inv_sum;
        }
    }

    // ======================== Step 9: Write O to gmem ========================
    // Convert float32 accumulator to fp16/bf16 and store
    // TODO: copy tOrO → smem → gmem (or direct register → gmem if layout matches)
}

} // namespace flash
