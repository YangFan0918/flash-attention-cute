#pragma once

#include "flash.h"
#include "flash_fwd_traits.h"
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

namespace flash {

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

template<typename Traits>
__global__ void flash_fwd_kernel(__grid_constant__ const ForwardParams params) {
    using Element = typename Traits::Element;
    using TiledMma = typename Traits::TiledMma;
    using SmemLayoutQ = typename Traits::SmemLayoutQ;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemLayoutV = typename Traits::SmemLayoutV;
    using Gmem2SmemTiledCopyQKV = typename Traits::Gmem2SmemTiledCopyQKV;
    using Smem2GmemTiledCopyO = typename Traits::Smem2GmemTiledCopyO;
    using Smem2RegCopyAtomA = typename Traits::Smem2RegCopyAtomA;
    using Smem2RegCopyAtomB = typename Traits::Smem2RegCopyAtomB;
    using Smem2RegCopyAtomBt = typename Traits::Smem2RegCopyAtomBt;
    using SmemLayoutVt = typename Traits::SmemLayoutVt;

    constexpr int kBlockM  = Traits::kBlockM;
    constexpr int kBlockN  = Traits::kBlockN;
    constexpr int kHeadDim = Traits::kHeadDim;

    const int m_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;

    if (m_block * kBlockM >= params.seqlen_q) return;

    auto Q = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                      + batch * params.q_batch_stride + head * params.q_head_stride),
        make_shape(params.seqlen_q, Int<kHeadDim>{}),
        make_stride(params.q_row_stride, Int<1>{})
    );
    auto K = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                      + batch * params.k_batch_stride + head * params.k_head_stride),
        make_shape(params.seqlen_k, Int<kHeadDim>{}),
        make_stride(params.k_row_stride, Int<1>{})
    );
    auto V = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                      + batch * params.v_batch_stride + head * params.v_head_stride),
        make_shape(params.seqlen_k, Int<kHeadDim>{}),
        make_stride(params.v_row_stride, Int<1>{})
    );
    auto O = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                      + batch * params.o_batch_stride + head * params.o_head_stride),
        make_shape(params.seqlen_q, Int<kHeadDim>{}),
        make_stride(params.o_row_stride, Int<1>{})
    );

    auto gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, 0));
    auto gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, 0));

    extern __shared__ char smem_[];
    Element* smem_q = reinterpret_cast<Element*>(smem_);
    Element* smem_k = smem_q + cosize(SmemLayoutQ{});
    Element* smem_v = smem_k + cosize(SmemLayoutK{});

    auto sQ = make_tensor(make_smem_ptr(smem_q), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(smem_v), SmemLayoutV{});

    Gmem2SmemTiledCopyQKV g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_thread_slice(threadIdx.x);

    auto tQgQ = g2s_thr_copy.partition_S(gQ);
    auto tQsQ = g2s_thr_copy.partition_D(sQ);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto tOrO = thr_mma.partition_fragment_C(sQ);
    clear(tOrO);

    auto row_max = make_tensor<float>(Shape<Int<size<0>(tOrO)>>{});
    auto row_sum = make_tensor<float>(Shape<Int<size<0>(tOrO)>>{});
    fill(row_max, -INFINITY);
    fill(row_sum, 0.f);

    auto s2r_tiled_copy_Q = make_tiled_copy_A(Smem2RegCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_Q   = s2r_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto tSsQ = s2r_thr_copy_Q.partition_S(sQ);

    auto s2r_tiled_copy_K = make_tiled_copy_B(Smem2RegCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_K   = s2r_tiled_copy_K.get_thread_slice(threadIdx.x);
    auto tSsK = s2r_thr_copy_K.partition_S(sK);

    auto sVt = make_tensor(sV.data(), SmemLayoutVt{});
    auto s2r_tiled_copy_V = make_tiled_copy_B(Smem2RegCopyAtomBt{}, tiled_mma);
    auto s2r_thr_copy_V   = s2r_tiled_copy_V.get_thread_slice(threadIdx.x);
    auto tOsV = s2r_thr_copy_V.partition_S(sVt);

    auto score_ref = make_tensor(static_cast<Element*>(nullptr),
                                 make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    auto c2a = left_inverse(thr_mma.partition_C(score_ref).layout())
               .compose(thr_mma.partition_A(score_ref).layout());

    cute::copy(g2s_tiled_copy, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    auto tSrQ = thr_mma.partition_fragment_A(sQ);
    auto tSrQ_copy = s2r_thr_copy_Q.retile_D(tSrQ);

    const int n_blocks = cute::ceil_div(params.seqlen_k, kBlockN);
    const int n_block_max = params.is_causal
        ? cute::ceil_div((m_block + 1) * kBlockM, kBlockN)
        : n_blocks;

    {
        auto gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, 0));
        cute::copy(g2s_tiled_copy, g2s_thr_copy.partition_S(gK), g2s_thr_copy.partition_D(sK));
        cp_async_fence();
    }

    for (int n_block = 0; n_block < n_block_max; n_block++) {
        cp_async_wait<0>();
        __syncthreads();

        auto gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(n_block, 0));
        cute::copy(g2s_tiled_copy, g2s_thr_copy.partition_S(gV), g2s_thr_copy.partition_D(sV));
        cp_async_fence();

        auto tSrS = thr_mma.partition_fragment_C(score_ref);
        clear(tSrS);

        auto tSrK = thr_mma.partition_fragment_B(sK);
        auto tSrK_copy = s2r_thr_copy_K.retile_D(tSrK);

        cute::copy(s2r_tiled_copy_K, tSsK(_, _, 0), tSrK_copy(_, _, 0));
        cute::copy(s2r_tiled_copy_Q, tSsQ(_, _, 0), tSrQ_copy(_, _, 0));
        #pragma unroll
        for (int ki = 0; ki < size<2>(tSrQ); ki++) {
            if (ki + 1 < size<2>(tSrQ)) {
                cute::copy(s2r_tiled_copy_K, tSsK(_, _, ki + 1), tSrK_copy(_, _, ki + 1));
                cute::copy(s2r_tiled_copy_Q, tSsQ(_, _, ki + 1), tSrQ_copy(_, _, ki + 1));
            }
            cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrS);
        }

        if (params.is_causal) {
        }

        auto new_row_max = make_tensor<float>(Shape<Int<size<0>(tSrS)>>{});
        fill(new_row_max, -INFINITY);
        #pragma unroll
        for (int mi = 0; mi < size<0>(tSrS); mi++) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(tSrS); ni++) {
                new_row_max(mi) = max(new_row_max(mi), tSrS(mi, 0, ni));
            }
        }
        #pragma unroll
        for (int i = 0; i < size<0>(tSrS); i += 2) {
            new_row_max(i) = new_row_max(i + 1) = max(new_row_max(i), new_row_max(i + 1));
        }
        #pragma unroll
        for (int mi = 0; mi < size<0>(tSrS); mi++) {
            new_row_max(mi) = warp_reduce_max<float, 4>(new_row_max(mi));
        }

        auto rescale = make_tensor<float>(Shape<Int<size<0>(tSrS)>>{});
        #pragma unroll
        for (int mi = 0; mi < size<0>(tSrS); mi++) {
            float old_max = row_max(mi);
            float cur_max = max(old_max, new_row_max(mi));
            row_max(mi) = cur_max;
            rescale(mi) = exp2f((old_max - cur_max) * params.softmax_scale_log2);
            row_sum(mi) *= rescale(mi);
        }

        #pragma unroll
        for (int mi = 0; mi < size<0>(tOrO); mi++) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(tOrO); ni++) {
                tOrO(mi, 0, ni) *= rescale(mi);
            }
        }

        #pragma unroll
        for (int mi = 0; mi < size<0>(tSrS); mi++) {
            float max_scaled = row_max(mi) * params.softmax_scale_log2;
            #pragma unroll
            for (int ni = 0; ni < size<2>(tSrS); ni++) {
                tSrS(mi, 0, ni) = exp2f(__fmaf_rn(tSrS(mi, 0, ni), params.softmax_scale_log2, -max_scaled));
                row_sum(mi) += tSrS(mi, 0, ni);
            }
        }

        cp_async_wait<0>();
        __syncthreads();

        if (n_block + 1 < n_block_max) {
            auto gK_next = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(n_block + 1, 0));
            cute::copy(g2s_tiled_copy, g2s_thr_copy.partition_S(gK_next), g2s_thr_copy.partition_D(sK));
            cp_async_fence();
        }

        auto tSrS_elem = make_tensor_like<Element>(tSrS);
        #pragma unroll
        for (int i = 0; i < size(tSrS); i++) {
            tSrS_elem(i) = Element(tSrS(i));
        }

        auto tOrP = tSrS_elem.compose(c2a);

        auto tOrV = thr_mma.partition_fragment_B(sVt);
        auto tOrV_copy = s2r_thr_copy_V.retile_D(tOrV);

        cute::copy(s2r_tiled_copy_V, tOsV(_, _, 0), tOrV_copy(_, _, 0));
        #pragma unroll
        for (int ki = 0; ki < size<2>(tOrP); ki++) {
            if (ki + 1 < size<2>(tOrP)) {
                cute::copy(s2r_tiled_copy_V, tOsV(_, _, ki + 1), tOrV_copy(_, _, ki + 1));
            }
            cute::gemm(tiled_mma, tOrP(_, _, ki), tOrV(_, _, ki), tOrO);
        }
    }

    #pragma unroll
    for (int i = 0; i < size<0>(tOrO); i += 2) {
        row_sum(i) = row_sum(i + 1) = row_sum(i) + row_sum(i + 1);
    }
    #pragma unroll
    for (int mi = 0; mi < size<0>(tOrO); mi++) {
        row_sum(mi) = warp_reduce_sum<float, 4>(row_sum(mi));
    }
    #pragma unroll
    for (int mi = 0; mi < size<0>(tOrO); mi++) {
        float inv_sum = (row_sum(mi) == 0.f) ? 1.f : 1.f / row_sum(mi);
        #pragma unroll
        for (int ni = 0; ni < size<2>(tOrO); ni++) {
            tOrO(mi, 0, ni) *= inv_sum;
        }
    }

    auto tOrO_elem = make_tensor_like<Element>(tOrO);
    #pragma unroll
    for (int i = 0; i < size(tOrO); i++) {
        tOrO_elem(i) = Element(tOrO(i));
    }
    auto tCsO = thr_mma.partition_C(sQ);
    cute::copy(tOrO_elem, tCsO);
    __syncthreads();

    Smem2GmemTiledCopyO s2g_tiled_copy_O;
    auto s2g_thr_copy_O = s2g_tiled_copy_O.get_thread_slice(threadIdx.x);
    auto tOsO = s2g_thr_copy_O.partition_S(sQ);
    auto tOgO = s2g_thr_copy_O.partition_D(gO);
    cute::copy(s2g_tiled_copy_O, tOsO, tOgO);
}

} // namespace flash
