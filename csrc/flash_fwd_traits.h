#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>
#include <cute/layout.hpp>

using namespace cute;

namespace flash {

template<int kHeadDim>
struct SwizzleSelector {
    using type = std::conditional_t<kHeadDim <= 64,
        Swizzle<3, 3, 3>,
        Swizzle<3, 3, 4>
    >;
};

template<typename Element_, int kHeadDim_, int kBlockM_ = 64, int kBlockN_ = 64>
struct FlashFwdTraits {
    using Element = Element_;
    static constexpr int kHeadDim  = kHeadDim_;
    static constexpr int kBlockM   = kBlockM_;
    static constexpr int kBlockN   = kBlockN_;
    static constexpr int kNWarps   = 4;
    static constexpr int kNThreads = kNWarps * 32;

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        SM80_16x8x16_F32F16F16F32_TN,
        SM80_16x8x16_F32BF16BF16F32_TN
    >;
    using TiledMma = TiledMMA<
        MMA_Atom<MMA_Atom_Arch>,
        Layout<Shape<Int<kNWarps>, _1, _1>>
    >;

    using SmemSwizzle = typename SwizzleSelector<kHeadDim>::type;

    using SmemLayoutAtom = decltype(composition(
        SmemSwizzle{},
        Layout<Shape<_8, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{}
    ));
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = SmemLayoutK;

    using SmemLayoutAtomVt = decltype(composition(
        SmemSwizzle{},
        Layout<Shape<Int<kHeadDim>, _8>, Stride<_1, Int<kHeadDim>>>{}
    ));
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomVt{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    using Gmem2SmemTiledCopyQKV = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
        Layout<Shape<Int<kNThreads / (kHeadDim / 8)>, Int<kHeadDim / 8>>,
               Stride<Int<kHeadDim / 8>, _1>>{},
        Layout<Shape<_1, _8>>{}
    ));

    using Smem2GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{},
        Layout<Shape<Int<kNThreads / (kHeadDim / 8)>, Int<kHeadDim / 8>>,
               Stride<Int<kHeadDim / 8>, _1>>{},
        Layout<Shape<_1, _8>>{}
    ));

    using Smem2RegCopyAtomA  = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using Smem2RegCopyAtomB  = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using Smem2RegCopyAtomBt = Copy_Atom<SM75_U16x4_LDSM_T, Element>;

    static constexpr int kSmemSize = (kBlockM + 2 * kBlockN) * kHeadDim * sizeof(Element);
};

} // namespace flash
