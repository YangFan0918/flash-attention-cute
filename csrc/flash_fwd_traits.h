#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>
#include <cute/layout.hpp>

using namespace cute;

namespace flash {

// Helper: select swizzle pattern based on head_dim
template<int kHeadDim>
struct SwizzleSelector {
    // head_dim=64 → row=128B → Swizzle<3,3,3>
    // head_dim=128 → row=256B → Swizzle<3,3,4>
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

    // ======================== TiledMMA ========================
    // SM80 MMA atom: 16x8x16, one warp
    // AtomLayout<_4,_1,_1>: 4 warps tile M → covers 64 rows in M
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        SM80_16x8x16_F32F16F16F32_TN,
        SM80_16x8x16_F32BF16BF16F32_TN
    >;
    using TiledMma = TiledMMA<
        MMA_Atom<MMA_Atom_Arch>,
        Layout<Shape<Int<kNWarps>, _1, _1>>  // 4 atoms in M direction
    >;

    // ======================== SmemLayout ========================
    // Compose swizzle into layout to avoid bank conflicts
    using SmemSwizzle = typename SwizzleSelector<kHeadDim>::type;

    // Atom: 8 rows x kHeadDim cols, row-major, with swizzle
    using SmemLayoutAtom = decltype(composition(
        SmemSwizzle{},
        Layout<Shape<_8, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{}
    ));
    // Tile atom to full block size
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = SmemLayoutK;

    // V transposed layout: (kHeadDim, kBlockN) — column-major view of same V data
    // Element (d, n) → offset d + n*kHeadDim, same physical location as (n, d) in SmemLayoutV
    using SmemLayoutAtomVt = decltype(composition(
        SmemSwizzle{},
        Layout<Shape<Int<kHeadDim>, _8>, Stride<_1, Int<kHeadDim>>>{}
    ));
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomVt{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    // ======================== TiledCopy: gmem ↔ smem ========================
    // gmem → smem: async copy, 128 bits per thread
    using Gmem2SmemTiledCopyQKV = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
        Layout<Shape<Int<kNThreads / (kHeadDim / 8)>, Int<kHeadDim / 8>>,
               Stride<Int<kHeadDim / 8>, _1>>{},
        Layout<Shape<_1, _8>>{}
    ));

    // smem → gmem: O write-back, same layout but non-async
    using Smem2GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{},
        Layout<Shape<Int<kNThreads / (kHeadDim / 8)>, Int<kHeadDim / 8>>,
               Stride<Int<kHeadDim / 8>, _1>>{},
        Layout<Shape<_1, _8>>{}
    ));

    // ======================== SmemTiledCopy ========================
    // smem → register copy, matched to TiledMMA's expected layout
    using Smem2RegCopyAtomA  = Copy_Atom<SM75_U32x4_LDSM_N, Element>;  // A operand: 4×u32 = 8 halfs/thread
    using Smem2RegCopyAtomB  = Copy_Atom<SM75_U32x2_LDSM_N, Element>;  // B operand: 2×u32 = 4 halfs/thread
    using Smem2RegCopyAtomBt = Copy_Atom<SM75_U16x4_LDSM_T, Element>;  // B operand transposed

    // ======================== Shared memory size ========================
    static constexpr int kSmemSize = (kBlockM + 2 * kBlockN) * kHeadDim * sizeof(Element);
};

} // namespace flash
