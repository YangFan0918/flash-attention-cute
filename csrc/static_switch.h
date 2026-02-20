#pragma once

/// Dispatch based on boolean (compile-time specialization)
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
    [&] {                                        \
        if (COND) {                              \
            constexpr bool CONST_NAME = true;    \
            return __VA_ARGS__();                 \
        } else {                                 \
            constexpr bool CONST_NAME = false;   \
            return __VA_ARGS__();                 \
        }                                        \
    }()

/// Dispatch based on fp16 vs bf16
#define FP16_SWITCH(IS_FP16, ...)                              \
    [&] {                                                       \
        if (IS_FP16) {                                          \
            using elem_type = cutlass::half_t;                  \
            return __VA_ARGS__();                                \
        } else {                                                \
            using elem_type = cutlass::bfloat16_t;              \
            return __VA_ARGS__();                                \
        }                                                       \
    }()

/// Dispatch based on head dimension
#define HEAD_DIM_SWITCH(HEAD_DIM, ...)                          \
    [&] {                                                       \
        if (HEAD_DIM == 64) {                                   \
            constexpr int kHeadDim = 64;                        \
            return __VA_ARGS__();                                \
        } else if (HEAD_DIM == 128) {                           \
            constexpr int kHeadDim = 128;                       \
            return __VA_ARGS__();                                \
        } else {                                                \
            assert(false && "Unsupported head_dim");            \
        }                                                       \
    }()
