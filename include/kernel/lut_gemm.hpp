#pragma once

#include "kernel/simd_utils.hpp"
#include <cstdint>
#include <vector>

namespace aie::kernel {

constexpr size_t LUT_GROUP_SIZE = 2;
constexpr size_t LUT_TABLE_ENTRIES = 16; // 4^2 = 16 linear combinations for 2-bit weights

class LutGemm {
public:
    LutGemm() = default;

    /**
     * Precomputes the Look-Up Table of linear combinations for activation vector X.
     * The table resides in the L1 cache / registers.
     * Table dimension: [K / 2, 16] float values.
     */
    static void build_activation_lut_2bit(
        const float* __restrict__ activation,
        size_t K,
        float* __restrict__ lut_table
    );

    /**
     * Executes Look-Up Table Matrix-Vector Multiply (T-MAC style).
     * Y = W * X
     * The inner loop operates strictly via vector table shuffles and additions.
     * NO RUNTIME DEQUANTIZATION OF WEIGHTS TO FLOATS IN REGISTERS.
     *
     * @param lut_table Precomputed activation table [K / 2, 16]
     * @param packed_w Packed 2-bit weights: 4 weights per byte [N, K / 4]
     * @param scales Block scales for weights [N, K / 32]
     * @param N Output rows
     * @param K Input dimension (must be multiple of 64)
     * @param out Output destination vector Y [N]
     */
    static void forward_2bit(
        const float* __restrict__ lut_table,
        const uint8_t* __restrict__ packed_w,
        const float* __restrict__ scales,
        size_t N,
        size_t K,
        float* __restrict__ out
    );

    /**
     * 3-bit LUT-GEMM Kernel.
     */
    static void forward_3bit(
        const float* __restrict__ activation,
        const uint8_t* __restrict__ packed_w,
        const float* __restrict__ scales,
        size_t N,
        size_t K,
        float* __restrict__ out
    );

    /**
     * 4-bit LUT-GEMM Kernel.
     */
    static void forward_4bit(
        const float* __restrict__ activation,
        const uint8_t* __restrict__ packed_w,
        const float* __restrict__ scales,
        size_t N,
        size_t K,
        float* __restrict__ out
    );
};

} // namespace aie::kernel
