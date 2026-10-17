#include "kernel/lut_gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace aie::kernel {

void LutGemm::build_activation_lut_2bit(
    const float* __restrict__ activation,
    size_t K,
    float* __restrict__ lut_table
) {
    // 2-bit weights map to signed values {-2, -1, 0, 1}
    // For a pair of inputs (a0, a1), precalculate:
    // LUT[w0, w1] = a0 * w0 + a1 * w1 for w0, w1 in {0, 1, 2, 3}
    const float val_map[4] = {-2.0f, -1.0f, 0.0f, 1.0f};

    size_t num_pairs = K / 2;
    for (size_t p = 0; p < num_pairs; ++p) {
        float a0 = activation[2 * p];
        float a1 = activation[2 * p + 1];
        float* pair_lut = lut_table + (p * 16);

        for (uint8_t idx = 0; idx < 16; ++idx) {
            uint8_t w0_idx = idx & 0x03;
            uint8_t w1_idx = (idx >> 2) & 0x03;
            pair_lut[idx] = a0 * val_map[w0_idx] + a1 * val_map[w1_idx];
        }
    }
}

void LutGemm::forward_2bit(
    const float* __restrict__ lut_table,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    size_t N,
    size_t K,
    float* __restrict__ out
) {
    size_t num_pairs = K / 2;
    size_t bytes_per_row = K / 4;

    for (size_t row = 0; row < N; ++row) {
        const uint8_t* w_row = packed_w + (row * bytes_per_row);
        float acc = 0.0f;
        prefetch_t0(w_row + 64);

        size_t pair_idx = 0;
        size_t byte_idx = 0;

#if defined(__AVX2__)
        __m256 acc_vec = _mm256_setzero_ps();

        while (pair_idx + 8 <= num_pairs) {
            uint32_t b4;
            std::memcpy(&b4, w_row + byte_idx, sizeof(uint32_t));

            alignas(32) float vals[8];
            for (int i = 0; i < 8; ++i) {
                uint8_t nibble = (b4 >> (i * 4)) & 0x0F;
                vals[i] = lut_table[(pair_idx + i) * 16 + nibble];
            }

            __m256 v = _mm256_load_ps(vals);
            acc_vec = _mm256_add_ps(acc_vec, v);

            pair_idx += 8;
            byte_idx += 4;
        }

        alignas(32) float hsum[8];
        _mm256_store_ps(hsum, acc_vec);
        for (int i = 0; i < 8; ++i) {
            acc += hsum[i];
        }
#endif

        while (pair_idx < num_pairs) {
            uint8_t byte_val = w_row[byte_idx];
            uint8_t nibble0 = byte_val & 0x0F;
            uint8_t nibble1 = (byte_val >> 4) & 0x0F;

            acc += lut_table[pair_idx * 16 + nibble0];
            pair_idx++;

            if (pair_idx < num_pairs) {
                acc += lut_table[pair_idx * 16 + nibble1];
                pair_idx++;
            }
            byte_idx++;
        }

        float scale = scales[row];
        out[row] = acc * scale;
    }
}

void LutGemm::forward_3bit(
    const float* __restrict__ activation,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    size_t N,
    size_t K,
    float* __restrict__ out
) {
    const float val_map[8] = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};

    for (size_t row = 0; row < N; ++row) {
        float acc = 0.0f;
        size_t bit_offset = 0;
        const uint8_t* row_bytes = packed_w + (row * (K * 3 / 8));

        for (size_t col = 0; col < K; ++col) {
            size_t byte_index = bit_offset / 8;
            size_t in_byte_offset = bit_offset % 8;

            uint32_t window = static_cast<uint32_t>(row_bytes[byte_index]) |
                             (static_cast<uint32_t>(row_bytes[byte_index + 1]) << 8);
            uint8_t code = (window >> in_byte_offset) & 0x07;

            acc += activation[col] * val_map[code];
            bit_offset += 3;
        }
        out[row] = acc * scales[row];
    }
}

void LutGemm::forward_4bit(
    const float* __restrict__ activation,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    size_t N,
    size_t K,
    float* __restrict__ out
) {
    size_t bytes_per_row = K / 2;

    for (size_t row = 0; row < N; ++row) {
        const uint8_t* w_row = packed_w + (row * bytes_per_row);
        float acc = 0.0f;

        for (size_t col = 0; col < K; col += 2) {
            uint8_t byte_val = w_row[col / 2];
            int8_t v0 = static_cast<int8_t>((byte_val & 0x0F) - 8);
            int8_t v1 = static_cast<int8_t>(((byte_val >> 4) & 0x0F) - 8);

            acc += activation[col] * static_cast<float>(v0);
            acc += activation[col + 1] * static_cast<float>(v1);
        }
        out[row] = acc * scales[row];
    }
}

} // namespace aie::kernel
