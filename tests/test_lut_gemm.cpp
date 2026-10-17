#include "kernel/lut_gemm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>

int main() {
    constexpr size_t N = 16;
    constexpr size_t K = 32;

    float* act = static_cast<float*>(aie::kernel::aligned_alloc_64(K * sizeof(float)));
    float* lut = static_cast<float*>(aie::kernel::aligned_alloc_64((K / 2) * 16 * sizeof(float)));
    uint8_t* packed_w = static_cast<uint8_t*>(aie::kernel::aligned_alloc_64(N * (K / 4)));
    float* scales = static_cast<float*>(aie::kernel::aligned_alloc_64(N * sizeof(float)));
    float* out = static_cast<float*>(aie::kernel::aligned_alloc_64(N * sizeof(float)));

    for (size_t i = 0; i < K; ++i) act[i] = 1.0f;
    for (size_t i = 0; i < N; ++i) scales[i] = 1.0f;
    std::memset(packed_w, 0xAA, N * (K / 4));

    aie::kernel::LutGemm::build_activation_lut_2bit(act, K, lut);
    aie::kernel::LutGemm::forward_2bit(lut, packed_w, scales, N, K, out);

    for (size_t i = 0; i < N; ++i) {
        assert(!std::isnan(out[i]));
    }

    aie::kernel::aligned_free_64(act);
    aie::kernel::aligned_free_64(lut);
    aie::kernel::aligned_free_64(packed_w);
    aie::kernel::aligned_free_64(scales);
    aie::kernel::aligned_free_64(out);

    std::cout << "[PASS] test_lut_gemm completed successfully.\n";
    return 0;
}
