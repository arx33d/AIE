#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#include <malloc.h>
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#else
#include <immintrin.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#endif

namespace aie::kernel {

inline void* aligned_alloc_64(size_t bytes) {
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        ptr = nullptr;
    }
#endif
    if (ptr) {
        std::memset(ptr, 0, bytes);
    }
    return ptr;
}

inline void aligned_free_64(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

inline void prefetch_t0(const void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);
#elif defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}

} // namespace aie::kernel
