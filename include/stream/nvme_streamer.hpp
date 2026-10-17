#pragma once

#include "kernel/simd_utils.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <mutex>

namespace aie::stream {

/**
 * High-performance double-buffered Out-Of-Core Direct NVMe DMA streamer.
 * While Layer N is computing in Buffer A, Layer N+1 is fetched asynchronously
 * into Buffer B.
 */
class NvmeStreamer {
public:
    NvmeStreamer(std::string weight_file, size_t max_layer_bytes);
    ~NvmeStreamer();

    NvmeStreamer(const NvmeStreamer&) = delete;
    NvmeStreamer& operator=(const NvmeStreamer&) = delete;

    /**
     * Submits an asynchronous read request for layer data.
     */
    void submit_prefetch(size_t buffer_id, uint64_t file_offset, size_t bytes);

    /**
     * Waits until the requested buffer has completed loading.
     */
    void wait_prefetch(size_t buffer_id);

    [[nodiscard]] uint8_t* get_buffer(size_t buffer_id) const {
        return buffers_[buffer_id % 2];
    }

private:
    std::string weight_file_;
    size_t max_layer_bytes_;
    uint8_t* buffers_[2]{nullptr, nullptr};

#if defined(_WIN32)
    void* file_handle_{nullptr};
#else
    int file_fd_{-1};
#endif

    struct Request {
        size_t buffer_id{0};
        uint64_t offset{0};
        size_t bytes{0};
        bool active{false};
    };

    Request active_reqs_[2];
    std::mutex cv_m_[2];
    std::condition_variable cv_[2];
    std::atomic<bool> ready_[2]{true, true};
    std::atomic<bool> stop_flag_{false};
    std::thread worker_threads_[2];

    void io_worker_loop(size_t buffer_id);
};

} // namespace aie::stream
