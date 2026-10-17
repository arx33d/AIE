#include "stream/nvme_streamer.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace aie::stream {

NvmeStreamer::NvmeStreamer(std::string weight_file, size_t max_layer_bytes)
    : weight_file_(std::move(weight_file)), max_layer_bytes_(max_layer_bytes) {

    size_t aligned_size = ((max_layer_bytes + 4095) / 4096) * 4096;
    buffers_[0] = static_cast<uint8_t*>(kernel::aligned_alloc_64(aligned_size));
    buffers_[1] = static_cast<uint8_t*>(kernel::aligned_alloc_64(aligned_size));

    if (!buffers_[0] || !buffers_[1]) {
        throw std::bad_alloc();
    }

#if defined(_WIN32)
    file_handle_ = CreateFileA(
        weight_file_.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("NvmeStreamer: failed to open " + weight_file_);
    }
#else
    file_fd_ = open(weight_file_.c_str(), O_RDONLY);
    if (file_fd_ < 0) {
        throw std::runtime_error("NvmeStreamer: failed to open " + weight_file_);
    }
#endif

    for (size_t i = 0; i < 2; ++i) {
        ready_[i].store(true);
        worker_threads_[i] = std::thread(&NvmeStreamer::io_worker_loop, this, i);
    }
}

NvmeStreamer::~NvmeStreamer() {
    stop_flag_.store(true);
    for (size_t i = 0; i < 2; ++i) {
        {
            std::lock_guard<std::mutex> lock(cv_m_[i]);
            active_reqs_[i].active = false;
        }
        cv_[i].notify_one();
        if (worker_threads_[i].joinable()) {
            worker_threads_[i].join();
        }
    }

#if defined(_WIN32)
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
    }
#else
    if (file_fd_ >= 0) {
        close(file_fd_);
    }
#endif

    kernel::aligned_free_64(buffers_[0]);
    kernel::aligned_free_64(buffers_[1]);
}

void NvmeStreamer::submit_prefetch(size_t buffer_id, uint64_t file_offset, size_t bytes) {
    size_t idx = buffer_id % 2;
    wait_prefetch(idx);

    {
        std::lock_guard<std::mutex> lock(cv_m_[idx]);
        active_reqs_[idx].buffer_id = idx;
        active_reqs_[idx].offset = file_offset;
        active_reqs_[idx].bytes = bytes;
        active_reqs_[idx].active = true;
        ready_[idx].store(false);
    }
    cv_[idx].notify_one();
}

void NvmeStreamer::wait_prefetch(size_t buffer_id) {
    size_t idx = buffer_id % 2;
    while (!ready_[idx].load()) {
        std::this_thread::yield();
    }
}

void NvmeStreamer::io_worker_loop(size_t buffer_id) {
    while (!stop_flag_.load()) {
        Request req;
        {
            std::unique_lock<std::mutex> lock(cv_m_[buffer_id]);
            cv_[buffer_id].wait(lock, [&]() {
                return active_reqs_[buffer_id].active || stop_flag_.load();
            });

            if (stop_flag_.load()) break;
            req = active_reqs_[buffer_id];
            active_reqs_[buffer_id].active = false;
        }

#if defined(_WIN32)
        OVERLAPPED ov;
        std::memset(&ov, 0, sizeof(ov));
        ov.Offset = static_cast<DWORD>(req.offset & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>((req.offset >> 32) & 0xFFFFFFFF);
        DWORD bytes_read = 0;
        ReadFile(file_handle_, buffers_[buffer_id], static_cast<DWORD>(req.bytes), &bytes_read, &ov);
#else
        ssize_t res = pread(file_fd_, buffers_[buffer_id], req.bytes, req.offset);
        (void)res;
#endif

        ready_[buffer_id].store(true);
    }
}

} // namespace aie::stream
