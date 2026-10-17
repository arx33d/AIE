#include "profile/hardware.hpp"
#include <iostream>
#include <thread>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/sysinfo.h>
#include <cpuid.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace aie::profile {

namespace {

void query_cpuid(int leaf, int subleaf, int* regs) {
#if defined(_WIN32)
    __cpuidex(regs, leaf, subleaf);
#elif defined(__linux__) || defined(__APPLE__)
#if defined(__x86_64__) || defined(__i386__)
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#else
    (void)leaf;
    (void)subleaf;
    std::memset(regs, 0, sizeof(int) * 4);
#endif
#else
    (void)leaf;
    (void)subleaf;
    std::memset(regs, 0, sizeof(int) * 4);
#endif
}

} // namespace

HardwareSpecs HardwareProfiler::discover() {
    HardwareSpecs specs;
    specs.logical_cores = std::thread::hardware_concurrency();

#if defined(_WIN32)
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        specs.total_ram_bytes = mem_status.ullTotalPhys;
        specs.available_ram_bytes = mem_status.ullAvailPhys;
    }

    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &buffer_size);
    if (buffer_size > 0) {
        std::vector<uint8_t> buffer(buffer_size);
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
        if (GetLogicalProcessorInformationEx(RelationCache, info, &buffer_size)) {
            DWORD offset = 0;
            while (offset < buffer_size) {
                auto* current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
                if (current->Relationship == RelationCache) {
                    if (current->Cache.Level == 1 && (current->Cache.Type == CacheData || current->Cache.Type == CacheUnified)) {
                        specs.l1d_cache_bytes = current->Cache.CacheSize;
                        specs.cache_line_size = current->Cache.LineSize;
                    } else if (current->Cache.Level == 2) {
                        specs.l2_cache_bytes = current->Cache.CacheSize;
                    } else if (current->Cache.Level == 3) {
                        specs.l3_cache_bytes = current->Cache.CacheSize;
                    }
                }
                offset += current->Size;
            }
        }
    }
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        specs.total_ram_bytes = static_cast<uint64_t>(si.totalram) * si.mem_unit;
        specs.available_ram_bytes = static_cast<uint64_t>(si.freeram) * si.mem_unit;
    }
    long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    long line = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (l1 > 0) specs.l1d_cache_bytes = static_cast<uint64_t>(l1);
    if (l2 > 0) specs.l2_cache_bytes = static_cast<uint64_t>(l2);
    if (l3 > 0) specs.l3_cache_bytes = static_cast<uint64_t>(l3);
    if (line > 0) specs.cache_line_size = static_cast<uint32_t>(line);
#endif

    if (specs.l1d_cache_bytes == 0) specs.l1d_cache_bytes = 32 * 1024;
    if (specs.l2_cache_bytes == 0) specs.l2_cache_bytes = 512 * 1024;
    if (specs.l3_cache_bytes == 0) specs.l3_cache_bytes = 16 * 1024 * 1024;

#if defined(__x86_64__) || defined(_M_X64)
    int regs[4] = {0};
    query_cpuid(0, 0, regs);
    int max_leaf = regs[0];

    if (max_leaf >= 7) {
        query_cpuid(7, 0, regs);
        specs.has_avx2 = (regs[1] & (1 << 5)) != 0;
        specs.has_avx512f = (regs[1] & (1 << 16)) != 0;
        specs.has_avx512bw = (regs[1] & (1 << 30)) != 0;
        specs.has_avx512vbmi = (regs[2] & (1 << 1)) != 0;
        specs.has_avx512vnni = (regs[2] & (1 << 11)) != 0;
        specs.has_amx_tile = (regs[3] & (1 << 24)) != 0;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    specs.has_neon = true;
#endif

    specs.estimated_dram_bandwidth_gbps = (specs.logical_cores >= 16) ? 68.0 : 38.4;
    return specs;
}

void HardwareProfiler::print_summary(const HardwareSpecs& specs) {
    std::cout << "[Hardware Profile]\n"
              << "  Total RAM:         " << (specs.total_ram_bytes / (1024 * 1024)) << " MB\n"
              << "  Available RAM:     " << (specs.available_ram_bytes / (1024 * 1024)) << " MB\n"
              << "  L1D Cache:         " << (specs.l1d_cache_bytes / 1024) << " KB\n"
              << "  L2 Cache:          " << (specs.l2_cache_bytes / 1024) << " KB\n"
              << "  L3 Cache:          " << (specs.l3_cache_bytes / (1024 * 1024)) << " MB\n"
              << "  Cache Line:        " << specs.cache_line_size << " bytes\n"
              << "  Logical Cores:     " << specs.logical_cores << "\n"
              << "  AVX2:              " << (specs.has_avx2 ? "YES" : "NO") << "\n"
              << "  AVX-512 F/BW/VNNI: " << (specs.has_avx512vnni ? "YES" : (specs.has_avx512f ? "PARTIAL" : "NO")) << "\n"
              << "  ARM NEON:          " << (specs.has_neon ? "YES" : "NO") << "\n"
              << "  Bandwidth Est.:    " << specs.estimated_dram_bandwidth_gbps << " GB/s\n";
}

} // namespace aie::profile
