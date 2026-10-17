#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aie::profile {

struct HardwareSpecs {
    uint64_t total_ram_bytes{0};
    uint64_t available_ram_bytes{0};
    uint64_t l1d_cache_bytes{0};
    uint64_t l2_cache_bytes{0};
    uint64_t l3_cache_bytes{0};
    uint32_t cache_line_size{64};
    uint32_t logical_cores{1};
    uint32_t physical_cores{1};

    // Vector extensions
    bool has_avx2{false};
    bool has_avx512f{false};
    bool has_avx512bw{false};
    bool has_avx512vnni{false};
    bool has_avx512vbmi{false};
    bool has_neon{false};
    bool has_amx_tile{false};

    double estimated_dram_bandwidth_gbps{0.0};
};

class HardwareProfiler {
public:
    static HardwareSpecs discover();
    static void print_summary(const HardwareSpecs& specs);
};

} // namespace aie::profile
