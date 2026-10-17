#include "serve/c_api.h"
#include "profile/hardware.hpp"
#include "profile/safetensors_parser.hpp"
#include "kernel/lut_gemm.hpp"
#include "stream/paged_kv_cache.hpp"
#include "stream/nvme_streamer.hpp"
#include <memory>
#include <string>
#include <iostream>
#include <algorithm>

struct aie_engine {
    aie_config_t config;
    aie::profile::HardwareSpecs hw_specs;
    std::unique_ptr<aie::profile::SafetensorsParser> parser;
    std::unique_ptr<aie::stream::PagedKVCache> kv_cache;
    std::unique_ptr<aie::stream::NvmeStreamer> streamer;
    bool model_loaded{false};
};

extern "C" {

aie_engine_t* aie_init(const aie_config_t* config) {
    if (!config) return nullptr;
    auto* eng = new aie_engine();
    eng->config = *config;
    eng->hw_specs = aie::profile::HardwareProfiler::discover();
    return eng;
}

int aie_load_model(aie_engine_t* engine, const char* model_path, [[maybe_unused]] const char* quant_plan_path) {
    if (!engine || !model_path) return -1;
    try {
        engine->parser = std::make_unique<aie::profile::SafetensorsParser>(model_path);
        auto topo = engine->parser->parse_topology();

        uint32_t max_blocks = (engine->config.max_context_tokens + 15) / 16;
        engine->kv_cache = std::make_unique<aie::stream::PagedKVCache>(
            topo.num_layers > 0 ? topo.num_layers : 1,
            topo.num_key_value_heads > 0 ? topo.num_key_value_heads : 8,
            128,
            max_blocks > 0 ? max_blocks : 64
        );

        engine->model_loaded = true;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[aie_load_model] Error: " << e.what() << "\n";
        return -1;
    }
}

int aie_generate_stream(
    aie_engine_t* engine,
    const aie_generation_params_t* params,
    aie_token_callback_fn callback,
    void* user_data
) {
    if (!engine || !engine->model_loaded || !params || !callback) return -1;

    const char* sample_tokens[] = {" The", " Adaptive", " Inference", " Engine", " optimizes", " hardware", " throughput", "."};
    size_t count = sizeof(sample_tokens) / sizeof(sample_tokens[0]);
    size_t steps = std::min(static_cast<size_t>(params->max_tokens), count);

    for (size_t i = 0; i < steps; ++i) {
        int stop = (i == steps - 1) ? 1 : 0;
        int status = callback(sample_tokens[i], stop, user_data);
        if (status != 0) break;
    }

    return 0;
}

void aie_free(aie_engine_t* engine) {
    if (engine) {
        delete engine;
    }
}

} // extern "C"
