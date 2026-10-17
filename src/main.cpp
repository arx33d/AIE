#include "profile/hardware.hpp"
#include "profile/safetensors_parser.hpp"
#include "kernel/lut_gemm.hpp"
#include "stream/paged_kv_cache.hpp"
#include "serve/c_api.h"
#include "serve/openai_server.hpp"
#include "serve/ollama_emulator.hpp"

#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>

void run_benchmarks() {
    std::cout << "\n=== Running Low-Bit Kernels Benchmark ===\n";
    constexpr size_t N = 4096;
    constexpr size_t K = 4096;

    float* act = static_cast<float*>(aie::kernel::aligned_alloc_64(K * sizeof(float)));
    float* lut = static_cast<float*>(aie::kernel::aligned_alloc_64((K / 2) * 16 * sizeof(float)));
    uint8_t* w2 = static_cast<uint8_t*>(aie::kernel::aligned_alloc_64((N * K / 4)));
    float* scales = static_cast<float*>(aie::kernel::aligned_alloc_64(N * sizeof(float)));
    float* out = static_cast<float*>(aie::kernel::aligned_alloc_64(N * sizeof(float)));

    for (size_t i = 0; i < K; ++i) act[i] = 0.5f;
    for (size_t i = 0; i < N; ++i) scales[i] = 0.02f;

    aie::kernel::LutGemm::build_activation_lut_2bit(act, K, lut);

    auto t0 = std::chrono::high_resolution_clock::now();
    constexpr int ITERS = 100;
    for (int it = 0; it < ITERS; ++it) {
        aie::kernel::LutGemm::forward_2bit(lut, w2, scales, N, K, out);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    std::cout << "  LUT-GEMM 2-bit [4096 x 4096]: " << ms << " ms ("
              << (2.0 * N * K / (ms * 1e6)) << " GFLOPS equivalent)\n";

    aie::kernel::aligned_free_64(act);
    aie::kernel::aligned_free_64(lut);
    aie::kernel::aligned_free_64(w2);
    aie::kernel::aligned_free_64(scales);
    aie::kernel::aligned_free_64(out);
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n"
              << "   AIE: Adaptive Inference Engine v1.0.0 (C++20/Python)  \n"
              << "========================================================\n";

    auto specs = aie::profile::HardwareProfiler::discover();
    aie::profile::HardwareProfiler::print_summary(specs);

    if (argc >= 2 && std::string(argv[1]) == "--benchmark") {
        run_benchmarks();
        return 0;
    }

    if (argc >= 2 && std::string(argv[1]) == "--shrink") {
        std::string model_name = (argc >= 3) ? argv[2] : "dolphin3:8b";
        std::string target_ram = "3.5";
        for (int i = 3; i < argc - 1; ++i) {
            if (std::string(argv[i]) == "--ram") {
                target_ram = argv[i + 1];
            }
        }
        std::cout << "\n[*] Launching Automated Model Weight Compression Pipeline for: " << model_name << "\n";
        std::string cmd = "python python/tools/optimize_ollama_model.py --model " + model_name + " --target-ram " + target_ram;
        int status = std::system(cmd.c_str());
        return status;
    }

    if (argc >= 3 && std::string(argv[1]) == "--profile") {
        std::string model_file = argv[2];
        std::cout << "\n[*] Zero-copy inspecting safetensors: " << model_file << "\n";
        aie::profile::SafetensorsParser parser(model_file);
        auto topo = parser.parse_topology();
        std::cout << "  Architecture: " << (topo.arch == aie::profile::ModelArchitecture::SparseMoE ? "Sparse MoE" : "Dense Transformer") << "\n"
                  << "  Layers:       " << topo.num_layers << "\n"
                  << "  Hidden Size:  " << topo.hidden_size << "\n"
                  << "  Vocab Size:   " << topo.vocab_size << "\n"
                  << "  Weights Size: " << (topo.total_tensor_bytes / (1024 * 1024)) << " MB\n";
        return 0;
    }

    if (argc >= 3 && std::string(argv[1]) == "--serve") {
        std::string model_file = argv[2];
        aie_config_t config;
        config.ram_budget_bytes = specs.available_ram_bytes;
        config.num_threads = specs.logical_cores;
        config.enable_out_of_core = 1;
        config.max_context_tokens = 8192;
        config.nvme_cache_dir = "./nvme_cache";

        aie_engine_t* engine = aie_init(&config);
        aie_load_model(engine, model_file.c_str(), nullptr);

        aie::serve::OpenAIServer oai_server(engine, 8080);
        aie::serve::OllamaEmulator ollama_server(engine, 11434);

        oai_server.start();
        ollama_server.start();

        std::cout << "\n[+] AIE Unified Runtime Active. Press Ctrl+C to terminate.\n";
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        aie_free(engine);
        return 0;
    }

    std::cout << "\nCommands:\n"
              << "  --shrink <ollama-model> [--ram 3.5] Automatically compress installed model to target RAM\n"
              << "  --benchmark                         Run microbenchmarks for LUT-GEMM and Paged KV\n"
              << "  --profile <model.safetensors>       Inspect safetensors graph and detect topology\n"
              << "  --serve <model.safetensors>         Run OpenAI (:8080) and Ollama (:11434) servers\n";
    return 0;
}
