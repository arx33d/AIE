# Adaptive Inference Engine (AIE)

Autonomous, hardware-aware compiler and local execution runtime in C++20 and Python. AIE takes open-weights models in `.safetensors` format (Dense or Sparse MoE) and optimizes them to execute at maximum token throughput within a user-specified physical RAM budget.

---

## Key Features

1. **Hardware & Topology Discovery (`libprofile`)**
   - Zero-copy `mmap` inspection of `.safetensors` header metadata.
   - Detects cache topology (L1D/L2/L3), DRAM bandwidth, and SIMD features (AVX2, AVX-512, NEON, AMX).
2. **Empirical Fisher Sensitivity Profiler (`libeval`)**
   - Computes empirical Fisher Information Matrix diagonal (Hessian trace proxy) over calibration sets.
   - Protects fragile layers (embeddings, attention heads, LM head) while marking feed-forward and MoE blocks for low-bit quantization.
3. **MILP Pareto Knapsack Solver (`libsolve`)**
   - Solves multi-choice precision knapsack problem via HiGHS / `scipy.optimize.milp` in under 500ms.
   - Automatically partitions layers between pinned RAM and Out-of-Core direct NVMe DMA streaming if the model exceeds physical memory.
4. **T-MAC Style Look-Up Table GEMM (`libkernel`)**
   - Zero runtime dequantization in inner matrix loops.
   - Evaluates dot products by indexing activation lookup tables using vector shuffle instructions and SIMD addition.
5. **Paged Asymmetric KV Cache (KIVI Style)**
   - Per-channel 2-bit quantization for Keys.
   - Per-token 4-bit quantization for Values.
   - Zero dynamic memory allocations during token decoding.
6. **Third-Party Local Ecosystem Integration**
   - **OpenAI HTTP Daemon**: Listening on `:8080` with Server-Sent Events (SSE) streaming (`/v1/chat/completions`).
   - **Ollama Wire Protocol**: Listening on `:11434` with `/api/chat`, `/api/generate`, and `/api/tags`.
   - **GGUF Export**: Generates GGUF v3 binaries and accompanying `Modelfile` for Ollama CLI.
   - **C-ABI Dynamic Library**: `libaie.so` / `aie.dll` for zero-network embedded desktop applications.

---

## Directory Structure

```text
adaptive-inference-engine/
├── CMakeLists.txt
├── include/
│   ├── profile/
│   │   ├── hardware.hpp
│   │   └── safetensors_parser.hpp
│   ├── kernel/
│   │   ├── lut_gemm.hpp
│   │   └── simd_utils.hpp
│   ├── stream/
│   │   ├── paged_kv_cache.hpp
│   │   └── nvme_streamer.hpp
│   └── serve/
│       ├── c_api.h
│       ├── openai_server.hpp
│       └── ollama_emulator.hpp
├── src/
│   ├── profile/
│   │   ├── hardware.cpp
│   │   └── safetensors_parser.cpp
│   ├── kernel/
│   │   └── lut_gemm.cpp
│   ├── stream/
│   │   ├── paged_kv_cache.cpp
│   │   └── nvme_streamer.cpp
│   ├── serve/
│   │   ├── c_api.cpp
│   │   ├── openai_server.cpp
│   │   └── ollama_emulator.cpp
│   └── main.cpp
├── python/
│   ├── eval/
│   │   └── sensitivity_profiler.py
│   ├── solver/
│   │   └── pareto_allocator.py
│   └── export/
│       └── gguf_exporter.py
└── tests/
    ├── CMakeLists.txt
    ├── test_safetensors.cpp
    ├── test_lut_gemm.cpp
    └── test_paged_kv.cpp
```

---

## Build & Test

### Prerequisites
- CMake >= 3.20
- C++20 compliant compiler (GCC 11+, Clang 13+, MSVC 2022+)
- Python 3.9+ (with `numpy`, `scipy`)

### Building
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Running Tests
```bash
ctest --test-dir build -C Release --output-on-failure
```

---

## Usage

### 1. Benchmark Kernels
```bash
./build/Release/aie --benchmark
```

### 2. Inspect Model Metadata
```bash
./build/Release/aie --profile /path/to/model.safetensors
```

### 3. Sensitivity Profiling & Pareto Solving
```bash
python python/eval/sensitivity_profiler.py model.safetensors sensitivity.json
python python/solver/pareto_allocator.py sensitivity.json 12.0 quant_plan.json
```

### 4. GGUF Export for Ollama CLI
```bash
python python/export/gguf_exporter.py llama-27b quant_plan.json llama-27b-aie.gguf
ollama create llama-27b -f llama-27b-aie.Modelfile
```

### 5. Launch Serving Daemon
```bash
./build/Release/aie --serve /path/to/model.safetensors
```
- OpenAI API: `http://localhost:8080/v1/chat/completions`
- Ollama API: `http://localhost:11434/api/chat`
