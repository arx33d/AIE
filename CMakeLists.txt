cmake_minimum_required(VERSION 3.20)
project(adaptive_inference_engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization and warning flags
if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /O2 /arch:AVX2)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX AIE_EXPORT)
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -O3
        -march=native
        -fPIC
    )
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core engine static library
add_library(aie_core STATIC
    src/profile/hardware.cpp
    src/profile/safetensors_parser.cpp
    src/kernel/lut_gemm.cpp
    src/stream/paged_kv_cache.cpp
    src/stream/nvme_streamer.cpp
    src/serve/c_api.cpp
    src/serve/openai_server.cpp
    src/serve/ollama_emulator.cpp
)

if(WIN32)
    target_link_libraries(aie_core PUBLIC ws2_32)
else()
    target_link_libraries(aie_core PUBLIC pthread)
endif()

# Standalone dynamic C-ABI library (libaie.so / aie.dll)
add_library(aie_shared SHARED
    src/serve/c_api.cpp
)
set_target_properties(aie_shared PROPERTIES OUTPUT_NAME "aie")
target_link_libraries(aie_shared PRIVATE aie_core)

# Main CLI Executable
add_executable(aie src/main.cpp)
target_link_libraries(aie PRIVATE aie_core)

# Tests
enable_testing()
add_subdirectory(tests)
