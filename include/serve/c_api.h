#ifndef AIE_C_API_H
#define AIE_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
    #if defined(AIE_EXPORT)
        #define AIE_API __declspec(dllexport)
    #else
        #define AIE_API __declspec(dllimport)
    #endif
#else
    #define AIE_API __attribute__((visibility("default")))
#endif

typedef struct aie_engine aie_engine_t;

typedef struct {
    uint64_t ram_budget_bytes;
    int32_t num_threads;
    const char* nvme_cache_dir;
    int32_t enable_out_of_core;
    uint32_t max_context_tokens;
} aie_config_t;

typedef struct {
    const char* prompt;
    uint32_t max_tokens;
    float temperature;
    float top_p;
} aie_generation_params_t;

typedef int (*aie_token_callback_fn)(const char* token_str, int is_finished, void* user_data);

AIE_API aie_engine_t* aie_init(const aie_config_t* config);

AIE_API int aie_load_model(aie_engine_t* engine, const char* model_path, const char* quant_plan_path);

AIE_API int aie_generate_stream(
    aie_engine_t* engine,
    const aie_generation_params_t* params,
    aie_token_callback_fn callback,
    void* user_data
);

AIE_API void aie_free(aie_engine_t* engine);

#ifdef __cplusplus
}
#endif

#endif // AIE_C_API_H
