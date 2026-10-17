#pragma once

#include "kernel/simd_utils.hpp"
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>

namespace aie::stream {

constexpr size_t KV_BLOCK_TOKENS = 16;

/**
 * Paged Asymmetric KV Cache (KIVI Style):
 * - Key Cache: Quantized per-channel to 2-bit with FP16 scale and bias vectors.
 * - Value Cache: Quantized per-token to 4-bit with FP16 scale and bias vectors.
 */
struct PhysicalKVBlock {
    // Keys: 16 tokens x 2-bit per channel = 4 bytes per channel
    // Values: 16 tokens x 4-bit per channel = 8 bytes per channel
    uint8_t* key_data{nullptr};      // [head_dim * (KV_BLOCK_TOKENS * 2 / 8)]
    float* key_scales{nullptr};      // [head_dim]
    float* key_biases{nullptr};      // [head_dim]

    uint8_t* val_data{nullptr};      // [KV_BLOCK_TOKENS * (head_dim * 4 / 8)]
    float* val_scales{nullptr};      // [KV_BLOCK_TOKENS]
    float* val_biases{nullptr};      // [KV_BLOCK_TOKENS]
};

class PagedKVCache {
public:
    PagedKVCache(
        uint32_t num_layers,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t max_total_blocks
    );
    ~PagedKVCache();

    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;

    /**
     * Appends single token Key and Value into the paged cache.
     * ZERO DYNAMIC ALLOCATIONS in the decode loop.
     */
    void append_token(
        uint32_t seq_id,
        uint32_t layer_idx,
        uint32_t kv_head_idx,
        const float* __restrict__ key,
        const float* __restrict__ value
    );

    /**
     * Computes attention context vector over paged asymmetric blocks.
     */
    void compute_attention(
        uint32_t seq_id,
        uint32_t layer_idx,
        uint32_t head_idx,
        const float* __restrict__ query,
        float* __restrict__ out_context
    ) const;

    [[nodiscard]] size_t get_context_length(uint32_t seq_id) const;

private:
    uint32_t allocate_block();
    void free_sequence(uint32_t seq_id);

    uint32_t num_layers_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    uint32_t max_blocks_;

    size_t key_bytes_per_block_;
    size_t val_bytes_per_block_;

    uint8_t* memory_pool_{nullptr};
    std::vector<PhysicalKVBlock> blocks_;
    std::vector<uint32_t> free_block_indices_;

    struct SequenceEntry {
        uint32_t token_count{0};
        std::vector<uint32_t> block_ids;
    };
    std::unordered_map<uint32_t, SequenceEntry> sequences_;
};

} // namespace aie::stream
