#include "stream/paged_kv_cache.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace aie::stream {

PagedKVCache::PagedKVCache(
    uint32_t num_layers,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_total_blocks
) : num_layers_(num_layers),
    num_kv_heads_(num_kv_heads),
    head_dim_(head_dim),
    max_blocks_(max_total_blocks) {

    key_bytes_per_block_ = head_dim_ * (KV_BLOCK_TOKENS * 2 / 8); // 4 bytes/channel
    val_bytes_per_block_ = KV_BLOCK_TOKENS * (head_dim_ * 4 / 8); // 8 bytes/channel

    size_t bytes_per_block_total = key_bytes_per_block_ + (head_dim_ * sizeof(float) * 2) +
                                   val_bytes_per_block_ + (KV_BLOCK_TOKENS * sizeof(float) * 2);

    memory_pool_ = static_cast<uint8_t*>(kernel::aligned_alloc_64(bytes_per_block_total * max_blocks_));
    if (!memory_pool_) {
        throw std::bad_alloc();
    }

    blocks_.resize(max_blocks_);
    free_block_indices_.reserve(max_blocks_);

    uint8_t* cursor = memory_pool_;
    for (uint32_t i = 0; i < max_blocks_; ++i) {
        blocks_[i].key_data = cursor;
        cursor += key_bytes_per_block_;

        blocks_[i].key_scales = reinterpret_cast<float*>(cursor);
        cursor += head_dim_ * sizeof(float);

        blocks_[i].key_biases = reinterpret_cast<float*>(cursor);
        cursor += head_dim_ * sizeof(float);

        blocks_[i].val_data = cursor;
        cursor += val_bytes_per_block_;

        blocks_[i].val_scales = reinterpret_cast<float*>(cursor);
        cursor += KV_BLOCK_TOKENS * sizeof(float);

        blocks_[i].val_biases = reinterpret_cast<float*>(cursor);
        cursor += KV_BLOCK_TOKENS * sizeof(float);

        free_block_indices_.push_back(i);
    }
}

PagedKVCache::~PagedKVCache() {
    if (memory_pool_) {
        kernel::aligned_free_64(memory_pool_);
        memory_pool_ = nullptr;
    }
}

uint32_t PagedKVCache::allocate_block() {
    if (free_block_indices_.empty()) {
        throw std::runtime_error("PagedKVCache: Out of physical memory blocks!");
    }
    uint32_t idx = free_block_indices_.back();
    free_block_indices_.pop_back();
    return idx;
}

void PagedKVCache::free_sequence(uint32_t seq_id) {
    auto it = sequences_.find(seq_id);
    if (it != sequences_.end()) {
        for (uint32_t b_id : it->second.block_ids) {
            free_block_indices_.push_back(b_id);
        }
        sequences_.erase(it);
    }
}

size_t PagedKVCache::get_context_length(uint32_t seq_id) const {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return 0;
    return it->second.token_count;
}

void PagedKVCache::append_token(
    uint32_t seq_id,
    [[maybe_unused]] uint32_t layer_idx,
    [[maybe_unused]] uint32_t kv_head_idx,
    const float* __restrict__ key,
    const float* __restrict__ value
) {
    auto& seq = sequences_[seq_id];
    size_t in_block_token_idx = seq.token_count % KV_BLOCK_TOKENS;

    if (in_block_token_idx == 0) {
        uint32_t block_id = allocate_block();
        seq.block_ids.push_back(block_id);
    }

    uint32_t active_block_id = seq.block_ids.back();
    PhysicalKVBlock& blk = blocks_[active_block_id];

    // Quantize Value per-token into 4-bit [0, 15]
    float v_min = value[0];
    float v_max = value[0];
    for (uint32_t c = 1; c < head_dim_; ++c) {
        v_min = std::min(v_min, value[c]);
        v_max = std::max(v_max, value[c]);
    }
    float v_scale = (v_max > v_min) ? (v_max - v_min) / 15.0f : 1.0f;
    blk.val_scales[in_block_token_idx] = v_scale;
    blk.val_biases[in_block_token_idx] = v_min;

    uint8_t* val_row = blk.val_data + (in_block_token_idx * (head_dim_ / 2));
    for (uint32_t c = 0; c < head_dim_; c += 2) {
        uint8_t q0 = static_cast<uint8_t>(std::clamp(std::round((value[c] - v_min) / v_scale), 0.0f, 15.0f));
        uint8_t q1 = static_cast<uint8_t>(std::clamp(std::round((value[c + 1] - v_min) / v_scale), 0.0f, 15.0f));
        val_row[c / 2] = static_cast<uint8_t>((q1 << 4) | (q0 & 0x0F));
    }

    // Key per-channel 2-bit quantization across block tokens
    for (uint32_t c = 0; c < head_dim_; ++c) {
        if (in_block_token_idx == 0) {
            blk.key_biases[c] = key[c];
            blk.key_scales[c] = 1.0f;
        }
        uint8_t* chan_bytes = blk.key_data + (c * 4);
        size_t byte_pos = in_block_token_idx / 4;
        size_t shift = (in_block_token_idx % 4) * 2;

        float k_val = key[c];
        float k_bias = blk.key_biases[c];
        float k_scale = blk.key_scales[c];
        uint8_t q = static_cast<uint8_t>(std::clamp(std::round((k_val - k_bias) / k_scale), 0.0f, 3.0f));

        chan_bytes[byte_pos] &= static_cast<uint8_t>(~(0x03 << shift));
        chan_bytes[byte_pos] |= static_cast<uint8_t>(q << shift);
    }

    seq.token_count++;
}

void PagedKVCache::compute_attention(
    uint32_t seq_id,
    [[maybe_unused]] uint32_t layer_idx,
    [[maybe_unused]] uint32_t head_idx,
    const float* __restrict__ query,
    float* __restrict__ out_context
) const {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end() || it->second.token_count == 0) {
        std::memset(out_context, 0, head_dim_ * sizeof(float));
        return;
    }

    size_t total_tokens = it->second.token_count;
    std::vector<float> scores(total_tokens, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    size_t tok_idx = 0;
    for (uint32_t blk_id : it->second.block_ids) {
        const PhysicalKVBlock& blk = blocks_[blk_id];
        size_t limit = std::min(static_cast<size_t>(KV_BLOCK_TOKENS), total_tokens - tok_idx);

        for (size_t t = 0; t < limit; ++t) {
            float dot = 0.0f;
            for (uint32_t c = 0; c < head_dim_; ++c) {
                const uint8_t* chan_bytes = blk.key_data + (c * 4);
                uint8_t q = (chan_bytes[t / 4] >> ((t % 4) * 2)) & 0x03;
                float k = blk.key_biases[c] + static_cast<float>(q) * blk.key_scales[c];
                dot += query[c] * k;
            }
            scores[tok_idx + t] = dot * scale;
        }
        tok_idx += limit;
    }

    // Softmax over scores
    float max_score = scores[0];
    for (size_t i = 1; i < total_tokens; ++i) {
        max_score = std::max(max_score, scores[i]);
    }
    float sum_exp = 0.0f;
    for (size_t i = 0; i < total_tokens; ++i) {
        scores[i] = std::exp(scores[i] - max_score);
        sum_exp += scores[i];
    }
    for (size_t i = 0; i < total_tokens; ++i) {
        scores[i] /= sum_exp;
    }

    // Weighted sum of Values
    std::memset(out_context, 0, head_dim_ * sizeof(float));
    tok_idx = 0;
    for (uint32_t blk_id : it->second.block_ids) {
        const PhysicalKVBlock& blk = blocks_[blk_id];
        size_t limit = std::min(static_cast<size_t>(KV_BLOCK_TOKENS), total_tokens - tok_idx);

        for (size_t t = 0; t < limit; ++t) {
            float s = scores[tok_idx + t];
            const uint8_t* val_row = blk.val_data + (t * (head_dim_ / 2));
            float v_scale = blk.val_scales[t];
            float v_bias = blk.val_biases[t];

            for (uint32_t c = 0; c < head_dim_; c += 2) {
                uint8_t packed = val_row[c / 2];
                float v0 = v_bias + static_cast<float>(packed & 0x0F) * v_scale;
                float v1 = v_bias + static_cast<float>((packed >> 4) & 0x0F) * v_scale;
                out_context[c] += s * v0;
                out_context[c + 1] += s * v1;
            }
        }
        tok_idx += limit;
    }
}

} // namespace aie::stream
