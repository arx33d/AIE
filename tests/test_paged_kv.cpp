#include "stream/paged_kv_cache.hpp"
#include <iostream>
#include <cassert>
#include <vector>

int main() {
    constexpr uint32_t num_layers = 2;
    constexpr uint32_t num_heads = 4;
    constexpr uint32_t head_dim = 64;
    constexpr uint32_t max_blocks = 32;

    aie::stream::PagedKVCache cache(num_layers, num_heads, head_dim, max_blocks);

    std::vector<float> k(head_dim, 0.5f);
    std::vector<float> v(head_dim, 1.2f);
    std::vector<float> q(head_dim, 0.5f);
    std::vector<float> ctx(head_dim, 0.0f);

    uint32_t seq_id = 1001;

    for (uint32_t t = 0; t < 32; ++t) {
        cache.append_token(seq_id, 0, 0, k.data(), v.data());
    }

    assert(cache.get_context_length(seq_id) == 32);

    cache.compute_attention(seq_id, 0, 0, q.data(), ctx.data());

    assert(ctx[0] > 0.0f);

    std::cout << "[PASS] test_paged_kv completed successfully.\n";
    return 0;
}
