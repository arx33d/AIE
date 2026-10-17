#include "profile/safetensors_parser.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>

void create_mock_safetensors(const std::string& path) {
    std::string json = "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[128,64],\"data_offsets\":[0,32768]},"
                       "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F16\",\"shape\":[64,64],\"data_offsets\":[32768,40960]}}";
    uint64_t header_len = json.size();

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(uint64_t));
    out.write(json.data(), json.size());

    std::vector<uint8_t> dummy(40960, 0);
    out.write(reinterpret_cast<const char*>(dummy.data()), dummy.size());
}

int main() {
    std::string mock_path = "mock_model.safetensors";
    create_mock_safetensors(mock_path);

    aie::profile::SafetensorsParser parser(mock_path);
    auto topo = parser.parse_topology();

    assert(topo.num_layers == 1);
    assert(topo.vocab_size == 128);
    assert(topo.hidden_size == 64);
    assert(topo.tensors.size() == 2);

    std::remove(mock_path.c_str());
    std::cout << "[PASS] test_safetensors completed successfully.\n";
    return 0;
}
