#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace aie::profile {

enum class TensorDType {
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    U8,
    UNKNOWN
};

enum class ModelArchitecture {
    DenseTransformer,
    SparseMoE,
    Unknown
};

struct TensorMeta {
    std::string name;
    TensorDType dtype{TensorDType::UNKNOWN};
    std::vector<int64_t> shape;
    uint64_t data_offset_start{0};
    uint64_t data_offset_end{0};

    [[nodiscard]] uint64_t nbytes() const {
        return data_offset_end - data_offset_start;
    }
};

struct ModelGraphTopology {
    ModelArchitecture arch{ModelArchitecture::Unknown};
    uint32_t num_layers{0};
    uint32_t hidden_size{0};
    uint32_t intermediate_size{0};
    uint32_t num_attention_heads{0};
    uint32_t num_key_value_heads{0};
    uint32_t vocab_size{0};
    uint32_t num_experts{0};
    uint32_t num_experts_per_tok{0};
    uint64_t total_tensor_bytes{0};
    std::unordered_map<std::string, TensorMeta> tensors;
};

class SafetensorsParser {
public:
    explicit SafetensorsParser(std::string file_path);
    ~SafetensorsParser();

    SafetensorsParser(const SafetensorsParser&) = delete;
    SafetensorsParser& operator=(const SafetensorsParser&) = delete;
    SafetensorsParser(SafetensorsParser&& other) noexcept;
    SafetensorsParser& operator=(SafetensorsParser&& other) noexcept;

    ModelGraphTopology parse_topology();
    [[nodiscard]] const uint8_t* get_tensor_data(const TensorMeta& meta) const;
    [[nodiscard]] uint64_t get_file_size() const { return file_size_; }

private:
    void map_file();
    void unmap_file();
    void parse_json_header(const char* json_str, size_t json_len, ModelGraphTopology& topology);

    std::string file_path_;
    uint64_t file_size_{0};
    uint64_t header_size_{0};
    const uint8_t* mmap_addr_{nullptr};
#if defined(_WIN32)
    void* file_handle_{nullptr};
    void* mapping_handle_{nullptr};
#else
    int file_fd_{-1};
#endif
};

} // namespace aie::profile
