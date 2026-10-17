#include "profile/safetensors_parser.hpp"
#include <stdexcept>
#include <cstring>
#include <regex>
#include <iostream>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace aie::profile {

namespace {

TensorDType parse_dtype(const std::string& s) {
    if (s == "F32") return TensorDType::F32;
    if (s == "F16") return TensorDType::F16;
    if (s == "BF16") return TensorDType::BF16;
    if (s == "I32") return TensorDType::I32;
    if (s == "I16") return TensorDType::I16;
    if (s == "I8") return TensorDType::I8;
    if (s == "U8") return TensorDType::U8;
    return TensorDType::UNKNOWN;
}

} // namespace

SafetensorsParser::SafetensorsParser(std::string file_path)
    : file_path_(std::move(file_path)) {
    map_file();
}

SafetensorsParser::~SafetensorsParser() {
    unmap_file();
}

SafetensorsParser::SafetensorsParser(SafetensorsParser&& other) noexcept
    : file_path_(std::move(other.file_path_)),
      file_size_(other.file_size_),
      header_size_(other.header_size_),
      mmap_addr_(other.mmap_addr_) {
#if defined(_WIN32)
    file_handle_ = other.file_handle_;
    mapping_handle_ = other.mapping_handle_;
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
#else
    file_fd_ = other.file_fd_;
    other.file_fd_ = -1;
#endif
    other.mmap_addr_ = nullptr;
    other.file_size_ = 0;
    other.header_size_ = 0;
}

SafetensorsParser& SafetensorsParser::operator=(SafetensorsParser&& other) noexcept {
    if (this != &other) {
        unmap_file();
        file_path_ = std::move(other.file_path_);
        file_size_ = other.file_size_;
        header_size_ = other.header_size_;
        mmap_addr_ = other.mmap_addr_;
#if defined(_WIN32)
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
#else
        file_fd_ = other.file_fd_;
        other.file_fd_ = -1;
#endif
        other.mmap_addr_ = nullptr;
        other.file_size_ = 0;
        other.header_size_ = 0;
    }
    return *this;
}

void SafetensorsParser::map_file() {
#if defined(_WIN32)
    file_handle_ = CreateFileA(
        file_path_.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open safetensors file: " + file_path_);
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file_handle_, &sz)) {
        CloseHandle(file_handle_);
        throw std::runtime_error("Failed to query file size: " + file_path_);
    }
    file_size_ = static_cast<uint64_t>(sz.QuadPart);

    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_handle_) {
        CloseHandle(file_handle_);
        throw std::runtime_error("CreateFileMapping failed: " + file_path_);
    }
    mmap_addr_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
    if (!mmap_addr_) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        throw std::runtime_error("MapViewOfFile failed: " + file_path_);
    }
#else
    file_fd_ = open(file_path_.c_str(), O_RDONLY);
    if (file_fd_ < 0) {
        throw std::runtime_error("Failed to open safetensors file: " + file_path_);
    }
    struct stat st;
    if (fstat(file_fd_, &st) < 0) {
        close(file_fd_);
        throw std::runtime_error("Failed to fstat file: " + file_path_);
    }
    file_size_ = static_cast<uint64_t>(st.st_size);

    void* addr = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, file_fd_, 0);
    if (addr == MAP_FAILED) {
        close(file_fd_);
        throw std::runtime_error("mmap failed for file: " + file_path_);
    }
    mmap_addr_ = static_cast<const uint8_t*>(addr);
#endif

    if (file_size_ < 8) {
        throw std::runtime_error("Invalid safetensors file: smaller than 8 bytes.");
    }
    std::memcpy(&header_size_, mmap_addr_, sizeof(uint64_t));
    if (8 + header_size_ > file_size_) {
        throw std::runtime_error("Header size exceeds file boundary.");
    }
}

void SafetensorsParser::unmap_file() {
#if defined(_WIN32)
    if (mmap_addr_) {
        UnmapViewOfFile(mmap_addr_);
        mmap_addr_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
    }
#else
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(const_cast<uint8_t*>(mmap_addr_), file_size_);
        mmap_addr_ = nullptr;
    }
    if (file_fd_ >= 0) {
        close(file_fd_);
        file_fd_ = -1;
    }
#endif
}

void SafetensorsParser::parse_json_header(const char* json_str, size_t json_len, ModelGraphTopology& topology) {
    std::string header(json_str, json_len);

    std::regex tensor_regex(
        R"raw("([^"\\]+)":\s*\{\s*"dtype":\s*"([^"]+)",\s*"shape":\s*\[([^\]]*)\],\s*"data_offsets":\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\]\s*\})raw"
    );

    auto words_begin = std::sregex_iterator(header.begin(), header.end(), tensor_regex);
    auto words_end = std::sregex_iterator();

    int max_layer_idx = -1;
    bool has_experts = false;

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        TensorMeta meta;
        meta.name = match[1].str();
        meta.dtype = parse_dtype(match[2].str());

        std::string shape_str = match[3].str();
        std::stringstream ss(shape_str);
        std::string dim_token;
        while (std::getline(ss, dim_token, ',')) {
            dim_token.erase(0, dim_token.find_first_not_of(" \t\n\r"));
            dim_token.erase(dim_token.find_last_not_of(" \t\n\r") + 1);
            if (!dim_token.empty()) {
                meta.shape.push_back(std::stoll(dim_token));
            }
        }

        meta.data_offset_start = std::stoull(match[4].str());
        meta.data_offset_end = std::stoull(match[5].str());

        topology.total_tensor_bytes += meta.nbytes();

        std::smatch layer_match;
        std::regex layer_idx_regex(R"(layers?\.([0-9]+)\.)");
        if (std::regex_search(meta.name, layer_match, layer_idx_regex)) {
            int l_idx = std::stoi(layer_match[1].str());
            if (l_idx > max_layer_idx) {
                max_layer_idx = l_idx;
            }
        }

        if (meta.name.find("embed_tokens") != std::string::npos && meta.shape.size() == 2) {
            topology.vocab_size = static_cast<uint32_t>(meta.shape[0]);
            topology.hidden_size = static_cast<uint32_t>(meta.shape[1]);
        }
        if (meta.name.find("q_proj") != std::string::npos && meta.shape.size() == 2) {
            topology.hidden_size = static_cast<uint32_t>(meta.shape[1]);
        }
        if (meta.name.find("block_sparse_moe") != std::string::npos || meta.name.find("experts") != std::string::npos) {
            has_experts = true;
        }

        topology.tensors[meta.name] = meta;
    }

    topology.num_layers = (max_layer_idx >= 0) ? static_cast<uint32_t>(max_layer_idx + 1) : 0;
    topology.arch = has_experts ? ModelArchitecture::SparseMoE : ModelArchitecture::DenseTransformer;
}

ModelGraphTopology SafetensorsParser::parse_topology() {
    ModelGraphTopology topo;
    const char* json_start = reinterpret_cast<const char*>(mmap_addr_ + 8);
    parse_json_header(json_start, header_size_, topo);
    return topo;
}

const uint8_t* SafetensorsParser::get_tensor_data(const TensorMeta& meta) const {
    uint64_t base_offset = 8 + header_size_;
    if (base_offset + meta.data_offset_end > file_size_) {
        throw std::runtime_error("Tensor data offset out of mapped file range.");
    }
    return mmap_addr_ + base_offset + meta.data_offset_start;
}

} // namespace aie::profile
