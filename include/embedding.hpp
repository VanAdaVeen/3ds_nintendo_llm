#pragma once

#include <cstdint>
#include <vector>
#include <string>

#pragma pack(push, 1)
struct EmbeddingHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t max_pos;
    uint32_t hidden_size;

    uint32_t wte_scales_offset;
    uint32_t wte_data_offset;
    uint32_t wpe_data_offset;
};
#pragma pack(pop)

class EmbeddingLayer {
private:
    std::vector<uint8_t> memory_blob;
    const EmbeddingHeader* header = nullptr;

    const float* wte_scales = nullptr;

    const int8_t* wte_data = nullptr;

    const float* wpe_data = nullptr;

public:

    bool load(const std::string& path);

    uint32_t getHiddenSize() const;

    bool forward(uint32_t token_id, uint32_t position, float* output_vector) const;
};