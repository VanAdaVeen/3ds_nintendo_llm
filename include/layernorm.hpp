#pragma once

#include <cstdint>
#include <vector>
#include <string>

#pragma pack(push,1)
struct LayerNormHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t hidden_size;
};
#pragma pack(pop)

class LayerNorm {
private:
    std::vector<uint8_t> memory_blob;
    const LayerNormHeader* header = nullptr;

    const float* gamma = nullptr;
    const float* beta = nullptr;
    float epsilon = 1e-5f;

public:
    bool load(const std::string& path);

    uint32_t getHiddenSize() const;

    void forward(float* x) const;
};