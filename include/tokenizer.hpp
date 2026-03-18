#pragma once

#include <cstdint>
#include <string>
#include <vector>

#pragma pack(push, 1)
struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t merges_count;
    int32_t unk_id;
    int32_t bos_id;
    int32_t eos_id;
    int32_t pad_id;
    uint32_t entries_offset;
    uint32_t blob_offset;
    uint32_t merges_offset;
};

struct Entry {
    uint32_t offset;
    uint16_t length;
};

struct MergeRule {
    uint32_t left_id;
    uint32_t right_id;  // <-- Correction : point-virgule ici
    uint32_t result_id;
    uint32_t rank;
};
#pragma pack(pop)

class TokenizerData {
private:
    std::vector<uint8_t> bytes;
    const Header* header = nullptr;
    const Entry* entries = nullptr;
    const char* blob = nullptr;
    const MergeRule* merges = nullptr; // <-- NOUVEAU : Pointeur vers la table de fusion

public:
    bool load(const std::string& path);
    std::string getToken(uint32_t id) const;
    const Entry* getEntry(uint32_t id) const;
    void debugPrint(uint32_t max_items = 20) const;
    
    // La fonction principale qui appliquera le BPE
    std::vector<uint32_t> tokenize(const std::string& text) const;

private:
    // NOUVEAU : Méthode interne ultra-rapide pour trouver si une fusion existe
    // Retourne le result_id si trouvé, sinon UINT32_MAX
    const MergeRule* findMerge(uint32_t left_id, uint32_t right_id) const;
};