#include "tokenizer.hpp"

#include <fstream>
#include <iostream>

bool TokenizerData::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    if (size < 0) {
        return false;
    }
    file.seekg(0, std::ios::beg);

    this->bytes.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(this->bytes.data()), size)) {
        return false;
    }

    if (this->bytes.size() < sizeof(Header)) {
        return false;
    }

    this->header = reinterpret_cast<const Header*>(this->bytes.data());

    if (this->header->magic != 0x4E4B4F54) { // "TOKN"
        return false;
    }

    // On passe à la version 3 pour le format BPE
    if (this->header->version != 3) {
        return false;
    }

    if (this->header->entries_offset < sizeof(Header)) {
        return false;
    }

    size_t entries_size = static_cast<size_t>(this->header->vocab_size) * sizeof(Entry);

    if (static_cast<size_t>(this->header->entries_offset) + entries_size > this->bytes.size()) {
        return false;
    }

    if (this->header->blob_offset > this->bytes.size()) {
        return false;
    }

    if (this->header->blob_offset < this->header->entries_offset + entries_size) {
        return false;
    }

    // NOUVEAU : Vérification de la table des fusions (merges)
    size_t merges_offset = static_cast<size_t>(this->header->merges_offset);
    size_t merges_size = static_cast<size_t>(this->header->merges_count) * sizeof(MergeRule);

    if (merges_offset > this->bytes.size()) {
        return false;
    }

    if (merges_offset + merges_size > this->bytes.size()) {
        return false;
    }

    if (merges_offset < this->header->blob_offset) {
        return false;
    }

    // Assignation des pointeurs
    this->entries = reinterpret_cast<const Entry*>(
        this->bytes.data() + this->header->entries_offset
    );

    this->blob = reinterpret_cast<const char*>(
        this->bytes.data() + this->header->blob_offset
    );

    this->merges = reinterpret_cast<const MergeRule*>(
        this->bytes.data() + this->header->merges_offset
    );

    return true;
}

std::string TokenizerData::getToken(uint32_t id) const {
    if (!this->header || id >= this->header->vocab_size) {
        return "";
    }

    const Entry& e = this->entries[id];

    // La taille du blob correspond maintenant à l'écart entre le blob et le début des merges
    size_t blob_size = static_cast<size_t>(this->header->merges_offset)
                     - static_cast<size_t>(this->header->blob_offset);

    if (static_cast<size_t>(e.offset) > blob_size) {
        return "";
    }

    if (static_cast<size_t>(e.offset) + static_cast<size_t>(e.length) > blob_size) {
        return "";
    }

    return std::string(this->blob + e.offset, this->blob + e.offset + e.length);
}

const Entry* TokenizerData::getEntry(uint32_t id) const {
    if (!this->header || id >= this->header->vocab_size) {
        return nullptr;
    }
    return &this->entries[id];
}

void TokenizerData::debugPrint(uint32_t max_items) const {
    if (!this->header) {
        return;
    }

    std::cout << "vocab_size=" << this->header->vocab_size << "\n";
    std::cout << "merges_count=" << this->header->merges_count << "\n";
    std::cout << "unk_id=" << this->header->unk_id
              << " bos_id=" << this->header->bos_id
              << " eos_id=" << this->header->eos_id
              << " pad_id=" << this->header->pad_id << "\n";

    std::cout << "--- Tokens ---\n";
    for (uint32_t i = 0; i < this->header->vocab_size && i < max_items; ++i) {
        std::cout << "id=" << i
                  << " token=" << getToken(i)
                  << "\n";
    }

    std::cout << "--- Merges ---\n";
    for (uint32_t i = 0; i < this->header->merges_count && i < max_items; ++i) {
        const MergeRule& m = this->merges[i];
        std::cout << "merge[" << i << "]: "
                  << m.left_id << " + " << m.right_id
                  << " -> " << m.result_id 
                  << " (rank " << m.rank << ")\n";
    }
}