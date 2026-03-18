#include "layernorm.hpp"
#include <cmath> // Pour std::sqrt
#include <fstream>
#include <iostream>

bool LayerNorm::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Erreur: Impossible d'ouvrir " << path << "\n";
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    if (size < 0) return false;
    file.seekg(0, std::ios::beg);

    this->memory_blob.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(this->memory_blob.data()), size)) {
        return false;
    }

    if (this->memory_blob.size() < sizeof(LayerNormHeader)) {
        return false;
    }

    this->header = reinterpret_cast<const LayerNormHeader*>(this->memory_blob.data());

    if (this->header->magic != 0x4D524E4C) { // "LNRM"
        std::cerr << "Erreur: Magic LNRM invalide.\n";
        return false;
    }

    if (this->header->version != 1) {
        return false;
    }

    // Calcul de la taille attendue des données : 2 tableaux de float de taille hidden_size
    size_t expected_data_size = this->header->hidden_size * sizeof(float) * 2;
    if (this->memory_blob.size() < sizeof(LayerNormHeader) + expected_data_size) {
        std::cerr << "Erreur: Fichier LayerNorm incomplet.\n";
        return false;
    }

    // gamma commence juste après le header
    this->gamma = reinterpret_cast<const float*>(this->memory_blob.data() + sizeof(LayerNormHeader));
    
    // beta commence juste après gamma
    this->beta = this->gamma + this->header->hidden_size;

    return true;
}

uint32_t LayerNorm::getHiddenSize() const {
    if (!this->header) return 0;
    return this->header->hidden_size;
}

void LayerNorm::forward(float* x) const {
    if (!this->header || !this->gamma || !this->beta) {
        return; 
    }

    uint32_t dim = this->header->hidden_size;

    // 1. Calcul de la moyenne
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        sum += x[i];
    }
    float mean = sum / static_cast<float>(dim);

    // 2. Calcul de la variance
    float variance_sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = x[i] - mean;
        variance_sum += diff * diff;
    }
    float variance = variance_sum / static_cast<float>(dim);

    // 3. Inverse de l'écart-type (optimisation pour éviter les divisions dans la boucle)
    float inv_stddev = 1.0f / std::sqrt(variance + this->epsilon);

    

    // 4. Normalisation et application des poids (IN-PLACE)
    for (uint32_t i = 0; i < dim; ++i) {
        float normalized = (x[i] - mean) * inv_stddev;
        x[i] = (normalized * this->gamma[i]) + this->beta[i];
    }
}