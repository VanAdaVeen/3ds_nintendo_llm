#include "linear.hpp"
#include <fstream>
#include <iostream>

bool Linear::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Erreur: Impossible d'ouvrir " << path << "\n";
        return false;
    }

    // Récupérer la taille du fichier
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    if (size < 0) return false;
    file.seekg(0, std::ios::beg);

    // Charger le fichier complet en RAM
    this->memory_blob.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(this->memory_blob.data()), size)) {
        return false;
    }

    if (this->memory_blob.size() < sizeof(LinearHeader)) {
        std::cerr << "Erreur: Fichier Linear trop petit.\n";
        return false;
    }

    this->header = reinterpret_cast<const LinearHeader*>(this->memory_blob.data());

    // Vérifications du Header
    if (this->header->magic != 0x524E494C) { // "LINR"
        std::cerr << "Erreur: Magic LINR invalide.\n";
        return false;
    }

    if (this->header->version != 1) {
        std::cerr << "Erreur: Version Linear non supportée.\n";
        return false;
    }

    // SÉCURITÉ 3DS : Vérification de l'alignement mémoire (Data Abort Check)
    if (this->header->scales_offset % 4 != 0 || 
        this->header->biases_offset % 4 != 0) {
        std::cerr << "Erreur FATALE: Les offsets des floats ne sont pas alignés sur 4 octets.\n";
        return false;
    }

    // Assignation des pointeurs
    this->scales = reinterpret_cast<const float*>(this->memory_blob.data() + this->header->scales_offset);
    this->weights = reinterpret_cast<const int8_t*>(this->memory_blob.data() + this->header->weights_offset);
    this->biases = reinterpret_cast<const float*>(this->memory_blob.data() + this->header->biases_offset);

    return true;
}

bool Linear::forward(const float* input, float* output) const {
    if (!this->header || !this->scales || !this->weights || !this->biases) {
        return false;
    }

    uint32_t in_feat = this->header->in_features;
    uint32_t out_feat = this->header->out_features;

    

    // LA BOUCLE DE MULTIPLICATION MATRICE-VECTEUR (GEMV)
    // On boucle sur chaque neurone de sortie (les lignes de notre matrice)
    for (uint32_t i = 0; i < out_feat; ++i) {
        
        // Produit scalaire (Dot Product) entre le vecteur d'entrée et la ligne quantifiée
        float dot_product = 0.0f;
        
        // Pointeur vers le début de la ligne actuelle dans la matrice 1D
        const int8_t* row_weights = this->weights + (i * in_feat);

        // Boucle interne : c'est ici que le CPU va passer 90% de son temps
        for (uint32_t j = 0; j < in_feat; ++j) {
            // On multiplie l'entrée (float) par le poids quantifié (converti en float)
            dot_product += input[j] * static_cast<float>(row_weights[j]);
        }

        // Équation finale : on applique l'échelle et on ajoute le biais
        output[i] = (dot_product * this->scales[i]) + this->biases[i];
    }

    return true;
}