#include "embedding.hpp"
#include <fstream>
#include <iostream>

bool EmbeddingLayer::load(const std::string& path) {
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

    // Charger tout le fichier en RAM (std::vector garantit un alignement de base correct)
    this->memory_blob.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(this->memory_blob.data()), size)) {
        return false;
    }

    if (this->memory_blob.size() < sizeof(EmbeddingHeader)) {
        return false;
    }

    // Mapper le header
    this->header = reinterpret_cast<const EmbeddingHeader*>(this->memory_blob.data());

    // Vérifications de sécurité (Magic = "EMBD")
    if (this->header->magic != 0x44424D45) {
        std::cerr << "Erreur: Magic EMBD invalide.\n";
        return false;
    }

    if (this->header->version != 1) {
        std::cerr << "Erreur: Version Embedding non supportée.\n";
        return false;
    }

    // SÉCURITÉ 3DS EXTRÊME : Vérification de l'alignement mémoire (Data Abort Check)
    // Les offsets pointant vers des floats DOIVENT être des multiples de 4.
    if (this->header->wte_scales_offset % 4 != 0 || 
        this->header->wpe_data_offset % 4 != 0) {
        std::cerr << "Erreur FATALE: Les offsets ne sont pas alignés sur 4 octets.\n";
        return false;
    }

    // Assigner les pointeurs vers les différentes zones mémoire
    this->wte_scales = reinterpret_cast<const float*>(
        this->memory_blob.data() + this->header->wte_scales_offset
    );
    
    this->wte_data = reinterpret_cast<const int8_t*>(
        this->memory_blob.data() + this->header->wte_data_offset
    );
    
    this->wpe_data = reinterpret_cast<const float*>(
        this->memory_blob.data() + this->header->wpe_data_offset
    );

    return true;
}

uint32_t EmbeddingLayer::getHiddenSize() const {
    return this->header ? this->header->hidden_size : 0;
}

bool EmbeddingLayer::forward(uint32_t token_id, uint32_t position, float* output_vector) const {
    if (!this->header || !this->wte_scales || !this->wte_data || !this->wpe_data) {
        return false;
    }

    // Sécurité anti-débordement
    if (token_id >= this->header->vocab_size) {
        token_id = 0; // Fallback d'urgence pour ne pas crasher
    }
    if (position >= this->header->max_pos) {
        position = this->header->max_pos - 1; // On bloque à la position max
    }

    uint32_t hidden_dim = this->header->hidden_size;

    // Calcul des adresses de départ dans nos tableaux 1D
    size_t wte_offset = static_cast<size_t>(token_id) * hidden_dim;
    size_t wpe_offset = static_cast<size_t>(position) * hidden_dim;

    // Récupération du facteur d'échelle pour ce token précis
    float scale = this->wte_scales[token_id];

    

    // LA BOUCLE CRITIQUE (Exécutée 768 fois par token généré)
    // On déquantifie wte et on ajoute wpe en une seule passe pour épargner le CPU
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        // 1. Lecture de l'entier 8 bits (rapide)
        int8_t quantized_val = this->wte_data[wte_offset + i];
        
        // 2. Conversion en float et application de l'échelle
        float token_val = static_cast<float>(quantized_val) * scale;
        
        // 3. Lecture du vecteur de position en float (rapide car déjà en FP32)
        float pos_val = this->wpe_data[wpe_offset + i];
        
        // 4. Écriture du résultat final
        output_vector[i] = token_val + pos_val;
    }

    return true;
}