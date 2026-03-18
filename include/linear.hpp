#pragma once

#include <cstdint>
#include <vector>
#include <string>

// Header pour le fichier binaire de la couche Linéaire
#pragma pack(push, 1)
struct LinearHeader {
    uint32_t magic;          // 'L' 'I' 'N' 'R' (0x524E494C)
    uint32_t version;        // 1
    
    // Dimensions de la matrice
    uint32_t in_features;    // La taille du vecteur d'entrée (ex: 768)
    uint32_t out_features;   // La taille du vecteur de sortie (ex: 2304 pour Q, K, V)
    
    // Adresses (offsets) dans le fichier pour éviter les Data Abort
    uint32_t scales_offset;  // Pointeur vers le tableau de float (out_features)
    uint32_t weights_offset; // Pointeur vers la matrice int8 (out_features * in_features)
    uint32_t biases_offset;  // Pointeur vers le tableau de float (out_features)
};
#pragma pack(pop)

class Linear {
private:
    std::vector<uint8_t> memory_blob;
    const LinearHeader* header = nullptr;

    // --- Les trois zones de la couche Linéaire ---
    
    // 1. Facteurs d'échelle (un par ligne/neurone de sortie)
    const float* scales = nullptr; 
    
    // 2. La grande matrice de poids compressée
    const int8_t* weights = nullptr; 
    
    // 3. Les biais (un par ligne/neurone de sortie)
    const float* biases = nullptr;

public:
    // Charge le fichier (ex: "attn_proj.bin")
    bool load(const std::string& path);

    // Accesseurs utiles pour préparer les buffers dans le main
    uint32_t getInFeatures() const { return header ? header->in_features : 0; }
    uint32_t getOutFeatures() const { return header ? header->out_features : 0; }

    // --- LA MULTIPLICATION MATRICE-VECTEUR (GEMV) ---
    // Prend le vecteur d'entrée (taille in_features)
    // Remplit le vecteur de sortie (taille out_features)
    // Attention : input et output doivent pointer vers des zones mémoire différentes !
    bool forward(const float* input, float* output) const;
};