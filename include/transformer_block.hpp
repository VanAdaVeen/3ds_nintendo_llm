#pragma once

#include "layernorm.hpp"
#include "linear.hpp"
#include "attention.hpp"
#include "gelu.hpp"
#include <vector>
#include <string>

class TransformerBlock {
private:
    uint32_t hidden_size = 768;

    // --- LES SOUS-MODULES (Les poids) ---
    LayerNorm ln_1;          // Normalisation avant l'attention
    Linear attn_c_attn;      // Création de Q, K, V (768 -> 2304)
    Attention attention;     // Le calcul multi-tête (KV Cache)
    Linear attn_c_proj;      // Projection de sortie d'attention (768 -> 768)
    
    LayerNorm ln_2;          // Normalisation avant le MLP
    Linear mlp_c_fc;         // Expansion du MLP (768 -> 3072)
    Linear mlp_c_proj;       // Réduction du MLP (3072 -> 768)

    // L'outil mathématique pur (sans poids)
    Gelu gelu;

    // --- LES BUFFERS DE TRAVAIL (Alloués une seule fois pour économiser la RAM) ---
    std::vector<float> norm_buffer;   // 768
    std::vector<float> qkv_buffer;    // 2304
    std::vector<float> attn_buffer;   // 768
    std::vector<float> mlp_buffer;    // 3072

public:
    // Charge tous les fichiers .bin d'une couche spécifique (ex: layer_idx = 0)
    // Va chercher "ln_1_0.bin", "attn_c_attn_0.bin", etc.
    bool load(const std::string& folder_path, uint32_t layer_idx);

    // Initialise le cache de l'Attention (doit être appelé après load)
    void initAttention(uint32_t max_seq_len = 1024);

    // Réinitialise la mémoire du contexte (à appeler au début d'une nouvelle phrase)
    void resetCache();

    // --- LE PIPELINE COMPLET ---
    // Prend le vecteur de 768 de la couche précédente, et le modifie IN-PLACE
    bool forward(float* x);
};