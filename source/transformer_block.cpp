#include "transformer_block.hpp"
#include <iostream>

bool TransformerBlock::load(const std::string& folder_path, uint32_t layer_idx) {
    std::string prefix = folder_path + "/";
    std::string suffix = "_" + std::to_string(layer_idx) + ".bin";

    // Chargement des 6 fichiers de poids pour cette couche spécifique
    if (!ln_1.load(prefix + "ln_1" + suffix)) return false;
    if (!attn_c_attn.load(prefix + "attn_c_attn" + suffix)) return false;
    if (!attn_c_proj.load(prefix + "attn_c_proj" + suffix)) return false;
    
    if (!ln_2.load(prefix + "ln_2" + suffix)) return false;
    if (!mlp_c_fc.load(prefix + "mlp_c_fc" + suffix)) return false;
    if (!mlp_c_proj.load(prefix + "mlp_c_proj" + suffix)) return false;

    return true;
}

void TransformerBlock::initAttention(uint32_t max_seq_len) {
    // Allocation des mémoires de travail internes (une seule fois)
    this->norm_buffer.resize(this->hidden_size, 0.0f);
    this->qkv_buffer.resize(this->hidden_size * 3, 0.0f); // 768 * 3 = 2304
    this->attn_buffer.resize(this->hidden_size, 0.0f);
    this->mlp_buffer.resize(this->hidden_size * 4, 0.0f); // 768 * 4 = 3072

    // Initialisation du module Attention (et de son KV Cache)
    this->attention.init(this->hidden_size, 12, max_seq_len);
}

void TransformerBlock::resetCache() {
    this->attention.resetCache();
}

bool TransformerBlock::forward(float* x) {
    // --------------------------------------------------------
    // BLOC 1 : ATTENTION MULTI-TÊTE
    // --------------------------------------------------------

    // 1. Copie du vecteur d'entrée x vers le buffer de normalisation
    for (uint32_t i = 0; i < this->hidden_size; ++i) {
        this->norm_buffer[i] = x[i];
    }

    // 2. LayerNorm 1 (IN-PLACE sur norm_buffer)
    this->ln_1.forward(this->norm_buffer.data());

    // 3. Projection QKV (Entrée: norm_buffer -> Sortie: qkv_buffer)
    if (!this->attn_c_attn.forward(this->norm_buffer.data(), this->qkv_buffer.data())) return false;

    // 4. Calcul de l'Attention (Entrée: qkv_buffer -> Sortie: norm_buffer)
    // Astuce 3DS : On recycle 'norm_buffer' comme sortie temporaire pour économiser la RAM !
    if (!this->attention.forward(this->qkv_buffer.data(), this->norm_buffer.data())) return false;

    // 5. Projection de sortie de l'Attention (Entrée: norm_buffer -> Sortie: attn_buffer)
    if (!this->attn_c_proj.forward(this->norm_buffer.data(), this->attn_buffer.data())) return false;

    // 6. CONNEXION RÉSIDUELLE 1 : x = x + Attention(x)
    for (uint32_t i = 0; i < this->hidden_size; ++i) {
        x[i] += this->attn_buffer[i];
    }

    // --------------------------------------------------------
    // BLOC 2 : MULTI-LAYER PERCEPTRON (MLP)
    // --------------------------------------------------------

    // 7. Copie du nouveau x (résiduel) vers le buffer de normalisation
    for (uint32_t i = 0; i < this->hidden_size; ++i) {
        this->norm_buffer[i] = x[i];
    }

    // 8. LayerNorm 2 (IN-PLACE sur norm_buffer)
    this->ln_2.forward(this->norm_buffer.data());

    // 9. Expansion du MLP (Entrée: norm_buffer -> Sortie: mlp_buffer) (768 vers 3072)
    if (!this->mlp_c_fc.forward(this->norm_buffer.data(), this->mlp_buffer.data())) return false;

    // 10. Fonction d'activation GELU (IN-PLACE sur mlp_buffer de taille 3072)
    this->gelu.forward(this->mlp_buffer.data(), this->hidden_size * 4);

    // 11. Réduction du MLP (Entrée: mlp_buffer -> Sortie: attn_buffer) (3072 vers 768)
    // On recycle à nouveau 'attn_buffer' pour stocker la sortie finale de la couche
    if (!this->mlp_c_proj.forward(this->mlp_buffer.data(), this->attn_buffer.data())) return false;

    // 12. CONNEXION RÉSIDUELLE 2 : x = x + MLP(x)
    for (uint32_t i = 0; i < this->hidden_size; ++i) {
        x[i] += this->attn_buffer[i];
    }

    return true; // Le vecteur x a terminé son voyage dans cette couche !
}