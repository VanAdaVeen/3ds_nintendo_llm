#pragma once

#include <cstdint>
#include <vector>

class Attention {
private:
    uint32_t num_heads = 12;      // DistilGPT-2 a 12 têtes
    uint32_t head_dim = 64;       // 768 / 12 = 64
    uint32_t hidden_size = 768;   // La dimension totale

    // --- LE CACHE KV (La mémoire de la conversation) ---
    // Les LLM génèrent un mot à la fois.
    // Pour éviter de recalculer "Hello, world" quand on génère "!", 
    // on garde en mémoire les Keys et les Values des mots précédents.
    // Taille : [max_seq_len * hidden_size]
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    
    // Le nombre de mots actuellement dans notre contexte
    uint32_t current_seq_len = 0; 
    uint32_t max_seq_len = 1024;  // La limite de DistilGPT-2

    // --- FONCTIONS MATHÉMATIQUES INTERNES ---
    // Applique l'équation du Softmax sur un tableau de scores
    void softmax(float* scores, uint32_t size) const;

public:
    // Prépare le module (alloue la mémoire pour le cache KV)
    void init(uint32_t hidden_size = 768, uint32_t num_heads = 12, uint32_t max_seq_len = 1024);

    // Remet le compteur de mots à zéro (à appeler avant une nouvelle phrase)
    void resetCache();

    // --- LE CŒUR DE L'IA ---
    // qkv_input : Le vecteur de 2304 floats sortant de ta couche Linear
    // output    : Le vecteur de 768 floats final
    // Retourne false si la limite de mots (1024) est atteinte.
    bool forward(const float* qkv_input, float* output);
};