#include "attention.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

void Attention::init(uint32_t hidden_dim, uint32_t heads, uint32_t max_len) {
    this->hidden_size = hidden_dim;
    this->num_heads = heads;
    this->head_dim = hidden_dim / heads; // 768 / 12 = 64
    this->max_seq_len = max_len;
    this->current_seq_len = 0;

    // Allocation du cache de contexte (KV Cache)
    // Taille : 1024 mots * 768 dimensions = ~786 Ko par cache. Très léger !
    this->k_cache.resize(max_seq_len * hidden_size, 0.0f);
    this->v_cache.resize(max_seq_len * hidden_size, 0.0f);
}

void Attention::resetCache() {
    this->current_seq_len = 0;
}

// L'équation du Softmax optimisée et sécurisée
void Attention::softmax(float* scores, uint32_t size) const {
    // 1. Trouver la valeur max pour la stabilité numérique
    // Si on calcule exp(100), l'ordinateur plante (Infinity).
    // Si on fait exp(100 - 100) = exp(0) = 1, la proportion reste mathématiquement identique et sûre.
    float max_val = scores[0];
    for (uint32_t i = 1; i < size; ++i) {
        if (scores[i] > max_val) max_val = scores[i];
    }

    // 2. Calculer les exponentielles et leur somme
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        scores[i] = std::exp(scores[i] - max_val);
        sum += scores[i];
    }

    // 3. Diviser par la somme pour obtenir des pourcentages (0.0 à 1.0)
    // Optimisation 3DS : On remplace la division par une multiplication de l'inverse
    float inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < size; ++i) {
        scores[i] *= inv_sum;
    }
}

bool Attention::forward(const float* qkv_input, float* output) {
    if (this->current_seq_len >= this->max_seq_len) {
        std::cerr << "Erreur : Contexte maximum (KV Cache) atteint.\n";
        return false;
    }

    // 1. Séparer Q, K et V depuis le méga-vecteur de 2304 floats
    const float* q = qkv_input;                       // Les 768 premiers
    const float* k = qkv_input + this->hidden_size;   // Les 768 du milieu
    const float* v = qkv_input + (2 * this->hidden_size); // Les 768 derniers

    // 2. Sauvegarder K et V dans le KV Cache à la ligne correspondant au mot actuel
    uint32_t cache_offset = this->current_seq_len * this->hidden_size;
    for (uint32_t i = 0; i < this->hidden_size; ++i) {
        this->k_cache[cache_offset + i] = k[i];
        this->v_cache[cache_offset + i] = v[i];
    }

    // Un tableau temporaire pour stocker les scores d'attention de chaque tête
    // Taille : le nombre de mots vus jusqu'ici + 1 (le mot actuel)
    uint32_t seq_len = this->current_seq_len + 1;
    std::vector<float> scores(seq_len);

    // 3. LA BOUCLE MULTI-TÊTE
    for (uint32_t h = 0; h < this->num_heads; ++h) {
        // Pointeur vers le début de la tête 'h' pour le mot actuel (Q)
        const float* q_head = q + (h * this->head_dim);
        
        // Pointeur vers la zone de sortie de cette tête
        float* out_head = output + (h * this->head_dim);

        // --- ETAPE A : Calcul des scores Q * K^T ---
        for (uint32_t t = 0; t < seq_len; ++t) {
            // Pointeur vers le K de la tête 'h' pour le mot historique 't'
            const float* k_head = this->k_cache.data() + (t * this->hidden_size) + (h * this->head_dim);
            
            float dot_product = 0.0f;
            for (uint32_t i = 0; i < this->head_dim; ++i) {
                dot_product += q_head[i] * k_head[i];
            }
            
            // Mise à l'échelle (Scaling) : diviser par sqrt(64), soit multiplier par 0.125f
            scores[t] = dot_product * 0.125f;
        }

        // --- ETAPE B : Transformation en pourcentages ---
        this->softmax(scores.data(), seq_len);

        // --- ETAPE C : Multiplication par V ---
        // On initialise la zone de sortie à zéro
        for (uint32_t i = 0; i < this->head_dim; ++i) {
            out_head[i] = 0.0f;
        }

        // On accumule le sens (V) de tous les mots, pondéré par leur score (pourcentage)
        for (uint32_t t = 0; t < seq_len; ++t) {
            const float* v_head = this->v_cache.data() + (t * this->hidden_size) + (h * this->head_dim);
            float current_score = scores[t];
            
            for (uint32_t i = 0; i < this->head_dim; ++i) {
                out_head[i] += current_score * v_head[i];
            }
        }
    }

    // Le mot a été traité, on incrémente le compteur pour le prochain appel
    this->current_seq_len++;

    return true;
}