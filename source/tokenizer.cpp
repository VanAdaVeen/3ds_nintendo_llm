#include "tokenizer.hpp"

// Fonction de recherche dichotomique (Binary Search) ultra-rapide
const MergeRule* TokenizerData::findMerge(uint32_t left_id, uint32_t right_id) const {
    if (!this->merges || this->header->merges_count == 0) {
        return nullptr;
    }

    // On combine les deux IDs 32 bits en une seule clé de 64 bits pour la comparaison
    uint64_t target = (static_cast<uint64_t>(left_id) << 32) | right_id;

    int32_t low = 0;
    int32_t high = static_cast<int32_t>(this->header->merges_count) - 1;

    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        const MergeRule& rule = this->merges[mid];
        
        uint64_t current = (static_cast<uint64_t>(rule.left_id) << 32) | rule.right_id;

        if (current == target) {
            return &rule; // Trouvé !
        } else if (current < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return nullptr; // Aucune fusion n'existe pour cette paire
}

std::vector<uint32_t> TokenizerData::tokenize(const std::string& text) const {
    std::vector<uint32_t> tokens;

    if (!this->header || text.empty()) {
        return tokens;
    }

    // 1. Initialisation : Découper le texte en octets bruts
    // On suppose ici que le vocabulaire généré par le script Python 
    // a assigné les IDs 0 à 255 aux 256 valeurs d'octets possibles.
    for (char c : text) {
        tokens.push_back(static_cast<uint8_t>(c));
    }

    // 2. Boucle principale du BPE
    // On continue tant qu'il y a au moins une paire de tokens à analyser
    while (tokens.size() >= 2) {
        uint32_t best_idx = UINT32_MAX;
        const MergeRule* best_rule = nullptr;
        uint32_t min_rank = UINT32_MAX;

        // Étape A : On scanne la phrase actuelle pour trouver la meilleure paire à fusionner
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            const MergeRule* rule = findMerge(tokens[i], tokens[i + 1]);
            
            // Si la paire existe ET qu'elle a une priorité plus forte (rank plus petit)
            if (rule && rule->rank < min_rank) {
                min_rank = rule->rank;
                best_rule = rule;
                best_idx = static_cast<uint32_t>(i);
            }
        }

        // Étape B : Si aucune paire n'a de règle de fusion, l'algorithme a terminé.
        if (best_rule == nullptr) {
            break; 
        }

        // Étape C : On applique la fusion sur la meilleure paire trouvée.
        // On remplace le token de gauche par le token fusionné.
        tokens[best_idx] = best_rule->result_id;
        // On supprime le token de droite qui a été absorbé.
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    return tokens;
}