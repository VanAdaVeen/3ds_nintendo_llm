#pragma once

#include <cstdint>

class Gelu {
public:
    // Applique l'approximation de GELU (GPT-2) IN-PLACE
    // x : Pointeur vers le vecteur d'entrée
    // size : La taille du vecteur (typiquement 3072 pour le MLP de DistilGPT-2)
    void forward(float* x, uint32_t size) const;
};