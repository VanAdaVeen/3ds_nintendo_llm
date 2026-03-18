#include "gelu.hpp"
#include <cmath>

void Gelu::forward(float* x, uint32_t size) const {
    // Précalcul des constantes pour éviter de faire les calculs dans la boucle
    // sqrt(2 / pi)
    const float sqrt_2_over_pi = 0.7978845608f; 
    const float coef = 0.044715f;

    // La boucle critique (exécutée 3072 fois par mot)
    for (uint32_t i = 0; i < size; ++i) {
        float val = x[i];
        
        // Optimisation : x^3 = x * x * x (plus rapide que std::pow)
        float cube = val * val * val;
        
        // L'intérieur de la parenthèse du tanh
        float inner = sqrt_2_over_pi * (val + coef * cube);
        
        // Le goulot d'étranglement de l'ARM11 : la tangente hyperbolique
        float tanh_res = std::tanh(inner);
        
        // L'équation finale : on écrase la valeur d'origine (in-place)
        x[i] = 0.5f * val * (1.0f + tanh_res);
    }
}