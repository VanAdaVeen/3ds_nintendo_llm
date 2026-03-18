import os
import struct
import numpy as np
from safetensors.numpy import load_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_linear():
    model_path = os.path.join(BASE_DIR, "model.safetensors")
    out_path = os.path.join(BASE_DIR, "attn_proj_0.bin")

    if not os.path.exists(model_path):
        print(f"Erreur : Le fichier {model_path} est introuvable.")
        return

    print("Chargement du modèle DistilGPT-2 (Safetensors)...")
    tensors = load_file(model_path)

    # Extraction des poids de la projection QKV de la toute première couche (couche 0)
    try:
        weight = tensors["transformer.h.0.attn.c_attn.weight"]
        bias = tensors["transformer.h.0.attn.c_attn.bias"]
    except KeyError:
        print("Erreur : Clés introuvables.")
        return

    # 1. LE FIX GPT-2 : Transposition de la matrice
    # Passe de (768, 2304) à (2304, 768) pour correspondre à notre boucle C++
    weight = weight.T 

    out_features, in_features = weight.shape
    print(f"Matrice Linéaire - in_features: {in_features}, out_features: {out_features}")

    # 2. Quantification INT8 ligne par ligne
    print("Quantification des poids en INT8...")
    max_abs = np.max(np.abs(weight), axis=1)
    scales = max_abs / 127.0
    scales[scales == 0] = 1e-9 # Sécurité division par zéro
    
    weight_quantized = np.round(weight / scales[:, None]).astype(np.int8)

    # --- CALCUL DES OFFSETS ---
    magic = 0x524E494C # 'LINR'
    version = 1
    header_size = 28 # 7 champs * 4 octets

    scales_offset = header_size
    scales_size = out_features * 4
    
    weights_offset = scales_offset + scales_size
    weights_size = out_features * in_features * 1
    
    # Sécurité alignement 4 octets pour les biais
    raw_biases_offset = weights_offset + weights_size
    padding = (4 - (raw_biases_offset % 4)) % 4
    biases_offset = raw_biases_offset + padding

    print(f"Génération de {os.path.basename(out_path)}...")

    with open(out_path, "wb") as f:
        # Header (7 uint32)
        f.write(struct.pack("<IIIIIII", 
                            magic, version, in_features, out_features,
                            scales_offset, weights_offset, biases_offset))

        # 1. Échelles (float32)
        f.write(scales.astype(np.float32).tobytes())

        # 2. Poids (int8)
        f.write(weight_quantized.tobytes())

        # Padding (zéros)
        if padding > 0:
            f.write(b'\0' * padding)

        # 3. Biais (float32)
        f.write(bias.astype(np.float32).tobytes())

    file_size_mo = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Succès ! Fichier généré (Taille: ~{file_size_mo:.2f} Mo)")

if __name__ == "__main__":
    build_linear()