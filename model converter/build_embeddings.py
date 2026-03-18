import os
import struct
import numpy as np
from safetensors.numpy import load_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_embeddings():
    model_path = os.path.join(BASE_DIR, "model.safetensors")
    out_path = os.path.join(BASE_DIR, "embeddings.bin")

    if not os.path.exists(model_path):
        print(f"Erreur : Le fichier {model_path} est introuvable.")
        return

    print("Chargement du modèle DistilGPT-2 (Safetensors)...")
    tensors = load_file(model_path)

    # Extraction des matrices qui nous intéressent
# Extraction des matrices qui nous intéressent
    wte = tensors["transformer.wte.weight"]  # (50257, 768)
    wpe = tensors["transformer.wpe.weight"]  # (1024, 768)

    vocab_size, hidden_size = wte.shape
    max_pos, _ = wpe.shape

    print(f"wte shape: {wte.shape}, wpe shape: {wpe.shape}")
    print("Quantification de wte en INT8 (calcul des scales)...")

    # 1. Calcul des scales (un float par ligne/token)
    # On trouve le max absolu de chaque ligne, on évite la division par zéro avec un petit epsilon
    max_abs = np.max(np.abs(wte), axis=1)
    scales = max_abs / 127.0
    scales[scales == 0] = 1e-9 

    # 2. Quantification des poids
    # On divise par le scale, on arrondit, et on force le type en entier 8 bits signés
    wte_quantized = np.round(wte / scales[:, None]).astype(np.int8)

    print("Préparation du fichier binaire EMBD...")

    # --- CALCUL DES OFFSETS POUR LA 3DS ---
    magic = 0x44424D45 # 'EMBD'
    version = 1
    header_size = 32 # 8 champs de 4 octets

    wte_scales_offset = header_size
    wte_scales_size = vocab_size * 4 # 4 octets pour un float32
    
    wte_data_offset = wte_scales_offset + wte_scales_size
    wte_data_size = vocab_size * hidden_size * 1 # 1 octet pour un int8
    
    # SÉCURITÉ 3DS : wpe_data doit commencer sur une adresse multiple de 4 !
    raw_wpe_offset = wte_data_offset + wte_data_size
    padding = (4 - (raw_wpe_offset % 4)) % 4
    wpe_data_offset = raw_wpe_offset + padding

    print(f"Taille finale estimée : ~{(wpe_data_offset + max_pos * hidden_size * 4) / (1024*1024):.2f} Mo")

    # --- ÉCRITURE DU FICHIER BINAIRE ---
    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<IIIIIIII", 
                            magic, version, vocab_size, max_pos, hidden_size,
                            wte_scales_offset, wte_data_offset, wpe_data_offset))

        # 1. Écriture des facteurs d'échelle (float32)
        f.write(scales.astype(np.float32).tobytes())

        # 2. Écriture de la matrice compressée (int8)
        f.write(wte_quantized.tobytes())

        # Padding pour l'alignement mémoire (rempli de zéros)
        if padding > 0:
            f.write(b'\0' * padding)

        # 3. Écriture de la matrice des positions (float32)
        f.write(wpe.astype(np.float32).tobytes())

    print(f"Succès ! Fichier généré : {out_path}")

if __name__ == "__main__":
    build_embeddings()