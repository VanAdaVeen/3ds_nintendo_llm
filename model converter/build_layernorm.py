import os
import struct
import numpy as np
from safetensors.numpy import load_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_layernorm():
    model_path = os.path.join(BASE_DIR, "model.safetensors")
    out_path = os.path.join(BASE_DIR, "ln_f.bin")

    if not os.path.exists(model_path):
        print(f"Erreur : Le fichier {model_path} est introuvable.")
        return

    print("Chargement du modèle DistilGPT-2 (Safetensors)...")
    tensors = load_file(model_path)

    # Extraction des poids de la LayerNorm finale ("ln_f")
    # Dans PyTorch/HuggingFace, gamma s'appelle "weight" et beta s'appelle "bias"
    gamma = tensors["transformer.ln_f.weight"]
    beta = tensors["transformer.ln_f.bias"]

    hidden_size = gamma.shape[0]
    print(f"LayerNorm hidden_size: {hidden_size}")

    # --- PRÉPARATION DU HEADER ---
    magic = 0x4D524E4C # 'LNRM' en ASCII (Little Endian)
    version = 1

    print("Génération de ln_f.bin...")

    with open(out_path, "wb") as f:
        # Header : magic (uint32), version (uint32), hidden_size (uint32)
        f.write(struct.pack("<III", magic, version, hidden_size))

        # Poids gamma en float32 (768 * 4 = 3072 octets)
        f.write(gamma.astype(np.float32).tobytes())

        # Biais beta en float32 (3072 octets)
        f.write(beta.astype(np.float32).tobytes())

    file_size = os.path.getsize(out_path)
    print(f"Succès ! Fichier généré : {out_path} (Taille: {file_size} octets, soit ~6 Ko)")

if __name__ == "__main__":
    build_layernorm()