import os
import sys
import struct
import numpy as np
from safetensors.numpy import load_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FONCTIONS D'EXPORTATION ---

def export_layernorm(tensors, hf_prefix, out_path):
    try:
        gamma = tensors[f"{hf_prefix}.weight"]
        beta = tensors[f"{hf_prefix}.bias"]
    except KeyError:
        print(f"  [X] Erreur : Clés introuvables pour {hf_prefix}")
        return False

    hidden_size = gamma.shape[0]
    magic = 0x4D524E4C # 'LNRM'
    version = 1

    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", magic, version, hidden_size))
        f.write(gamma.astype(np.float32).tobytes())
        f.write(beta.astype(np.float32).tobytes())
    
    print(f"  [V] {os.path.basename(out_path)} (LayerNorm, {hidden_size} dims)")
    return True

def export_linear(tensors, hf_prefix, out_path):
    try:
        weight = tensors[f"{hf_prefix}.weight"]
        bias = tensors[f"{hf_prefix}.bias"]
    except KeyError:
        print(f"  [X] Erreur : Clés introuvables pour {hf_prefix}")
        return False

    # FIX GPT-2 : Transposition vitale pour notre C++ (Conv1D -> Linear)
    weight = weight.T 

    out_features, in_features = weight.shape

    # Quantification INT8
    max_abs = np.max(np.abs(weight), axis=1)
    scales = max_abs / 127.0
    scales[scales == 0] = 1e-9 
    weight_quantized = np.round(weight / scales[:, None]).astype(np.int8)

    # Offsets et alignement mémoire (4 octets)
    magic = 0x524E494C # 'LINR'
    version = 1
    header_size = 28 

    scales_offset = header_size
    weights_offset = scales_offset + (out_features * 4)
    raw_biases_offset = weights_offset + (out_features * in_features * 1)
    
    padding = (4 - (raw_biases_offset % 4)) % 4
    biases_offset = raw_biases_offset + padding

    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIIIIII", magic, version, in_features, out_features,
                            scales_offset, weights_offset, biases_offset))
        f.write(scales.astype(np.float32).tobytes())
        f.write(weight_quantized.tobytes())
        if padding > 0:
            f.write(b'\0' * padding)
        f.write(bias.astype(np.float32).tobytes())

    print(f"  [V] {os.path.basename(out_path)} (Linear, in={in_features}, out={out_features})")
    return True


# --- LE CŒUR DU SCRIPT ---

def build_layer(layer_idx):
    model_path = os.path.join(BASE_DIR, "model.safetensors")
    
    if not os.path.exists(model_path):
        print(f"Erreur : Le fichier {model_path} est introuvable.")
        return

    print(f"Chargement du modèle DistilGPT-2...")
    tensors = load_file(model_path)
    
    print(f"\n--- EXTRACTION DE LA COUCHE {layer_idx} ---")

    # Définition des chemins Hugging Face et des noms de fichiers de sortie
    # Format : (Nom HuggingFace interne, Nom de notre fichier, Fonction d'export)
    modules = [
        (f"transformer.h.{layer_idx}.ln_1", f"ln_1_{layer_idx}.bin", export_layernorm),
        (f"transformer.h.{layer_idx}.attn.c_attn", f"attn_c_attn_{layer_idx}.bin", export_linear),
        (f"transformer.h.{layer_idx}.attn.c_proj", f"attn_c_proj_{layer_idx}.bin", export_linear),
        (f"transformer.h.{layer_idx}.ln_2", f"ln_2_{layer_idx}.bin", export_layernorm),
        (f"transformer.h.{layer_idx}.mlp.c_fc", f"mlp_c_fc_{layer_idx}.bin", export_linear),
        (f"transformer.h.{layer_idx}.mlp.c_proj", f"mlp_c_proj_{layer_idx}.bin", export_linear),
    ]

    for hf_name, file_name, export_func in modules:
        out_path = os.path.join(BASE_DIR, file_name)
        export_func(tensors, hf_name, out_path)
        
    print(f"--- COUCHE {layer_idx} EXPORTÉE AVEC SUCCÈS ! ---\n")

if __name__ == "__main__":
    # Par défaut, on extrait la couche 0, mais on peut passer un argument dans la console
    target_layer = 0
    if len(sys.argv) > 1:
        try:
            target_layer = int(sys.argv[1])
        except ValueError:
            print("Veuillez fournir un numéro de couche valide (ex: python build_layer.py 0)")
            sys.exit(1)
            
    build_layer(target_layer)