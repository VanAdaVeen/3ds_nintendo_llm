import json
import os

# 1. On récupère le dossier absolu dans lequel se trouve ce script Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# L'algorithme officiel d'OpenAI pour le Byte-Level BPE
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def main():
    # 2. Utilisation de os.path.join pour un chemin propre et cross-platform
    vocab_path = os.path.join(BASE_DIR, "vocab.json")
    
    if not os.path.exists(vocab_path):
        print(f"Erreur : Le fichier {vocab_path} est introuvable.")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    byte_to_unicode = bytes_to_unicode()

    # 3. Génération du code C++ directement dans un fichier header
    output_path = os.path.join(BASE_DIR, "gpt2_byte_map.hpp")
    
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("#pragma once\n\n")
        out.write("#include <cstdint>\n\n")
        out.write("// Généré automatiquement pour DistilGPT-2\n")
        out.write("const uint32_t gpt2_byte_to_id[256] = {\n")

        for i in range(256):
            unicode_char = byte_to_unicode[i]
            token_id = vocab[unicode_char]
            out.write(f"    {token_id}, // octet {i}\n")

        out.write("};\n")
        
    print(f"Succès ! Fichier généré ici : {output_path}")

if __name__ == "__main__":
    main()