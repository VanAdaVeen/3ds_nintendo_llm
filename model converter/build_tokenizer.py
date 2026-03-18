import json
import struct
import os

def build_tokenizer():
    # Trouve le dossier exact où se trouve build_tokenizer.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construit le chemin absolu vers tes fichiers
    vocab_path = os.path.join(script_dir, "../datas/original model/vocab.json")
    merges_path = os.path.join(script_dir, "../datas/original model/merges.txt")

    print("Chargement de vocab.json...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    print("Chargement de merges.txt...")
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 1. Parsing des règles de fusion
    merges = []
    rank = 0
    # On ignore la première ligne qui est le header "#version: 0.2..."
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
            
        left_str, right_str = parts
        merged_str = left_str + right_str

        # On vérifie que les tokens existent bien dans le vocabulaire
        if left_str in vocab and right_str in vocab and merged_str in vocab:
            left_id = vocab[left_str]
            right_id = vocab[right_str]
            result_id = vocab[merged_str]
            merges.append((left_id, right_id, result_id, rank))
            rank += 1

    # LE SECRET DE LA VITESSE SUR 3DS : On trie par left_id puis right_id
    print("Tri des règles de fusion pour la recherche dichotomique...")
    merges.sort(key=lambda x: (x[0], x[1]))

    # 2. Préparation des Entrées et du Blob de texte
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab_size = len(sorted_vocab)
    
    blob = bytearray()
    entries = []
    
    for token_str, token_id in sorted_vocab:
        # GPT-2 utilise des caractères Unicode spéciaux, on encode en UTF-8
        token_bytes = token_str.encode('utf-8')
        offset = len(blob)
        length = len(token_bytes)
        entries.append((offset, length))
        blob.extend(token_bytes)

    # ALIGNEMENT MÉMOIRE : L'ARM11 déteste lire des uint32_t sur des adresses non multiples de 4
    # On ajoute des octets nuls (padding) au blob pour que la section suivante (merges) soit alignée.
    padding_len = (4 - (len(blob) % 4)) % 4
    blob.extend(b'\0' * padding_len)

    # 3. Calcul des offsets et création du Header
    magic = 0x4E4B4F54 # "TOKN" en ASCII (Little Endian)
    version = 3
    merges_count = len(merges)
    
    # GPT-2 utilise le même ID (50256) pour EOS, BOS, UNK et PAD
    special_id = vocab.get("<|endoftext|>", 50256) 

    header_size = 44 # 11 champs * 4 octets
    entries_size = vocab_size * 6 # offset(4) + length(2)
    
    entries_offset = header_size
    blob_offset = entries_offset + entries_size
    blob_size = len(blob)
    merges_offset = blob_offset + blob_size

    print(f"Génération de tokenizer.bin (Vocab: {vocab_size}, Merges: {merges_count})...")
    
    with open("tokenizer.bin", "wb") as f:
        # < indique Little Endian (format natif de la 3DS)
        # I = uint32, i = int32
        header = struct.pack("<IIIIiiiiIII", 
                             magic, version, vocab_size, merges_count,
                             special_id, special_id, special_id, special_id,
                             entries_offset, blob_offset, merges_offset)
        f.write(header)

        # Écriture des Entries (offset uint32, length uint16)
        for offset, length in entries:
            f.write(struct.pack("<IH", offset, length))

        # Écriture du texte brut
        f.write(blob)

        # Écriture des Merges (4 * uint32 = 16 octets par règle)
        for left_id, right_id, result_id, rnk in merges:
            f.write(struct.pack("<IIII", left_id, right_id, result_id, rnk))

    print("Terminé ! Fichier tokenizer.bin prêt pour la carte SD.")

if __name__ == "__main__":
    build_tokenizer()