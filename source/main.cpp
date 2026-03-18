#include "tokenizer.hpp"
#include "embedding.hpp"
#include "layernorm.hpp"

#include <iostream>
#include <string>
#include <vector>

static void test(const TokenizerData& tokenizer, const std::string& text) {
    std::cout << "Input : \"" << text << "\"\n";
    std::vector<uint32_t> ids = tokenizer.tokenize(text);
    std::cout << "IDs    (" << ids.size() << "):";
    for (uint32_t id : ids) {
        std::cout << " " << id;
    }
    std::cout << "\nTokens :";
    for (uint32_t id : ids) {
        std::cout << " [" << tokenizer.getToken(id) << "]";
    }
    std::cout << "\n\n";
}

static void test_embedding(const EmbeddingLayer& emb, uint32_t token_id, uint32_t position) {
    uint32_t hidden_size = emb.getHiddenSize();
    std::vector<float> output(hidden_size);

    std::cout << "Embedding forward(token=" << token_id << ", pos=" << position << "): ";
    if (!emb.forward(token_id, position, output.data())) {
        std::cout << "ECHEC\n";
        return;
    }

    // Afficher les 8 premières valeurs du vecteur de sortie
    uint32_t preview = (hidden_size < 8) ? hidden_size : 8;
    std::cout << "[";
    for (uint32_t i = 0; i < preview; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << output[i];
    }
    if (hidden_size > preview) std::cout << ", ...";
    std::cout << "] (dim=" << hidden_size << ")\n";
}

static void test_layernorm(const LayerNorm& ln, const std::vector<float>& input) {
    uint32_t dim = ln.getHiddenSize();
    std::vector<float> x(input.begin(), input.end());

    std::cout << "LayerNorm forward (dim=" << dim << ")\n";
    std::cout << "  Avant  : [";
    uint32_t preview = (dim < 8) ? dim : 8;
    for (uint32_t i = 0; i < preview; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << x[i];
    }
    if (dim > preview) std::cout << ", ...";
    std::cout << "]\n";

    ln.forward(x.data());

    std::cout << "  Apres  : [";
    for (uint32_t i = 0; i < preview; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << x[i];
    }
    if (dim > preview) std::cout << ", ...";
    std::cout << "]\n\n";
}

int main(int argc, char* argv[]) {
    const char* tok_path = (argc > 1) ? argv[1] : "tokenizer.bin";
    const char* emb_path = (argc > 2) ? argv[2] : "embeddings.bin";
    const char* ln_path  = (argc > 3) ? argv[3] : "layernorm.bin";

    // --- Test Tokenizer ---
    std::cout << "=== TEST TOKENIZER ===\n\n";
    TokenizerData tokenizer;
    if (!tokenizer.load(tok_path)) {
        std::cerr << "Erreur: impossible de charger " << tok_path << "\n";
        return 1;
    }

    tokenizer.debugPrint(10);
    std::cout << "\n";

    test(tokenizer, "Hello, world!");
    test(tokenizer, "The quick brown fox");
    test(tokenizer, "Je suis un tokenizer.");
    test(tokenizer, "The capital of France is");

    // --- Test Embedding ---
    std::cout << "=== TEST EMBEDDING ===\n\n";
    EmbeddingLayer emb;
    if (!emb.load(emb_path)) {
        std::cerr << "Erreur: impossible de charger " << emb_path << "\n";
        return 1;
    }

    std::cout << "hidden_size = " << emb.getHiddenSize() << "\n\n";

    // Tokeniser une phrase et passer chaque token dans l'embedding
    const std::string sentence = "Hello, world!";
    std::vector<uint32_t> ids = tokenizer.tokenize(sentence);
    std::cout << "Embeddings pour \"" << sentence << "\":\n";
    for (uint32_t pos = 0; pos < static_cast<uint32_t>(ids.size()); ++pos) {
        test_embedding(emb, ids[pos], pos);
    }
    std::cout << "\n";

    // Quelques tests individuels
    test_embedding(emb, 0, 0);
    test_embedding(emb, 1, 1);
    test_embedding(emb, 50256, 0); // OOV / token limite

    // --- Test LayerNorm ---
    std::cout << "=== TEST LAYERNORM ===\n\n";
    LayerNorm ln;
    if (!ln.load(ln_path)) {
        std::cerr << "Erreur: impossible de charger " << ln_path << "\n";
        return 1;
    }

    uint32_t hidden_size = ln.getHiddenSize();
    std::cout << "hidden_size = " << hidden_size << "\n\n";

    // Test 1 : vecteur nul (cas limite)
    {
        std::vector<float> zeros(hidden_size, 0.0f);
        std::cout << "Test vecteur nul :\n";
        test_layernorm(ln, zeros);
    }

    // Test 2 : vecteur constant (variance=0, cas numeriquement delicat)
    {
        std::vector<float> constant(hidden_size, 3.14f);
        std::cout << "Test vecteur constant :\n";
        test_layernorm(ln, constant);
    }

    // Test 3 : sortie embedding du premier token de "Hello, world!"
    {
        const std::vector<uint32_t> ids2 = tokenizer.tokenize("Hello, world!");
        if (!ids2.empty()) {
            std::vector<float> emb_out(hidden_size);
            if (emb.forward(ids2[0], 0, emb_out.data())) {
                std::cout << "Test sur embedding(token=" << ids2[0] << ", pos=0) :\n";
                test_layernorm(ln, emb_out);
            }
        }
    }

    return 0;
}