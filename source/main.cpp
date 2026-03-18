#include "tokenizer.hpp"
#include "embedding.hpp"
#include "layernorm.hpp"
#include "linear.hpp"
#include "attention.hpp"
#include "gelu.hpp"
#include "transformer_block.hpp"

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

static void print_preview(const float* vec, uint32_t dim) {
    uint32_t preview = (dim < 8) ? dim : 8;
    std::cout << "[";
    for (uint32_t i = 0; i < preview; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << vec[i];
    }
    if (dim > preview) std::cout << ", ...";
    std::cout << "]";
}

static void test_layernorm(const LayerNorm& ln, const std::vector<float>& input) {
    uint32_t dim = ln.getHiddenSize();
    std::vector<float> x(input.begin(), input.end());

    std::cout << "LayerNorm forward (dim=" << dim << ")\n";
    std::cout << "  Avant  : "; print_preview(x.data(), dim); std::cout << "\n";
    ln.forward(x.data());
    std::cout << "  Apres  : "; print_preview(x.data(), dim); std::cout << "\n\n";
}

static void test_gelu(const Gelu& gelu, const std::vector<float>& input, const std::string& label) {
    std::vector<float> x(input);
    gelu.forward(x.data(), static_cast<uint32_t>(x.size()));

    std::cout << "GELU forward \"" << label << "\" (size=" << x.size() << ")\n";
    std::cout << "  Avant  : "; print_preview(input.data(), static_cast<uint32_t>(input.size())); std::cout << "\n";
    std::cout << "  Apres  : "; print_preview(x.data(),     static_cast<uint32_t>(x.size()));     std::cout << "\n\n";
}

static void test_transformer_block(TransformerBlock& block, const float* input,
                                   const std::string& label) {
    const uint32_t hidden_size = 768;
    std::vector<float> x(input, input + hidden_size);

    std::cout << "TransformerBlock forward \"" << label << "\"\n";
    std::cout << "  Avant  : "; print_preview(x.data(), hidden_size); std::cout << "\n";
    if (!block.forward(x.data())) {
        std::cout << "  ECHEC\n\n";
        return;
    }
    std::cout << "  Apres  : "; print_preview(x.data(), hidden_size); std::cout << "\n\n";
}

static void test_attention(Attention& attn, const std::vector<float>& qkv_input,
                           uint32_t hidden_size, const std::string& label) {
    std::cout << "Attention forward \"" << label << "\" (qkv_size=" << qkv_input.size() << ")\n";

    std::vector<float> output(hidden_size);
    if (!attn.forward(qkv_input.data(), output.data())) {
        std::cout << "  ECHEC\n\n";
        return;
    }

    std::cout << "  QKV[Q] : "; print_preview(qkv_input.data(), hidden_size);         std::cout << "\n";
    std::cout << "  Sortie : "; print_preview(output.data(), hidden_size);             std::cout << "\n\n";
}

static void test_linear(const Linear& lin, const std::vector<float>& input, const std::string& label) {
    uint32_t in_feat  = lin.getInFeatures();
    uint32_t out_feat = lin.getOutFeatures();

    std::cout << "Linear forward \"" << label << "\" (in=" << in_feat << ", out=" << out_feat << ")\n";

    if (static_cast<uint32_t>(input.size()) != in_feat) {
        std::cout << "  SKIP : taille d'entree incorrecte (" << input.size() << " != " << in_feat << ")\n\n";
        return;
    }

    std::vector<float> output(out_feat);
    if (!lin.forward(input.data(), output.data())) {
        std::cout << "  ECHEC\n\n";
        return;
    }

    std::cout << "  Entree : "; print_preview(input.data(), in_feat);   std::cout << "\n";
    std::cout << "  Sortie : "; print_preview(output.data(), out_feat); std::cout << "\n\n";
}

int main(int argc, char* argv[]) {
    const char* tok_path    = (argc > 1) ? argv[1] : "datas/adapted model/tokenizer.bin";
    const char* emb_path    = (argc > 2) ? argv[2] : "datas/adapted model/embeddings.bin";
    const char* ln_path     = (argc > 3) ? argv[3] : "datas/adapted model/ln_f.bin";
    const char* linear_path = (argc > 4) ? argv[4] : "datas/adapted model/attn_c_attn_0.bin";
    const char* layer_path  = (argc > 5) ? argv[5] : "datas/adapted model";

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

    // --- Test Linear ---
    std::cout << "=== TEST LINEAR ===\n\n";
    Linear lin;
    if (!lin.load(linear_path)) {
        std::cerr << "Erreur: impossible de charger " << linear_path << "\n";
        return 1;
    }

    uint32_t in_feat  = lin.getInFeatures();
    uint32_t out_feat = lin.getOutFeatures();
    std::cout << "in_features=" << in_feat << ", out_features=" << out_feat << "\n\n";

    // Test 1 : vecteur nul
    {
        std::vector<float> zeros(in_feat, 0.0f);
        test_linear(lin, zeros, "vecteur nul");
    }

    // Test 2 : vecteur constant (1.0)
    {
        std::vector<float> ones(in_feat, 1.0f);
        test_linear(lin, ones, "vecteur constant 1.0");
    }

    // Test 3 : sortie embedding + layernorm du premier token de "Hello, world!"
    {
        const std::vector<uint32_t> ids3 = tokenizer.tokenize("Hello, world!");
        if (!ids3.empty() && in_feat == ln.getHiddenSize()) {
            std::vector<float> emb_out(in_feat);
            if (emb.forward(ids3[0], 0, emb_out.data())) {
                ln.forward(emb_out.data());
                test_linear(lin, emb_out, "emb+ln(token=" + std::to_string(ids3[0]) + ")");
            }
        }
    }

    // --- Test GELU ---
    std::cout << "=== TEST GELU ===\n\n";

    Gelu gelu;

    // Test 1 : vecteur nul — GELU(0) = 0
    {
        std::vector<float> zeros(8, 0.0f);
        test_gelu(gelu, zeros, "vecteur nul");
    }

    // Test 2 : valeurs connues — GELU(1.0) ≈ 0.841, GELU(-1.0) ≈ -0.159
    {
        std::vector<float> known = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 10.0f};
        test_gelu(gelu, known, "valeurs connues");
    }

    // Test 3 : sortie Linear (taille out_features, typiquement 3072 pour le MLP DistilGPT-2)
    {
        std::vector<float> ones(out_feat, 1.0f);
        test_gelu(gelu, ones, "vecteur 1.0 (taille out_features=" + std::to_string(out_feat) + ")");
    }

    // Test 4 : pipeline emb -> ln -> linear -> GELU
    if (in_feat == 768) {
        const std::vector<uint32_t> ids5 = tokenizer.tokenize("Hello, world!");
        if (!ids5.empty()) {
            std::vector<float> emb_out(768);
            if (emb.forward(ids5[0], 0, emb_out.data())) {
                ln.forward(emb_out.data());
                std::vector<float> lin_out(out_feat);
                if (lin.forward(emb_out.data(), lin_out.data())) {
                    test_gelu(gelu, lin_out,
                              "emb+ln+linear(token=" + std::to_string(ids5[0]) + ")");
                }
            }
        }
    }

    // --- Test Attention ---
    std::cout << "=== TEST ATTENTION ===\n\n";

    Attention attn;
    attn.init(768, 12, 1024);
    std::cout << "init(hidden=768, heads=12, max_seq=1024)\n\n";

    // Test 1 : vecteur QKV nul (un seul token)
    {
        std::vector<float> qkv_zeros(768 * 3, 0.0f);
        attn.resetCache();
        test_attention(attn, qkv_zeros, 768, "QKV nul");
    }

    // Test 2 : vecteur QKV constant 0.1 (un seul token)
    {
        std::vector<float> qkv_const(768 * 3, 0.1f);
        attn.resetCache();
        test_attention(attn, qkv_const, 768, "QKV constant 0.1");
    }

    // Test 3 : pipeline complet emb -> ln -> linear(QKV) -> attention
    // On passe chaque token de "Hello, world!" séquentiellement pour exercer le KV cache
    if (in_feat == 768 && out_feat == 768 * 3) {
        const std::string pipeline_sentence = "Hello, world!";
        const std::vector<uint32_t> ids4 = tokenizer.tokenize(pipeline_sentence);
        attn.resetCache();
        std::cout << "Pipeline emb+ln+linear+attn sur \"" << pipeline_sentence
                  << "\" (" << ids4.size() << " tokens) :\n\n";

        for (uint32_t pos = 0; pos < static_cast<uint32_t>(ids4.size()); ++pos) {
            std::vector<float> emb_out(768);
            if (!emb.forward(ids4[pos], pos, emb_out.data())) continue;

            ln.forward(emb_out.data());

            std::vector<float> qkv(768 * 3);
            if (!lin.forward(emb_out.data(), qkv.data())) continue;

            test_attention(attn, qkv, 768,
                           "token=" + std::to_string(ids4[pos]) + " pos=" + std::to_string(pos));
        }
    } else {
        std::cout << "SKIP pipeline : linear.bin n'a pas les dimensions QKV attendues "
                  << "(in=768, out=2304)\n\n";
    }

    // --- Test TransformerBlock ---
    std::cout << "=== TEST TRANSFORMER BLOCK (couche 0) ===\n\n";

    TransformerBlock block;
    if (!block.load(layer_path, 0)) {
        std::cerr << "Erreur: impossible de charger la couche 0 depuis \"" << layer_path << "\"\n"
                  << "Fichiers attendus : ln_1_0.bin, attn_c_attn_0.bin, attn_c_proj_0.bin,\n"
                  << "                   ln_2_0.bin, mlp_c_fc_0.bin, mlp_c_proj_0.bin\n";
        return 1;
    }
    block.initAttention(1024);
    std::cout << "load(\"" << layer_path << "\", 0) OK\n\n";

    // Test 1 : vecteur nul (un seul token)
    {
        std::vector<float> zeros(768, 0.0f);
        block.resetCache();
        test_transformer_block(block, zeros.data(), "vecteur nul");
    }

    // Test 2 : vecteur constant 1.0 (un seul token)
    {
        std::vector<float> ones(768, 1.0f);
        block.resetCache();
        test_transformer_block(block, ones.data(), "vecteur constant 1.0");
    }

    // Test 3 : pipeline complet emb -> transformer_block pour chaque token de "Hello, world!"
    // Exerce le KV cache (le bloc voit les tokens un par un, en séquence)
    {
        const std::string sentence = "Hello, world!";
        const std::vector<uint32_t> ids6 = tokenizer.tokenize(sentence);
        block.resetCache();
        std::cout << "Pipeline emb+block sur \"" << sentence
                  << "\" (" << ids6.size() << " tokens) :\n\n";

        for (uint32_t pos = 0; pos < static_cast<uint32_t>(ids6.size()); ++pos) {
            std::vector<float> emb_out(768);
            if (!emb.forward(ids6[pos], pos, emb_out.data())) continue;
            test_transformer_block(block, emb_out.data(),
                                   "token=" + std::to_string(ids6[pos]) + " pos=" + std::to_string(pos));
        }
    }

    return 0;
}