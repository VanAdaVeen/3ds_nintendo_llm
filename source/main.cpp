#include "tokenizer.hpp"

#include <iostream>
#include <string>

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

int main(int argc, char* argv[]) {
    const char* path = (argc > 1) ? argv[1] : "tokenizer.bin";

    TokenizerData tokenizer;
    if (!tokenizer.load(path)) {
        std::cerr << "Erreur: impossible de charger " << path << "\n";
        return 1;
    }

    tokenizer.debugPrint(10);
    std::cout << "\n";

    test(tokenizer, "Hello, world!");
    test(tokenizer, "The quick brown fox");
    test(tokenizer, "Je suis un tokenizer.");
    test(tokenizer, "The capital of France is");

    return 0;
}