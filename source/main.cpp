#include "tokenizer.hpp"

#include <iostream>
#include <string>

static void test(const TokenizerData& tokenizer, const std::string& text) {
    std::cout << "Input: [" << text << "]\n";
    std::vector<uint32_t> tokens = tokenizer.tokenize(text);
    std::cout << "Tokens (" << tokens.size() << "):";
    for (uint32_t id : tokens) {
        std::cout << " " << id;
    }
    std::cout << "\n\n";
}

int main() {
    TokenizerData tokenizer;

    if (!tokenizer.load("tokenizer.bin")) {
        std::cerr << "Erreur: impossible de charger tokenizer.bin\n";
        return 1;
    }

    test(tokenizer, "Hello, world!");
    test(tokenizer, "The quick brown fox");
    test(tokenizer, "Je suis un tokenizer.");

    return 0;
}