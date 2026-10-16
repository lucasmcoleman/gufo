// A6 parity probe: encodes every line of a corpus with the GLM-5.3-Flash
// tokenizer for comparison with llama.cpp's llama-tokenize.
//
// usage: glm53_flash_tokenize_probe MODEL.gguf CORPUS.hex
// Each CORPUS line is the hex of one UTF-8 string. For each, prints
//   <index> special <ids...>
//   <index> plain <ids...>
//   <index> roundtrip <0|1>   (both modes decode back to the input)

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/tokenizer.hpp"

namespace glm = gufo::models::glm53_flash;

namespace {

std::string FromHex(const std::string& hex) {
  std::string out;
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
  }
  return out;
}

void PrintIds(std::size_t index, const char* mode,
              const std::vector<glm::TokenId>& ids) {
  std::printf("%zu %s", index, mode);
  for (const auto id : ids) {
    std::printf(" %u", id);
  }
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " MODEL.gguf CORPUS.hex\n";
    return 2;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::cerr << "open failed: " << error << "\n";
    return 1;
  }
  const auto tokenizer = glm::Tokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::cerr << "tokenizer failed: " << error << "\n";
    return 1;
  }
  std::ifstream corpus(argv[2]);
  std::string line;
  for (std::size_t index = 0; std::getline(corpus, line); ++index) {
    const std::string text = FromHex(line);
    const auto special = tokenizer->Encode(text, true);
    const auto plain = tokenizer->Encode(text, false);
    PrintIds(index, "special", special);
    PrintIds(index, "plain", plain);
    const bool ok =
        tokenizer->Decode(special) == text && tokenizer->Decode(plain) == text;
    std::printf("%zu roundtrip %d\n", index, ok ? 1 : 0);
  }
  return 0;
}
