#ifndef GUFO_TESTS_MODELS_LAGUNA_ENGINE_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_ENGINE_FIXTURE_HPP_

// A tiny random Laguna model (executor_gpu.hpp) behind the real serving
// engine: a synthetic GPT-2 byte-level vocabulary laid out like the real one
// where it matters (BOS/EOS 2, PAD 9, `<think>` 18, `</think>` 19,
// `<assistant>` 23, EOT `</assistant>` 24, `<tool_call>` 25/26) so the chat
// template tokenizes the way it does on the real model, and a Model built
// from the parts. No model file is read.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "src/models/laguna/engine.hpp"
#include "tests/models/laguna/executor_gpu.hpp"
#include "tests/models/laguna/support.hpp"

namespace gufo::testing::laguna {

/// GPT-2's bytes_to_unicode: the character each byte token is spelled with.
inline std::string ByteLevelChar(unsigned byte) {
  unsigned cp = byte;
  const bool direct = (byte >= 33 && byte <= 126) ||
                      (byte >= 161 && byte <= 172) ||
                      (byte >= 174 && byte <= 255);
  if (!direct) {
    static const auto table = [] {
      std::array<unsigned, 256> t{};
      unsigned next = 0;
      for (unsigned b = 0; b < 256; ++b) {
        const bool d = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                       (b >= 174 && b <= 255);
        if (!d) {
          t[b] = 256 + next++;
        }
      }
      return t;
    }();
    cp = table[byte];
  }
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

/// Tokenizer metadata for a `vocab`-token model (at least 300).
inline Metadata TinyTokenizerMetadata(std::uint32_t vocab) {
  Require(vocab >= 300, "the tiny vocabulary needs room for 256 byte tokens");
  std::vector<std::string> tokens(vocab);
  std::vector<std::uint32_t> types(vocab, 1);
  const auto control = [&](std::uint32_t id, std::string text) {
    tokens[id] = std::move(text);
    types[id] = 3;
  };
  const auto user = [&](std::uint32_t id, std::string text) {
    tokens[id] = std::move(text);
    types[id] = 4;
  };
  for (std::uint32_t id = 0; id < 27; ++id) {
    control(id, "〈|S" + std::to_string(id) + "|〉");
  }
  control(0, "〈|UNK|〉");
  control(2, "〈|EOS|〉");
  control(9, "〈|PAD|〉");
  user(18, "<think>");
  user(19, "</think>");
  user(23, "<assistant>");
  control(24, "</assistant>");
  user(25, "<tool_call>");
  user(26, "</tool_call>");
  for (unsigned b = 0; b < 256; ++b) {
    tokens[27 + b] = ByteLevelChar(b);
  }
  tokens[283] = "th";
  tokens[284] = "the";
  for (std::uint32_t id = 285; id < vocab; ++id) {
    tokens[id] = "~f" + std::to_string(id);
  }
  Metadata m;
  m["general.architecture"] = std::string{"laguna"};
  m["tokenizer.ggml.model"] = std::string{"gpt2"};
  m["tokenizer.ggml.pre"] = std::string{"laguna"};
  m["tokenizer.ggml.tokens"] = tokens;
  m["tokenizer.ggml.merges"] = std::vector<std::string>{"t h", "th e"};
  m["tokenizer.ggml.token_type"] = types;
  m["tokenizer.ggml.bos_token_id"] = std::uint32_t{2};
  m["tokenizer.ggml.eos_token_id"] = std::uint32_t{2};
  m["tokenizer.ggml.padding_token_id"] = std::uint32_t{9};
  m["tokenizer.ggml.eot_token_id"] = std::uint32_t{24};
  m["tokenizer.ggml.add_bos_token"] = true;
  return m;
}

/// The tiny model resident on the GPU, with its tokenizer.
struct EngineFixture {
  std::shared_ptr<TinyModel> tiny;
  std::shared_ptr<gufo::models::laguna::Model> model;
};

inline EngineFixture MakeEngineFixture(
    gufo::models::laguna::ModelOptions options = {.max_context = 512,
                                                  .prefill_chunk = 16},
    const ModelSpec& spec = {}) {
  EngineFixture f;
  f.tiny = std::make_shared<TinyModel>(spec);
  std::vector<std::uint8_t> storage;
  const auto reader =
      OpenSynthetic(TinyTokenizerMetadata(spec.vocab), &storage);
  std::string error;
  auto tokenizer = gufo::models::laguna::Tokenizer::FromGguf(*reader, &error);
  Require(tokenizer != nullptr, "tiny tokenizer: " + error);
  auto weights =
      std::make_unique<gufo::models::laguna::ModelWeights>(f.tiny->weights());
  auto device = gufo::models::laguna::rocm::DeviceModel::Upload(
      *weights, nullptr, &error);
  Require(device != nullptr, "tiny upload: " + error);
  f.model = gufo::models::laguna::Model::FromParts(
      f.tiny, std::move(weights), std::move(device), std::move(tokenizer),
      options, "Laguna-tiny", &error);
  Require(f.model != nullptr, "tiny model: " + error);
  return f;
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_ENGINE_FIXTURE_HPP_
