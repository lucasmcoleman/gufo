#ifndef GUFO_MODELS_LAGUNA_TOKENIZER_HPP_
#define GUFO_MODELS_LAGUNA_TOKENIZER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::laguna {

using TokenId = tokenization::TokenId;

struct EncodeOptions {
  /// Prepend the BOS token (id 2 in Laguna-S-2.1). Never implicit: the default
  /// is false even though the GGUF sets `tokenizer.ggml.add_bos_token`, so a
  /// caller that wants llama.cpp's default framing must ask for it, and a
  /// comparison against llama-tokenize must pass the same choice on both sides
  /// (`llama-tokenize` adds BOS unless given `--no-bos`).
  bool add_bos{false};
  /// Match control-token text such as `〈|EOS|〉` as tokens. When false the
  /// user-defined tokens (`<think>`, `</think>`, `<assistant>`, `<tool_call>`,
  /// `</tool_call>`) are still matched, exactly as llama.cpp does.
  bool parse_special{true};
};

/// GPT-2 byte-level BPE with the Laguna pre-tokenizer (`tokenizer.ggml.pre ==
/// "laguna"`). The merge engine and vocabulary handling are the Qwen BPE
/// tokenizer's, driven with no pre-tokenizer of its own; this class owns the
/// special-token partition and the Laguna split, both ported from llama.cpp
/// d854d98 (`tokenizer_st_partition`, and the `[^\n]+|[\n]+` stage followed by
/// the Qwen2-style regex of LLAMA_VOCAB_PRE_TYPE_LAGUNA). Neither stage
/// normalizes the text.
class Tokenizer {
public:
  /// Builds the tokenizer from `tokenizer.ggml.*` metadata. Requires model
  /// `gpt2`, pre `laguna`, tokens, merges, token types, and BOS/EOS/PAD ids.
  [[nodiscard]] static std::unique_ptr<Tokenizer> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<TokenId> Encode(
      std::string_view text, const EncodeOptions& options = {}) const;

  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;

  /// One token's text (byte-level pieces, so a multi-byte character can span
  /// several tokens). Empty for an id outside the vocabulary.
  [[nodiscard]] std::string DecodeToken(TokenId token) const;

  /// The Laguna pre-tokenizer split alone: views into `text` that concatenate
  /// back to it. Invalid UTF-8 bytes each count as one symbol (llama.cpp
  /// throws instead).
  [[nodiscard]] static std::vector<std::string_view> PreTokenize(
      std::string_view text);

  [[nodiscard]] std::size_t vocab_size() const noexcept {
    return bpe_->GetVocabSize();
  }
  [[nodiscard]] TokenId bos_token_id() const noexcept { return bos_; }
  [[nodiscard]] TokenId eos_token_id() const noexcept { return eos_; }
  [[nodiscard]] TokenId pad_token_id() const noexcept { return pad_; }
  /// `tokenizer.ggml.eot_token_id` (`</assistant>`, 24), or kInvalidTokenId.
  [[nodiscard]] TokenId eot_token_id() const noexcept { return eot_; }
  /// Generation stops at EOS (2) or EOT (24), as generation_config.json's
  /// eos_token_id [2, 24] lists them.
  [[nodiscard]] bool IsStopToken(TokenId token) const noexcept {
    return token != tokenization::kInvalidTokenId &&
           (token == eos_ || token == eot_);
  }
  /// What the GGUF asks for (`tokenizer.ggml.add_bos_token`); informational,
  /// Encode never consults it.
  [[nodiscard]] bool gguf_adds_bos() const noexcept { return gguf_adds_bos_; }

private:
  struct Special {
    std::string text;
    TokenId id;
    bool user_defined;
  };

  Tokenizer() = default;

  std::unique_ptr<tokenization::QwenTokenizer> bpe_;
  /// Longest text first, as llama.cpp tries them; ties keep id order.
  std::vector<Special> specials_;
  TokenId bos_{tokenization::kInvalidTokenId};
  TokenId eos_{tokenization::kInvalidTokenId};
  TokenId pad_{tokenization::kInvalidTokenId};
  TokenId eot_{tokenization::kInvalidTokenId};
  bool gguf_adds_bos_{false};
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_TOKENIZER_HPP_
