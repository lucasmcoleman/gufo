#ifndef GUFO_MODELS_GLM53_FLASH_TOKENIZER_HPP_
#define GUFO_MODELS_GLM53_FLASH_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm53_flash {

using TokenId = std::uint32_t;

constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;

/// Ids named by the GGUF `tokenizer.ggml.*_token_id` keys.
struct SpecialIds {
  TokenId eos{kInvalidTokenId};
  TokenId pad{kInvalidTokenId};
  TokenId unk{kInvalidTokenId};
  TokenId bos{kInvalidTokenId};
  TokenId eot{kInvalidTokenId};
  TokenId eom{kInvalidTokenId};
};

/// Ids of the chat-template control tokens, looked up by their vocabulary
/// strings. A token the vocabulary lacks stays kInvalidTokenId.
struct ChatTokens {
  TokenId mask{kInvalidTokenId};               ///< [MASK]
  TokenId gmask{kInvalidTokenId};              ///< [gMASK]
  TokenId smask{kInvalidTokenId};              ///< [sMASK]
  TokenId sop{kInvalidTokenId};                ///< <sop>
  TokenId eop{kInvalidTokenId};                ///< <eop>
  TokenId system{kInvalidTokenId};             ///< <|system|>
  TokenId user{kInvalidTokenId};               ///< <|user|>
  TokenId assistant{kInvalidTokenId};          ///< <|assistant|>
  TokenId observation{kInvalidTokenId};        ///< <|observation|>
  TokenId think{kInvalidTokenId};              ///< <think>
  TokenId think_end{kInvalidTokenId};          ///< </think>
  TokenId tool_call{kInvalidTokenId};          ///< <tool_call>
  TokenId tool_call_end{kInvalidTokenId};      ///< </tool_call>
  TokenId tool_response{kInvalidTokenId};      ///< <tool_response>
  TokenId tool_response_end{kInvalidTokenId};  ///< </tool_response>
  TokenId arg_key{kInvalidTokenId};            ///< <arg_key>
  TokenId arg_key_end{kInvalidTokenId};        ///< </arg_key>
  TokenId arg_value{kInvalidTokenId};          ///< <arg_value>
  TokenId arg_value_end{kInvalidTokenId};      ///< </arg_value>
  TokenId nothink{kInvalidTokenId};            ///< /nothink
  TokenId box_begin{kInvalidTokenId};          ///< <|begin_of_box|>
  TokenId box_end{kInvalidTokenId};            ///< <|end_of_box|>
};

/// GLM-5.3-Flash byte-level BPE tokenizer (`tokenizer.ggml.model=gpt2`,
/// `tokenizer.ggml.pre=glm4`), matching llama.cpp's
/// LLAMA_VOCAB_PRE_TYPE_CHATGLM4 with ignore_merges. Encode adds no BOS/EOS:
/// the chat template writes [gMASK]<sop> itself.
class Tokenizer {
public:
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) noexcept = default;
  Tokenizer& operator=(Tokenizer&&) noexcept = default;
  ~Tokenizer() = default;

  /// Reads tokens, merges, token types and special ids from GGUF metadata.
  [[nodiscard]] static std::unique_ptr<Tokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Encodes UTF-8 text. The text of a user-defined vocabulary token
  /// (<think>, <tool_call>, /nothink, ...) always becomes its id, as in
  /// llama.cpp. Control and unknown tokens ([gMASK], <|user|>, ...) do so only
  /// with `parse_special`; otherwise they are tokenized as ordinary text.
  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text,
                                            bool parse_special) const;

  /// Decodes ids to UTF-8. Control and user-defined tokens decode to their
  /// vocabulary text; ids beyond the vocabulary decode to nothing.
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;
  [[nodiscard]] std::string DecodeToken(TokenId token) const;

  [[nodiscard]] std::size_t VocabSize() const noexcept {
    return id_to_token_.size();
  }
  [[nodiscard]] const SpecialIds& special_ids() const noexcept {
    return special_ids_;
  }
  [[nodiscard]] const ChatTokens& chat_tokens() const noexcept {
    return chat_tokens_;
  }
  /// True for control and user-defined vocabulary entries.
  [[nodiscard]] bool IsSpecialToken(TokenId id) const noexcept;
  [[nodiscard]] TokenId FindToken(std::string_view text) const noexcept;

private:
  struct PairHash {
    std::size_t operator()(
        const std::pair<TokenId, TokenId>& p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32) ^
             static_cast<std::size_t>(p.second);
    }
  };
  struct Merge {
    std::uint32_t rank;
    TokenId token;
  };

  Tokenizer() = default;

  void EncodeOrdinary(std::string_view text, std::vector<TokenId>& out) const;
  void EncodePiece(std::string_view piece, std::vector<TokenId>& out) const;

  std::vector<std::string> id_to_token_;
  std::vector<bool> is_special_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::pair<TokenId, TokenId>, Merge, PairHash> merges_;
  struct SpecialEntry {
    std::string text;
    TokenId id;
    bool user_defined;  ///< Matched even when parse_special is off.
  };
  /// Special vocabulary entries, longest text first.
  std::vector<SpecialEntry> specials_;
  std::array<TokenId, 256> byte_tokens_{};
  SpecialIds special_ids_;
  ChatTokens chat_tokens_;
};

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_TOKENIZER_HPP_
