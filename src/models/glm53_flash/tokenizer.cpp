#include "src/models/glm53_flash/tokenizer.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace gufo::models::glm53_flash {

namespace {

// GGUF tokenizer.ggml.token_type values (llama.cpp's llama_token_type).
constexpr std::int64_t kTypeUnknown = 2;
constexpr std::int64_t kTypeControl = 3;
constexpr std::int64_t kTypeUserDefined = 4;

/// The GPT-2 byte-to-unicode table: printable bytes map to themselves, the
/// rest to U+0100 upward in byte order.
const std::array<std::string, 256>& ByteToUtf8Table() {
  static const auto table = [] {
    std::array<std::string, 256> t;
    const auto encode = [](int cp) {
      std::string s;
      if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
      } else {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
      return s;
    };
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      const bool printable =
          (b >= '!' && b <= '~') || (b >= 161 && b <= 172) || b >= 174;
      t[b] = encode(printable ? b : 256 + extra++);
    }
    return t;
  }();
  return table;
}

/// Inverse of ByteToUtf8Table over two-byte (or ASCII) code points; any
/// sequence that is not a table entry is copied through unchanged.
std::string UnescapeBytes(std::string_view text) {
  static const auto inverse = [] {
    std::unordered_map<std::string, std::uint8_t> m;
    const auto& table = ByteToUtf8Table();
    for (int b = 0; b < 256; ++b) {
      m.emplace(table[b], static_cast<std::uint8_t>(b));
    }
    return m;
  }();
  std::string out;
  out.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    const auto lead = static_cast<unsigned char>(text[i]);
    const std::size_t len = lead < 0x80 ? 1 : (lead & 0xE0U) == 0xC0U ? 2 : 1;
    const auto found = i + len <= text.size()
                           ? inverse.find(std::string(text.substr(i, len)))
                           : inverse.end();
    if (found != inverse.end()) {
      out.push_back(static_cast<char>(found->second));
    } else {
      out.append(text.substr(i, len));
    }
    i += len;
  }
  return out;
}

struct CodePoint {
  UChar32 value{0};
  std::size_t length{1};
};

CodePoint DecodeUtf8(std::string_view text, std::size_t offset) noexcept {
  const auto first = static_cast<std::uint8_t>(text[offset]);
  if (first < 0x80U) {
    return {.value = first, .length = 1};
  }
  std::size_t length = 0;
  UChar32 value = 0;
  if ((first & 0xE0U) == 0xC0U) {
    length = 2;
    value = static_cast<UChar32>(first & 0x1FU);
  } else if ((first & 0xF0U) == 0xE0U) {
    length = 3;
    value = static_cast<UChar32>(first & 0x0FU);
  } else if ((first & 0xF8U) == 0xF0U) {
    length = 4;
    value = static_cast<UChar32>(first & 0x07U);
  } else {
    return {.value = first, .length = 1};
  }
  if (offset + length > text.size()) {
    return {.value = first, .length = 1};
  }
  for (std::size_t i = 1; i < length; ++i) {
    const auto byte = static_cast<std::uint8_t>(text[offset + i]);
    if ((byte & 0xC0U) != 0x80U) {
      return {.value = first, .length = 1};
    }
    value = static_cast<UChar32>((value << 6U) | (byte & 0x3FU));
  }
  return {.value = value, .length = length};
}

bool IsLetter(UChar32 c) noexcept {
  return (U_GET_GC_MASK(c) & U_GC_L_MASK) != 0;
}

bool IsNumber(UChar32 c) noexcept {
  return (U_GET_GC_MASK(c) & U_GC_N_MASK) != 0;
}

bool IsSpace(UChar32 c) noexcept {
  return u_isUWhiteSpace(c) != 0;
}

bool IsNewline(UChar32 c) noexcept {
  return c == '\r' || c == '\n';
}

/// Length of the case-insensitive contraction at `offset` ('s 't 're 've 'm
/// 'll 'd), or 0.
std::size_t ContractionLength(std::string_view text, std::size_t offset) {
  if (text[offset] != '\'') {
    return 0;
  }
  constexpr std::array<std::string_view, 7> kSuffixes = {"s", "t",  "re", "ve",
                                                         "m", "ll", "d"};
  const std::string_view rest = text.substr(offset + 1);
  for (const std::string_view suffix : kSuffixes) {
    if (rest.size() < suffix.size()) {
      continue;
    }
    bool match = true;
    for (std::size_t i = 0; i < suffix.size(); ++i) {
      const char c = rest[i];
      const char lower = c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c;
      if (lower != suffix[i]) {
        match = false;
        break;
      }
    }
    if (match) {
      return 1 + suffix.size();
    }
  }
  return 0;
}

std::size_t LetterRunEnd(std::string_view text, std::size_t pos) {
  while (pos < text.size()) {
    const CodePoint c = DecodeUtf8(text, pos);
    if (!IsLetter(c.value)) {
      break;
    }
    pos += c.length;
  }
  return pos;
}

/// End of the next pre-tokenizer piece of llama.cpp's CHATGLM4 split:
///   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
///   |[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*
///   |\s*[\r\n]+|\s+(?!\S)|\s+
std::size_t PieceEnd(std::string_view text, std::size_t offset) {
  if (const std::size_t n = ContractionLength(text, offset); n != 0) {
    return offset + n;
  }

  const CodePoint first = DecodeUtf8(text, offset);
  const std::size_t after_first = offset + first.length;

  if (IsLetter(first.value)) {
    return LetterRunEnd(text, after_first);
  }
  if (!IsNewline(first.value) && !IsNumber(first.value) &&
      after_first < text.size() &&
      IsLetter(DecodeUtf8(text, after_first).value)) {
    return LetterRunEnd(text, after_first);
  }

  if (IsNumber(first.value)) {
    std::size_t end = offset;
    for (int digits = 0; digits < 3 && end < text.size(); ++digits) {
      const CodePoint c = DecodeUtf8(text, end);
      if (!IsNumber(c.value)) {
        break;
      }
      end += c.length;
    }
    return end;
  }

  const auto is_symbol = [](UChar32 c) {
    return !IsSpace(c) && !IsLetter(c) && !IsNumber(c);
  };
  std::size_t symbol_start = offset;
  if (first.value == ' ' && after_first < text.size() &&
      is_symbol(DecodeUtf8(text, after_first).value)) {
    symbol_start = after_first;
  }
  if (is_symbol(DecodeUtf8(text, symbol_start).value)) {
    std::size_t end = symbol_start;
    while (end < text.size()) {
      const CodePoint c = DecodeUtf8(text, end);
      if (!is_symbol(c.value)) {
        break;
      }
      end += c.length;
    }
    while (end < text.size()) {
      const CodePoint c = DecodeUtf8(text, end);
      if (!IsNewline(c.value)) {
        break;
      }
      end += c.length;
    }
    return end;
  }

  if (IsSpace(first.value)) {
    std::size_t end = offset;
    std::size_t last_start = offset;
    std::size_t last_newline_end = offset;
    while (end < text.size()) {
      const CodePoint c = DecodeUtf8(text, end);
      if (!IsSpace(c.value)) {
        break;
      }
      last_start = end;
      end += c.length;
      if (IsNewline(c.value)) {
        last_newline_end = end;
      }
    }
    if (last_newline_end != offset) {
      return last_newline_end;  // \s*[\r\n]+ ends at the last newline.
    }
    // \s+(?!\S) gives the final space back to the next piece.
    return end < text.size() && last_start > offset ? last_start : end;
  }

  return after_first;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return nullptr;
  };

  if (reader.GetMetadataString("tokenizer.ggml.model") !=
      std::optional<std::string_view>{"gpt2"}) {
    return fail("tokenizer.ggml.model is not gpt2");
  }
  const auto pre = reader.GetMetadataString("tokenizer.ggml.pre");
  if (pre != std::optional<std::string_view>{"glm4"}) {
    return fail("tokenizer.ggml.pre is not glm4");
  }
  const auto token_views =
      reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (token_views.empty()) {
    return fail("GGUF metadata missing tokenizer.ggml.tokens");
  }
  const auto merge_views =
      reader.GetMetadataStringArray("tokenizer.ggml.merges");
  if (merge_views.empty()) {
    return fail("GGUF metadata missing tokenizer.ggml.merges");
  }

  auto tok = std::unique_ptr<Tokenizer>(new Tokenizer());
  tok->id_to_token_.reserve(token_views.size());
  tok->token_to_id_.reserve(token_views.size());
  for (std::size_t i = 0; i < token_views.size(); ++i) {
    tok->id_to_token_.emplace_back(token_views[i]);
    tok->token_to_id_.emplace(tok->id_to_token_.back(),
                              static_cast<TokenId>(i));
  }

  // Control, user-defined and unknown entries are the ones llama.cpp matches
  // in the raw text before BPE.
  tok->is_special_.assign(token_views.size(), false);
  const auto* types = reader.FindMetadata("tokenizer.ggml.token_type");
  if (types == nullptr) {
    return fail("GGUF metadata missing tokenizer.ggml.token_type");
  }
  const auto type_at = [&](std::size_t i) -> std::int64_t {
    if (const auto* s = std::get_if<std::vector<std::int64_t>>(&types->value)) {
      return i < s->size() ? (*s)[i] : 1;
    }
    if (const auto* u =
            std::get_if<std::vector<std::uint64_t>>(&types->value)) {
      return i < u->size() ? static_cast<std::int64_t>((*u)[i]) : 1;
    }
    return 1;
  };
  for (std::size_t i = 0; i < token_views.size(); ++i) {
    const std::int64_t type = type_at(i);
    if (type == kTypeControl || type == kTypeUserDefined ||
        type == kTypeUnknown) {
      tok->is_special_[i] = true;
      tok->specials_.push_back({.text = tok->id_to_token_[i],
                                .id = static_cast<TokenId>(i),
                                .user_defined = type == kTypeUserDefined});
    }
  }
  std::stable_sort(tok->specials_.begin(), tok->specials_.end(),
                   [](const SpecialEntry& a, const SpecialEntry& b) {
                     return a.text.size() > b.text.size();
                   });

  tok->merges_.reserve(merge_views.size());
  for (std::uint32_t rank = 0; rank < merge_views.size(); ++rank) {
    const std::string_view merge = merge_views[rank];
    const std::size_t space = merge.find(' ');
    if (space == std::string_view::npos) {
      return fail("malformed merge entry at rank " + std::to_string(rank));
    }
    const auto left =
        tok->token_to_id_.find(std::string(merge.substr(0, space)));
    const auto right =
        tok->token_to_id_.find(std::string(merge.substr(space + 1)));
    if (left == tok->token_to_id_.end() || right == tok->token_to_id_.end()) {
      return fail("merge at rank " + std::to_string(rank) +
                  " names a token missing from the vocabulary");
    }
    std::string joined(merge.substr(0, space));
    joined.append(merge.substr(space + 1));
    const auto result = tok->token_to_id_.find(joined);
    tok->merges_.try_emplace(
        {left->second, right->second},
        Merge{.rank = rank,
              .token = result != tok->token_to_id_.end() ? result->second
                                                         : kInvalidTokenId});
  }

  const auto& table = ByteToUtf8Table();
  for (std::size_t b = 0; b < 256; ++b) {
    const auto it = tok->token_to_id_.find(table[b]);
    if (it == tok->token_to_id_.end()) {
      return fail("vocabulary has no token for byte " + std::to_string(b));
    }
    tok->byte_tokens_[b] = it->second;
  }

  const auto id_key = [&](std::string_view key) {
    const auto v = reader.GetMetadataUint32(key);
    return v && *v < tok->id_to_token_.size() ? *v : kInvalidTokenId;
  };
  tok->special_ids_ = {
      .eos = id_key("tokenizer.ggml.eos_token_id"),
      .pad = id_key("tokenizer.ggml.padding_token_id"),
      .unk = id_key("tokenizer.ggml.unknown_token_id"),
      .bos = id_key("tokenizer.ggml.bos_token_id"),
      .eot = id_key("tokenizer.ggml.eot_token_id"),
      .eom = id_key("tokenizer.ggml.eom_token_id"),
  };

  ChatTokens& c = tok->chat_tokens_;
  const std::pair<TokenId*, std::string_view> chat_names[] = {
      {&c.mask, "[MASK]"},
      {&c.gmask, "[gMASK]"},
      {&c.smask, "[sMASK]"},
      {&c.sop, "<sop>"},
      {&c.eop, "<eop>"},
      {&c.system, "<|system|>"},
      {&c.user, "<|user|>"},
      {&c.assistant, "<|assistant|>"},
      {&c.observation, "<|observation|>"},
      {&c.think, "<think>"},
      {&c.think_end, "</think>"},
      {&c.tool_call, "<tool_call>"},
      {&c.tool_call_end, "</tool_call>"},
      {&c.tool_response, "<tool_response>"},
      {&c.tool_response_end, "</tool_response>"},
      {&c.arg_key, "<arg_key>"},
      {&c.arg_key_end, "</arg_key>"},
      {&c.arg_value, "<arg_value>"},
      {&c.arg_value_end, "</arg_value>"},
      {&c.nothink, "/nothink"},
      {&c.box_begin, "<|begin_of_box|>"},
      {&c.box_end, "<|end_of_box|>"},
  };
  for (const auto& [slot, name] : chat_names) {
    *slot = tok->FindToken(name);
  }
  return tok;
}

TokenId Tokenizer::FindToken(std::string_view text) const noexcept {
  const auto it = token_to_id_.find(std::string(text));
  return it == token_to_id_.end() ? kInvalidTokenId : it->second;
}

bool Tokenizer::IsSpecialToken(TokenId id) const noexcept {
  return id < is_special_.size() && is_special_[id];
}

void Tokenizer::EncodePiece(std::string_view piece,
                            std::vector<TokenId>& out) const {
  const auto& table = ByteToUtf8Table();
  std::string encoded;
  encoded.reserve(piece.size() * 2);
  std::vector<TokenId> word;
  word.reserve(piece.size());
  for (const unsigned char byte : piece) {
    encoded += table[byte];
    word.push_back(byte_tokens_[byte]);
  }

  // glm4 sets ignore_merges: a piece that is itself a token is emitted whole.
  if (const auto whole = token_to_id_.find(encoded);
      whole != token_to_id_.end()) {
    out.push_back(whole->second);
    return;
  }
  if (word.size() <= 1) {
    out.insert(out.end(), word.begin(), word.end());
    return;
  }

  // Lowest-ranked adjacent pair first, leftmost on a tie; a linked list and a
  // heap keep long pieces linear-ish. Stale heap entries are skipped.
  constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
  const std::size_t count = word.size();
  std::vector<std::size_t> previous(count);
  std::vector<std::size_t> following(count);
  for (std::size_t i = 0; i < count; ++i) {
    previous[i] = i == 0 ? kNone : i - 1;
    following[i] = i + 1 < count ? i + 1 : kNone;
  }
  struct Candidate {
    std::uint32_t rank;
    std::size_t index;
    TokenId left;
    TokenId right;
  };
  const auto later = [](const Candidate& a, const Candidate& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.index > b.index;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)>
      candidates(later);
  const auto offer = [&](std::size_t left, std::size_t right) {
    const auto merge = merges_.find({word[left], word[right]});
    if (merge != merges_.end()) {
      candidates.push({.rank = merge->second.rank,
                       .index = left,
                       .left = word[left],
                       .right = word[right]});
    }
  };
  for (std::size_t i = 0; i + 1 < count; ++i) {
    offer(i, i + 1);
  }
  while (!candidates.empty()) {
    const Candidate best = candidates.top();
    candidates.pop();
    const std::size_t right = following[best.index];
    if (word[best.index] != best.left || right == kNone ||
        word[right] != best.right) {
      continue;
    }
    const auto merge = merges_.find({best.left, best.right});
    if (merge == merges_.end() || merge->second.token == kInvalidTokenId) {
      break;
    }
    word[best.index] = merge->second.token;
    const std::size_t after = following[right];
    following[best.index] = after;
    if (after != kNone) {
      previous[after] = best.index;
    }
    word[right] = kInvalidTokenId;
    if (previous[best.index] != kNone) {
      offer(previous[best.index], best.index);
    }
    if (after != kNone) {
      offer(best.index, after);
    }
  }
  for (std::size_t i = 0; i != kNone; i = following[i]) {
    out.push_back(word[i]);
  }
}

void Tokenizer::EncodeOrdinary(std::string_view text,
                               std::vector<TokenId>& out) const {
  std::size_t offset = 0;
  while (offset < text.size()) {
    const std::size_t end = PieceEnd(text, offset);
    EncodePiece(text.substr(offset, end - offset), out);
    offset = end;
  }
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text,
                                       bool parse_special) const {
  std::vector<TokenId> out;
  if (text.empty()) {
    return out;
  }
  // llama.cpp partitions the text once per special token, longest first, so a
  // longer token wins over a shorter one that starts earlier.
  struct Fragment {
    std::string_view raw;
    TokenId token;  ///< kInvalidTokenId for ordinary text.
  };
  std::vector<Fragment> fragments{{text, kInvalidTokenId}};
  for (const auto& [name, id, user_defined] : specials_) {
    if (!parse_special && !user_defined) {
      continue;
    }
    std::vector<Fragment> next;
    next.reserve(fragments.size());
    for (const Fragment& fragment : fragments) {
      if (fragment.token != kInvalidTokenId) {
        next.push_back(fragment);
        continue;
      }
      std::string_view rest = fragment.raw;
      for (std::size_t at = rest.find(name); at != std::string_view::npos;
           at = rest.find(name)) {
        if (at > 0) {
          next.push_back({rest.substr(0, at), kInvalidTokenId});
        }
        next.push_back({{}, id});
        rest.remove_prefix(at + name.size());
      }
      if (!rest.empty()) {
        next.push_back({rest, kInvalidTokenId});
      }
    }
    fragments = std::move(next);
  }
  for (const Fragment& fragment : fragments) {
    if (fragment.token != kInvalidTokenId) {
      out.push_back(fragment.token);
    } else {
      EncodeOrdinary(fragment.raw, out);
    }
  }
  return out;
}

std::string Tokenizer::DecodeToken(TokenId token) const {
  if (token >= id_to_token_.size()) {
    return {};
  }
  if (is_special_[token]) {
    return id_to_token_[token];
  }
  return UnescapeBytes(id_to_token_[token]);
}

std::string Tokenizer::Decode(std::span<const TokenId> tokens) const {
  std::string out;
  for (const TokenId token : tokens) {
    out += DecodeToken(token);
  }
  return out;
}

}  // namespace gufo::models::glm53_flash
