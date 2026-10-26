#include "src/models/laguna/tokenizer.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <unordered_map>
#include <variant>

namespace gufo::models::laguna {
namespace {

// tokenizer.ggml.token_type values (llama.cpp LLAMA_TOKEN_TYPE_*).
constexpr std::int64_t kTokenUnknown = 2;
constexpr std::int64_t kTokenControl = 3;
constexpr std::int64_t kTokenUserDefined = 4;

constexpr char32_t kReplacement = 0xFFFD;
constexpr char32_t kOutOfRange = 0xFFFFFFFF;

/// UTF-8 as code points with the byte offset of each; offsets has one extra
/// entry holding the text size. Invalid bytes become U+FFFD, one per byte.
struct Decoded {
  std::vector<char32_t> cpts;
  std::vector<std::size_t> offsets;
};

Decoded DecodeUtf8(std::string_view text) {
  Decoded out;
  out.cpts.reserve(text.size());
  out.offsets.reserve(text.size() + 1);
  const auto byte = [&](std::size_t i) {
    return static_cast<unsigned char>(text[i]);
  };
  std::size_t i = 0;
  while (i < text.size()) {
    const unsigned char lead = byte(i);
    std::size_t length = 1;
    char32_t value = kReplacement;
    if (lead < 0x80) {
      value = lead;
    } else if (lead >= 0xC2 && lead <= 0xF4) {
      const std::size_t need = lead < 0xE0 ? 2 : (lead < 0xF0 ? 3 : 4);
      if (i + need <= text.size()) {
        char32_t v = lead & (0xFFU >> (need + 1));
        bool ok = true;
        for (std::size_t k = 1; k < need; ++k) {
          if ((byte(i + k) & 0xC0) != 0x80) {
            ok = false;
            break;
          }
          v = (v << 6) | (byte(i + k) & 0x3F);
        }
        const char32_t minimum =
            need == 2 ? 0x80 : (need == 3 ? 0x800 : 0x10000);
        if (ok && v >= minimum && v <= 0x10FFFF &&
            !(v >= 0xD800 && v <= 0xDFFF)) {
          value = v;
          length = need;
        }
      }
    }
    out.cpts.push_back(value);
    out.offsets.push_back(i);
    i += length;
  }
  out.offsets.push_back(text.size());
  return out;
}

/// The three categories the Laguna regex distinguishes. `valid` is false only
/// outside a segment, the way llama.cpp's flags are empty out of range.
struct Flags {
  bool valid{false};
  bool letter{false};
  bool number{false};
  bool whitespace{false};
};

/// llama.cpp d854d98 classifies code points from a Unicode 15.1 table; later
/// additions (for instance CJK Extension J, U+323B0..U+3347F) are "undefined"
/// there, so they match `[^\s\p{L}\p{N}]` rather than `\p{L}`. ICU here knows
/// newer Unicode, so code points added after 15.1 are classified as undefined
/// too, which keeps the split identical to llama-tokenize's.
bool AddedAfterUnicode151(UChar32 c) {
  if (c < 0x80) {
    return false;
  }
  UVersionInfo age;
  u_charAge(c, age);
  return age[0] > 15 || (age[0] == 15 && age[1] > 1);
}

Flags FlagsOf(char32_t cpt) {
  const auto c = static_cast<UChar32>(cpt);
  if (AddedAfterUnicode151(c)) {
    return {.valid = true};
  }
  const std::uint32_t mask = U_GET_GC_MASK(c);
  return {.valid = true,
          .letter = (mask & U_GC_L_MASK) != 0,
          .number = (mask & U_GC_N_MASK) != 0,
          .whitespace = u_isUWhiteSpace(c) != 0};
}

constexpr char32_t AsciiLower(char32_t c) {
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

/// Splits code points [ini, end) with
///   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
///   |[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*
///   |\s*[\r\n]+|\s+(?!\S)|\s+
/// following llama.cpp's hand-written matcher for this expression
/// (unicode_regex_split_custom_qwen2), including how it treats the segment
/// edges, and appends each piece's code point count to `lengths`.
void SplitSegment(const std::vector<char32_t>& cpts, std::size_t ini,
                  std::size_t end, std::vector<std::size_t>& lengths) {
  const auto cpt_at = [&](std::size_t pos) -> char32_t {
    return (ini <= pos && pos < end) ? cpts[pos] : kOutOfRange;
  };
  const auto flags_at = [&](std::size_t pos) -> Flags {
    return (ini <= pos && pos < end) ? FlagsOf(cpts[pos]) : Flags{};
  };
  const auto emit = [&](std::size_t from, std::size_t to) {
    if (to > from) {
      lengths.push_back(to - from);
    }
  };

  std::size_t pos = ini;
  while (pos < end) {
    const char32_t cpt = cpt_at(pos);
    const Flags flags = flags_at(pos);

    // 's 't 'm 'd 're 've 'll, case-insensitive.
    if (cpt == '\'' && pos + 1 < end) {
      const char32_t next = AsciiLower(cpt_at(pos + 1));
      if (next == 's' || next == 't' || next == 'm' || next == 'd') {
        emit(pos, pos + 2);
        pos += 2;
        continue;
      }
      if (pos + 2 < end) {
        const char32_t after = AsciiLower(cpt_at(pos + 2));
        if ((next == 'r' && after == 'e') || (next == 'v' && after == 'e') ||
            (next == 'l' && after == 'l')) {
          emit(pos, pos + 3);
          pos += 3;
          continue;
        }
      }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (!(cpt == '\r' || cpt == '\n' || flags.number)) {
      if (flags.letter || flags_at(pos + 1).letter) {
        const std::size_t start = pos++;
        while (flags_at(pos).letter) {
          ++pos;
        }
        emit(start, pos);
        continue;
      }
    }

    // \p{N}: one digit per piece.
    if (flags.number) {
      emit(pos, pos + 1);
      ++pos;
      continue;
    }

    // ' '?[^\s\p{L}\p{N}]+[\r\n]*
    Flags flags2 = cpt == ' ' ? flags_at(pos + 1) : flags;
    if (!(flags2.whitespace || flags2.letter || flags2.number) && flags.valid) {
      const std::size_t start = pos;
      pos += cpt == ' ' ? 1 : 0;
      while (!(flags2.whitespace || flags2.letter || flags2.number) &&
             flags2.valid) {
        flags2 = flags_at(++pos);
      }
      char32_t tail = cpt_at(pos);
      while (tail == '\r' || tail == '\n') {
        tail = cpt_at(++pos);
      }
      emit(start, pos);
      continue;
    }

    std::size_t spaces = 0;
    std::size_t last_end_r_or_n = 0;
    while (flags_at(pos + spaces).whitespace) {
      const char32_t c = cpt_at(pos + spaces);
      if (c == '\r' || c == '\n') {
        last_end_r_or_n = pos + spaces + 1;
      }
      ++spaces;
    }

    // \s*[\r\n]+
    if (last_end_r_or_n > 0) {
      emit(pos, last_end_r_or_n);
      pos = last_end_r_or_n;
      continue;
    }
    // \s+(?!\S): leave the last space for the next piece.
    if (spaces > 1 && cpt_at(pos + spaces) != kOutOfRange) {
      emit(pos, pos + spaces - 1);
      pos += spaces - 1;
      continue;
    }
    // \s+
    if (spaces > 0) {
      emit(pos, pos + spaces);
      pos += spaces;
      continue;
    }
    emit(pos, pos + 1);
    ++pos;
  }
}

/// Integer metadata array of any stored width.
std::vector<std::int64_t> IntArray(const core::GgufReader& gguf,
                                   std::string_view key, bool* present) {
  std::vector<std::int64_t> out;
  const auto* meta = gguf.FindMetadata(key);
  *present = meta != nullptr;
  if (meta == nullptr) {
    return out;
  }
  if (const auto* s = std::get_if<std::vector<std::int64_t>>(&meta->value)) {
    out = *s;
  } else if (const auto* u =
                 std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
    for (const auto v : *u) {
      out.push_back(static_cast<std::int64_t>(v));
    }
  } else {
    *present = false;
  }
  return out;
}

}  // namespace

std::vector<std::string_view> Tokenizer::PreTokenize(std::string_view text) {
  const Decoded decoded = DecodeUtf8(text);
  const auto& cpts = decoded.cpts;
  std::vector<std::size_t> lengths;
  lengths.reserve(cpts.size() / 3 + 1);
  // Stage one, `[^\n]+|[\n]+`: runs of newlines versus everything else.
  std::size_t begin = 0;
  while (begin < cpts.size()) {
    const bool newline = cpts[begin] == '\n';
    std::size_t stop = begin;
    while (stop < cpts.size() && (cpts[stop] == '\n') == newline) {
      ++stop;
    }
    // Stage two runs inside each run; the newline runs are one piece each
    // because `\s*[\r\n]+` consumes them whole.
    SplitSegment(cpts, begin, stop, lengths);
    begin = stop;
  }
  std::vector<std::string_view> pieces;
  pieces.reserve(lengths.size());
  std::size_t at = 0;
  for (const std::size_t length : lengths) {
    const std::size_t from = decoded.offsets[at];
    const std::size_t to = decoded.offsets[at + length];
    pieces.push_back(text.substr(from, to - from));
    at += length;
  }
  return pieces;
}

std::unique_ptr<Tokenizer> Tokenizer::FromGguf(const core::GgufReader& gguf,
                                               std::string* error_msg) {
  const auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return nullptr;
  };
  if (gguf.GetMetadataString("general.architecture") != "laguna") {
    return fail("GGUF architecture is not laguna");
  }
  if (gguf.GetMetadataString("tokenizer.ggml.model") != "gpt2") {
    return fail("tokenizer.ggml.model must be gpt2");
  }
  if (gguf.GetMetadataString("tokenizer.ggml.pre") != "laguna") {
    return fail("tokenizer.ggml.pre must be laguna");
  }

  const auto token_views = gguf.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto merge_views = gguf.GetMetadataStringArray("tokenizer.ggml.merges");
  if (token_views.empty() || merge_views.empty()) {
    return fail("GGUF has no tokenizer.ggml.tokens or tokenizer.ggml.merges");
  }
  bool has_types = false;
  const auto types = IntArray(gguf, "tokenizer.ggml.token_type", &has_types);
  if (!has_types || types.size() != token_views.size()) {
    return fail("tokenizer.ggml.token_type must describe every token");
  }

  auto tokenizer = std::unique_ptr<Tokenizer>(new Tokenizer());
  std::vector<std::string> tokens(token_views.begin(), token_views.end());
  std::vector<std::string> merges(merge_views.begin(), merge_views.end());
  std::unordered_map<std::string, TokenId> special_map;
  for (std::size_t id = 0; id < tokens.size(); ++id) {
    const auto type = types[id];
    if (type == kTokenUnknown || type == kTokenControl ||
        type == kTokenUserDefined) {
      special_map[tokens[id]] = static_cast<TokenId>(id);
      tokenizer->specials_.push_back(
          {tokens[id], static_cast<TokenId>(id), type == kTokenUserDefined});
    }
  }
  std::stable_sort(tokenizer->specials_.begin(), tokenizer->specials_.end(),
                   [](const Special& a, const Special& b) {
                     return a.text.size() > b.text.size();
                   });

  std::string error;
  tokenizer->bpe_ = tokenization::QwenTokenizer::CreateFromVocabulary(
      tokens, merges, special_map, &error);
  if (tokenizer->bpe_ == nullptr) {
    return fail("BPE vocabulary rejected: " + error);
  }

  const auto id_of = [&](std::string_view key, TokenId* out) {
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value() || *value >= tokens.size()) {
      return false;
    }
    *out = *value;
    return true;
  };
  if (!id_of("tokenizer.ggml.bos_token_id", &tokenizer->bos_) ||
      !id_of("tokenizer.ggml.eos_token_id", &tokenizer->eos_) ||
      !id_of("tokenizer.ggml.padding_token_id", &tokenizer->pad_)) {
    return fail("GGUF lacks a valid BOS, EOS or padding token id");
  }
  tokenizer->gguf_adds_bos_ =
      gguf.GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(false);
  return tokenizer;
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text,
                                       const EncodeOptions& options) const {
  std::vector<TokenId> out;
  if (options.add_bos) {
    out.push_back(bos_);
  }

  // llama.cpp's tokenizer_st_partition: for each special token, longest text
  // first, cut every occurrence out of the still-raw fragments.
  struct Fragment {
    bool special;
    TokenId id;
    std::size_t offset;
    std::size_t length;
  };
  std::vector<Fragment> fragments;
  if (!text.empty()) {
    fragments.push_back({false, 0, 0, text.size()});
  }
  for (const Special& special : specials_) {
    if (!options.parse_special && !special.user_defined) {
      continue;
    }
    std::vector<Fragment> next;
    next.reserve(fragments.size() + 2);
    for (const Fragment& fragment : fragments) {
      if (fragment.special) {
        next.push_back(fragment);
        continue;
      }
      const std::string_view raw =
          text.substr(fragment.offset, fragment.length);
      std::size_t cursor = 0;
      while (true) {
        const std::size_t match = raw.find(special.text, cursor);
        if (match == std::string_view::npos) {
          break;
        }
        if (match > cursor) {
          next.push_back({false, 0, fragment.offset + cursor, match - cursor});
        }
        next.push_back({true, special.id, 0, 0});
        cursor = match + special.text.size();
      }
      if (cursor < raw.size()) {
        next.push_back(
            {false, 0, fragment.offset + cursor, raw.size() - cursor});
      }
    }
    fragments = std::move(next);
  }

  const tokenization::TokenizerOptions bpe_options{
      .add_bos = false, .add_eos = false, .parse_special_tokens = false};
  for (const Fragment& fragment : fragments) {
    if (fragment.special) {
      out.push_back(fragment.id);
      continue;
    }
    for (const std::string_view piece :
         PreTokenize(text.substr(fragment.offset, fragment.length))) {
      const auto ids = bpe_->Encode(piece, bpe_options);
      out.insert(out.end(), ids.begin(), ids.end());
    }
  }
  return out;
}

std::string Tokenizer::Decode(std::span<const TokenId> tokens) const {
  return bpe_->Decode(tokens);
}

}  // namespace gufo::models::laguna
