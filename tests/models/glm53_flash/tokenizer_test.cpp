// A6 tokenizer check for GLM-5.3-Flash on the real Maya-S vocabulary: special
// ids, chat-template control tokens, the glm4 pre-tokenizer split, special
// token parsing and byte-exact round trips. Set GLM53_FLASH_GGUF to the first
// shard; without it the test reports a skip (which is not a pass for model
// qualification). Parity with llama-tokenize is checked by
// /server/staging/glm53-gufo/a6/run_parity.py.

#include "src/models/glm53_flash/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace glm = gufo::models::glm53_flash;

namespace {

constexpr int kSkip = 77;  // CTest SKIP_RETURN_CODE convention.

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

using Ids = std::vector<glm::TokenId>;

}  // namespace

int main() {
  const char* path = std::getenv("GLM53_FLASH_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GLM53_FLASH_GGUF is not set\n";
    return kSkip;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, "open " + std::string(path) + ": " + error);
    const auto tok = glm::Tokenizer::CreateFromGguf(*reader, &error);
    Require(tok != nullptr, "tokenizer: " + error);

    Require(tok->VocabSize() == 154880, "vocab size");
    const auto& s = tok->special_ids();
    Require(s.eos == 154820 && s.pad == 154820 && s.unk == 154820,
            "eos/pad/unk ids");
    Require(s.bos == 154822, "bos id");
    Require(s.eot == 154827, "eot id");
    Require(s.eom == 154829, "eom id");

    const auto& c = tok->chat_tokens();
    Require(c.gmask == 154822 && c.sop == 154824, "[gMASK]/<sop>");
    Require(c.system == 154826 && c.user == 154827 && c.assistant == 154828 &&
                c.observation == 154829,
            "role tokens");
    Require(c.think == 154841 && c.think_end == 154842, "think tokens");
    Require(c.tool_call == 154843 && c.tool_call_end == 154844 &&
                c.tool_response == 154845 && c.tool_response_end == 154846 &&
                c.arg_key == 154847 && c.arg_key_end == 154848 &&
                c.arg_value == 154849 && c.arg_value_end == 154850,
            "tool tokens");
    Require(c.mask != glm::kInvalidTokenId && c.smask != glm::kInvalidTokenId &&
                c.eop != glm::kInvalidTokenId &&
                c.nothink != glm::kInvalidTokenId &&
                c.box_begin != glm::kInvalidTokenId &&
                c.box_end != glm::kInvalidTokenId,
            "remaining control tokens resolve");

    // Special-token parsing: a chat prompt is control ids around ordinary text.
    const Ids prompt =
        tok->Encode("[gMASK]<sop><|user|>hi<|assistant|><think>", true);
    Require(prompt.size() == 6 && prompt[0] == c.gmask && prompt[1] == c.sop &&
                prompt[2] == c.user && prompt[4] == c.assistant &&
                prompt[5] == c.think,
            "chat prompt layout");
    // Without parsing, the same text never yields a control id.
    for (const auto id : tok->Encode("[gMASK]<sop><|user|>hi", false)) {
      Require(!tok->IsSpecialToken(id), "control id leaked with parse off");
    }

    // User-defined tokens are matched even with parse_special off.
    Require(tok->Encode("<think>", false) == Ids{c.think} &&
                tok->Encode("</tool_call>", false) == Ids{c.tool_call_end},
            "user-defined tokens parse with parse_special off");

    // glm4 split: digits in groups of three, contraction, whitespace runs.
    Ids expected = tok->Encode("123", false);
    for (const char* group : {"456", "7"}) {
      const Ids part = tok->Encode(group, false);
      expected.insert(expected.end(), part.begin(), part.end());
    }
    Require(tok->Encode("1234567", false) == expected,
            "digits split in groups of three");
    Ids contraction = tok->Encode("don", false);
    const Ids suffix = tok->Encode("'t", false);
    contraction.insert(contraction.end(), suffix.begin(), suffix.end());
    Require(tok->Encode("don't", false) == contraction,
            "contraction splits before the apostrophe");
    Require(tok->Encode("", true).empty(), "empty text");

    const std::vector<std::string> round_trip = {
        "hello world",
        "  two  spaces\n\n\ttabs\r\n",
        "def f(x):\n    return x**2  # \xe2\x9c\x93",
        "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c",
        "emoji \xf0\x9f\x98\x80\xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd",
        "<|user|>x<|assistant|><think>y</think>",
        std::string("\xff\xfe invalid utf-8 \xc3", 18),
    };
    for (const auto& text : round_trip) {
      for (const bool parse : {false, true}) {
        const Ids ids = tok->Encode(text, parse);
        Require(tok->Decode(ids) == text,
                "round trip failed: parse_special=" + std::to_string(parse));
      }
    }
    std::cout << "glm53_flash tokenizer: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
