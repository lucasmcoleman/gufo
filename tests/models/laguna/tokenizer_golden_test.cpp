// Token-id goldens: gufo's Laguna tokenizer against llama.cpp d854d98's
// llama-tokenize over fixtures/tokenizer_goldens.json (regenerate with
// fixtures/generate_tokenizer_goldens.py). The vocabulary comes from the real
// GGUF; without it the test ends with the CTest skip status and verifies
// nothing.
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/laguna/tokenizer.hpp"
#include "tests/models/laguna/support.hpp"

namespace laguna = gufo::models::laguna;
namespace json = gufo::json;
using gufo::testing::laguna::Require;

namespace {

std::vector<laguna::TokenId> Ids(const json::Value& value) {
  Require(value.is_array(), "fixture ids are not an array");
  std::vector<laguna::TokenId> ids;
  for (const auto& item : value.items()) {
    ids.push_back(static_cast<laguna::TokenId>(item.as_size(0)));
  }
  return ids;
}

std::string Show(const std::vector<laguna::TokenId>& ids) {
  std::string out = "[";
  for (std::size_t i = 0; i < ids.size() && i < 24; ++i) {
    out += (i ? ", " : "") + std::to_string(ids[i]);
  }
  return out + (ids.size() > 24 ? ", ...]" : "]");
}

}  // namespace

int main() {
  try {
    const auto reader =
        gufo::testing::laguna::OpenModelOrSkip("laguna.tokenizer_golden");
    std::string error;
    const auto tokenizer = laguna::Tokenizer::FromGguf(*reader, &error);
    Require(tokenizer != nullptr, "Tokenizer::FromGguf failed: " + error);
    Require(tokenizer->vocab_size() == 100352, "vocabulary size");
    Require(tokenizer->bos_token_id() == 2 && tokenizer->eos_token_id() == 2 &&
                tokenizer->pad_token_id() == 9,
            "BOS/EOS/PAD ids");
    Require(tokenizer->gguf_adds_bos(),
            "the GGUF is expected to request BOS (add_bos_token)");

    std::ifstream in(GUFO_LAGUNA_TOKENIZER_GOLDENS, std::ios::binary);
    Require(static_cast<bool>(in), "cannot open the golden fixture");
    std::stringstream buffer;
    buffer << in.rdbuf();
    const json::Value fixture = json::parse(buffer.str());
    const json::Value* cases = fixture.find("cases");
    Require(cases != nullptr && cases->is_array(), "fixture has no cases");
    Require(cases->items().size() >= 40, "fixture needs at least 40 cases");

    std::size_t checked = 0;
    std::size_t failures = 0;
    std::set<std::string> categories;
    const auto compare = [&](const std::string& name, const char* variant,
                             const std::vector<laguna::TokenId>& got,
                             const std::vector<laguna::TokenId>& want) {
      ++checked;
      if (got == want) {
        return;
      }
      ++failures;
      std::cerr << "MISMATCH " << name << " (" << variant << ")\n  expected "
                << Show(want) << " (" << want.size() << " ids)\n  got      "
                << Show(got) << " (" << got.size() << " ids)\n";
    };
    for (const auto& c : cases->items()) {
      const std::string name = c.member_str("name");
      const std::string text = c.member_str("text");
      categories.insert(c.member_str("category"));
      Require(text.size() == c.member_size("utf8_bytes"),
              "fixture text did not load byte-exact: " + name);
      const auto bos = Ids(*c.find("ids_bos"));
      const auto no_bos = Ids(*c.find("ids_no_bos"));
      const auto no_special = Ids(*c.find("ids_no_special"));

      // The BOS policy is explicit on both sides. llama-tokenize's default
      // prepends BOS; gufo prepends it only on request.
      compare(name, "add_bos", tokenizer->Encode(text, {.add_bos = true}), bos);
      compare(name, "no bos (the default)", tokenizer->Encode(text), no_bos);
      compare(name, "no bos, explicit",
              tokenizer->Encode(text, {.add_bos = false}), no_bos);
      compare(
          name, "no bos, special text as plain text",
          tokenizer->Encode(text, {.add_bos = false, .parse_special = false}),
          no_special);

      // Decoding the ids returns the exact bytes (specials decode to their
      // own text).
      const auto round_trip = tokenizer->Decode(no_bos);
      ++checked;
      if (round_trip != text) {
        ++failures;
        std::cerr << "ROUND TRIP " << name << ": decoded text differs\n";
      }
    }
    for (const char* wanted :
         {"prose", "code", "numbers", "whitespace", "tabs_newlines",
          "unicode_cjk", "unicode_emoji", "unicode_accents", "unicode_rtl",
          "contraction", "punctuation", "special"}) {
      Require(categories.contains(wanted),
              std::string("fixture lacks category ") + wanted);
    }
    Require(failures == 0, std::to_string(failures) + " of " +
                               std::to_string(checked) +
                               " tokenizer comparisons failed");
    std::cout << "PASS laguna.tokenizer_golden: " << cases->items().size()
              << " texts, " << checked << " comparisons\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL laguna.tokenizer_golden: " << e.what() << '\n';
    return 1;
  }
}
