// Laguna-S-2.1 chat prompts through the real GGUF: the artifact's embedded
// template must be the pinned one, and every golden prompt must tokenize to
// the ids llama.cpp's llama-tokenize produces for the same text. Reads only
// the GGUF header and vocabulary; without the file the test ends with the
// CTest skip status and verifies nothing.
#include <iostream>
#include <string>
#include <vector>

#include "src/models/laguna/chat_template.hpp"
#include "src/models/laguna/tokenizer.hpp"
#include "tests/models/laguna/chat_golden.hpp"
#include "tests/models/laguna/support.hpp"

namespace ml = gufo::models::laguna;
namespace json = gufo::json;
using gufo::testing::laguna::ChatInputOf;
using gufo::testing::laguna::LoadChatGoldens;
using gufo::testing::laguna::Require;

int main() {
  try {
    const auto reader =
        gufo::testing::laguna::OpenModelOrSkip("laguna.chat_template_tokens");
    std::string error;
    Require(ml::ValidateGgufTemplate(*reader, &error),
            "the GGUF's chat template is not the pinned one: " + error);
    const auto tokenizer = ml::Tokenizer::FromGguf(*reader, &error);
    Require(tokenizer != nullptr, "tokenizer: " + error);
    Require(tokenizer->eot_token_id() == 24 && tokenizer->IsStopToken(2) &&
                tokenizer->IsStopToken(24) && !tokenizer->IsStopToken(23),
            "stop tokens are exactly EOS 2 and EOT 24");

    const json::Value fixture = LoadChatGoldens();
    std::size_t checked = 0;
    std::size_t failures = 0;
    for (const auto& c : fixture.find("cases")->items()) {
      const std::string name = c.member_str("name");
      const auto input = ChatInputOf(c);
      const std::string rendered =
          ml::RenderChat(input.messages, input.tools, input.options);
      // Exactly what Model::EncodeChat does: the template carries BOS.
      const auto got = tokenizer->Encode(
          rendered, ml::EncodeOptions{.add_bos = false, .parse_special = true});
      std::vector<ml::TokenId> want;
      for (const auto& id : c.find("token_ids")->items()) {
        want.push_back(static_cast<ml::TokenId>(id.as_size(0)));
      }
      ++checked;
      if (got != want) {
        ++failures;
        std::cerr << "TOKEN MISMATCH " << name << ": expected " << want.size()
                  << " ids, got " << got.size() << '\n';
        continue;
      }
      Require(!got.empty() && got.front() == tokenizer->bos_token_id(),
              name + ": the prompt must start with BOS");
    }
    Require(failures == 0, std::to_string(failures) + " of " +
                               std::to_string(checked) +
                               " prompts tokenize differently from llama.cpp");
    std::cout << "laguna.chat_template_tokens: " << checked
              << " prompts match llama-tokenize\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
