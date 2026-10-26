#ifndef GUFO_TESTS_MODELS_LAGUNA_CHAT_GOLDEN_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_CHAT_GOLDEN_HPP_

// Loads fixtures/chat_goldens.json (see generate_chat_goldens.py) and turns a
// case's OpenAI-shaped messages and tools into the renderer's inputs.

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/laguna/chat_template.hpp"
#include "tests/models/laguna/support.hpp"

namespace gufo::testing::laguna {

namespace json = gufo::json;
namespace ml = gufo::models::laguna;

inline json::Value LoadChatGoldens() {
  std::ifstream in(GUFO_LAGUNA_CHAT_GOLDENS, std::ios::binary);
  Require(static_cast<bool>(in), "cannot open chat_goldens.json");
  std::stringstream buffer;
  buffer << in.rdbuf();
  return json::parse(buffer.str());
}

struct ChatGoldenInput {
  std::vector<ml::ChatMessage> messages;
  std::vector<ml::ChatTool> tools;
  ml::ChatTemplateOptions options;
};

inline ChatGoldenInput ChatInputOf(const json::Value& c) {
  ChatGoldenInput input;
  input.options.enable_thinking = c.find("enable_thinking")->as_bool();
  input.options.preserve_thinking = c.find("preserve_thinking")->as_bool();
  input.options.add_generation_prompt =
      c.find("add_generation_prompt")->as_bool();
  for (const auto& m : c.find("messages")->items()) {
    ml::ChatMessage message;
    message.role = m.member_str("role");
    if (const auto* content = m.find("content");
        content && content->is_string()) {
      message.content = content->get_str();
    }
    // `reasoning` and `reasoning_content` are the template's two field names.
    for (const char* key : {"reasoning", "reasoning_content"}) {
      if (const auto* r = m.find(key); r && r->is_string()) {
        message.reasoning_content = r->get_str();
        break;
      }
    }
    if (const auto* calls = m.find("tool_calls")) {
      for (const auto& call : calls->items()) {
        const json::Value& function = *call.find("function");
        ml::ChatMessage::ToolCall converted;
        converted.name = function.member_str("name");
        for (const auto& [name, value] :
             function.find("arguments")->members()) {
          converted.arguments.push_back(
              {name, value.is_string() ? value.get_str() : value.dump(),
               value.is_string()});
        }
        message.tool_calls.push_back(std::move(converted));
      }
    }
    input.messages.push_back(std::move(message));
  }
  if (const auto* tools = c.find("tools"); tools && tools->is_array()) {
    for (const auto& tool : tools->items()) {
      ml::ChatTool converted;
      const json::Value& function = *tool.find("function");
      converted.name = function.member_str("name");
      converted.description = function.member_str("description");
      converted.definition_json = tool.dump();
      input.tools.push_back(std::move(converted));
    }
  }
  return input;
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_CHAT_GOLDEN_HPP_
