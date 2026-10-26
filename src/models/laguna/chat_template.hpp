#ifndef GUFO_MODELS_LAGUNA_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_LAGUNA_CHAT_TEMPLATE_HPP_

// Poolside's Laguna-S-2.1 chat template (`chat_template.jinja` in
// poolside/Laguna-S-2.1, byte-identical to the GGUF's
// `tokenizer.chat_template`) as a compiled renderer, the way the other model
// families do it: the GGUF's template must hash to the pinned value, and the
// renderer is checked against the Jinja template's output in
// tests/models/laguna/chat_template_test.cpp.
//
// Framing, from the template:
//   "〈|EOS|〉" (token 2, BOS)
//   "<system>" [system text] ["\n\n### Tools\n\n...<available_tools>\n" one
//   `tojson(tool)` per line "</available_tools>"] "</system>\n"
//   "<user>" text "</user>\n"
//   "<assistant>" ("<think>" reasoning "</think>" | "</think>") text
//       ("<tool_call>" name ("<arg_key>" k "</arg_key><arg_value>" v
//       "</arg_value>")* "</tool_call>")* "</assistant>\n"
//   "<tool_response>" text "</tool_response>\n"
//   generation prompt: "<assistant>" ("<think>" | "</think>")
//
// Thinking defaults to on. A default system message ("You are a helpful,
// conversationally-fluent assistant made by Poolside...") applies when the
// conversation has none; a system message with blank content removes it.

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/laguna/tool_calls.hpp"

namespace gufo::models::laguna {

struct ChatMessage {
  struct ToolArgument {
    std::string name;
    std::string value;  ///< Raw text for a string, JSON text otherwise.
    bool is_string{true};
  };
  struct ToolCall {
    std::string name;
    std::vector<ToolArgument> arguments;
    std::string id;
  };

  std::string role;  ///< system, user, assistant or tool.
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// The complete OpenAI tool object `{"type":"function","function":{...}}`,
  /// whose key order and extra fields the template serializes as given. When
  /// empty it is rebuilt from the three fields above.
  std::string definition_json;
};

struct ChatTemplateOptions {
  bool enable_thinking{true};
  bool preserve_thinking{false};
  bool add_generation_prompt{true};
};

/// Provider-neutral reasoning controls onto the template's two flags. The
/// template has no effort levels, so an effort only implies thinking on.
[[nodiscard]] ChatTemplateOptions ResolveLagunaChatOptions(
    const ReasoningOptions& reasoning);

/// The conversation as the Jinja template renders it, `tools` serialized with
/// Python's `json.dumps(ensure_ascii=False)` spacing.
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools,
                                     const ChatTemplateOptions& options = {});

/// Rejects a Laguna artifact whose embedded template is not the pinned one
/// this renderer implements.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg = nullptr);

[[nodiscard]] constexpr std::string_view ChatTemplateId() noexcept {
  return "laguna-s-2.1-compiled-v1";
}

/// SHA-256 of `chat_template.jinja` (3875 bytes), equal to the GGUF's.
[[nodiscard]] constexpr std::string_view ArtifactTemplateSha256() noexcept {
  return "444819b8ad4612870827ac05b9147fe9e3344d3850cae8c2790898fc514099ff";
}

/// The renderer's first bytes: the BOS token's text.
inline constexpr std::string_view kBosText = "〈|EOS|〉";
inline constexpr std::string_view kDefaultSystemMessage =
    "You are a helpful, conversationally-fluent assistant made by Poolside. "
    "You are here to be helpful to users through natural language "
    "conversations.";

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_CHAT_TEMPLATE_HPP_
