#include "src/models/laguna/chat_template.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::laguna {
namespace {

// Python's str.isspace() code points, which str.strip() / str.rstrip() remove.
bool IsPythonSpace(std::uint32_t cp) noexcept {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) ||
         cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

/// Decodes the code point starting at `at`; `length` receives its bytes.
/// Malformed bytes decode as 0xFFFFFFFF (never whitespace), one byte each.
std::uint32_t CodePointAt(std::string_view text, std::size_t at,
                          std::size_t* length) noexcept {
  const auto byte = [&](std::size_t i) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(text[i]));
  };
  const std::uint32_t first = byte(at);
  *length = 1;
  if (first < 0x80U) {
    return first;
  }
  const std::size_t need = first >= 0xF8U   ? 0
                           : first >= 0xF0U ? 3
                           : first >= 0xE0U ? 2
                           : first >= 0xC0U ? 1
                                            : 0;
  if (need == 0 || at + need >= text.size()) {
    return 0xFFFFFFFFU;
  }
  std::uint32_t cp = first & (0x3FU >> need);
  for (std::size_t i = 1; i <= need; ++i) {
    const std::uint32_t next = byte(at + i);
    if ((next & 0xC0U) != 0x80U) {
      return 0xFFFFFFFFU;
    }
    cp = (cp << 6U) | (next & 0x3FU);
  }
  *length = need + 1;
  return cp;
}

bool IsBlank(std::string_view text) noexcept {
  for (std::size_t at = 0; at < text.size();) {
    std::size_t length = 1;
    if (!IsPythonSpace(CodePointAt(text, at, &length))) {
      return false;
    }
    at += length;
  }
  return true;
}

/// Python's str.rstrip().
std::string_view RStrip(std::string_view text) noexcept {
  std::size_t end = 0;
  for (std::size_t at = 0; at < text.size();) {
    std::size_t length = 1;
    const std::uint32_t cp = CodePointAt(text, at, &length);
    at += length;
    if (!IsPythonSpace(cp)) {
      end = at;
    }
  }
  return text.substr(0, end);
}

/// `value | tojson(ensure_ascii=False)` for an argument the model or client
/// supplied as JSON text. Unparseable text is kept as written.
std::string ArgumentJson(const std::string& value) {
  try {
    return json::parse(value).tojson();
  } catch (const std::exception&) {
    return value;
  }
}

std::string ToolJson(const ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    try {
      return json::parse(tool.definition_json).tojson();
    } catch (const std::exception&) {
      // Fall through to the rebuilt object.
    }
  }
  json::Value function = json::Value::object();
  function["name"] = tool.name;
  if (!tool.description.empty()) {
    function["description"] = tool.description;
  }
  try {
    function["parameters"] = json::parse(tool.parameters_json);
  } catch (const std::exception&) {
    function["parameters"] = json::Value::object();
  }
  json::Value object = json::Value::object();
  object["type"] = "function";
  object["function"] = std::move(function);
  return object.tojson();
}

}  // namespace

ChatTemplateOptions ResolveLagunaChatOptions(
    const ReasoningOptions& reasoning) {
  ChatTemplateOptions options;
  // The template's default (`enable_thinking | default(true)`) and the
  // checkpoint's generation_config.json both turn thinking on.
  options.enable_thinking = reasoning.enabled.value_or(true);
  options.preserve_thinking = reasoning.preserve_thinking.value_or(false);
  return options;
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       const ChatTemplateOptions& options) {
  std::string out(kBosText);

  std::string_view system_message = kDefaultSystemMessage;
  if (!messages.empty() && messages.front().role == "system") {
    system_message = messages.front().content;
    messages = messages.subspan(1);
  }
  const bool has_sys = !IsBlank(system_message);
  const bool has_tools = !tools.empty();
  if (has_sys || has_tools || options.enable_thinking) {
    out += "<system>";
    if (has_sys) {
      out += RStrip(system_message);
      if (has_tools) {
        out += "\n\n";
      }
    }
    if (has_tools) {
      out +=
          "### Tools\n\n"
          "You may call functions to assist with the user query.\n"
          "All available function signatures are listed below:\n"
          "<available_tools>\n";
      for (const ChatTool& tool : tools) {
        out += ToolJson(tool);
        out += '\n';
      }
      out += "</available_tools>";
    }
    out += "</system>\n";
  }

  for (const ChatMessage& message : messages) {
    if (message.role == "user") {
      out += "<user>";
      out += message.content;
      out += "</user>\n";
    } else if (message.role == "assistant") {
      out += "<assistant>";
      if (options.enable_thinking || options.preserve_thinking) {
        out += "<think>";
        out += message.reasoning_content;
        out += "</think>";
      } else {
        out += "</think>";
      }
      out += message.content;
      for (const ChatMessage::ToolCall& call : message.tool_calls) {
        out += "<tool_call>";
        out += call.name;
        for (const ChatMessage::ToolArgument& argument : call.arguments) {
          out += "<arg_key>";
          out += argument.name;
          out += "</arg_key><arg_value>";
          out += argument.is_string ? argument.value
                                    : ArgumentJson(argument.value);
          out += "</arg_value>";
        }
        out += "</tool_call>";
      }
      out += "</assistant>\n";
    } else if (message.role == "tool") {
      out += "<tool_response>";
      out += message.content;
      out += "</tool_response>\n";
    } else if (message.role == "system") {
      out += "<system>";
      out += message.content;
      out += "</system>\n";
    }
  }

  if (options.add_generation_prompt) {
    out += "<assistant>";
    out += options.enable_thinking ? "<think>" : "</think>";
  }
  return out;
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  const auto fail = [&](std::string message) {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return false;
  };
  if (reader.GetMetadataString("general.architecture") != "laguna") {
    return fail("artifact architecture is not laguna");
  }
  const auto raw = reader.GetMetadataString("tokenizer.chat_template");
  if (!raw.has_value() || raw->empty()) {
    return fail("GGUF is missing tokenizer.chat_template metadata");
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(raw->data());
  const std::string hash = crypto::Sha256Hex({bytes, raw->size()});
  if (hash != ArtifactTemplateSha256()) {
    return fail(
        "GGUF chat template SHA-256 is not the pinned Laguna-S-2.1 template: " +
        hash);
  }
  return true;
}

}  // namespace gufo::models::laguna
