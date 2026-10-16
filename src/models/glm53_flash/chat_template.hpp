#ifndef GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/reasoning.hpp"
#include "src/models/glm53_flash/tokenizer.hpp"

namespace gufo::models::glm53_flash {

/// Roles the GLM-5.3-Flash template renders. The template has no developer
/// role (such a message renders nothing); callers map it to kSystem.
enum class ChatRole : std::uint8_t {
  kSystem = 0,
  kUser = 1,
  kAssistant = 2,
  kTool = 3,
};

/// A structured input message. Content is plain text: the template's
/// image/video/audio parts, `tool_reference` results and list-of-outputs tool
/// messages are not modelled.
struct ChatMessage {
  struct ToolArgument {
    std::string name;
    /// The string itself when `is_string`, otherwise JSON text. The template
    /// re-serialises JSON the way Python's json.dumps does, so a client's
    /// `2.0` stays `2.0` and `{"a":1}` becomes `{"a": 1}`.
    std::string value;
    bool is_string{true};
  };

  struct ToolCall {
    std::string id;
    std::string name;
    std::vector<ToolArgument> arguments;
  };

  ChatRole role{ChatRole::kUser};
  std::string content;
  /// The assistant's `reasoning_content`. Absent is not the same as empty: an
  /// absent field falls back to splitting `<think>...</think>` out of
  /// `content`.
  std::optional<std::string> reasoning_content;
  std::string tool_call_id;  ///< Tool messages: the call this answers.
  std::vector<ToolCall> tool_calls;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// The complete HTTP tool object (`{"type":"function","function":{...}}` or
  /// a bare function object), in client field order. Preferred over the three
  /// fields above when present, as the template iterates the object's keys.
  std::string definition_json;
};

/// The template's `reasoning_effort` global. Anything other than low or high
/// renders as Max.
enum class ReasoningLevel : std::uint8_t {
  kLow,
  kHigh,
  kMax,
};

struct ChatTemplateOptions {
  bool add_generation_prompt{true};
  /// The template always opens the turn with `<think>`; it has no switch.
  /// False swaps the generation prompt for `<|assistant|><think></think>`, the
  /// shape the template itself writes for an assistant turn without
  /// reasoning. That is a gufo extension, outside the golden parity.
  bool enable_thinking{true};
  ReasoningLevel reasoning_effort{ReasoningLevel::kMax};
  /// The template's `clear_thinking` global: drop reasoning from assistant
  /// turns that precede the last user message.
  bool clear_thinking{false};
};

/// Maps provider-neutral controls onto the template: minimal/low -> low,
/// medium/high -> high, xhigh/max -> max; `preserve_thinking` is the inverse
/// of `clear_thinking`.
[[nodiscard]] ChatTemplateOptions ResolveChatOptions(
    const ReasoningOptions& reasoning);

/// A byte span of the rendered prompt that came from client data (message
/// text, tool names, arguments, tool definitions) rather than template
/// framing. Spans are in increasing offset order and never overlap.
struct ContentSpan {
  std::size_t offset{0};
  std::size_t size{0};
};

/// Suffix that opens a new assistant turn.
[[nodiscard]] std::string_view GenerationPrompt(bool enable_thinking);

/// Renders `messages` byte-identically to the model's Jinja chat template
/// (checked against jinja2 goldens). Returns nullopt, with `error_msg` set,
/// when a tool definition or JSON argument is not valid JSON.
[[nodiscard]] std::optional<std::string> RenderChat(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options = {}, std::string* error_msg = nullptr,
    std::vector<ContentSpan>* content_spans = nullptr);

/// Tokenizes a rendered prompt: framing bytes with `parse_special` so
/// `<|user|>` is the control token, content spans as text so a `<|user|>` a
/// client typed stays characters. (User-defined tokens such as `<think>` are
/// always recognised by the tokenizer, as in llama.cpp.)
[[nodiscard]] std::vector<TokenId> EncodeRendered(
    const Tokenizer& tokenizer, std::string_view rendered,
    std::span<const ContentSpan> content_spans);

/// Python's `json.dumps(json.loads(text), ensure_ascii=False)` for the
/// template's `tojson`: ", " and ": " separators, number spelling as Python
/// prints it (2.0 stays 2.0, 1E3 becomes 1000.0, big integers verbatim).
/// Nullopt for text json.loads would reject.
[[nodiscard]] std::optional<std::string> PythonJsonDumps(
    std::string_view json_text);

/// The literals that frame a GLM tool call. RenderChat writes historical calls
/// with these, and ScanToolCall reads the same bytes back.
struct ToolSyntax {
  static constexpr std::string_view kCallOpen = "<tool_call>";
  static constexpr std::string_view kCallClose = "</tool_call>";
  static constexpr std::string_view kKeyOpen = "<arg_key>";
  static constexpr std::string_view kKeyClose = "</arg_key>";
  static constexpr std::string_view kValueOpen = "<arg_value>";
  static constexpr std::string_view kValueClose = "</arg_value>";
  static constexpr std::string_view kThinkClose = "</think>";
};

/// One argument exactly as the model wrote it, before the API surface decides
/// whether it is a string or JSON.
struct ToolCallArgumentSyntax {
  std::string name;   ///< Between <arg_key> and </arg_key>, whitespace-trimmed.
  std::string value;  ///< Between <arg_value> and </arg_value>, verbatim.
};

struct ToolCallSyntax {
  std::string name;  ///< Text after <tool_call>, whitespace-trimmed.
  std::vector<ToolCallArgumentSyntax> arguments;
  /// The argument whose value ran to the end of the scan without
  /// </arg_value>. `value` holds back a possible partial closing tag.
  std::optional<ToolCallArgumentSyntax> unfinished;
  /// </tool_call> was found and every argument was well formed.
  bool complete{false};
};

struct ToolCallScan {
  bool matched{false};  ///< `text` begins with ToolSyntax::kCallOpen.
  /// Bytes of `text` to skip before scanning resumes when the call is not
  /// accepted: the whole call when it completed, the end of the scanned
  /// prefix of a still-growing one, or just the opener when malformed.
  std::size_t progress{0};
  std::size_t end{0};  ///< Bytes of a complete call, through </tool_call>.
  /// Absent while the name is still arriving, or when the call is malformed
  /// (empty name, text between elements that is not a tag).
  std::optional<ToolCallSyntax> call;
  bool malformed{false};
};

/// Scans the call beginning at `text[0]`, if any. Whitespace between elements
/// is accepted (the GLM-4.5 family writes newlines there); the template
/// writes none.
[[nodiscard]] ToolCallScan ScanToolCall(std::string_view text);

/// A call with arguments typed for the API surface.
struct ParsedToolCall {
  std::string name;
  std::vector<ChatMessage::ToolArgument> arguments;
};

/// A generated response split into its parts.
struct ParsedResponse {
  std::string reasoning;
  std::string content;  ///< Text outside reasoning and calls, verbatim.
  std::vector<ParsedToolCall> tool_calls;
  bool reasoning_closed{false};  ///< </think> was seen.
  /// Trailing bytes of the input withheld because they may begin a tag
  /// (partial mode) or an unfinished call; they are not in any field.
  std::size_t held_back{0};
};

/// Splits generated text. `prompt_opened_think` is true when the prompt ended
/// in `<think>` (the template's generation prompt), so the text starts inside
/// reasoning. With `partial` set the text is a stream prefix: a trailing
/// partial tag and an incomplete call are withheld rather than emitted. When
/// not partial, an unterminated call is kept as content text.
///
/// An argument value is a string unless the tool's schema (from `tools`) does
/// not restrict it to strings and it parses as non-string JSON, in which case
/// the trimmed JSON text is kept. The template writes strings and JSON alike
/// without quotes, so this is the only way to tell "123" from 123.
[[nodiscard]] ParsedResponse ParseResponse(std::string_view text,
                                           std::span<const ChatTool> tools,
                                           bool prompt_opened_think = true,
                                           bool partial = false);

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_CHAT_TEMPLATE_HPP_
