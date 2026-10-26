#ifndef GUFO_MODELS_LAGUNA_TOOL_CALLS_HPP_
#define GUFO_MODELS_LAGUNA_TOOL_CALLS_HPP_

// Laguna-S-2.1's native tool-call dialect, as its chat template renders
// historical calls:
//   <tool_call>NAME(<arg_key>K</arg_key><arg_value>V</arg_value>)*</tool_call>
// String values are raw text and other values JSON; nothing marks the type,
// so the API surface decides from the tool schema, as for Qwen. The template
// writes both verbatim (`v | tojson if v is not string else v`), so a JSON
// string may hold a literal </arg_value>; a JSON value therefore ends at the
// first closer outside its strings, while raw text ends at the first closer,
// as llama.cpp's tagged parser reads them. The scanner is header-light so the
// HTTP layer can link it without the model engine.

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::laguna {

/// The literals that frame a native Laguna tool call. RenderChat writes
/// historical calls with these and ScanToolCall reads the same bytes back, so
/// the wire format cannot change in one direction only.
struct ToolSyntax {
  static constexpr std::string_view kCallOpen = "<tool_call>";
  static constexpr std::string_view kCallClose = "</tool_call>";
  static constexpr std::string_view kKeyOpen = "<arg_key>";
  static constexpr std::string_view kKeyClose = "</arg_key>";
  static constexpr std::string_view kValueOpen = "<arg_value>";
  static constexpr std::string_view kValueClose = "</arg_value>";
};

/// One argument exactly as the model wrote it, before the API surface decides
/// whether it is a string or JSON (the dialect carries no type flag).
struct ScannedToolArgument {
  std::string name;   ///< Between <arg_key> and </arg_key>.
  std::string value;  ///< Between <arg_value> and </arg_value>, verbatim.
  /// Bytes of the scanned text consumed through this argument.
  std::size_t progress{0};
};

/// One call as written. `name_complete` is set once a tag after the name
/// proves it ended; `unfinished` is the argument whose value ran to the end
/// of the text, with a possible partial </arg_value> held back.
struct ScannedToolCall {
  std::string name;
  bool name_complete{false};
  std::vector<ScannedToolArgument> arguments;
  std::optional<ScannedToolArgument> unfinished;
  bool complete{false};  ///< Every delimiter through </tool_call> was present.
};

struct ToolCallScan {
  bool matched{false};      ///< `text` begins with ToolSyntax::kCallOpen.
  std::size_t progress{0};  ///< Bytes to skip when the call is rejected.
  std::size_t end{0};       ///< Bytes of a complete call.
  std::optional<ScannedToolCall> call;  ///< Absent until a name is present.
};

/// Whether argument `key` of tool `name` holds JSON, i.e. its schema admits
/// no string. The native grammar then constrains it as a JSON value instead
/// of raw text that excludes the closer.
using JsonArgument =
    std::function<bool(std::string_view name, std::string_view key)>;

/// Scans the Laguna call beginning at `text[0]`, if any. ASCII whitespace
/// between tags is tolerated; values are kept verbatim. Without
/// `json_argument`, every value is raw text.
[[nodiscard]] ToolCallScan ScanToolCall(std::string_view text,
                                        const JsonArgument& json_argument = {});

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_TOOL_CALLS_HPP_
