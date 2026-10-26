#include "src/models/laguna/tool_calls.hpp"

#include <algorithm>

namespace gufo::models::laguna {

namespace {

bool IsAsciiSpace(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

std::string_view TrimAscii(std::string_view value) noexcept {
  while (!value.empty() && IsAsciiSpace(value.front()))
    value.remove_prefix(1);
  while (!value.empty() && IsAsciiSpace(value.back()))
    value.remove_suffix(1);
  return value;
}

// Hold a possible closing delimiter until its next byte disambiguates it.
std::string_view BeforePartialDelimiter(std::string_view value,
                                        std::string_view delimiter) noexcept {
  for (auto size = std::min(value.size(), delimiter.size()); size > 0; --size)
    if (delimiter.starts_with(value.substr(value.size() - size)))
      return value.substr(0, value.size() - size);
  return value;
}

// Offset of the </arg_value> ending a JSON value: the first one outside a
// JSON string, since only strings can hold `<`. `quoted` reports whether an
// unterminated value ends inside a string.
std::size_t JsonValueEnd(std::string_view value, bool* quoted) noexcept {
  bool escaped = false;
  *quoted = false;
  for (std::size_t i = 0; i < value.size(); ++i) {
    const char c = value[i];
    if (*quoted) {
      if (escaped)
        escaped = false;
      else if (c == '\\')
        escaped = true;
      else if (c == '"')
        *quoted = false;
    } else if (c == '"') {
      *quoted = true;
    } else if (value.substr(i).starts_with(ToolSyntax::kValueClose)) {
      return i;
    }
  }
  return std::string_view::npos;
}

}  // namespace

ToolCallScan ScanToolCall(std::string_view text,
                          const JsonArgument& json_argument) {
  ToolCallScan result;
  if (!text.starts_with(ToolSyntax::kCallOpen))
    return result;
  result.matched = true;
  result.progress = ToolSyntax::kCallOpen.size();
  result.end = result.progress;
  std::string_view body = text.substr(ToolSyntax::kCallOpen.size());
  const auto offset = [&] {
    return static_cast<std::size_t>(body.data() - text.data());
  };
  // The name runs to the first tag. A name is never a tag, so `<` ends it.
  const auto name_end = body.find('<');
  ScannedToolCall call;
  call.name =
      std::string(TrimAscii(body.substr(0, std::min(name_end, body.size()))));
  if (call.name.empty())
    return result;
  if (name_end == std::string_view::npos) {
    result.call = std::move(call);  // The name may still be growing.
    return result;
  }
  body.remove_prefix(name_end);
  const auto rest = [&](std::string_view tag) {
    // True when what remains could still become `tag`.
    return tag.starts_with(body);
  };
  call.name_complete = body.starts_with(ToolSyntax::kKeyOpen) ||
                       body.starts_with(ToolSyntax::kCallClose);
  bool valid = true;
  while (true) {
    body = TrimAscii(body);
    if (!body.starts_with(ToolSyntax::kKeyOpen))
      break;
    body.remove_prefix(ToolSyntax::kKeyOpen.size());
    const auto key_end = body.find(ToolSyntax::kKeyClose);
    if (key_end == std::string_view::npos) {
      valid = false;
      break;
    }
    ScannedToolArgument argument{.name = std::string(body.substr(0, key_end))};
    body.remove_prefix(key_end + ToolSyntax::kKeyClose.size());
    body = TrimAscii(body);
    if (!body.starts_with(ToolSyntax::kValueOpen)) {
      valid = false;
      break;
    }
    body.remove_prefix(ToolSyntax::kValueOpen.size());
    bool quoted = false;
    const auto value_end =
        json_argument && json_argument(call.name, argument.name)
            ? JsonValueEnd(body, &quoted)
            : body.find(ToolSyntax::kValueClose);
    if (value_end == std::string_view::npos) {
      // Never recover a nested example from inside an unfinished value.
      result.progress = text.size();
      argument.value = std::string(
          quoted ? body
                 : BeforePartialDelimiter(body, ToolSyntax::kValueClose));
      argument.progress = text.size();
      call.unfinished = std::move(argument);
      valid = false;
      break;
    }
    argument.value = std::string(body.substr(0, value_end));
    body.remove_prefix(value_end + ToolSyntax::kValueClose.size());
    argument.progress = offset();
    result.progress = argument.progress;
    call.arguments.push_back(std::move(argument));
  }
  if (valid) {
    body = TrimAscii(body);
    if (body.starts_with(ToolSyntax::kCallClose)) {
      body.remove_prefix(ToolSyntax::kCallClose.size());
      call.complete = true;
      result.end = offset();
      result.progress = result.end;
    } else if (!body.empty() && !rest(ToolSyntax::kCallClose) &&
               !rest(ToolSyntax::kKeyOpen)) {
      // Neither another argument nor the closer can follow: malformed.
      result.progress = std::max(result.progress, offset());
    }
  }
  result.call = std::move(call);
  return result;
}

}  // namespace gufo::models::laguna
