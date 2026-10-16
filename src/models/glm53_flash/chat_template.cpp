#include "src/models/glm53_flash/chat_template.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/quote_tracker.hpp"
#include "src/core/json.hpp"

namespace gufo::models::glm53_flash {

namespace {

// ---------------------------------------------------------------------------
// A small JSON document model that keeps what Python's json.loads keeps and
// what the template's `tojson` prints: member order, int vs float spelling.
// ---------------------------------------------------------------------------

struct Node {
  enum class Kind : std::uint8_t {
    kNull,
    kBool,
    kInt,
    kFloat,
    kString,
    kArray,
    kObject
  };
  Kind kind{Kind::kNull};
  bool flag{false};
  std::string text;  ///< String payload, or the number's source spelling.
  std::vector<Node> items;
  std::vector<std::pair<std::string, Node>> members;

  [[nodiscard]] const Node* Find(std::string_view key) const {
    for (const auto& [name, value] : members)
      if (name == key)
        return &value;
    return nullptr;
  }
};

void AppendUtf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

class JsonReader {
public:
  explicit JsonReader(std::string_view text) : s_(text) {}

  [[nodiscard]] std::optional<Node> Parse() {
    SkipWs();
    Node node;
    if (!Value(node, 0))
      return std::nullopt;
    SkipWs();
    if (i_ != s_.size())
      return std::nullopt;
    return node;
  }

private:
  static constexpr int kMaxDepth = 128;

  void SkipWs() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' ||
                              s_[i_] == '\n' || s_[i_] == '\r'))
      ++i_;
  }

  bool Literal(std::string_view word) {
    if (s_.substr(i_, word.size()) != word)
      return false;
    i_ += word.size();
    return true;
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  bool Value(Node& out, int depth) {
    if (i_ >= s_.size() || depth > kMaxDepth)
      return false;
    const char c = s_[i_];
    if (c == '{')
      return Object(out, depth);
    if (c == '[')
      return Array(out, depth);
    if (c == '"') {
      out.kind = Node::Kind::kString;
      return String(out.text);
    }
    if (c == 't') {
      out.kind = Node::Kind::kBool;
      out.flag = true;
      return Literal("true");
    }
    if (c == 'f') {
      out.kind = Node::Kind::kBool;
      return Literal("false");
    }
    if (c == 'n') {
      out.kind = Node::Kind::kNull;
      return Literal("null");
    }
    return Number(out);
  }

  bool Number(Node& out) {
    const std::size_t begin = i_;
    if (i_ < s_.size() && s_[i_] == '-')
      ++i_;
    if (i_ >= s_.size())
      return false;
    if (s_[i_] == '0') {
      ++i_;
    } else if (s_[i_] >= '1' && s_[i_] <= '9') {
      while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_])))
        ++i_;
    } else {
      return false;
    }
    bool is_float = false;
    if (i_ < s_.size() && s_[i_] == '.') {
      is_float = true;
      ++i_;
      const std::size_t digits = i_;
      while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_])))
        ++i_;
      if (i_ == digits)
        return false;
    }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      is_float = true;
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-'))
        ++i_;
      const std::size_t digits = i_;
      while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_])))
        ++i_;
      if (i_ == digits)
        return false;
    }
    out.kind = is_float ? Node::Kind::kFloat : Node::Kind::kInt;
    out.text = std::string(s_.substr(begin, i_ - begin));
    return true;
  }

  bool Hex4(std::uint32_t& value) {
    if (i_ + 4 > s_.size())
      return false;
    value = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_ + k];
      value <<= 4;
      if (c >= '0' && c <= '9')
        value |= static_cast<std::uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f')
        value |= static_cast<std::uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        value |= static_cast<std::uint32_t>(c - 'A' + 10);
      else
        return false;
    }
    i_ += 4;
    return true;
  }

  bool String(std::string& out) {
    ++i_;  // opening quote
    out.clear();
    while (i_ < s_.size()) {
      const char c = s_[i_++];
      if (c == '"')
        return true;
      if (static_cast<unsigned char>(c) < 0x20)
        return false;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (i_ >= s_.size())
        return false;
      const char e = s_[i_++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          out.push_back(e);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          std::uint32_t cp = 0;
          if (!Hex4(cp))
            return false;
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            std::uint32_t low = 0;
            if (!Literal("\\u") || !Hex4(low) || low < 0xDC00 || low > 0xDFFF)
              return false;
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            return false;  // Lone surrogates cannot be written as UTF-8.
          }
          AppendUtf8(out, cp);
          break;
        }
        default:
          return false;
      }
    }
    return false;
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  bool Array(Node& out, int depth) {
    out.kind = Node::Kind::kArray;
    ++i_;
    SkipWs();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      return true;
    }
    while (true) {
      SkipWs();
      Node item;
      if (!Value(item, depth + 1))
        return false;
      out.items.push_back(std::move(item));
      SkipWs();
      if (i_ >= s_.size())
        return false;
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == ']') {
        ++i_;
        return true;
      }
      return false;
    }
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  bool Object(Node& out, int depth) {
    out.kind = Node::Kind::kObject;
    ++i_;
    SkipWs();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      return true;
    }
    while (true) {
      SkipWs();
      if (i_ >= s_.size() || s_[i_] != '"')
        return false;
      std::string key;
      if (!String(key))
        return false;
      SkipWs();
      if (i_ >= s_.size() || s_[i_] != ':')
        return false;
      ++i_;
      SkipWs();
      Node value;
      if (!Value(value, depth + 1))
        return false;
      // As in a Python dict: a repeated key keeps its first position and its
      // last value.
      auto existing = std::ranges::find(
          out.members, key, [](const auto& kv) { return kv.first; });
      if (existing != out.members.end())
        existing->second = std::move(value);
      else
        out.members.emplace_back(std::move(key), std::move(value));
      SkipWs();
      if (i_ >= s_.size())
        return false;
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == '}') {
        ++i_;
        return true;
      }
      return false;
    }
  }

  std::string_view s_;
  std::size_t i_{0};
};

// Python's float repr: shortest round-trip digits, fixed notation for decimal
// exponents in [-4, 16), otherwise d[.ddd]e+XX.
std::string PythonFloatRepr(double value) {
  if (std::isnan(value))
    return "NaN";
  if (std::isinf(value))
    return value < 0 ? "-Infinity" : "Infinity";
  std::array<char, 64> buffer{};
  const auto [end, error] =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                    std::chars_format::scientific);
  if (error != std::errc{})
    return "0.0";
  std::string_view sci(buffer.data(),
                       static_cast<std::size_t>(end - buffer.data()));
  std::string out;
  if (sci.starts_with('-')) {
    out.push_back('-');
    sci.remove_prefix(1);
  }
  const auto e_pos = sci.find('e');
  std::string digits;
  for (const char c : sci.substr(0, e_pos))
    if (c != '.')
      digits.push_back(c);
  const int exp10 = std::atoi(std::string(sci.substr(e_pos + 1)).c_str());
  if (exp10 >= -4 && exp10 < 16) {
    if (exp10 >= 0) {
      const auto int_len = static_cast<std::size_t>(exp10) + 1;
      if (digits.size() < int_len)
        digits.append(int_len - digits.size(), '0');
      out.append(digits, 0, int_len);
      out.push_back('.');
      out.append(digits.size() > int_len ? digits.substr(int_len) : "0");
    } else {
      out.append("0.");
      out.append(static_cast<std::size_t>(-exp10 - 1), '0');
      out.append(digits);
    }
  } else {
    out.push_back(digits[0]);
    if (digits.size() > 1) {
      out.push_back('.');
      out.append(digits, 1, std::string::npos);
    }
    char tail[16];
    std::snprintf(tail, sizeof(tail), "e%c%02d", exp10 < 0 ? '-' : '+',
                  std::abs(exp10));
    out.append(tail);
  }
  return out;
}

void DumpString(std::string& out, std::string_view text) {
  out.push_back('"');
  for (const char ch : text) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':
        out.append("\\\"");
        break;
      case '\\':
        out.append("\\\\");
        break;
      case '\b':
        out.append("\\b");
        break;
      case '\f':
        out.append("\\f");
        break;
      case '\n':
        out.append("\\n");
        break;
      case '\r':
        out.append("\\r");
        break;
      case '\t':
        out.append("\\t");
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out.append(buf);
        } else {
          out.push_back(ch);
        }
    }
  }
  out.push_back('"');
}

// json.dumps(ensure_ascii=False) with the default ", " / ": " separators.
// NOLINTNEXTLINE(misc-no-recursion)
void DumpNode(std::string& out, const Node& node) {
  switch (node.kind) {
    case Node::Kind::kNull:
      out.append("null");
      break;
    case Node::Kind::kBool:
      out.append(node.flag ? "true" : "false");
      break;
    case Node::Kind::kInt:
      // int("-0") prints as 0; every other integer keeps its digits.
      out.append(node.text == "-0" ? "0" : node.text);
      break;
    case Node::Kind::kFloat:
      out.append(PythonFloatRepr(std::strtod(node.text.c_str(), nullptr)));
      break;
    case Node::Kind::kString:
      DumpString(out, node.text);
      break;
    case Node::Kind::kArray:
      out.push_back('[');
      for (std::size_t i = 0; i < node.items.size(); ++i) {
        if (i != 0)
          out.append(", ");
        DumpNode(out, node.items[i]);
      }
      out.push_back(']');
      break;
    case Node::Kind::kObject:
      out.push_back('{');
      for (std::size_t i = 0; i < node.members.size(); ++i) {
        if (i != 0)
          out.append(", ");
        DumpString(out, node.members[i].first);
        out.append(": ");
        DumpNode(out, node.members[i].second);
      }
      out.push_back('}');
      break;
  }
}

[[nodiscard]] bool Truthy(const Node& node) {
  switch (node.kind) {
    case Node::Kind::kNull:
      return false;
    case Node::Kind::kBool:
      return node.flag;
    case Node::Kind::kInt:
      return node.text.find_first_not_of("-0") != std::string::npos;
    case Node::Kind::kFloat:
      return std::strtod(node.text.c_str(), nullptr) != 0.0;
    case Node::Kind::kString:
      return !node.text.empty();
    case Node::Kind::kArray:
      return !node.items.empty();
    case Node::Kind::kObject:
      return !node.members.empty();
  }
  return false;
}

// ---------------------------------------------------------------------------
// Python str.strip() (Unicode whitespace) on UTF-8.
// ---------------------------------------------------------------------------

[[nodiscard]] bool IsPythonSpace(std::uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) ||
         cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// Decodes the code point starting at s[i]; `len` receives its byte length.
[[nodiscard]] std::uint32_t DecodeAt(std::string_view s, std::size_t i,
                                     std::size_t& len) {
  const auto b0 = static_cast<unsigned char>(s[i]);
  std::size_t need = b0 < 0x80    ? 1
                     : b0 >= 0xF0 ? 4
                     : b0 >= 0xE0 ? 3
                     : b0 >= 0xC0 ? 2
                                  : 1;
  if (i + need > s.size())
    need = 1;
  len = need;
  if (need == 1)
    return b0;
  std::uint32_t cp = b0 & (0x3F >> (need - 1));
  for (std::size_t k = 1; k < need; ++k) {
    const auto b = static_cast<unsigned char>(s[i + k]);
    if ((b & 0xC0) != 0x80) {
      len = 1;
      return b0;
    }
    cp = (cp << 6) | (b & 0x3F);
  }
  return cp;
}

[[nodiscard]] std::string_view PythonStrip(std::string_view s) {
  std::size_t begin = 0;
  while (begin < s.size()) {
    std::size_t len = 1;
    if (!IsPythonSpace(DecodeAt(s, begin, len)))
      break;
    begin += len;
  }
  std::size_t end = s.size();
  while (end > begin) {
    std::size_t start = end - 1;
    while (start > begin &&
           (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80)
      --start;
    std::size_t len = 1;
    if (!IsPythonSpace(DecodeAt(s, start, len)) || start + len != end)
      break;
    end = start;
  }
  return s.substr(begin, end - begin);
}

// ---------------------------------------------------------------------------
// Rendering.
// ---------------------------------------------------------------------------

class Output {
public:
  explicit Output(std::vector<ContentSpan>* spans) : spans_(spans) {}

  void Frame(std::string_view text) { text_.append(text); }
  void Content(std::string_view text) {
    if (spans_ != nullptr && !text.empty()) {
      if (!spans_->empty() &&
          spans_->back().offset + spans_->back().size == text_.size())
        spans_->back().size += text.size();
      else
        spans_->push_back({text_.size(), text.size()});
    }
    text_.append(text);
  }
  std::string Take() { return std::move(text_); }

private:
  std::string text_;
  std::vector<ContentSpan>* spans_;
};

constexpr std::string_view kToolsHeader =
    "<|system|>\n# Tools\n\nYou may call one or more functions to assist with "
    "the user query.\n\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\n<tools>\n";
constexpr std::string_view kToolsFooter =
    "</tools>\n\nFor each function call, output the function name and "
    "arguments within the following XML format:\n"
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-"
    "value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-"
    "2}</arg_value>...</tool_call>";

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
  return false;
}

// One tool as the template's `tool_to_json` writes it: an empty inner optional
// when the tool is deferred (the template skips it), nullopt on invalid input.
std::optional<std::optional<std::string>> RenderTool(const ChatTool& tool,
                                                     std::string* error_msg) {
  Node root;
  if (!tool.definition_json.empty()) {
    auto parsed = JsonReader(tool.definition_json).Parse();
    if (!parsed || parsed->kind != Node::Kind::kObject) {
      Fail(error_msg, "tool definition is not a JSON object");
      return std::nullopt;
    }
    root = std::move(*parsed);
  } else {
    root.kind = Node::Kind::kObject;
    Node name;
    name.kind = Node::Kind::kString;
    name.text = tool.name;
    root.members.emplace_back("name", std::move(name));
    if (!tool.description.empty()) {
      Node description;
      description.kind = Node::Kind::kString;
      description.text = tool.description;
      root.members.emplace_back("description", std::move(description));
    }
    auto parameters = JsonReader(tool.parameters_json).Parse();
    if (!parameters) {
      Fail(error_msg, "tool '" + tool.name + "' parameters are not valid JSON");
      return std::nullopt;
    }
    root.members.emplace_back("parameters", std::move(*parameters));
  }
  const Node* fn = &root;
  if (const Node* inner = root.Find("function")) {
    if (inner->kind != Node::Kind::kObject) {
      Fail(error_msg, "tool 'function' is not a JSON object");
      return std::nullopt;
    }
    fn = inner;
  }
  if (const Node* deferred = fn->Find("defer_loading");
      deferred != nullptr && Truthy(*deferred))
    return std::optional<std::string>{};
  std::string out = "{";
  bool first = true;
  for (const auto& [key, value] : fn->members) {
    if (key == "defer_loading" || key == "strict")
      continue;
    if (!first)
      out.append(", ");
    first = false;
    // The template writes the key unescaped: "{{ k }}".
    out.append("\"").append(key).append("\": ");
    DumpNode(out, value);
  }
  out.push_back('}');
  return std::optional<std::string>{std::move(out)};
}

std::string_view EffortName(ReasoningLevel level) {
  switch (level) {
    case ReasoningLevel::kLow:
      return "Low";
    case ReasoningLevel::kHigh:
      return "High";
    case ReasoningLevel::kMax:
      break;
  }
  return "Max";
}

// `<think>` text and the visible body of an assistant message.
struct AssistantParts {
  std::optional<std::string> reasoning;
  std::string body;  ///< Already stripped.
};

AssistantParts SplitAssistant(const ChatMessage& message) {
  AssistantParts parts;
  std::string_view content = message.content;
  if (message.reasoning_content.has_value()) {
    parts.reasoning = *message.reasoning_content;
  } else if (const auto close = content.find(ToolSyntax::kThinkClose);
             close != std::string_view::npos) {
    // content.split('</think>')[0].split('<think>')[-1]
    const auto head = content.substr(0, close);
    const auto open = head.rfind("<think>");
    parts.reasoning = std::string(
        open == std::string_view::npos ? head : head.substr(open + 7));
    // content.split('</think>')[-1]
    content = content.substr(content.rfind(ToolSyntax::kThinkClose) +
                             ToolSyntax::kThinkClose.size());
  }
  parts.body = std::string(PythonStrip(content));
  return parts;
}

bool RenderToolArgument(Output& out, const ChatMessage::ToolArgument& arg,
                        std::string* error_msg) {
  out.Frame(ToolSyntax::kKeyOpen);
  out.Content(arg.name);
  out.Frame(ToolSyntax::kKeyClose);
  out.Frame(ToolSyntax::kValueOpen);
  if (arg.is_string) {
    out.Content(arg.value);
  } else {
    auto parsed = JsonReader(arg.value).Parse();
    if (!parsed)
      return Fail(error_msg, "argument '" + arg.name + "' is not valid JSON");
    // A JSON string is a Python str: the template prints it unquoted.
    if (parsed->kind == Node::Kind::kString) {
      out.Content(parsed->text);
    } else {
      std::string dumped;
      DumpNode(dumped, *parsed);
      out.Content(dumped);
    }
  }
  out.Frame(ToolSyntax::kValueClose);
  return true;
}

void RenderToolBlock(Output& out, std::span<const ChatMessage> messages,
                     std::size_t block_start) {
  std::size_t block_end = block_start;
  while (block_end + 1 < messages.size() &&
         messages[block_end + 1].role == ChatRole::kTool)
    ++block_end;
  const std::vector<ChatMessage::ToolCall>* calls = nullptr;
  if (block_start > 0 &&
      messages[block_start - 1].role == ChatRole::kAssistant &&
      !messages[block_start - 1].tool_calls.empty())
    calls = &messages[block_start - 1].tool_calls;

  // The template reorders results to the order of the calls, but only when
  // every result and call carries an id, ids are unique, and they pair off.
  bool can_sort = calls != nullptr;
  if (can_sort) {
    for (std::size_t k = block_start; k <= block_end && can_sort; ++k) {
      const auto& id = messages[k].tool_call_id;
      if (id.empty()) {
        can_sort = false;
        break;
      }
      std::size_t seen = 0;
      for (std::size_t j = block_start; j <= block_end; ++j)
        seen += messages[j].tool_call_id == id ? 1 : 0;
      const bool known = std::ranges::any_of(
          *calls, [&](const auto& call) { return call.id == id; });
      if (seen > 1 || !known)
        can_sort = false;
    }
    for (std::size_t i = 0; i < calls->size() && can_sort; ++i) {
      if ((*calls)[i].id.empty())
        can_sort = false;
      for (std::size_t j = i + 1; j < calls->size(); ++j)
        if ((*calls)[j].id == (*calls)[i].id)
          can_sort = false;
    }
  }

  const auto response = [&](const ChatMessage& message) {
    out.Frame("<tool_response>");
    out.Content(message.content);
    out.Frame("</tool_response>");
  };
  out.Frame("<|observation|>");
  if (can_sort) {
    for (const auto& call : *calls)
      for (std::size_t k = block_start; k <= block_end; ++k)
        if (messages[k].tool_call_id == call.id)
          response(messages[k]);
  } else {
    for (std::size_t k = block_start; k <= block_end; ++k)
      response(messages[k]);
  }
}

}  // namespace

std::optional<std::string> PythonJsonDumps(std::string_view json_text) {
  auto parsed = JsonReader(json_text).Parse();
  if (!parsed)
    return std::nullopt;
  std::string out;
  DumpNode(out, *parsed);
  return out;
}

ChatTemplateOptions ResolveChatOptions(const ReasoningOptions& reasoning) {
  ChatTemplateOptions options;
  options.enable_thinking = reasoning.enabled.value_or(options.enable_thinking);
  if (reasoning.preserve_thinking.has_value())
    options.clear_thinking = !*reasoning.preserve_thinking;
  switch (reasoning.effort.value_or(gufo::ReasoningEffort::kMax)) {
    case gufo::ReasoningEffort::kMinimal:
    case gufo::ReasoningEffort::kLow:
      options.reasoning_effort = ReasoningLevel::kLow;
      break;
    case gufo::ReasoningEffort::kMedium:
    case gufo::ReasoningEffort::kHigh:
      options.reasoning_effort = ReasoningLevel::kHigh;
      break;
    case gufo::ReasoningEffort::kXHigh:
    case gufo::ReasoningEffort::kMax:
      options.reasoning_effort = ReasoningLevel::kMax;
      break;
  }
  return options;
}

std::string_view GenerationPrompt(bool enable_thinking) {
  return enable_thinking ? "<|assistant|><think>"
                         : "<|assistant|><think></think>";
}

std::optional<std::string> RenderChat(std::span<const ChatMessage> messages,
                                      std::span<const ChatTool> tools,
                                      const ChatTemplateOptions& options,
                                      std::string* error_msg,
                                      std::vector<ContentSpan>* content_spans) {
  if (content_spans != nullptr)
    content_spans->clear();
  Output out(content_spans);
  out.Frame("[gMASK]<sop><|system|>Reasoning Effort: ");
  out.Frame(EffortName(options.reasoning_effort));

  if (!tools.empty()) {
    out.Frame(kToolsHeader);
    for (const auto& tool : tools) {
      auto rendered = RenderTool(tool, error_msg);
      if (!rendered)
        return std::nullopt;
      if (!rendered->has_value())
        continue;
      out.Content(**rendered);
      out.Frame("\n");
    }
    out.Frame(kToolsFooter);
  }

  std::size_t last_user = 0;
  bool have_user = false;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (messages[i].role == ChatRole::kUser) {
      last_user = i;
      have_user = true;
    }
  }

  for (std::size_t i = 0; i < messages.size(); ++i) {
    const ChatMessage& message = messages[i];
    switch (message.role) {
      case ChatRole::kUser:
        out.Frame("<|user|>");
        out.Content(message.content);
        break;
      case ChatRole::kSystem:
        out.Frame("<|system|>");
        out.Content(message.content);
        break;
      case ChatRole::kAssistant: {
        out.Frame("<|assistant|>");
        const AssistantParts parts = SplitAssistant(message);
        // loop.index0 > last_user_index, with last_user_index = -1 when no
        // user message exists.
        const bool after_last_user = !have_user || i > last_user;
        if ((!options.clear_thinking || after_last_user) &&
            parts.reasoning.has_value()) {
          out.Frame("<think>");
          out.Content(*parts.reasoning);
          out.Frame("</think>");
        } else {
          out.Frame("<think></think>");
        }
        out.Content(parts.body);
        for (const auto& call : message.tool_calls) {
          out.Frame(ToolSyntax::kCallOpen);
          out.Content(call.name);
          for (const auto& arg : call.arguments)
            if (!RenderToolArgument(out, arg, error_msg))
              return std::nullopt;
          out.Frame(ToolSyntax::kCallClose);
        }
        break;
      }
      case ChatRole::kTool:
        if (i == 0 || messages[i - 1].role != ChatRole::kTool)
          RenderToolBlock(out, messages, i);
        break;
    }
  }
  if (options.add_generation_prompt)
    out.Frame(GenerationPrompt(options.enable_thinking));
  return out.Take();
}

std::vector<TokenId> EncodeRendered(
    const Tokenizer& tokenizer, std::string_view rendered,
    std::span<const ContentSpan> content_spans) {
  std::vector<TokenId> ids;
  const auto append = [&](std::string_view text, bool parse_special) {
    if (text.empty())
      return;
    const auto part = tokenizer.Encode(text, parse_special);
    ids.insert(ids.end(), part.begin(), part.end());
  };
  std::size_t cursor = 0;
  for (const auto& span : content_spans) {
    if (span.offset < cursor || span.offset + span.size > rendered.size())
      continue;
    append(rendered.substr(cursor, span.offset - cursor), true);
    append(rendered.substr(span.offset, span.size), false);
    cursor = span.offset + span.size;
  }
  append(rendered.substr(cursor), true);
  return ids;
}

// ---------------------------------------------------------------------------
// Tool-call parsing.
// ---------------------------------------------------------------------------

namespace {

bool IsAsciiSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
         c == '\f';
}

std::string_view TrimAscii(std::string_view s) {
  while (!s.empty() && IsAsciiSpace(s.front()))
    s.remove_prefix(1);
  while (!s.empty() && IsAsciiSpace(s.back()))
    s.remove_suffix(1);
  return s;
}

// Length of the longest proper prefix of `tag` that `text` ends with.
std::size_t PartialSuffix(std::string_view text, std::string_view tag) {
  const std::size_t most = std::min(text.size(), tag.size() - 1);
  for (std::size_t k = most; k > 0; --k)
    if (text.ends_with(tag.substr(0, k)))
      return k;
  return 0;
}

}  // namespace

ToolCallScan ScanToolCall(std::string_view text) {
  ToolCallScan result;
  if (!text.starts_with(ToolSyntax::kCallOpen))
    return result;
  result.matched = true;
  const std::size_t begin = ToolSyntax::kCallOpen.size();
  result.progress = begin;
  result.end = begin;

  std::string_view body = text.substr(begin);
  const auto offset = [&]() {
    return static_cast<std::size_t>(body.data() - text.data());
  };
  const auto skip_ws = [&]() {
    while (!body.empty() && IsAsciiSpace(body.front()))
      body.remove_prefix(1);
  };

  // The name runs to the first argument or to the closing tag.
  const auto key = body.find(ToolSyntax::kKeyOpen);
  const auto close = body.find(ToolSyntax::kCallClose);
  const auto name_end = std::min(key, close);
  if (name_end == std::string_view::npos) {
    result.progress = text.size();  // Name still arriving.
    return result;
  }
  ToolCallSyntax call;
  call.name = std::string(TrimAscii(body.substr(0, name_end)));
  // A function name has no markup or spaces; anything else is prose.
  if (call.name.empty() || std::ranges::any_of(call.name, [](char c) {
        return c == '<' || c == '>' || IsAsciiSpace(c);
      })) {
    result.malformed = true;
    return result;
  }
  body.remove_prefix(name_end);

  while (true) {
    skip_ws();
    if (body.starts_with(ToolSyntax::kCallClose)) {
      body.remove_prefix(ToolSyntax::kCallClose.size());
      call.complete = true;
      result.end = offset();
      result.progress = result.end;
      break;
    }
    if (!body.starts_with(ToolSyntax::kKeyOpen)) {
      // Either the text ran out inside a tag, or it is not a call.
      const bool prefix = body.empty() ||
                          ToolSyntax::kKeyOpen.starts_with(body) ||
                          ToolSyntax::kCallClose.starts_with(body);
      if (!prefix) {
        result.malformed = true;
        result.progress = begin;
        return result;
      }
      result.progress = text.size();
      break;
    }
    body.remove_prefix(ToolSyntax::kKeyOpen.size());
    const auto key_end = body.find(ToolSyntax::kKeyClose);
    if (key_end == std::string_view::npos) {
      result.progress = text.size();
      break;
    }
    ToolCallArgumentSyntax arg;
    arg.name = std::string(TrimAscii(body.substr(0, key_end)));
    body.remove_prefix(key_end + ToolSyntax::kKeyClose.size());
    skip_ws();
    if (!body.starts_with(ToolSyntax::kValueOpen)) {
      const bool prefix =
          body.empty() || ToolSyntax::kValueOpen.starts_with(body);
      if (!prefix) {
        result.malformed = true;
        result.progress = begin;
        return result;
      }
      result.progress = text.size();
      break;
    }
    body.remove_prefix(ToolSyntax::kValueOpen.size());
    const auto value_end = body.find(ToolSyntax::kValueClose);
    if (value_end == std::string_view::npos) {
      // Hold back a possible partial </arg_value> so a stream never emits it
      // as value text.
      const auto held = PartialSuffix(body, ToolSyntax::kValueClose);
      arg.value = std::string(body.substr(0, body.size() - held));
      call.unfinished = std::move(arg);
      result.progress = text.size();
      break;
    }
    arg.value = std::string(body.substr(0, value_end));
    body.remove_prefix(value_end + ToolSyntax::kValueClose.size());
    call.arguments.push_back(std::move(arg));
  }
  result.call = std::move(call);
  return result;
}

namespace {

// JSON value types a schema permits, as a bit set over Node::Kind. Numbers
// are one bit: the parser keeps int and float spellings apart, a schema does
// not.
using TypeMask = unsigned;
constexpr TypeMask kMaskNull = 1U << 0;
constexpr TypeMask kMaskBool = 1U << 1;
constexpr TypeMask kMaskNumber = 1U << 2;
constexpr TypeMask kMaskString = 1U << 3;
constexpr TypeMask kMaskArray = 1U << 4;
constexpr TypeMask kMaskObject = 1U << 5;
constexpr TypeMask kMaskAny = 0x3FU;

TypeMask MaskOfKind(Node::Kind kind) {
  switch (kind) {
    case Node::Kind::kNull:
      return kMaskNull;
    case Node::Kind::kBool:
      return kMaskBool;
    case Node::Kind::kInt:
    case Node::Kind::kFloat:
      return kMaskNumber;
    case Node::Kind::kString:
      return kMaskString;
    case Node::Kind::kArray:
      return kMaskArray;
    case Node::Kind::kObject:
      break;
  }
  return kMaskObject;
}

// The mask of a JSON value in a schema's enum or const.
TypeMask MaskOfSchemaValue(const json::Value& value) {
  if (value.is_null())
    return kMaskNull;
  if (value.is_bool())
    return kMaskBool;
  if (value.is_number())
    return kMaskNumber;
  if (value.is_string())
    return kMaskString;
  return value.is_array() ? kMaskArray : kMaskObject;
}

TypeMask MaskOfTypeName(std::string_view name) {
  if (name == "string")
    return kMaskString;
  if (name == "integer" || name == "number")
    return kMaskNumber;
  if (name == "boolean")
    return kMaskBool;
  if (name == "null")
    return kMaskNull;
  if (name == "array")
    return kMaskArray;
  if (name == "object")
    return kMaskObject;
  return kMaskAny;  // Unknown name: no restriction rather than a guess.
}

// Types a property schema permits: the intersection of what `type`, `enum`,
// `const` and `anyOf`/`oneOf` each allow. A keyword that is absent, or that a
// reader cannot interpret, restricts nothing, so a schema without any of them
// keeps the unrestricted default.
TypeMask SchemaTypeMask(const json::Value& schema) {
  TypeMask mask = kMaskAny;
  if (!schema.is_object())
    return mask;
  if (const auto* type = schema.find("type"); type != nullptr) {
    if (type->is_string()) {
      mask &= MaskOfTypeName(type->str());
    } else if (type->is_array() && !type->items().empty()) {
      TypeMask named = 0;
      for (const auto& item : type->items())
        named |= item.is_string() ? MaskOfTypeName(item.str()) : kMaskAny;
      mask &= named;
    }
  }
  if (const auto* values = schema.find("enum");
      values != nullptr && values->is_array() && !values->items().empty()) {
    TypeMask named = 0;
    for (const auto& item : values->items())
      named |= MaskOfSchemaValue(item);
    mask &= named;
  }
  if (const auto* constant = schema.find("const"); constant != nullptr)
    mask &= MaskOfSchemaValue(*constant);
  for (const char* keyword : {"anyOf", "oneOf"}) {
    const auto* branches = schema.find(keyword);
    if (branches == nullptr || !branches->is_array() ||
        branches->items().empty())
      continue;
    TypeMask joined = 0;
    for (const auto& branch : branches->items())
      joined |= SchemaTypeMask(branch);
    mask &= joined;
  }
  return mask;
}

// The types argument `name` of `tool` may take. Unknown tools and properties
// are unrestricted.
TypeMask ArgumentTypeMask(std::span<const ChatTool> tools,
                          std::string_view tool, std::string_view name) {
  for (const auto& candidate : tools) {
    std::string tool_name = candidate.name;
    std::string parameters = candidate.parameters_json;
    if (!candidate.definition_json.empty()) {
      try {
        const auto def = json::parse(candidate.definition_json);
        const auto* fn = def.find("function");
        const auto& body = fn != nullptr ? *fn : def;
        tool_name = body.member_str("name");
        const auto* p = body.find("parameters");
        parameters = p != nullptr ? p->dump() : "{}";
      } catch (const std::exception&) {
        continue;
      }
    }
    if (tool_name != tool)
      continue;
    try {
      const auto schema = json::parse(parameters);
      const auto* properties = schema.find("properties");
      const auto* property =
          properties != nullptr ? properties->find(std::string(name)) : nullptr;
      return property != nullptr ? SchemaTypeMask(*property) : kMaskAny;
    } catch (const std::exception&) {
      return kMaskAny;
    }
  }
  return kMaskAny;
}

ParsedToolCall InterpretCall(const ToolCallSyntax& syntax,
                             std::span<const ChatTool> tools) {
  ParsedToolCall call;
  call.name = syntax.name;
  for (const auto& arg : syntax.arguments) {
    ChatMessage::ToolArgument out;
    out.name = arg.name;
    out.value = arg.value;
    out.is_string = true;
    const TypeMask allowed = ArgumentTypeMask(tools, syntax.name, arg.name);
    if (allowed != kMaskString) {
      const auto trimmed = TrimAscii(arg.value);
      if (!trimmed.empty()) {
        const auto parsed = JsonReader(trimmed).Parse();
        // The decoded value stands only when the schema permits its type;
        // otherwise "123" for a ["string","null"] property is the string.
        if (parsed && parsed->kind != Node::Kind::kString &&
            (allowed & MaskOfKind(parsed->kind)) != 0) {
          out.value = std::string(trimmed);
          out.is_string = false;
        }
      }
    }
    call.arguments.push_back(std::move(out));
  }
  return call;
}

}  // namespace

ParsedToolCall InterpretToolCall(const ToolCallSyntax& syntax,
                                 std::span<const ChatTool> tools) {
  return InterpretCall(syntax, tools);
}

bool IsStringArgument(std::span<const ChatTool> tools, std::string_view tool,
                      std::string_view argument) {
  return ArgumentTypeMask(tools, tool, argument) == kMaskString;
}

namespace {
constexpr std::string_view kThinkOpen = "<think>";
}  // namespace

ParsedResponse ParseResponse(std::string_view text,
                             std::span<const ChatTool> tools,
                             bool prompt_opened_think, bool partial) {
  ParsedResponse response;
  std::string_view rest = text;

  if (prompt_opened_think) {
    const auto close = rest.find(ToolSyntax::kThinkClose);
    if (close == std::string_view::npos) {
      std::size_t held =
          partial ? PartialSuffix(rest, ToolSyntax::kThinkClose) : 0;
      response.reasoning = std::string(rest.substr(0, rest.size() - held));
      response.held_back = held;
      return response;
    }
    response.reasoning = std::string(rest.substr(0, close));
    response.reasoning_closed = true;
    rest.remove_prefix(close + ToolSyntax::kThinkClose.size());
  } else if (rest.starts_with(kThinkOpen)) {
    const auto close = rest.find(ToolSyntax::kThinkClose);
    if (close == std::string_view::npos) {
      const std::size_t held =
          partial ? PartialSuffix(rest, ToolSyntax::kThinkClose) : 0;
      response.reasoning = std::string(rest.substr(
          kThinkOpen.size(), rest.size() - kThinkOpen.size() - held));
      response.held_back = held;
      return response;
    }
    response.reasoning =
        std::string(rest.substr(kThinkOpen.size(), close - kThinkOpen.size()));
    response.reasoning_closed = true;
    rest.remove_prefix(close + ToolSyntax::kThinkClose.size());
  } else if (partial && !rest.empty() && rest.size() < kThinkOpen.size() &&
             kThinkOpen.starts_with(rest)) {
    // A stream may still be spelling "<think>": it is not content yet.
    response.held_back = rest.size();
    return response;
  }

  // A marker inside a backtick span or code fence is the model quoting the
  // format, not calling it. Every candidate is asked, not just the first;
  // a call's own argument values are skipped whole, so markup nested in one
  // never starts a second call.
  gufo::server::QuoteTracker quotes;
  quotes.Reset(rest);
  std::size_t cursor = 0;
  while (cursor < rest.size()) {
    const auto open = rest.find(ToolSyntax::kCallOpen, cursor);
    if (open == std::string_view::npos) {
      auto tail = rest.substr(cursor);
      const std::size_t held =
          partial ? PartialSuffix(tail, ToolSyntax::kCallOpen) : 0;
      response.content.append(tail.substr(0, tail.size() - held));
      response.held_back = held;
      return response;
    }
    response.content.append(rest.substr(cursor, open - cursor));
    if (quotes.QuotedAt(open)) {
      response.content.append(ToolSyntax::kCallOpen);
      cursor = open + ToolSyntax::kCallOpen.size();
      continue;
    }
    // An unfinished quote may yet cover this marker: a stream waits.
    if (partial && quotes.UnclosedAt(open)) {
      response.held_back = rest.size() - open;
      return response;
    }
    const auto scan = ScanToolCall(rest.substr(open));
    if (scan.call && scan.call->complete) {
      response.tool_calls.push_back(InterpretCall(*scan.call, tools));
      cursor = open + scan.end;
      continue;
    }
    if (scan.malformed) {
      response.content.append(ToolSyntax::kCallOpen);
      cursor = open + ToolSyntax::kCallOpen.size();
      continue;
    }
    // Unfinished: a stream withholds it, a finished response keeps the text.
    if (partial) {
      response.held_back = rest.size() - open;
    } else {
      response.content.append(rest.substr(open));
    }
    return response;
  }
  return response;
}

}  // namespace gufo::models::glm53_flash
