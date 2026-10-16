// A6 chat template check for GLM-5.3-Flash: the hand-written renderer must
// reproduce the model's Jinja template byte-for-byte on every golden in
// fixtures/chat_template_goldens.json (rendered by jinja2 from the template in
// the Maya-S GGUF; see fixtures/make_chat_template_fixture.py). The tool-call
// parser is checked on the template's own output and on streamed prefixes.
// Needs no GGUF; with GLM53_FLASH_GGUF set it also tokenizes a rendered prompt.

#include "src/models/glm53_flash/chat_template.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/glm53_flash/tokenizer.hpp"

namespace glm = gufo::models::glm53_flash;
namespace json = gufo::json;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireEq(const std::string& actual, const std::string& expected,
               const std::string& what) {
  if (actual == expected)
    return;
  std::size_t at = 0;
  while (at < actual.size() && at < expected.size() &&
         actual[at] == expected[at])
    ++at;
  throw std::runtime_error(
      what + ": first difference at byte " + std::to_string(at) +
      "\n  actual:   " + actual.substr(at > 40 ? at - 40 : 0, 120) +
      "\n  expected: " + expected.substr(at > 40 ? at - 40 : 0, 120));
}

std::string ReadFile(const char* path) {
  std::ifstream in(path, std::ios::binary);
  Require(in.good(), std::string("cannot open ") + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

glm::ChatMessage Msg(glm::ChatRole role, std::string content) {
  glm::ChatMessage message;
  message.role = role;
  message.content = std::move(content);
  return message;
}

glm::ChatRole RoleOf(const std::string& role) {
  if (role == "system")
    return glm::ChatRole::kSystem;
  if (role == "user")
    return glm::ChatRole::kUser;
  if (role == "assistant")
    return glm::ChatRole::kAssistant;
  if (role == "tool")
    return glm::ChatRole::kTool;
  throw std::runtime_error("unknown role " + role);
}

glm::ChatMessage MessageFrom(const json::Value& m) {
  glm::ChatMessage message;
  message.role = RoleOf(m.member_str("role"));
  message.content = m.member_str("content");
  if (const auto* r = m.find("reasoning_content"))
    message.reasoning_content = r->str();
  message.tool_call_id = m.member_str("tool_call_id");
  if (const auto* calls = m.find("tool_calls")) {
    for (const auto& c : calls->items()) {
      glm::ChatMessage::ToolCall call;
      call.id = c.member_str("id");
      call.name = c.member_str("name");
      for (const auto& a : c.find("arguments")->items())
        call.arguments.push_back({a.member_str("name"), a.member_str("value"),
                                  a.find("is_string")->as_bool()});
      message.tool_calls.push_back(std::move(call));
    }
  }
  return message;
}

int RunGoldens() {
  const auto goldens = json::parse(ReadFile(GLM53_CHAT_TEMPLATE_GOLDENS));
  int count = 0;
  for (const auto& golden : goldens.items()) {
    const std::string name = golden.member_str("name");
    std::vector<glm::ChatMessage> messages;
    for (const auto& m : golden.find("messages")->items())
      messages.push_back(MessageFrom(m));
    std::vector<glm::ChatTool> tools;
    for (const auto& t : golden.find("tools")->items()) {
      glm::ChatTool tool;
      tool.definition_json = t.str();
      tools.push_back(std::move(tool));
    }
    glm::ChatTemplateOptions options;
    const auto& o = *golden.find("options");
    options.add_generation_prompt = o.find("add_generation_prompt")->as_bool();
    const std::string effort = o.member_str("reasoning_effort", "max");
    options.reasoning_effort = effort == "low"    ? glm::ReasoningLevel::kLow
                               : effort == "high" ? glm::ReasoningLevel::kHigh
                                                  : glm::ReasoningLevel::kMax;
    if (const auto* clear = o.find("clear_thinking");
        clear != nullptr && clear->is_bool())
      options.clear_thinking = clear->as_bool();

    std::string error;
    std::vector<glm::ContentSpan> spans;
    const auto rendered =
        glm::RenderChat(messages, tools, options, &error, &spans);
    Require(rendered.has_value(), name + ": render failed: " + error);
    RequireEq(*rendered, golden.member_str("expected"), "golden " + name);
    // Spans must lie inside the prompt, in order, without overlap.
    std::size_t cursor = 0;
    for (const auto& span : spans) {
      Require(span.offset >= cursor && span.size > 0 &&
                  span.offset + span.size <= rendered->size(),
              name + ": bad content span");
      cursor = span.offset + span.size;
    }
    ++count;
  }
  Require(count >= 12, "need at least 12 goldens");
  return count;
}

void TestPythonJson() {
  const auto dump = [](const char* text) {
    const auto out = glm::PythonJsonDumps(text);
    Require(out.has_value(), std::string("rejected ") + text);
    return *out;
  };
  RequireEq(dump("{\"a\":1,\"b\":[1,2.0,{\"c\":null}]}"),
            "{\"a\": 1, \"b\": [1, 2.0, {\"c\": null}]}", "separators");
  RequireEq(dump("1E3"), "1000.0", "1E3");
  RequireEq(dump("1e-7"), "1e-07", "1e-7");
  RequireEq(dump("0.0001"), "0.0001", "0.0001");
  RequireEq(dump("0.00001"), "1e-05", "0.00001");
  RequireEq(dump("1e16"), "1e+16", "1e16");
  RequireEq(dump("1e15"), "1000000000000000.0", "1e15");
  RequireEq(dump("123456789012345678.0"), "1.2345678901234568e+17", "1.2e17");
  RequireEq(dump("-0"), "0", "-0");
  RequireEq(dump("-0.0"), "-0.0", "-0.0");
  RequireEq(dump("0.1"), "0.1", "0.1");
  RequireEq(dump("100"), "100", "100");
  RequireEq(dump("12345678901234567890123"), "12345678901234567890123",
            "bigint");
  RequireEq(dump("{\"a\":1,\"b\":2,\"a\":3}"), "{\"a\": 3, \"b\": 2}",
            "duplicate key");
  RequireEq(dump("\"\\u00e9\\ud83d\\ude00\\u0001\\u007f/\""),
            "\"\xC3\xA9\xF0\x9F\x98\x80\\u0001\x7f/\"", "string escapes");
  for (const char* bad : {"", "{", "[1,]", "01", "1.", "\"a", "tru", "{\"a\"}",
                          "[1] x", "\"\\ud800\"", "\"\t\""})
    Require(!glm::PythonJsonDumps(bad).has_value(),
            std::string("accepted ") + bad);
}

void TestOptionsAndErrors() {
  const glm::ChatMessage user = Msg(glm::ChatRole::kUser, "hi");
  const std::vector<glm::ChatMessage> messages{user};
  glm::ChatTemplateOptions off;
  off.enable_thinking = false;
  RequireEq(*glm::RenderChat(messages, {}, off),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|>"
            "<think></think>",
            "thinking off");

  gufo::ReasoningOptions reasoning;
  reasoning.effort = gufo::ReasoningEffort::kMinimal;
  reasoning.preserve_thinking = false;
  auto resolved = glm::ResolveChatOptions(reasoning);
  Require(resolved.reasoning_effort == glm::ReasoningLevel::kLow &&
              resolved.clear_thinking,
          "minimal maps to low, preserve=false clears");
  reasoning.effort = gufo::ReasoningEffort::kMedium;
  Require(glm::ResolveChatOptions(reasoning).reasoning_effort ==
              glm::ReasoningLevel::kHigh,
          "medium maps to high");

  std::string error;
  glm::ChatTool bad;
  bad.definition_json = "{not json";
  Require(
      !glm::RenderChat(messages, std::vector{bad}, {}, &error).has_value() &&
          !error.empty(),
      "bad tool definition is an error");
  glm::ChatMessage call_message;
  call_message.role = glm::ChatRole::kAssistant;
  call_message.tool_calls.push_back({"c", "f", {{"x", "{oops", false}}});
  Require(
      !glm::RenderChat(std::vector{call_message}, {}, {}, &error).has_value(),
      "bad argument JSON is an error");

  // Built from fields when no definition_json is given.
  glm::ChatTool fields;
  fields.name = "add";
  fields.description = "Add";
  fields.parameters_json = "{\"type\":\"object\"}";
  const auto with_fields = glm::RenderChat(messages, std::vector{fields});
  Require(with_fields.has_value() &&
              with_fields->find("{\"name\": \"add\", \"description\": \"Add\", "
                                "\"parameters\": {\"type\": \"object\"}}\n") !=
                  std::string::npos,
          "tool from fields");
}

void TestScanner() {
  using glm::ToolSyntax;
  // Exactly what the template writes for a history call.
  const std::string written =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris"
      "</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
      "tail";
  auto scan = glm::ScanToolCall(written);
  Require(scan.matched && scan.call && scan.call->complete, "scan complete");
  Require(scan.call->name == "get_weather" &&
              scan.call->arguments.size() == 2 &&
              scan.call->arguments[0].name == "city" &&
              scan.call->arguments[0].value == "Paris" &&
              scan.call->arguments[1].value == "3",
          "scan fields");
  Require(written.substr(scan.end) == "tail", "scan end");

  // GLM-4.5 style newlines between elements are tolerated; values stay
  // verbatim.
  scan = glm::ScanToolCall(
      "<tool_call>f\n<arg_key> k </arg_key>\n<arg_value> v\n</arg_value>\n"
      "</tool_call>");
  Require(scan.call && scan.call->complete && scan.call->name == "f" &&
              scan.call->arguments.at(0).name == "k" &&
              scan.call->arguments[0].value == " v\n",
          "newline style");

  scan = glm::ScanToolCall("<tool_call>noargs</tool_call>");
  Require(scan.call && scan.call->complete && scan.call->arguments.empty(),
          "no arguments");

  // Every prefix of a call is either pending or a faithful partial.
  const std::string whole =
      "<tool_call>f<arg_key>a</arg_key><arg_value>hello</arg_value>"
      "</tool_call>";
  for (std::size_t n = 0; n < whole.size(); ++n) {
    const auto prefix = glm::ScanToolCall(whole.substr(0, n));
    Require(!prefix.malformed, "prefix " + std::to_string(n) + " malformed");
    Require(!prefix.call || !prefix.call->complete,
            "prefix " + std::to_string(n) + " complete");
  }
  scan = glm::ScanToolCall(
      "<tool_call>f<arg_key>a</arg_key><arg_value>hello</arg_val");
  Require(scan.call && !scan.call->complete && scan.call->unfinished &&
              scan.call->unfinished->name == "a" &&
              scan.call->unfinished->value == "hello",
          "partial closing tag held back");
  scan = glm::ScanToolCall("<tool_call>f<arg_key>a</arg_key><arg_value>hel");
  Require(scan.call && scan.call->unfinished->value == "hel", "partial value");
  scan = glm::ScanToolCall("<tool_call>f");
  Require(scan.matched && !scan.call && !scan.malformed, "name arriving");

  scan = glm::ScanToolCall("<tool_call></tool_call>");
  Require(scan.malformed, "empty name");
  scan = glm::ScanToolCall("<tool_call>f junk <arg_key>");
  Require(scan.malformed && scan.progress == ToolSyntax::kCallOpen.size(),
          "junk in name");
  scan = glm::ScanToolCall("<tool_call>f<arg_key>a</arg_key>garbage");
  Require(scan.malformed, "garbage between key and value");
  Require(!glm::ScanToolCall("plain").matched, "not a call");
}

void TestParseResponse() {
  glm::ChatTool weather;
  weather.definition_json =
      "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":"
      "\"string\"},\"days\":{\"type\":\"integer\"},\"note\":{\"type\":"
      "[\"string\",\"null\"]}}}}}";
  const std::vector<glm::ChatTool> tools{weather};

  const std::string generated =
      "Need the forecast.</think>Checking now.\n<tool_call>get_weather"
      "<arg_key>city</arg_key><arg_value>123</arg_value><arg_key>days"
      "</arg_key><arg_value>3</arg_value><arg_key>extra</arg_key><arg_value>"
      "[1, \"a\"]</arg_value><arg_key>word</arg_key><arg_value>plain text"
      "</arg_value><arg_key>note</arg_key><arg_value>7</arg_value></tool_call>"
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome"
      "</arg_value></tool_call>";
  auto parsed = glm::ParseResponse(generated, tools);
  Require(parsed.reasoning == "Need the forecast." && parsed.reasoning_closed,
          "reasoning");
  Require(parsed.content == "Checking now.\n", "content: " + parsed.content);
  Require(parsed.tool_calls.size() == 2, "two calls");
  const auto& args = parsed.tool_calls[0].arguments;
  Require(args.size() == 5, "five arguments");
  Require(args[0].is_string && args[0].value == "123",
          "schema string stays string");
  Require(!args[1].is_string && args[1].value == "3", "integer is JSON");
  Require(!args[2].is_string && args[2].value == "[1, \"a\"]",
          "undeclared JSON array");
  Require(args[3].is_string && args[3].value == "plain text", "plain string");
  // A ["string","null"] property permits a string or null, never a number:
  // "7" is the string "7".
  Require(args[4].is_string && args[4].value == "7",
          "string-or-null keeps a numeric spelling as a string");
  Require(parsed.tool_calls[1].arguments[0].value == "Rome", "second call");
  Require(parsed.held_back == 0, "nothing held back");

  // Streaming: cumulative prefixes never emit a partial tag or call, and the
  // final prefix equals the whole parse.
  std::string seen_reasoning;
  std::string seen_content;
  for (std::size_t n = 0; n <= generated.size(); ++n) {
    const auto p =
        glm::ParseResponse(generated.substr(0, n), tools, true, true);
    Require(p.reasoning.starts_with(
                seen_reasoning.substr(0, seen_reasoning.size())),
            "reasoning only grows");
    Require(p.content.starts_with(seen_content), "content only grows");
    Require(p.reasoning.find("</") == std::string::npos &&
                p.content.find("<tool") == std::string::npos,
            "no partial tag emitted at " + std::to_string(n));
    Require(p.tool_calls.size() <= 2, "call count");
    seen_reasoning = p.reasoning;
    seen_content = p.content;
  }
  const auto last = glm::ParseResponse(generated, tools, true, true);
  Require(last.tool_calls.size() == 2 && last.content == parsed.content &&
              last.reasoning == parsed.reasoning,
          "stream end equals full parse");

  // A finished response keeps an unterminated call as text.
  parsed = glm::ParseResponse("hi</think>ok<tool_call>f<arg_key>a", {});
  Require(
      parsed.tool_calls.empty() && parsed.content == "ok<tool_call>f<arg_key>a",
      "unterminated call kept as content");
  // A literal tag that is not a call is content.
  parsed = glm::ParseResponse("x</think>a <tool_call> b", {});
  Require(parsed.content == "a <tool_call> b", "stray tag: " + parsed.content);
  // No </think>: all reasoning when the prompt opened one, content otherwise.
  parsed = glm::ParseResponse("still thinking", {});
  Require(parsed.reasoning == "still thinking" && parsed.content.empty() &&
              !parsed.reasoning_closed,
          "unclosed reasoning");
  parsed = glm::ParseResponse("direct answer", {}, false);
  Require(parsed.content == "direct answer" && parsed.reasoning.empty(),
          "no think prefix");
  parsed = glm::ParseResponse("<think>r</think>a", {}, false);
  Require(parsed.reasoning == "r" && parsed.content == "a", "explicit think");
  parsed = glm::ParseResponse("abc</thi", {}, true, true);
  Require(parsed.reasoning == "abc" && parsed.held_back == 5, "held </thi");
}

// An argument's decoded JSON is used only when the schema permits its type.
void TestSchemaTypedArguments() {
  const auto make_tool = [](const std::string& property) {
    glm::ChatTool tool;
    tool.definition_json =
        "{\"type\":\"function\",\"function\":{\"name\":\"f\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"p\":" +
        property + "}}}}";
    return tool;
  };
  // {schema, value, expect the value stays a string}
  struct Case {
    std::string schema;
    std::string value;
    bool is_string;
  };
  const std::vector<Case> cases{
      {R"({"type":["string","null"]})", "123", true},
      {R"({"type":["string","null"]})", "null", false},
      {R"({"type":["string","null"]})", "true", true},
      {R"({"type":["string","null"]})", "[1]", true},
      {R"({"type":["integer","null"]})", "5", false},
      {R"({"type":["integer","null"]})", "null", false},
      {R"({"type":"boolean"})", "7", true},
      {R"({"type":"boolean"})", "true", false},
      {R"({"type":"object"})", "123", true},
      {R"({"type":"object"})", "{\"a\":1}", false},
      {R"({"type":"array"})", "{\"a\":1}", true},
      {R"({"enum":["a","b"]})", "123", true},
      {R"({"enum":["a",null]})", "5", true},
      {R"({"enum":["a",null]})", "null", false},
      {R"({"enum":[1,2]})", "2", false},
      {R"({"enum":["1","2"]})", "2", true},
      {R"({"const":"7"})", "7", true},
      {R"({"const":7})", "7", false},
      {R"({"anyOf":[{"type":"string"},{"type":"null"}]})", "123", true},
      {R"({"anyOf":[{"type":"string"},{"type":"null"}]})", "null", false},
      {R"({"anyOf":[{"type":"string"},{"type":"integer"}]})", "123", false},
      {R"({"oneOf":[{"type":"string"},{"enum":["x"]}]})", "123", true},
      {R"({"anyOf":[{"type":"string"},{"description":"anything"}]})", "123",
       false},
      {R"({"description":"no type"})", "123", false},
  };
  for (const auto& c : cases) {
    const std::vector<glm::ChatTool> tools{make_tool(c.schema)};
    const auto parsed =
        glm::ParseResponse("<tool_call>f<arg_key>p</arg_key><arg_value>" +
                               c.value + "</arg_value></tool_call>",
                           tools, false);
    Require(parsed.tool_calls.size() == 1, "call for " + c.schema);
    const auto& arg = parsed.tool_calls[0].arguments.at(0);
    Require(arg.is_string == c.is_string && arg.value == c.value,
            "schema " + c.schema + " value " + c.value +
                ": is_string=" + (arg.is_string ? "true" : "false"));
  }
  const std::vector<glm::ChatTool> plain{make_tool(R"({"type":"string"})")};
  Require(glm::IsStringArgument(plain, "f", "p"), "string-only property");
  Require(
      glm::IsStringArgument({{make_tool(R"({"enum":["a","b"]})")}}, "f", "p"),
      "string enum is string-only");
  Require(!glm::IsStringArgument({{make_tool(R"({"type":["string","null"]})")}},
                                 "f", "p"),
          "nullable string is not string-only");
}

// A call quoted after a real call is documentation, not a second call.
void TestQuotedExampleAfterCall() {
  const std::string call =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris"
      "</arg_value></tool_call>";
  const std::string text = call + " Example: `" + call + "`";
  const auto parsed = glm::ParseResponse(text, {}, false);
  Require(parsed.tool_calls.size() == 1,
          "quoted copy is not a call, got " +
              std::to_string(parsed.tool_calls.size()));
  Require(parsed.content == " Example: `" + call + "`",
          "quoted copy stays content: " + parsed.content);

  const std::string fenced = call + "\n```\n" + call + "\n```\n";
  Require(glm::ParseResponse(fenced, {}, false).tool_calls.size() == 1,
          "fenced copy is not a call");
  // A quoted call that opens the response is prose as well.
  Require(glm::ParseResponse("`" + call + "`", {}, false).tool_calls.empty(),
          "quoted first call is not a call");
  // Two real calls, and markup nested in an argument value, are unchanged.
  Require(glm::ParseResponse(call + call, {}, false).tool_calls.size() == 2,
          "two real calls");
  const auto nested = glm::ParseResponse(
      "<tool_call>f<arg_key>a</arg_key><arg_value>see <tool_call>g"
      "</arg_value></tool_call>",
      {}, false);
  Require(nested.tool_calls.size() == 1 && nested.content.empty(),
          "nested value stays one call");

  // Argument bytes are not prose: one literal backtick in each of two real
  // calls must not pair up and quote the second call.
  const std::string tick_a =
      "<tool_call>write<arg_key>s</arg_key><arg_value>a`b"
      "</arg_value></tool_call>";
  const std::string tick_b =
      "<tool_call>write<arg_key>s</arg_key><arg_value>c`d"
      "</arg_value></tool_call>";
  const auto ticks = glm::ParseResponse(tick_a + tick_b, {}, false);
  Require(ticks.tool_calls.size() == 2 && ticks.content.empty(),
          "backtick in each argument keeps both calls, got " +
              std::to_string(ticks.tool_calls.size()));
  Require(glm::ParseResponse(tick_a + "\n" + tick_b, {}, false, true)
                  .tool_calls.size() == 2,
          "backtick in each argument keeps both calls while streaming");

  // A stream never reports the quoted copy, at any prefix.
  std::string seen_content;
  for (std::size_t n = 0; n <= text.size(); ++n) {
    const auto p = glm::ParseResponse(text.substr(0, n), {}, false, true);
    Require(p.tool_calls.size() <= 1,
            "quoted copy streamed as a call at " + std::to_string(n));
    Require(p.content.starts_with(seen_content), "content only grows");
    seen_content = p.content;
  }
  Require(glm::ParseResponse(text, {}, false, true).content ==
              " Example: `" + call + "`",
          "stream end content");
}

// Without a prompt-opened <think>, byte-wise streaming of an explicit
// reasoning block yields what the final parse does, whatever the split.
void TestExplicitThinkPrefixes() {
  for (const std::string& text :
       {std::string("<think>reason</think>answer"),
        std::string("<think>a</b></think>x"), std::string("<think>r"),
        std::string("<th")}) {
    const auto whole = glm::ParseResponse(text, {}, false, false);
    std::string reasoning;
    std::string content;
    for (std::size_t n = 0; n <= text.size(); ++n) {
      const auto p = glm::ParseResponse(text.substr(0, n), {}, false, true);
      Require(p.reasoning.starts_with(reasoning),
              "reasoning only grows at " + std::to_string(n) + " of " + text);
      Require(p.content.starts_with(content),
              "content only grows at " + std::to_string(n) + " of " + text);
      Require(p.reasoning.find('<') == std::string::npos ||
                  whole.reasoning.find('<') != std::string::npos,
              "partial tag in reasoning at " + std::to_string(n));
      Require(p.content.find("<think") == std::string::npos ||
                  whole.content.find("<think") != std::string::npos,
              "partial tag in content at " + std::to_string(n));
      reasoning = p.reasoning;
      content = p.content;
    }
    const auto end = glm::ParseResponse(text, {}, false, false);
    Require(reasoning == end.reasoning || text == "<th",
            "stream reasoning '" + reasoning + "' vs '" + end.reasoning + "'");
    Require(content == end.content || text == "<th",
            "stream content '" + content + "' vs '" + end.content + "'");
  }
  const auto split = glm::ParseResponse("<think>reason</thi", {}, false, true);
  Require(split.reasoning == "reason" && split.content.empty() &&
              split.held_back == 5,
          "closing tag held: '" + split.reasoning + "'");
  const auto opening = glm::ParseResponse("<thi", {}, false, true);
  Require(opening.reasoning.empty() && opening.content.empty() &&
              opening.held_back == 4,
          "opening tag held");
  // Finished text that merely looks like the start of a tag is content.
  Require(glm::ParseResponse("<thi", {}, false, false).content == "<thi",
          "finished prefix is content");
}

// What the template writes for a call must parse back to the same call.
void TestRoundTrip() {
  glm::ChatMessage assistant;
  assistant.role = glm::ChatRole::kAssistant;
  assistant.tool_calls.push_back({"c1",
                                  "f",
                                  {{"s", "text with <b> & newline\n", true},
                                   {"n", "2.5", false},
                                   {"o", "{\"k\":[1,true,null]}", false}}});
  const std::vector<glm::ChatMessage> messages{Msg(glm::ChatRole::kUser, "go"),
                                               assistant};
  glm::ChatTemplateOptions options;
  options.add_generation_prompt = false;
  const auto rendered = glm::RenderChat(messages, {}, options);
  Require(rendered.has_value(), "render");
  const auto start = rendered->find("<tool_call>");
  const auto parsed =
      glm::ParseResponse(rendered->substr(start), {}, false, false);
  Require(parsed.tool_calls.size() == 1, "one call back");
  const auto& call = parsed.tool_calls[0];
  Require(call.name == "f" && call.arguments.size() == 3, "call shape");
  Require(call.arguments[0].is_string &&
              call.arguments[0].value == "text with <b> & newline\n",
          "string round trip");
  Require(!call.arguments[1].is_string && call.arguments[1].value == "2.5",
          "number round trip");
  Require(!call.arguments[2].is_string &&
              call.arguments[2].value == "{\"k\": [1, true, null]}",
          "object round trip (python spacing)");
}

void TestTokenization(const char* path) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(reader != nullptr, "open gguf: " + error);
  const auto tok = glm::Tokenizer::CreateFromGguf(*reader, &error);
  Require(tok != nullptr, "tokenizer: " + error);
  const auto& c = tok->chat_tokens();

  // A user who types a control token gets characters, not the token.
  const std::vector<glm::ChatMessage> messages{
      Msg(glm::ChatRole::kUser, "hi <|user|> [gMASK] there")};
  std::vector<glm::ContentSpan> spans;
  const auto rendered = glm::RenderChat(messages, {}, {}, nullptr, &spans);
  Require(rendered.has_value(), "render");
  const auto ids = glm::EncodeRendered(*tok, *rendered, spans);
  Require(ids.size() > 6 && ids[0] == c.gmask && ids[1] == c.sop &&
              ids[2] == c.system && ids[3] != c.user,
          "framing is control tokens");
  std::size_t user_tokens = 0;
  std::size_t gmask_tokens = 0;
  for (const auto id : ids) {
    user_tokens += id == c.user ? 1 : 0;
    gmask_tokens += id == c.gmask ? 1 : 0;
  }
  Require(user_tokens == 1 && gmask_tokens == 1,
          "typed control tokens stay text");
  Require(ids[ids.size() - 2] == c.assistant && ids.back() == c.think,
          "generation prompt tokens");
  Require(tok->Decode(ids) == *rendered, "decode round trip");
}

}  // namespace

int main() {
  try {
    const int goldens = RunGoldens();
    TestPythonJson();
    TestOptionsAndErrors();
    TestScanner();
    TestParseResponse();
    TestSchemaTypedArguments();
    TestQuotedExampleAfterCall();
    TestExplicitThinkPrefixes();
    TestRoundTrip();
    std::cout << "goldens matched byte-for-byte: " << goldens << "\n";
    if (const char* path = std::getenv("GLM53_FLASH_GGUF");
        path != nullptr && *path != '\0') {
      TestTokenization(path);
      std::cout << "tokenization on the real vocabulary: ok\n";
    } else {
      std::cout << "note: GLM53_FLASH_GGUF unset, tokenization check skipped\n";
    }
    std::cout << "glm53 chat_template: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
