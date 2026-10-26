// Laguna-S-2.1 chat template: the compiled renderer against the bytes that
// Poolside's Jinja template renders (fixtures/chat_goldens.json, regenerate
// with fixtures/generate_chat_goldens.py), plus the renderer's own edge cases.
// Needs no model file, so nothing here is skipped.
#include "src/models/laguna/chat_template.hpp"

#include <iostream>
#include <string>

#include "src/core/crypto/sha256.hpp"
#include "tests/models/laguna/chat_golden.hpp"
#include "tests/models/laguna/support.hpp"

namespace ml = gufo::models::laguna;
namespace json = gufo::json;
using gufo::testing::laguna::ChatInputOf;
using gufo::testing::laguna::LoadChatGoldens;
using gufo::testing::laguna::Require;

namespace {

std::string Sha(const std::string& text) {
  return gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

std::string FirstDifference(const std::string& got, const std::string& want) {
  std::size_t at = 0;
  while (at < got.size() && at < want.size() && got[at] == want[at]) {
    ++at;
  }
  const auto clip = [&](const std::string& text) {
    return text.substr(at > 40 ? at - 40 : 0, 120);
  };
  return "first difference at byte " + std::to_string(at) +
         "\n  expected: " + clip(want) + "\n  got:      " + clip(got);
}

void GoldenCases() {
  const json::Value fixture = LoadChatGoldens();
  Require(fixture.member_str("template_sha256") ==
              std::string(ml::ArtifactTemplateSha256()),
          "the fixture was rendered from a different template than the "
          "pinned one");
  const auto* cases = fixture.find("cases");
  Require(cases != nullptr && cases->items().size() >= 20,
          "the fixture needs at least 20 cases");
  std::size_t failures = 0;
  for (const auto& c : cases->items()) {
    const std::string name = c.member_str("name");
    const auto input = ChatInputOf(c);
    const std::string got =
        ml::RenderChat(input.messages, input.tools, input.options);
    const std::string want = c.member_str("rendered");
    if (got != want) {
      ++failures;
      std::cerr << "MISMATCH " << name << ": " << FirstDifference(got, want)
                << '\n';
    } else if (Sha(got) != c.member_str("rendered_sha256")) {
      ++failures;
      std::cerr << "fixture hash does not match its own text: " << name << '\n';
    }
  }
  Require(failures == 0, std::to_string(failures) + " chat golden(s) differ");
  std::cout << "chat goldens: " << cases->items().size() << " rendered, "
            << "0 differ\n";
}

void EdgeCases() {
  using M = ml::ChatMessage;
  const auto user = [](std::string text) {
    M m;
    m.role = "user";
    m.content = std::move(text);
    return m;
  };
  // No tool: the tools block and its separator are absent.
  {
    const std::vector<M> messages{user("hi")};
    const std::string out = ml::RenderChat(messages, {}, {});
    Require(out.starts_with("〈|EOS|〉<system>You are a helpful"),
            "default system message");
    Require(out.ends_with("<user>hi</user>\n<assistant><think>"),
            "generation prompt with thinking");
    Require(out.find("### Tools") == std::string::npos, "no tools block");
  }
  // A tool without a rebuilt definition is serialized from its fields.
  {
    ml::ChatTool tool;
    tool.name = "f";
    tool.description = "d";
    tool.parameters_json =
        R"({"type":"object","properties":{"a":{"type":"integer"}}})";
    const std::vector<M> messages{user("hi")};
    const std::vector<ml::ChatTool> tools{tool};
    const std::string out = ml::RenderChat(messages, tools, {});
    Require(out.find("{\"type\": \"function\", \"function\": {\"name\": \"f\", "
                     "\"description\": \"d\", \"parameters\": {\"type\": "
                     "\"object\", \"properties\": {\"a\": {\"type\": "
                     "\"integer\"}}}}}\n</available_tools></system>") !=
                std::string::npos,
            "rebuilt tool JSON uses Python's separators");
  }
  // Non-string arguments are re-serialized with Python's spacing, string
  // arguments are written raw, and an unparseable value is kept as written.
  {
    M assistant;
    assistant.role = "assistant";
    ml::ChatMessage::ToolCall call;
    call.name = "t";
    call.arguments = {{"s", "a <b> \"q\"", true},
                      {"o", R"({"x":[1,2],"y":null})", false},
                      {"n", "5", false},
                      {"bad", "not json", false}};
    assistant.tool_calls.push_back(call);
    ml::ChatTemplateOptions options;
    options.add_generation_prompt = false;
    const std::vector<M> messages{user("go"), assistant};
    const std::string out = ml::RenderChat(messages, {}, options);
    Require(out.ends_with(
                "<assistant><think></think><tool_call>t<arg_key>s</arg_key>"
                "<arg_value>a <b> \"q\"</arg_value><arg_key>o</arg_key>"
                "<arg_value>{\"x\": [1, 2], \"y\": null}</arg_value>"
                "<arg_key>n</arg_key><arg_value>5</arg_value>"
                "<arg_key>bad</arg_key><arg_value>not json</arg_value>"
                "</tool_call></assistant>\n"),
            "tool call arguments");
  }
  // Only the first message can be the header; roles the template does not
  // know are skipped.
  {
    M odd;
    odd.role = "developer";
    odd.content = "ignored";
    const std::vector<M> messages{user("a"), odd, user("b")};
    const std::string out = ml::RenderChat(messages, {}, {});
    Require(out.find("ignored") == std::string::npos, "unknown role skipped");
    Require(out.find("<user>a</user>\n<user>b</user>\n") != std::string::npos,
            "user messages kept in order");
  }
  // Options from the provider-neutral reasoning controls.
  {
    gufo::ReasoningOptions reasoning;
    Require(ml::ResolveLagunaChatOptions(reasoning).enable_thinking,
            "thinking defaults on");
    reasoning.enabled = false;
    Require(!ml::ResolveLagunaChatOptions(reasoning).enable_thinking,
            "thinking can be turned off");
    reasoning.enabled = std::nullopt;
    reasoning.effort = gufo::ReasoningEffort::kLow;
    Require(ml::ResolveLagunaChatOptions(reasoning).enable_thinking,
            "an effort keeps thinking on");
    reasoning.preserve_thinking = true;
    Require(ml::ResolveLagunaChatOptions(reasoning).preserve_thinking,
            "preserve_thinking is passed through");
  }
}

}  // namespace

int main() {
  try {
    GoldenCases();
    EdgeCases();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "laguna.chat_template: ok\n";
  return 0;
}
