// Generation loop of GLM-5.3-Flash (generator.hpp) on a scripted
// ForwardSession: the executor seam returns logits that make greedy sampling
// emit a fixed token script. The tokenizer is a synthetic glm4 GGUF (byte
// tokens plus the template's control tokens), so no model is needed.
// Checks:
//   - stop tokens: eos / eot / eom and the role tokens end a turn and are
//     neither emitted nor run through the model; stop_at_eos = false runs
//     through them; max_tokens, a full context and a false callback;
//   - prefix reuse: a prompt extending the held tokens prefills only the
//     suffix, anything else resets the session;
//   - streaming chat: reasoning, content and a tool call arrive as deltas
//     through ResponseStream, the call only once </tool_call> is generated,
//     never as partial tag text, with arguments typed by the tool schema;
//   - load refusal: a build without Lane B's GPU operators refuses a
//     glm5-next GGUF with kLaneBMissingError before reading weights (and the
//     real GGUF from GLM53_FLASH_GGUF, when set).

#include "src/models/glm53_flash/generator.hpp"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/chat_template.hpp"
#include "src/models/glm53_flash/engine.hpp"
#include "src/models/glm53_flash/tokenizer.hpp"

namespace glm = gufo::models::glm53_flash;
using glm::TokenId;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// ---- Synthetic GGUF ----

class GgufBuilder {
public:
  GgufBuilder() {
    AppendBytes("GGUF", 4);
    AppendPod(std::uint32_t{3});
    AppendPod(std::uint64_t{0});  // tensors
    count_pos_ = buffer_.size();
    AppendPod(std::uint64_t{0});  // metadata entries
  }

  void AddString(std::string_view key, std::string_view value) {
    Key(key, gufo::core::GgufValueType::kString);
    AppendString(value);
  }
  void AddUint32(std::string_view key, std::uint32_t value) {
    Key(key, gufo::core::GgufValueType::kUint32);
    AppendPod(value);
  }
  void AddStrings(std::string_view key, const std::vector<std::string>& v) {
    Key(key, gufo::core::GgufValueType::kArray);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendPod(static_cast<std::uint64_t>(v.size()));
    for (const auto& s : v)
      AppendString(s);
  }
  void AddInt32s(std::string_view key, const std::vector<std::int32_t>& v) {
    Key(key, gufo::core::GgufValueType::kArray);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kInt32));
    AppendPod(static_cast<std::uint64_t>(v.size()));
    for (const std::int32_t x : v)
      AppendPod(x);
  }

  std::vector<std::uint8_t> Build() {
    std::memcpy(buffer_.data() + count_pos_, &count_, sizeof(count_));
    if (const std::size_t r = buffer_.size() % 32; r != 0)
      buffer_.resize(buffer_.size() + 32 - r, 0);
    return buffer_;
  }

private:
  void Key(std::string_view key, gufo::core::GgufValueType type) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(type));
    ++count_;
  }
  void AppendBytes(const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + n);
  }
  template<typename T>
  void AppendPod(T value) {
    AppendBytes(&value, sizeof(value));
  }
  void AppendString(std::string_view s) {
    AppendPod(static_cast<std::uint64_t>(s.size()));
    AppendBytes(s.data(), s.size());
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t count_pos_{0};
  std::uint64_t count_{0};
};

/// GPT-2 bytes_to_unicode: the vocabulary spelling of byte `b`.
std::string ByteToken(int b) {
  const auto direct = [](int c) {
    return (c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) ||
           (c >= 0xAE && c <= 0xFF);
  };
  int cp = b;
  if (!direct(b)) {
    int n = 0;
    for (int c = 0; c < b; ++c)
      n += direct(c) ? 0 : 1;
    cp = 256 + n;
  }
  std::string out;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

constexpr std::int32_t kNormal = 1;
constexpr std::int32_t kControl = 3;
constexpr std::int32_t kUserDefined = 4;

/// Byte tokens 0..255 (id == byte), one merge, then the template's tokens.
std::vector<std::uint8_t> TokenizerGguf(std::string_view architecture) {
  std::vector<std::string> tokens;
  std::vector<std::int32_t> types;
  for (int b = 0; b < 256; ++b) {
    tokens.push_back(ByteToken(b));
    types.push_back(kNormal);
  }
  tokens.emplace_back("he");
  types.push_back(kNormal);
  for (const char* control :
       {"<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"}) {
    tokens.emplace_back(control);
    types.push_back(kControl);
  }
  for (const char* user : {"<think>", "</think>", "<tool_call>", "</tool_call>",
                           "<tool_response>", "</tool_response>", "<arg_key>",
                           "</arg_key>", "<arg_value>", "</arg_value>"}) {
    tokens.emplace_back(user);
    types.push_back(kUserDefined);
  }
  const auto id = [&](std::string_view text) {
    for (std::size_t i = 0; i < tokens.size(); ++i)
      if (tokens[i] == text)
        return static_cast<std::uint32_t>(i);
    throw std::runtime_error("no token " + std::string(text));
  };
  GgufBuilder b;
  b.AddString("general.architecture", architecture);
  b.AddString("tokenizer.ggml.model", "gpt2");
  b.AddString("tokenizer.ggml.pre", "glm4");
  b.AddStrings("tokenizer.ggml.tokens", tokens);
  b.AddInt32s("tokenizer.ggml.token_type", types);
  b.AddStrings("tokenizer.ggml.merges", {"h e"});
  b.AddUint32("tokenizer.ggml.eos_token_id", id("<|endoftext|>"));
  b.AddUint32("tokenizer.ggml.padding_token_id", id("<|endoftext|>"));
  b.AddUint32("tokenizer.ggml.bos_token_id", id("[gMASK]"));
  b.AddUint32("tokenizer.ggml.eot_token_id", id("<|user|>"));
  b.AddUint32("tokenizer.ggml.eom_token_id", id("<|observation|>"));
  return b.Build();
}

struct Vocab {
  std::vector<std::uint8_t> gguf;
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::unique_ptr<glm::Tokenizer> tokenizer;

  Vocab() : gguf(TokenizerGguf(glm::kArchitecture)) {
    std::string error;
    reader =
        gufo::core::GgufReader::OpenMemory(gguf.data(), gguf.size(), &error);
    Require(reader != nullptr, "synthetic GGUF opens: " + error);
    tokenizer = glm::Tokenizer::CreateFromGguf(*reader, &error);
    Require(tokenizer != nullptr, "synthetic tokenizer: " + error);
  }
  [[nodiscard]] TokenId Id(std::string_view special) const {
    const TokenId id = tokenizer->FindToken(special);
    Require(id != glm::kInvalidTokenId, "token " + std::string(special));
    return id;
  }
  /// Plain ASCII as byte tokens (id == byte).
  static std::vector<TokenId> Text(std::string_view text) {
    std::vector<TokenId> ids;
    for (const char c : text)
      ids.push_back(static_cast<unsigned char>(c));
    return ids;
  }
  [[nodiscard]] std::uint32_t Size() const {
    return static_cast<std::uint32_t>(tokenizer->VocabSize());
  }
};

/// Concatenates token lists.
std::vector<TokenId> Cat(std::initializer_list<std::vector<TokenId>> parts) {
  std::vector<TokenId> out;
  for (const auto& p : parts)
    out.insert(out.end(), p.begin(), p.end());
  return out;
}

// ---- Scripted executor seam ----

/// Each Forward that asks for logits makes the next script token the argmax
/// (the end of the script repeats `fallback`). Records every call.
class ScriptedSession final : public glm::ForwardSession {
public:
  ScriptedSession(std::uint32_t vocab, std::uint32_t capacity, TokenId fallback)
      : vocab_(vocab), capacity_(capacity), fallback_(fallback) {}

  void SetScript(std::vector<TokenId> script) {
    script_ = std::move(script);
    cursor_ = 0;
  }

  bool Forward(std::span<const TokenId> tokens, std::vector<float>* logits,
               std::string* error) override {
    if (position_ + tokens.size() > capacity_) {
      *error = "scripted session overflow";
      return false;
    }
    position_ += static_cast<std::uint32_t>(tokens.size());
    calls.emplace_back(tokens.begin(), tokens.end());
    fed.insert(fed.end(), tokens.begin(), tokens.end());
    if (logits != nullptr) {
      const TokenId next =
          cursor_ < script_.size() ? script_[cursor_++] : fallback_;
      logits->assign(vocab_, 0.0F);
      (*logits)[next] = 8.0F;
    }
    return true;
  }
  bool Reset(std::string*) override {
    position_ = 0;
    fed.clear();
    ++resets;
    return true;
  }
  std::uint32_t Position() const noexcept override { return position_; }
  std::uint32_t Capacity() const noexcept override { return capacity_; }

  std::vector<std::vector<TokenId>> calls;
  std::vector<TokenId> fed;  ///< Tokens of the sequence since the last reset.
  int resets{0};

private:
  std::uint32_t vocab_;
  std::uint32_t capacity_;
  TokenId fallback_;
  std::vector<TokenId> script_;
  std::size_t cursor_{0};
  std::uint32_t position_{0};
};

// ---- Stop tokens ----

void TestStopTokens(const Vocab& v) {
  const glm::StopTokens stops = glm::ResolveStopTokens(*v.tokenizer);
  for (const char* name :
       {"<|endoftext|>", "<|user|>", "<|observation|>", "<|system|>"})
    Require(stops.Contains(v.Id(name)), std::string("stop token ") + name);
  Require(!stops.Contains(v.Id("<|assistant|>")) &&
              !stops.Contains(v.Id("</tool_call>")),
          "assistant and tag tokens do not stop");
  Require(stops.ids.size() == 4, "eot / eom alias role tokens once");

  const std::vector<TokenId> prompt =
      Cat({{v.Id("[gMASK]"), v.Id("<sop>"), v.Id("<|user|>")},
           Vocab::Text("Hi"),
           {v.Id("<|assistant|>")}});
  for (const char* name :
       {"<|endoftext|>", "<|user|>", "<|observation|>", "<|system|>"}) {
    ScriptedSession session(v.Size(), 64, v.Id("<|endoftext|>"));
    glm::Generator generator(session, stops, v.tokenizer.get());
    session.SetScript(Cat({Vocab::Text("Yo"), {v.Id(name)}, Vocab::Text("x")}));
    glm::GenerationOptions options;
    options.max_tokens = 16;
    std::vector<std::string> pieces;
    glm::GenerationResult r;
    std::string error;
    Require(generator.Generate(
                prompt, options,
                [&](TokenId, std::string_view piece) {
                  pieces.emplace_back(piece);
                  return true;
                },
                &r, &error),
            "generate: " + error);
    const std::string label = std::string("stop on ") + name;
    Require(r.finish_reason == glm::FinishReason::kStop, label + ": finish");
    Require(r.stop_token == v.Id(name), label + ": stop token reported");
    Require(r.text == "Yo" && r.tokens == Vocab::Text("Yo"),
            label + ": output excludes the stop token, got '" + r.text + "'");
    Require(pieces == std::vector<std::string>{"Y", "o"},
            label + ": streamed pieces");
    // Prompt, then "Y" and "o"; the stop token never reaches the model.
    Require(session.fed == Cat({prompt, Vocab::Text("Yo")}),
            label + ": stop token not run through the model");
    Require(r.prompt_tokens == prompt.size() && r.reused_tokens == 0,
            label + ": prompt accounting");
  }

  {  // stop_at_eos = false samples through the stop token.
    ScriptedSession session(v.Size(), 64, v.Id("<|endoftext|>"));
    glm::Generator generator(session, stops, v.tokenizer.get());
    session.SetScript(
        Cat({Vocab::Text("a"), {v.Id("<|user|>")}, Vocab::Text("b")}));
    glm::GenerationOptions options;
    options.max_tokens = 3;
    options.stop_at_eos = false;
    glm::GenerationResult r;
    std::string error;
    Require(generator.Generate(prompt, options, {}, &r, &error), error);
    Require(
        r.finish_reason == glm::FinishReason::kLength &&
            r.tokens ==
                Cat({Vocab::Text("a"), {v.Id("<|user|>")}, Vocab::Text("b")}),
        "stop_at_eos=false runs through stop tokens to max_tokens");
    // The last token of a length-limited turn is not run.
    Require(session.fed.size() == prompt.size() + 2,
            "final token of a length-limited turn is not forwarded");
  }

  {  // Full context.
    ScriptedSession session(v.Size(),
                            static_cast<std::uint32_t>(prompt.size() + 2),
                            v.Id("<|endoftext|>"));
    glm::Generator generator(session, stops, v.tokenizer.get());
    session.SetScript(Vocab::Text("abcdef"));
    glm::GenerationOptions options;
    options.max_tokens = 10;
    glm::GenerationResult r;
    std::string error;
    Require(generator.Generate(prompt, options, {}, &r, &error), error);
    Require(r.finish_reason == glm::FinishReason::kContext && r.text == "abc",
            "full context ends the turn, got '" + r.text + "'");
  }

  {  // Callback cancellation.
    ScriptedSession session(v.Size(), 64, v.Id("<|endoftext|>"));
    glm::Generator generator(session, stops, v.tokenizer.get());
    session.SetScript(Vocab::Text("abcdef"));
    glm::GenerationOptions options;
    options.max_tokens = 10;
    glm::GenerationResult r;
    std::string error;
    int seen = 0;
    Require(
        generator.Generate(
            prompt, options,
            [&](TokenId, std::string_view) { return ++seen < 2; }, &r, &error),
        error);
    Require(r.finish_reason == glm::FinishReason::kCancelled && r.text == "ab",
            "false callback cancels after its token");
  }

  {  // Prompt longer than the context.
    ScriptedSession session(v.Size(), 4, v.Id("<|endoftext|>"));
    glm::Generator generator(session, stops, v.tokenizer.get());
    glm::GenerationResult r;
    std::string error;
    Require(!generator.Generate(prompt, {}, {}, &r, &error) &&
                error.find("exceeds the context") != std::string::npos,
            "oversized prompt refused: " + error);
  }
  std::cout << "ok   stop tokens, length, context, cancellation\n";
}

// ---- Prefix reuse ----

void TestPrefixReuse(const Vocab& v) {
  const glm::StopTokens stops = glm::ResolveStopTokens(*v.tokenizer);
  ScriptedSession session(v.Size(), 128, v.Id("<|endoftext|>"));
  glm::Generator generator(session, stops, v.tokenizer.get());
  glm::GenerationOptions options;
  options.max_tokens = 8;
  std::string error;

  const std::vector<TokenId> first =
      Cat({{v.Id("[gMASK]"), v.Id("<sop>"), v.Id("<|user|>")},
           Vocab::Text("one"),
           {v.Id("<|assistant|>")}});
  session.SetScript(Cat({Vocab::Text("ok"), {v.Id("<|user|>")}}));
  glm::GenerationResult r;
  Require(generator.Generate(first, options, {}, &r, &error), error);
  Require(r.text == "ok" && generator.tokens().size() == first.size() + 2,
          "first turn holds prompt + reply");

  // The next turn extends everything held: only the suffix is prefilled.
  const std::vector<TokenId> second = Cat({first,
                                           Vocab::Text("ok"),
                                           {v.Id("<|user|>")},
                                           Vocab::Text("two"),
                                           {v.Id("<|assistant|>")}});
  const std::size_t calls_before = session.calls.size();
  session.SetScript(Cat({Vocab::Text("k"), {v.Id("<|endoftext|>")}}));
  Require(generator.Generate(second, options, {}, &r, &error), error);
  Require(r.reused_tokens == first.size() + 2, "second turn reuses the prefix");
  Require(session.resets == 1, "extension does not reset");
  Require(session.calls[calls_before] ==
              std::vector<TokenId>(second.begin() + static_cast<std::ptrdiff_t>(
                                                        first.size() + 2),
                                   second.end()),
          "only the new suffix is prefilled");
  Require(session.fed == Cat({second, Vocab::Text("k")}),
          "sequence equals prompt + reply");

  // A diverging prompt starts over.
  const std::vector<TokenId> other =
      Cat({{v.Id("[gMASK]"), v.Id("<sop>"), v.Id("<|user|>")},
           Vocab::Text("new"),
           {v.Id("<|assistant|>")}});
  session.SetScript({v.Id("<|endoftext|>")});
  Require(generator.Generate(other, options, {}, &r, &error), error);
  Require(r.reused_tokens == 0 && session.resets == 2 && session.fed == other,
          "diverging prompt resets and prefills in full");

  // The same prompt again: its logits are still held, nothing runs.
  const std::size_t calls_now = session.calls.size();
  session.SetScript({v.Id("<|endoftext|>")});
  Require(generator.Generate(other, options, {}, &r, &error), error);
  Require(r.reused_tokens == other.size() && session.calls.size() == calls_now,
          "identical prompt with held logits needs no forward");
  std::cout << "ok   prefix reuse\n";
}

// ---- Streaming chat with a tool call ----

const char* kWeatherTool = R"({"type":"function","function":{
  "name":"get_weather","description":"Weather forecast",
  "parameters":{"type":"object","properties":{
    "city":{"type":"string"},"days":{"type":"integer"}},
    "required":["city"]}}})";

std::vector<glm::ChatTool> WeatherTools() {
  glm::ChatTool tool;
  tool.name = "get_weather";
  tool.description = "Weather forecast";
  tool.parameters_json =
      R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]})";
  tool.definition_json = kWeatherTool;
  return {tool};
}

void TestStreamingToolCall(const Vocab& v) {
  const glm::StopTokens stops = glm::ResolveStopTokens(*v.tokenizer);
  ScriptedSession session(v.Size(), 4096, v.Id("<|endoftext|>"));
  glm::Generator generator(session, stops, v.tokenizer.get());
  const auto tools = WeatherTools();
  std::vector<glm::ChatMessage> messages(1);
  messages[0].role = glm::ChatRole::kUser;
  messages[0].content = "Weather in Paris?";

  // The model's turn after "<|assistant|><think>".
  const std::vector<TokenId> call = Cat({{v.Id("<tool_call>")},
                                         Vocab::Text("get_weather"),
                                         {v.Id("<arg_key>")},
                                         Vocab::Text("city"),
                                         {v.Id("</arg_key>")},
                                         {v.Id("<arg_value>")},
                                         Vocab::Text("Paris"),
                                         {v.Id("</arg_value>")},
                                         {v.Id("<arg_key>")},
                                         Vocab::Text("days"),
                                         {v.Id("</arg_key>")},
                                         {v.Id("<arg_value>")},
                                         Vocab::Text("3"),
                                         {v.Id("</arg_value>")},
                                         {v.Id("</tool_call>")}});
  const std::vector<TokenId> script = Cat({Vocab::Text("Check it."),
                                           {v.Id("</think>")},
                                           Vocab::Text("Sure."),
                                           call,
                                           {v.Id("<|observation|>")}});
  session.SetScript(script);

  std::string reasoning;
  std::string content;
  std::vector<glm::ParsedToolCall> calls;
  std::size_t call_delta_at = 0;
  glm::GenerationOptions options;
  options.max_tokens = 256;
  glm::ChatTemplateOptions template_options;  // thinking on
  glm::ChatResult result;
  std::string error;
  const std::size_t call_end = 9 + 1 + 5 + call.size();  // through </tool_call>
  Require(generator.Chat(
              messages, tools, template_options, options,
              [&](const glm::ChatDelta& d) {
                reasoning += d.reasoning;
                content += d.content;
                for (const auto& tc : d.tool_calls) {
                  calls.push_back(tc);
                  call_delta_at = session.fed.size();
                }
                for (const std::string_view text : {d.reasoning, d.content}) {
                  Require(
                      text.find('<') == std::string_view::npos,
                      "no tag text in a delta: '" + std::string(text) + "'");
                }
                return true;
              },
              &result, &error),
          "chat: " + error);
  Require(result.generation.finish_reason == glm::FinishReason::kStop &&
              result.generation.stop_token == v.Id("<|observation|>"),
          "the call turn stops on <|observation|>");
  Require(reasoning == "Check it.",
          "streamed reasoning, got '" + reasoning + "'");
  Require(content == "Sure.", "streamed content, got '" + content + "'");
  Require(calls.size() == 1, "one streamed call");
  // A token's delta fires before the token is run, so the model then holds
  // the prompt and every earlier token: the call's delta must come with
  // </tool_call> (script index call_end - 1), not before.
  const std::size_t prompt_size = result.generation.prompt_tokens;
  Require(call_delta_at == prompt_size + call_end - 1,
          "call delivered only once </tool_call> is generated (at " +
              std::to_string(call_delta_at) + ", expected " +
              std::to_string(prompt_size + call_end - 1) + ")");
  const auto& tc = calls[0];
  Require(tc.name == "get_weather" && tc.arguments.size() == 2,
          "call name and arity");
  Require(tc.arguments[0].name == "city" && tc.arguments[0].value == "Paris" &&
              tc.arguments[0].is_string,
          "string argument");
  Require(tc.arguments[1].name == "days" && tc.arguments[1].value == "3" &&
              !tc.arguments[1].is_string,
          "integer argument typed by the schema");
  Require(result.response.reasoning == "Check it." &&
              result.response.content == "Sure." &&
              result.response.tool_calls.size() == 1 &&
              result.response.reasoning_closed,
          "final ParseResponse agrees with the stream");

  // The prompt is the template's: it ends with the thinking generation prompt.
  const auto prompt =
      glm::PrepareChat(*v.tokenizer, messages, tools, template_options, &error);
  Require(
      prompt && prompt->opened_think &&
          prompt->tokens.back() == v.Id("<think>") &&
          prompt->tokens[prompt->tokens.size() - 2] == v.Id("<|assistant|>"),
      "chat prompt ends in <|assistant|><think>");
  std::cout << "ok   streaming tool call through the generator\n";
}

void TestUnfinishedCallAtLength(const Vocab& v) {
  const glm::StopTokens stops = glm::ResolveStopTokens(*v.tokenizer);
  ScriptedSession session(v.Size(), 4096, v.Id("<|endoftext|>"));
  glm::Generator generator(session, stops, v.tokenizer.get());
  std::vector<glm::ChatMessage> messages(1);
  messages[0].content = "Weather?";
  const std::vector<TokenId> script = Cat({Vocab::Text("ok"),
                                           {v.Id("<tool_call>")},
                                           Vocab::Text("get_weather"),
                                           {v.Id("<arg_key>")}});
  session.SetScript(script);
  glm::ChatTemplateOptions template_options;
  template_options.enable_thinking = false;
  glm::GenerationOptions options;
  options.max_tokens = script.size();
  std::string content;
  std::size_t deltas_with_calls = 0;
  glm::ChatResult result;
  std::string error;
  Require(generator.Chat(
              messages, WeatherTools(), template_options, options,
              [&](const glm::ChatDelta& d) {
                Require(d.reasoning.empty(), "no reasoning without thinking");
                content += d.content;
                deltas_with_calls += d.tool_calls.empty() ? 0 : 1;
                return true;
              },
              &result, &error),
          error);
  Require(result.generation.finish_reason == glm::FinishReason::kLength,
          "length finish");
  // Withheld while streaming, released as text when the turn ends unfinished.
  Require(content == "ok<tool_call>get_weather<arg_key>",
          "unfinished call is content at the end, got '" + content + "'");
  Require(deltas_with_calls == 0 && result.response.tool_calls.empty(),
          "an unfinished call is not a call");
  std::cout << "ok   unfinished call at max_tokens\n";
}

// ---- Load refusal ----

std::filesystem::path WriteTemp(const std::vector<std::uint8_t>& bytes,
                                const char* name) {
  const auto path =
      std::filesystem::temp_directory_path() /
      (std::string(name) + "-" +
       std::to_string(static_cast<long long>(::getpid())) + ".gguf");
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  Require(out.good(), "write " + path.string());
  return path;
}

void TestLoadRefusal() {
  std::string why;
  if (glm::RuntimeAvailable(&why)) {
    std::cout << "skip load refusal: this build has the GLM-5.3 runtime\n";
    return;
  }
  Require(!why.empty(), "a refusal says why");
  const bool lane_b_missing = why == glm::kLaneBMissingError;
#if defined(ENGINE_ENABLE_HIP)
  Require(lane_b_missing, "a ROCm build refuses only for Lane B: " + why);
#endif
  // A glm5-next GGUF with a vocabulary and no tensors: the refusal comes
  // before config, tokenizer or weights are read.
  const auto glm_path =
      WriteTemp(TokenizerGguf(glm::kArchitecture), "glm53-refusal");
  std::string error;
  auto model = glm::Model::Load(glm_path.string(), {}, &error);
  std::filesystem::remove(glm_path);
  Require(model == nullptr, "load refused");
  Require(error == why, "load reports the runtime refusal: " + error);
  if (lane_b_missing)
    Require(error ==
                "GLM-5.3-Flash: GPU MLA/MoE operators not built yet "
                "(Lane B)",
            "Lane B message text");

  // Another architecture is not mistaken for GLM-5.3.
  const auto other_path = WriteTemp(TokenizerGguf("qwen4exp"), "glm53-other");
  model = glm::Model::Load(other_path.string(), {}, &error);
  std::filesystem::remove(other_path);
  Require(model == nullptr &&
              error.find("not a GLM-5.3-Flash GGUF") != std::string::npos &&
              error.find("qwen4exp") != std::string::npos,
          "wrong architecture: " + error);

  model = glm::Model::Load("/nonexistent/glm53.gguf", {}, &error);
  Require(model == nullptr && error.starts_with("failed to open GGUF"),
          "missing file: " + error);

  if (const char* real = std::getenv("GLM53_FLASH_GGUF");
      real != nullptr && *real != '\0') {
    model = glm::Model::Load(real, {}, &error);
    Require(model == nullptr && error == why,
            "real GGUF refused with the runtime reason: " + error);
    std::cout << "ok   real GGUF refused: " << error << "\n";
  } else {
    std::cout << "skip real GGUF refusal (GLM53_FLASH_GGUF unset)\n";
  }
  std::cout << "ok   load refusal: " << why << "\n";
}

}  // namespace

int main() {
  try {
    const Vocab v;
    TestStopTokens(v);
    TestPrefixReuse(v);
    TestStreamingToolCall(v);
    TestUnfinishedCallAtLength(v);
    TestLoadRefusal();
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << "\n";
    return 1;
  }
  std::cout << "glm53_flash generator tests passed\n";
  return 0;
}
