// Laguna through the serve path on a tiny random model (no model file,
// nothing skipped): InferenceBackend -> scheduler -> TextRunnerPool ->
// LagunaTextRunner -> Session -> GPU executor, the same objects `gufo serve`
// builds for the real model.
//
//  * registration: sampling preset, model id, context, and the unsupported
//    combinations refused with a message; `load(path)` reaches the Laguna
//    loader for general.architecture == "laguna".
//  * chat and raw completion equal the engine's own greedy decode (chat
//    template, BOS policy, chunked prefill, stop tokens, streaming pieces).
//  * snapshots: an identical prompt is a complete hit with no prefill, and a
//    conversation that extends it restores the stored prompt and continues
//    bit for bit like a cold run.
//  * cancellation (queued, mid-decode) and retry, untapped; the retry equals
//    the cold run and every session returns to idle.
//  * several requests admitted together on three sessions equal their
//    sequential results.
//  * the context limit ends a request exactly at capacity.
//  * a device error in a prefill pass, a decode pass, or a request restored
//    from a snapshot fails that request only; the next requests are correct.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include "src/cli/serve/inference_backend.hpp"
#include "src/core/text_sampling_defaults.hpp"
#include "tests/models/laguna/engine_fixture.hpp"

using namespace gufo::testing::laguna;
namespace ml = gufo::models::laguna;
namespace srv = gufo::server;
namespace smp = gufo::sampling;

namespace {

using Request = std::shared_ptr<srv::TextGenerationBackend::GenerationRequest>;
using Result = srv::TextGenerationBackend::Result;
using Tokens = std::vector<gufo::tokenization::TokenId>;

smp::SamplingConfig MakeGreedy() {
  smp::SamplingConfig config;
  config.temperature = 0.0F;
  return config;
}
const smp::SamplingConfig kGreedy = MakeGreedy();

srv::ChatRequest ChatOf(const std::string& text) {
  return srv::ChatRequest(
      {{gufo::tokenization::ChatRole::kUser, text, "", ""}});
}

/// A user prompt whose rendered length is a multiple of the prefill chunk, so
/// that a cold run and a run restored from a snapshot of it share one grid of
/// chunks (rows rounded identically).
srv::ChatRequest AlignedChat(srv::InferenceBackend& backend, std::string text,
                             std::size_t chunk = 16) {
  for (int i = 0; i < 64; ++i) {
    auto request = ChatOf(text);
    const auto n = backend.count_chat_tokens(request);
    Require(n.has_value(), "count_chat_tokens");
    if (*n % chunk == 0) {
      return request;
    }
    text += "x";  // one byte token, so every length is reachable
  }
  Require(false, "could not align the prompt to the chunk");
  return ChatOf(text);
}

srv::ChatRequest Continue(srv::ChatRequest request, const std::string& reply,
                          const std::string& next) {
  request.messages.emplace_back(gufo::tokenization::ChatRole::kAssistant, reply,
                                "", "");
  request.messages.emplace_back(gufo::tokenization::ChatRole::kUser, next, "",
                                "");
  return request;
}

/// The engine's own greedy decode of `prompt`, the way the runner does it.
Tokens Direct(const ml::Model& model, const std::vector<std::int32_t>& prompt,
              std::size_t max_tokens) {
  std::string error;
  auto session = model.CreateSession(&error);
  Require(session != nullptr, error);
  Require(session->Prefill(prompt, &error), error);
  const std::vector<smp::TokenId> history(prompt.begin(), prompt.end());
  smp::SamplerState sampler(kGreedy, history);
  Tokens out;
  while (out.size() < max_tokens) {
    const auto token = sampler.Sample(session->Logits());
    if (model.IsStopToken(static_cast<std::int32_t>(token))) {
      break;
    }
    sampler.Accept(token);
    out.push_back(token);
    if (out.size() < max_tokens) {
      Require(session->Evaluate(static_cast<std::int32_t>(token), &error),
              error);
    }
  }
  return out;
}

std::vector<std::int32_t> PromptOf(const ml::Model& model,
                                   const std::string& text,
                                   bool thinking = true) {
  std::vector<ml::ChatMessage> messages(1);
  messages[0].role = "user";
  messages[0].content = text;
  ml::ChatTemplateOptions options;
  options.enable_thinking = thinking;
  return model.EncodeChat(messages, {}, options);
}

void AllIdle(srv::InferenceBackend& backend, const char* when) {
  for (const auto& state : backend.session_states()) {
    Require(!state.processing, std::string("a session is still busy ") + when);
  }
}

// ---- registration --------------------------------------------------------

void Registration(const EngineFixture& f) {
  std::string error;
  srv::InferenceBackend backend;
  Require(!backend.ready(), "not ready before load");
  Require(backend.load(f.model, &error, 256, 2), "load: " + error);
  Require(backend.ready() && backend.model_id() == "Laguna-tiny" &&
              backend.max_context() == 256 &&
              backend.session_states().size() == 2,
          "model id, context and sessions");
  const auto defaults = backend.sampling_defaults();
  Require(defaults.model == smp::TextModelPreset::kLaguna,
          "the Laguna sampling preset is registered");
  for (const bool thinking : {true, false}) {
    const auto config = defaults.Resolve(thinking);
    Require(config.temperature == 1.0F && config.top_k == 20 &&
                config.top_p == 1.0F && config.min_p == 0.0F &&
                config.presence_penalty == 0.0F,
            "GGUF defaults: temperature 1.0, top_k 20, top_p 1.0, min_p 0");
  }
  gufo::core::ModelConfig config;
  config.architecture = "laguna";
  Require(smp::TextPreset(config) == smp::TextModelPreset::kLaguna,
          "TextPreset maps architecture laguna");
  // An explicit request value still wins over the preset.
  smp::SamplingConfig supplied;
  supplied.top_p = 0.5F;
  smp::SamplingOverrides only_top_p;
  only_top_p.top_p = true;
  Require(smp::ResolveTextSampling(smp::TextModelPreset::kLaguna, true,
                                   supplied, only_top_p)
                  .top_p == 0.5F,
          "request values override the Laguna preset");

  // Unsupported combinations are refused with a message, not ignored.
  srv::InferenceBackend refuse;
  srv::TextSpeculativeConfig speculative;
  speculative.backend = srv::TextSpeculativeBackend::kDFlash;
  Require(!refuse.load(f.model, &error, 256, 1, {}, {}, speculative) &&
              error.find("speculative") != std::string::npos,
          "speculative decoding is refused");
  srv::TextDiskCacheConfig disk;
  disk.directory = "/nonexistent-laguna-cache";
  Require(!refuse.load(f.model, &error, 256, 1, {}, {}, {}, disk) &&
              error.find("persistent") != std::string::npos,
          "a disk cache is refused");
  Require(!refuse.load(f.model, &error, 256, 0), "zero sessions are refused");
  Require(!refuse.load(f.model, &error, 100000, 1) &&
              error.find("context") != std::string::npos,
          "a context above the model's is refused");
  Require(!refuse.ready(), "refused loads leave the backend unloaded");

  // `load(path)` dispatches on general.architecture == "laguna": a tensorless
  // artifact reaches the Laguna loader, which names itself in the error.
  {
    const auto path =
        std::filesystem::temp_directory_path() /
        ("laguna-registration-" + std::to_string(::getpid()) + ".gguf");
    Metadata m;
    m["general.architecture"] = std::string{"laguna"};
    m["laguna.context_length"] = std::uint32_t{262144};
    std::vector<std::uint8_t> bytes;
    (void)OpenSynthetic(m, &bytes);
    {
      std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    }
    srv::InferenceBackend by_path;
    const bool loaded = by_path.load(path.string(), &error, 0, 1);
    Require(!loaded && error.find("Laguna") != std::string::npos,
            "the Laguna loader handles architecture laguna: " + error);
    Require(!by_path.load(path.string(), &error, 0, 1, {}, {}, {}, {},
                          "mmproj.gguf") &&
                error.find("--mmproj") != std::string::npos,
            "--mmproj is refused for Laguna: " + error);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
  std::cout << "registration ok\n";
}

// ---- chat, completion, streaming ----------------------------------------

void ChatAndCompletion(const EngineFixture& f, srv::InferenceBackend& backend) {
  constexpr std::size_t kMax = 24;
  const std::string text = "the then there is the end";
  const auto prompt = PromptOf(*f.model, text);
  const Tokens expected = Direct(*f.model, prompt, kMax);
  Require(expected.size() > 4, "the reference generated something");

  std::vector<std::string> pieces;
  const Result result = backend.chat(ChatOf(text), kMax, kGreedy, {},
                                     [&](std::string_view piece) {
                                       pieces.emplace_back(piece);
                                       return true;
                                     });
  Require(result.prompt_tokens == prompt.size(),
          "the backend renders the same prompt as EncodeChat: " +
              std::to_string(result.prompt_tokens) + " vs " +
              std::to_string(prompt.size()));
  Require(result.tokens == expected, "chat equals the engine's greedy decode");
  Require(result.finish_reason ==
              (expected.size() == kMax
                   ? srv::TextGenerationBackend::FinishReason::kLength
                   : srv::TextGenerationBackend::FinishReason::kStop),
          "finish reason");
  std::string joined;
  for (const auto& piece : pieces) {
    joined += piece;
  }
  std::vector<std::int32_t> ids(result.tokens.begin(), result.tokens.end());
  Require(joined == result.text && result.text == f.model->Decode(ids) &&
              pieces.size() == result.tokens.size(),
          "streamed pieces concatenate to the decoded text");
  Require(result.reasoning_tokens == 0 ||
              result.reasoning_tokens <= result.tokens.size(),
          "reasoning accounting is bounded");

  // Thinking off changes the prompt (and the output state), nothing else.
  auto off = ChatOf(text);
  off.reasoning.enabled = false;
  Require(backend.initial_output_state(off) ==
                  srv::TextGenerationBackend::InitialOutputState::kContent &&
              backend.initial_output_state(ChatOf(text)) ==
                  srv::TextGenerationBackend::InitialOutputState::kReasoning,
          "thinking selects the initial output state");
  const Result no_think = backend.chat(off, kMax, kGreedy);
  Require(no_think.prompt_tokens == PromptOf(*f.model, text, false).size() &&
              no_think.tokens ==
                  Direct(*f.model, PromptOf(*f.model, text, false), kMax),
          "thinking off renders and generates like the engine");

  // Raw completion: BOS is added, as llama.cpp does for this GGUF.
  const Result raw = backend.complete("the then there", 8, kGreedy);
  const auto raw_prompt = f.model->Tokenize("the then there", true);
  Require(raw.prompt_tokens == raw_prompt.size() &&
              backend.count_tokens("the then there") == raw_prompt.size(),
          "raw completion tokenizes with BOS");
  Require(raw.tokens == Direct(*f.model, raw_prompt, 8),
          "completion equals the engine's greedy decode");

  // A seeded sample replays, cold and warm, under the Laguna preset.
  auto sampled = backend.sampling_defaults().Resolve(true);
  sampled.seed = 17;
  const Result s1 = backend.chat(ChatOf(text), 16, sampled);
  const Result s2 = backend.chat(ChatOf(text), 16, sampled);
  auto cold = ChatOf(text);
  cold.cache_prompt = false;
  const Result s3 = backend.chat(cold, 16, sampled);
  Require(s1.tokens == s2.tokens && s1.tokens == s3.tokens,
          "a seeded sample replays across cold and warm runs");
  sampled.seed = 18;
  Require(backend.chat(ChatOf(text), 16, sampled).tokens != s1.tokens,
          "a different seed samples differently");
  AllIdle(backend, "after chat");
  std::cout << "chat and completion ok\n";
}

// ---- snapshots and prefix reuse -----------------------------------------

void Snapshots(const EngineFixture& f, srv::InferenceBackend& backend) {
  constexpr std::size_t kMax = 12;
  const auto first = AlignedChat(backend, "Shared prefix for the cache test:");
  const Result cold = backend.chat(first, kMax, kGreedy);
  Require(!cold.cache_hit && cold.prefill_tokens == cold.prompt_tokens,
          "the first request is cold");
  Require(cold.cache_snapshot_bytes > 0, "a snapshot was retained");
  const Result again = backend.chat(first, kMax, kGreedy);
  Require(again.cache_hit && again.prefill_tokens == 0 &&
              again.cached_prompt_tokens > 0 && again.tokens == cold.tokens,
          "an identical prompt is a complete hit with no prefill");

  // The conversation continues: the stored prompt is a prefix of the next
  // one. The restored run must equal a run that never saw the cache.
  const auto second = Continue(first, "ok", "And now a follow-up question.");
  const Result warm = backend.chat(second, kMax, kGreedy);
  auto fresh = second;
  fresh.cache_prompt = false;
  const Result reference = backend.chat(fresh, kMax, kGreedy);
  Require(warm.cache_hit && warm.cached_prompt_tokens >= cold.prompt_tokens &&
              warm.prefill_tokens < warm.prompt_tokens,
          "the follow-up restores the stored prompt (cached " +
              std::to_string(warm.cached_prompt_tokens) + " of " +
              std::to_string(warm.prompt_tokens) + ")");
  Require(warm.tokens == reference.tokens && warm.text == reference.text,
          "a continuation from a snapshot equals a cold run");
  Require(warm.tokens == Direct(*f.model,
                                f.model->EncodeChat(
                                    [&] {
                                      std::vector<ml::ChatMessage> v(3);
                                      v[0].role = "user";
                                      v[0].content = first.messages[0].content;
                                      v[1].role = "assistant";
                                      v[1].content = "ok";
                                      v[2].role = "user";
                                      v[2].content =
                                          "And now a follow-up "
                                          "question.";
                                      return v;
                                    }(),
                                    {}, {}),
                                kMax),
          "the follow-up equals the engine's own decode");
  // Rewriting the earlier turn diverges inside the cached prefix: still
  // correct.
  auto rewritten = Continue(AlignedChat(backend,
                                        "Shared prefix for the cache "
                                        "test, rewritten:"),
                            "ok", "Again?");
  const Result rewrite = backend.chat(rewritten, kMax, kGreedy);
  rewritten.cache_prompt = false;
  Require(rewrite.tokens == backend.chat(rewritten, kMax, kGreedy).tokens,
          "an edited history equals its cold run");
  AllIdle(backend, "after snapshot requests");
  std::cout << "snapshots and prefix reuse ok\n";
}

// ---- cancellation --------------------------------------------------------

void Cancellation(const EngineFixture&, srv::InferenceBackend& backend) {
  constexpr std::size_t kMax = 40;
  const auto request = AlignedChat(
      backend,
      "A prompt long enough to take several prefill chunks. "
      "More words follow so that the prefill is not a single pass.");
  auto cold = request;
  cold.cache_prompt = false;
  const Result baseline = backend.chat(cold, kMax, kGreedy);
  Require(baseline.tokens.size() > 8, "baseline has tokens");

  for (int variant = 0; variant < 3; ++variant) {
    auto cancelled_request = variant == 2 ? cold : request;
    Request pending = backend.start_chat(cancelled_request, kMax, kGreedy, {},
                                         /*stream_output=*/true);
    std::size_t seen = 0;
    if (variant == 0) {
      pending->Cancel();  // while queued or in prefill
    }
    const Result result = pending->Wait([&](std::string_view) {
      if (++seen == 5 && variant != 0) {
        pending->Cancel();  // mid-decode
      }
      return true;
    });
    Require(result.cancelled &&
                result.finish_reason ==
                    srv::TextGenerationBackend::FinishReason::kCancelled &&
                result.tokens.size() < kMax,
            "variant " + std::to_string(variant) + " is cancelled");
    // Retry at once, with and without the cache.
    const Result retry = backend.chat(request, kMax, kGreedy);
    auto retry_cold = request;
    retry_cold.cache_prompt = false;
    Require(
        retry.tokens == baseline.tokens &&
            backend.chat(retry_cold, kMax, kGreedy).tokens == baseline.tokens,
        "the retry after cancellation variant " + std::to_string(variant) +
            " equals the cold run");
    AllIdle(backend, "after a cancelled request");
  }
  // Cancelled requests that overlap others.
  std::vector<Request> pending;
  for (int i = 0; i < 3; ++i) {
    auto r = request;
    r.client_id = "cancel-" + std::to_string(i);
    r.cache_prompt = false;
    pending.push_back(backend.start_chat(r, kMax, kGreedy, {}, true));
  }
  pending[1]->Cancel();
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const Result result = pending[i]->Wait();
    if (i == 1) {
      Require(result.cancelled || result.tokens == baseline.tokens,
              "the cancelled peer ends");
    } else {
      Require(result.tokens == baseline.tokens && !result.cancelled,
              "an uncancelled peer is unaffected by the cancellation");
    }
  }
  AllIdle(backend, "after overlapping cancellations");
  std::cout << "cancellation and retry ok\n";
}

// ---- several sessions ----------------------------------------------------

void MultiSession(const EngineFixture&, srv::InferenceBackend& backend) {
  constexpr std::size_t kMax = 20;
  std::vector<srv::ChatRequest> requests;
  std::vector<Tokens> expected;
  for (int i = 0; i < 6; ++i) {
    auto r = ChatOf("Request number " + std::to_string(i) +
                    " asks about topic " + std::string(1 + 7 * i, 'x'));
    r.cache_prompt = false;
    r.client_id = "multi-" + std::to_string(i);
    expected.push_back(backend.chat(r, kMax, kGreedy).tokens);
    requests.push_back(std::move(r));
  }
  for (int round = 0; round < 2; ++round) {
    std::vector<Request> pending;
    for (const auto& r : requests) {
      pending.push_back(backend.start_chat(r, kMax, kGreedy, {}, true));
    }
    for (std::size_t i = 0; i < pending.size(); ++i) {
      Require(pending[i]->Wait().tokens == expected[i],
              "request " + std::to_string(i) +
                  " admitted with its peers "
                  "equals its solo result");
    }
  }
  AllIdle(backend, "after concurrent requests");
  std::cout << "multi-session ok\n";
}

// ---- context limit -------------------------------------------------------

void ContextLimit(const EngineFixture& f) {
  std::string error;
  srv::InferenceBackend backend;
  Require(backend.load(f.model, &error, 256, 2), error);
  const Result result = backend.chat(ChatOf("fill the context"), 500, kGreedy);
  Require(result.finish_reason ==
                  srv::TextGenerationBackend::FinishReason::kLength ||
              result.finish_reason ==
                  srv::TextGenerationBackend::FinishReason::kStop,
          "ends normally");
  if (result.finish_reason ==
      srv::TextGenerationBackend::FinishReason::kLength) {
    Require(result.prompt_tokens + result.completion_tokens == 256,
            "a request fills the context exactly and stops: " +
                std::to_string(result.prompt_tokens) + " + " +
                std::to_string(result.completion_tokens));
  }
  // Next to it, a request that does not fit is refused, not truncated.
  auto huge = ChatOf(std::string(300, 'y'));
  bool refused = false;
  try {
    (void)backend.chat(huge, 8, kGreedy);
  } catch (const std::exception&) {
    refused = true;
  }
  Require(refused, "a prompt beyond the context is refused");
  const Result after = backend.chat(ChatOf("fill the context"), 500, kGreedy);
  Require(after.tokens == result.tokens, "serving continues after the refusal");
  std::cout << "context limit ok\n";
}

// ---- failure injection ---------------------------------------------------

void Failures(const EngineFixture& f, srv::InferenceBackend& backend) {
  constexpr std::size_t kMax = 16;
  auto request = AlignedChat(backend,
                             "A prompt for the failure checks, with "
                             "enough words to need three chunks of "
                             "prefill.");
  auto cold = request;
  cold.cache_prompt = false;
  const Result baseline = backend.chat(cold, kMax, kGreedy);
  const auto passes = (baseline.prompt_tokens + 15) / 16;

  const auto fails = [&](const srv::ChatRequest& r, std::uint32_t skip,
                         const char* stage, const char* what,
                         bool expect_pieces) {
    f.model->InjectForwardFailureForTesting(skip, stage);
    std::size_t pieces = 0;
    bool failed = false;
    try {
      (void)backend.chat(r, kMax, kGreedy, {}, [&](std::string_view) {
        ++pieces;
        return true;
      });
    } catch (const std::exception& e) {
      failed = true;
    }
    Require(failed,
            std::string("the injected failure fails the request: ") + what);
    Require((pieces > 0) == expect_pieces,
            std::string("streamed output before the failure: ") + what);
    // Nothing continues on the poisoned state: every later request is right.
    AllIdle(backend, what);
    Require(backend.chat(cold, kMax, kGreedy).tokens == baseline.tokens,
            std::string("a cold request after the failure: ") + what);
    Require(backend.chat(request, kMax, kGreedy).tokens == baseline.tokens,
            std::string("a cached request after the failure: ") + what);
  };
  fails(cold, 0, "attn", "first prefill pass", false);
  fails(cold, static_cast<std::uint32_t>(passes - 1), "attn",
        "last prefill pass", false);
  fails(cold, static_cast<std::uint32_t>(passes + 3), "o", "a decode pass",
        true);
  // A request restored from a snapshot, failing in its first pass.
  (void)backend.chat(request, kMax, kGreedy);  // retain a snapshot
  fails(Continue(request, "ok", "Follow up?"), 0, "attn",
        "a request restored from a snapshot", false);
  std::cout << "failure injection ok\n";
}

// ---- tools in the template ------------------------------------------------

void ToolPrompts(const EngineFixture& f, srv::InferenceBackend& backend) {
  using gufo::tokenization::ChatRole;
  srv::ChatRequest request;
  request.messages.emplace_back(ChatRole::kUser, "Weather in Oslo?", "", "");
  gufo::tokenization::ChatMessage call(ChatRole::kAssistant, "", "", "");
  call.tool_calls.push_back(
      {"call_1",
       "get_weather",
       {{"location", "Oslo", true}, {"days", "3", false}}});
  request.messages.push_back(call);
  request.messages.emplace_back(ChatRole::kTool, "{\"temp_c\": 3}", "", "");
  request.tool_choice = srv::ChatRequest::ToolChoice::kNone;
  request.tools.push_back(
      {"get_weather", "Get the weather.",
       R"({"type":"object","properties":{"location":{"type":"string"}}})",
       R"({"type":"function","function":{"name":"get_weather","description":"Get the weather.","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}})"});
  // tool_choice none renders no tools; the history still renders natively.
  std::vector<ml::ChatMessage> messages(3);
  messages[0].role = "user";
  messages[0].content = "Weather in Oslo?";
  messages[1].role = "assistant";
  messages[1].tool_calls.push_back(
      {"get_weather", {{"location", "Oslo", true}, {"days", "3", false}}, ""});
  messages[2].role = "tool";
  messages[2].content = "{\"temp_c\": 3}";
  const auto expected = f.model->EncodeChat(messages, {}, {});
  const auto counted = backend.count_chat_tokens(request);
  Require(counted && *counted == expected.size(),
          "history with a tool call renders through the runner exactly like "
          "the template");
  request.tool_choice = srv::ChatRequest::ToolChoice::kAuto;
  const auto with_tools = backend.count_chat_tokens(request);
  Require(with_tools && *with_tools > *counted,
          "advertised tools add the tools block to the prompt");
  std::cout << "tool prompts ok\n";
}

}  // namespace

int main() {
  try {
    const EngineFixture fixture = MakeEngineFixture();
    Registration(fixture);
    srv::InferenceBackend backend;
    std::string error;
    Require(backend.load(fixture.model, &error, 512, 3), "load: " + error);
    ChatAndCompletion(fixture, backend);
    Snapshots(fixture, backend);
    Cancellation(fixture, backend);
    MultiSession(fixture, backend);
    Failures(fixture, backend);
    ToolPrompts(fixture, backend);
    ContextLimit(fixture);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "laguna.serving: ok\n";
  return 0;
}
