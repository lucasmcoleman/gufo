#include "src/cli/serve/glm53_text_runner.hpp"

#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/glm53_flash/chat_template.hpp"
#include "src/models/glm53_flash/engine.hpp"
#include "src/models/glm53_flash/generator.hpp"

namespace gufo::server {

namespace {

namespace glm = models::glm53_flash;

constexpr std::string_view kGlm53StateAbi = "glm53-flash-executor-v1";

glm::ChatRole Glm53Role(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
    case tokenization::ChatRole::kDeveloper:
      return glm::ChatRole::kSystem;  // The template has no developer role.
    case tokenization::ChatRole::kAssistant:
      return glm::ChatRole::kAssistant;
    case tokenization::ChatRole::kTool:
      return glm::ChatRole::kTool;
    case tokenization::ChatRole::kUser:
      return glm::ChatRole::kUser;
  }
  return glm::ChatRole::kUser;
}

class Glm53TextRunnerState final : public TextRunnerState {
public:
  Glm53TextRunnerState(std::unique_ptr<glm::ForwardSession> session,
                       const glm::Model& model)
      : session_(std::move(session)),
        generator_(*session_, model.stop_tokens(), &model.tokenizer()) {}

  void Invalidate() noexcept override { generator_.Invalidate(); }

  [[nodiscard]] glm::Generator& generator() noexcept { return generator_; }
  [[nodiscard]] const glm::Generator& generator() const noexcept {
    return generator_;
  }

private:
  std::unique_ptr<glm::ForwardSession> session_;
  glm::Generator generator_;
};

Glm53TextRunnerState& RequireGlm53State(TextRunnerState& state) {
  auto* glm_state = dynamic_cast<Glm53TextRunnerState*>(&state);
  if (glm_state == nullptr)
    throw std::logic_error("text runner state is not GLM-5.3-Flash");
  return *glm_state;
}

const Glm53TextRunnerState& RequireGlm53State(const TextRunnerState& state) {
  const auto* glm_state = dynamic_cast<const Glm53TextRunnerState*>(&state);
  if (glm_state == nullptr)
    throw std::logic_error("text runner state is not GLM-5.3-Flash");
  return *glm_state;
}

class Glm53TextRunner final : public TextModelRunner {
public:
  Glm53TextRunner(std::shared_ptr<glm::Model> model, std::uint32_t max_context,
                  std::uint32_t prefill_chunk)
      : model_(std::move(model)),
        max_context_(max_context),
        prefill_chunk_(std::max<std::uint32_t>(prefill_chunk, 1)) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kGlm53StateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = model_->Mtp(),
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0,
                .prefix_reuse = true,
                .in_pass_checkpoint = false,
                .partial_prefill_retention = true,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = model_->DeviceBytes(),
        .state_capacity_bytes = std::nullopt,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = std::nullopt,
        .retained_snapshot_ceiling_bytes = std::nullopt,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              model_->tokenizer().DecodeToken(id),
              model_->stop_tokens().Contains(id)};
        });
  }

  [[nodiscard]] sampling::JsonConstraint::ToolFormat ToolFormat()
      const override {
    return sampling::JsonConstraint::ToolFormat::kGlm;
  }

  /// Raw completions may spell the template's control tokens.
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text, /*parse_special=*/true);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    std::vector<glm::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      glm::ChatMessage converted;
      converted.role = Glm53Role(message.role);
      converted.content = message.content + message.framing_suffix;
      if (!message.thought.empty())
        converted.reasoning_content = message.thought;
      converted.tool_call_id = message.tool_call_id;
      for (const auto& call : message.tool_calls) {
        glm::ChatMessage::ToolCall converted_call;
        converted_call.id = call.id;
        converted_call.name = call.name;
        for (const auto& argument : call.arguments)
          converted_call.arguments.push_back({.name = argument.name,
                                              .value = argument.value,
                                              .is_string = argument.is_string});
        converted.tool_calls.push_back(std::move(converted_call));
      }
      messages.push_back(std::move(converted));
    }
    std::vector<glm::ChatTool> tools;
    if (request.tool_choice != ChatRequest::ToolChoice::kNone) {
      tools.reserve(request.tools.size());
      for (const auto& tool : request.tools)
        tools.push_back({.name = tool.name,
                         .description = tool.description,
                         .parameters_json = tool.parameters_json,
                         .definition_json = tool.definition_json});
    }
    std::string error;
    auto prompt =
        glm::PrepareChat(model_->tokenizer(), messages, tools,
                         glm::ResolveChatOptions(request.reasoning), &error);
    if (!prompt)
      throw std::invalid_argument("GLM-5.3-Flash chat template: " + error);
    return std::move(prompt->tokens);
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return glm::ResolveChatOptions(request.reasoning).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    std::string error;
    auto session = model_->CreateSession(&error);
    if (session == nullptr)
      throw std::runtime_error("Failed to create GLM-5.3-Flash session: " +
                               error);
    return std::make_unique<Glm53TextRunnerState>(std::move(session), *model_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    if (RequireGlm53State(state).generator().tokens().size() != prefix.size())
      throw std::logic_error(
          "GLM-5.3-Flash reused prefix does not match checkpoint");
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& generator = RequireGlm53State(state).generator();
    if (offset != generator.tokens().size()) {
      if (offset != 0)
        throw std::logic_error(
            "GLM-5.3-Flash prefill offset does not match retained state");
      generator.Invalidate();  // The next Extend starts a new sequence.
    }
    if (offset >= prompt.size())
      throw std::logic_error("GLM-5.3-Flash prefill has no remaining input");
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, prefill_chunk_});
    if (consumed == 0)
      throw std::logic_error("GLM-5.3-Flash prefill was granted no tokens");
    const std::size_t next_position = offset + consumed;
    std::string error;
    if (!generator.Extend(prompt.subspan(offset, consumed),
                          next_position == prompt.size(), &error))
      throw std::runtime_error("GLM-5.3-Flash prefill failed: " + error);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
        .checkpoint_ms = 0,
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& generator = RequireGlm53State(state).generator();
    if (generator.tokens().size() >= max_context_)
      return {.stop = true, .token = 0, .piece = {}};
    const auto logits = generator.logits();
    if (logits.empty())
      throw std::runtime_error("GLM-5.3-Flash has no logits to sample");
    const auto token = sampler.Sample(logits);
    if (state.stop_at_eos() && generator.IsStop(token))
      return {.stop = true, .token = 0, .piece = {}};
    return {
        .stop = false,
        .token = token,
        .piece = model_->tokenizer().DecodeToken(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    std::string error;
    if (!RequireGlm53State(state).generator().Extend(
            std::span<const TextRunnerToken>(&token, 1), /*want_logits=*/true,
            &error))
      throw std::runtime_error("GLM-5.3-Flash decode failed: " + error);
  }

  // MTP: forwards the sampled token with the session's drafts and returns the
  // drafts the sampler matched. It samples on a copy (the pool Accepts the
  // returned tokens) and publishes the RNG and a deferred mismatch.
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& generator = RequireGlm53State(state).generator();
    if (max_tokens <= 1 || !generator.session().CanDraft())
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    if (generator.tokens().size() >= max_context_)
      return {.selections = {}, .stop = true};
    const auto logits = generator.logits();
    if (logits.empty())
      throw std::runtime_error("GLM-5.3-Flash has no logits to sample");
    sampling::SamplerState working = sampler;
    const auto token = working.Sample(logits);
    if (state.stop_at_eos() && generator.IsStop(token)) {
      sampler.CopyDrawStateFrom(working);
      return {.selections = {}, .stop = true};
    }
    working.Accept(token);
    std::vector<TextRunnerToken> kept;
    std::size_t drafted = 0;
    std::string error;
    if (!generator.ExtendDrafted(token, working, max_tokens - 1,
                                 state.stop_at_eos(), &kept, &drafted, &error))
      throw std::runtime_error("GLM-5.3-Flash decode failed: " + error);
    sampler.CopyDrawStateFrom(working);
    TextDecodeStep step;
    step.selections.reserve(kept.size() + 1);
    const auto select = [&](TextRunnerToken id) {
      step.selections.push_back({.stop = false,
                                 .token = id,
                                 .piece = model_->tokenizer().DecodeToken(id)});
    };
    select(token);
    for (const auto id : kept)
      select(id);
    step.draft_rounds = drafted == 0 ? 0 : 1;
    step.draft_tokens = drafted;
    step.draft_accepted_tokens = kept.size();
    return step;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireGlm53State(state).generator().tokens().size();
  }

private:
  std::shared_ptr<glm::Model> model_;
  std::uint32_t max_context_;
  std::uint32_t prefill_chunk_;
};

}  // namespace

std::shared_ptr<TextModelRunner> MakeGlm53TextRunner(
    std::shared_ptr<models::glm53_flash::Model> model,
    std::uint32_t max_context, std::uint32_t prefill_chunk) {
  if (model == nullptr)
    throw std::invalid_argument("GLM-5.3-Flash model must not be null");
  return std::make_shared<Glm53TextRunner>(std::move(model), max_context,
                                           prefill_chunk);
}

}  // namespace gufo::server
