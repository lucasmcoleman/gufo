#include "src/models/glm53_flash/engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/glm53_flash/weights.hpp"

#if defined(GLM53_ROCM_BUILD)
#include "src/models/glm53_flash/executor.hpp"
#endif

namespace gufo::models::glm53_flash {

namespace {

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

#if defined(GLM53_ROCM_BUILD)
/// Lane B's operators, or kLaneBMissingError in a build without them.
bool ProductionOps(LayerOps* ops, std::string* error) {
#if defined(GLM53_LANE_B_OPS)
  *ops = LaneBOps();
  (void)error;
  return true;
#else
  (void)ops;
  return Fail(error, std::string(kLaneBMissingError));
#endif
}

class ExecutorSession final : public ForwardSession {
public:
  ExecutorSession(Executor& executor, std::uint32_t capacity,
                  std::uint32_t vocab, bool fused_last_logits)
      : executor_(executor),
        capacity_(capacity),
        vocab_(vocab),
        fused_(fused_last_logits) {}

  bool Forward(std::span<const TokenId> tokens, std::vector<float>* logits,
               std::string* error) override {
    if (tokens.empty())
      return Fail(error, "GLM-5.3-Flash forward needs at least one token");
    ids_.clear();
    for (const TokenId token : tokens) {
      if (token >
          static_cast<TokenId>(std::numeric_limits<std::int32_t>::max()))
        return Fail(error, "GLM-5.3-Flash token id out of range");
      ids_.push_back(static_cast<std::int32_t>(token));
    }
    const std::span<const std::int32_t> all(ids_);
    if (fused_) {
      // One pass per chunk: the final pass projects its last row only, and
      // no pass downloads the hidden states.
      if (!executor_.ForwardLast(all, logits != nullptr, &result_, nullptr,
                                 error))
        return false;
      if (logits == nullptr)
        return true;
      if (result_.logits.size() != vocab_) {
        return Fail(error, "GLM-5.3-Flash executor returned " +
                               std::to_string(result_.logits.size()) +
                               " logits for a vocabulary of " +
                               std::to_string(vocab_));
      }
      logits->assign(result_.logits.begin(), result_.logits.end());
      return true;
    }
    if (logits == nullptr)
      return executor_.Forward(all, false, &result_, nullptr, error);
    // Logits of the last token only: the executor returns a row per token,
    // so the rest of the call runs without them.
    if (all.size() > 1 && !executor_.Forward(all.first(all.size() - 1), false,
                                             &result_, nullptr, error))
      return false;
    if (!executor_.Forward(all.last(1), true, &result_, nullptr, error))
      return false;
    if (result_.logits.size() < vocab_) {
      return Fail(error, "GLM-5.3-Flash executor returned " +
                             std::to_string(result_.logits.size()) +
                             " logits for a vocabulary of " +
                             std::to_string(vocab_));
    }
    logits->assign(result_.logits.end() - vocab_, result_.logits.end());
    return true;
  }

  bool Reset(std::string* error) override {
    return executor_.Reset() ||
           Fail(error, "GLM-5.3-Flash executor reset failed");
  }

  std::uint32_t Position() const noexcept override {
    return executor_.Position();
  }
  std::uint32_t Capacity() const noexcept override { return capacity_; }

private:
  Executor& executor_;
  std::uint32_t capacity_;
  std::uint32_t vocab_;
  bool fused_;
  std::vector<std::int32_t> ids_;
  ForwardResult result_;
};
#endif

}  // namespace

bool IsGlm53Flash(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == kArchitecture;
}

bool RuntimeAvailable(std::string* error) {
#if !defined(GLM53_LANE_B_OPS)
  return Fail(error, std::string(kLaneBMissingError));
#elif !defined(GLM53_ROCM_BUILD)
  return Fail(error,
              "GLM-5.3-Flash requires the ROCm backend (ENGINE_ENABLE_HIP=ON)");
#else
  (void)error;
  return true;
#endif
}

#if defined(GLM53_ROCM_BUILD)
std::unique_ptr<ForwardSession> MakeExecutorSession(Executor& executor,
                                                    std::uint32_t capacity,
                                                    std::uint32_t vocab,
                                                    bool fused_last_logits) {
  return std::make_unique<ExecutorSession>(executor, capacity, vocab,
                                           fused_last_logits);
}
#endif

struct Model::Impl {
  std::shared_ptr<const core::GgufReader> reader;
  std::unique_ptr<ModelWeights> weights;
#if defined(GLM53_ROCM_BUILD)
  std::unique_ptr<Executor> executor;
  bool fused_last_logits{true};
#endif
};

/// Hands the model's executor to one owner at a time.
class Model::Session final : public ForwardSession {
public:
  Session(std::shared_ptr<Model> model, std::unique_ptr<ForwardSession> inner)
      : model_(std::move(model)), inner_(std::move(inner)) {}
  ~Session() override { model_->session_alive_ = false; }

  bool Forward(std::span<const TokenId> tokens, std::vector<float>* logits,
               std::string* error) override {
    return inner_->Forward(tokens, logits, error);
  }
  bool Reset(std::string* error) override { return inner_->Reset(error); }
  std::uint32_t Position() const noexcept override {
    return inner_->Position();
  }
  std::uint32_t Capacity() const noexcept override {
    return inner_->Capacity();
  }

private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<ForwardSession> inner_;
};

Model::Model() : impl_(std::make_unique<Impl>()) {}

Model::~Model() = default;

const Config& Model::config() const noexcept {
  return impl_->weights->config;
}

std::uint32_t Model::VocabSize() const noexcept {
  return impl_->weights->config.vocab_size;
}

std::size_t Model::DeviceBytes() const noexcept {
#if defined(GLM53_ROCM_BUILD)
  return impl_->executor != nullptr ? impl_->executor->DeviceBytes() : 0;
#else
  return 0;
#endif
}

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error) {
  std::string detail;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &detail);
  if (reader_owner == nullptr) {
    Fail(error, "failed to open GGUF: " + detail);
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  if (!IsGlm53Flash(*reader)) {
    Fail(error,
         "not a GLM-5.3-Flash GGUF: general.architecture is '" +
             std::string(reader->GetMetadataString("general.architecture")
                             .value_or(std::string_view{})) +
             "', expected '" + std::string(kArchitecture) + "'");
    return nullptr;
  }
  // Refuse before touching weights or the device.
  if (!RuntimeAvailable(error))
    return nullptr;
#if defined(GLM53_ROCM_BUILD)
  const auto config = Config::FromGguf(*reader, &detail);
  if (!config) {
    Fail(error, "GLM-5.3-Flash config: " + detail);
    return nullptr;
  }
  const std::uint32_t max_context =
      options.max_context != 0 ? options.max_context : config->context_length;
  if (max_context > config->context_length) {
    Fail(error, "GLM-5.3-Flash context " + std::to_string(max_context) +
                    " exceeds the model's " +
                    std::to_string(config->context_length));
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*reader, /*bind_nextn=*/false, &detail);
  if (!weights) {
    Fail(error, "GLM-5.3-Flash weights: " + detail);
    return nullptr;
  }
  auto tokenizer = Tokenizer::CreateFromGguf(*reader, &detail);
  if (tokenizer == nullptr) {
    Fail(error, "GLM-5.3-Flash tokenizer: " + detail);
    return nullptr;
  }
  LayerOps ops;
  if (!ProductionOps(&ops, error))
    return nullptr;

  std::shared_ptr<Model> model(new Model());
  model->impl_->reader = reader;
  model->impl_->weights = std::make_unique<ModelWeights>(std::move(*weights));
  ExecutorOptions executor_options;
  executor_options.max_tokens =
      std::max<std::uint32_t>(options.prefill_chunk, 1);
  executor_options.capacity = max_context;
  executor_options.shards = reader->GetMappedRegions();
  model->impl_->executor = Executor::Create(
      *model->impl_->weights, executor_options, std::move(ops), &detail);
  if (model->impl_->executor == nullptr) {
    Fail(error, "GLM-5.3-Flash executor: " + detail);
    return nullptr;
  }
  const char* fused_env = std::getenv("GUFO_GLM53_FUSED_LAST_LOGITS");
  model->impl_->fused_last_logits =
      options.fused_last_logits &&
      !(fused_env != nullptr && std::string_view(fused_env) == "0");
  model->stops_ = ResolveStopTokens(*tokenizer);
  model->tokenizer_ = std::move(tokenizer);
  model->max_context_ = max_context;
  model->name_ = std::string(
      reader->GetMetadataString("general.name").value_or(kArchitecture));
  return model;
#else
  (void)options;
  return nullptr;  // RuntimeAvailable refused above.
#endif
}

std::unique_ptr<ForwardSession> Model::CreateSession(std::string* error) {
#if defined(GLM53_ROCM_BUILD)
  if (impl_->executor == nullptr) {
    Fail(error, "GLM-5.3-Flash model has no executor");
    return nullptr;
  }
  if (session_alive_.exchange(true)) {
    Fail(error,
         "GLM-5.3-Flash runs one sequence at a time; its session is in use");
    return nullptr;
  }
  return std::make_unique<Session>(
      shared_from_this(),
      MakeExecutorSession(*impl_->executor, max_context_, VocabSize(),
                          impl_->fused_last_logits));
#else
  Fail(error, std::string(kLaneBMissingError));
  return nullptr;
#endif
}

}  // namespace gufo::models::glm53_flash
