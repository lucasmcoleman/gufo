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
    if (!ToIds(tokens, error))
      return false;
    const std::span<const std::int32_t> all(ids_);
    if (draft_) {
      // A Draft not followed by Verify left the anchor in the draft block.
      if (drafted_ && !executor_.RewindNextn(executor_.NextnPosition() - 1))
        return Fail(error, "GLM-5.3-Flash draft rewind failed");
      drafted_ = false;
      return ForwardWithDraft(all, logits, error);
    }
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
    // The draft block's first token pairs with a zero h (llama.cpp's
    // draft-mtp pending row).
    h_last_.assign(hidden_, 0.0F);
    drafted_ = false;
    return executor_.Reset() ||
           Fail(error, "GLM-5.3-Flash executor reset failed");
  }

  std::uint32_t Position() const noexcept override {
    return executor_.Position();
  }
  std::uint32_t Capacity() const noexcept override { return capacity_; }

  bool CanDraft() const noexcept override { return draft_; }

  bool Draft(TokenId anchor, TokenId* draft, std::string* error) override {
    if (!draft_ || draft == nullptr)
      return Fail(error, "GLM-5.3-Flash session has no draft block");
    const std::int32_t id[1] = {static_cast<std::int32_t>(anchor)};
    if (drafted_ && !executor_.RewindNextn(executor_.NextnPosition() - 1))
      return Fail(error, "GLM-5.3-Flash draft rewind failed");
    drafted_ = false;
    if (!executor_.ForwardNextn(id, h_last_, true, &draft_result_, error))
      return false;
    drafted_ = true;
    if (draft_result_.logits.size() != vocab_)
      return Fail(error, "GLM-5.3-Flash draft returned no logits");
    *draft = static_cast<TokenId>(
        std::max_element(draft_result_.logits.begin(),
                         draft_result_.logits.end()) -
        draft_result_.logits.begin());
    return true;
  }

  bool Verify(std::span<const TokenId> tokens, std::vector<float>* logits,
              std::string* error) override {
    if (!draft_ || logits == nullptr || tokens.empty())
      return Fail(error, "GLM-5.3-Flash verify needs a draft block");
    if (!drafted_ || !ToIds(tokens, error))
      return Fail(error, "GLM-5.3-Flash verify needs a Draft first");
    drafted_ = false;
    verify_ids_ = ids_;
    if (!executor_.ForwardVerify(ids_, true, &result_, error))
      return false;
    if (result_.logits.size() != tokens.size() * vocab_ ||
        result_.hidden.size() != tokens.size() * hidden_)
      return Fail(error, "GLM-5.3-Flash verify returned short rows");
    logits->assign(result_.logits.begin(), result_.logits.end());
    return true;
  }

  bool Accept(std::size_t keep, std::string* error) override {
    if (keep == 0 || keep > verify_ids_.size() ||
        !executor_.Rollback(static_cast<std::uint32_t>(keep)))
      return Fail(error, "GLM-5.3-Flash rollback failed");
    // Draft() already ran the anchor through the draft block; the other
    // kept tokens pair with the trunk h of the row before them.
    if (keep > 1) {
      const std::span<const std::int32_t> rest(verify_ids_.data() + 1,
                                               keep - 1);
      const std::span<const float> h(result_.hidden.data(),
                                     (keep - 1) * hidden_);
      if (!executor_.ForwardNextn(rest, h, false, &draft_result_, error))
        return false;
    }
    h_last_.assign(result_.hidden.begin() + ((keep - 1) * hidden_),
                   result_.hidden.begin() + (keep * hidden_));
    return true;
  }

  /// Enables Draft/Verify/Accept; the executor must have the NextN block.
  void EnableDraft(std::uint32_t hidden) {
    draft_ = true;
    hidden_ = hidden;
    h_last_.assign(hidden_, 0.0F);
  }

private:
  bool ToIds(std::span<const TokenId> tokens, std::string* error) {
    ids_.clear();
    for (const TokenId token : tokens) {
      if (token >
          static_cast<TokenId>(std::numeric_limits<std::int32_t>::max()))
        return Fail(error, "GLM-5.3-Flash token id out of range");
      ids_.push_back(static_cast<std::int32_t>(token));
    }
    return true;
  }

  // Trunk forward that keeps every row's h, then the draft block's catch-up
  // over the same tokens (token p with h at p - 1).
  bool ForwardWithDraft(std::span<const std::int32_t> all,
                        std::vector<float>* logits, std::string* error) {
    std::vector<float> h_prev = h_last_;
    if (all.size() > 1) {
      if (!executor_.Forward(all.first(all.size() - 1), false, &result_,
                             nullptr, error))
        return false;
      h_prev.insert(h_prev.end(), result_.hidden.begin(),
                    result_.hidden.end());
    }
    if (!executor_.Forward(all.last(1), logits != nullptr, &result_, nullptr,
                           error))
      return false;
    if (result_.hidden.size() != hidden_)
      return Fail(error, "GLM-5.3-Flash forward returned no hidden row");
    if (!executor_.ForwardNextn(all, h_prev, false, &draft_result_, error))
      return false;
    h_last_ = result_.hidden;
    if (logits == nullptr)
      return true;
    if (result_.logits.size() != vocab_)
      return Fail(error, "GLM-5.3-Flash executor returned no logits");
    logits->assign(result_.logits.begin(), result_.logits.end());
    return true;
  }

  Executor& executor_;
  std::uint32_t capacity_;
  std::uint32_t vocab_;
  bool fused_;
  bool draft_{false};
  bool drafted_{false};  ///< Draft ran; the draft block holds the anchor.
  std::uint32_t hidden_{0};
  std::vector<float> h_last_;
  std::vector<std::int32_t> ids_;
  std::vector<std::int32_t> verify_ids_;
  ForwardResult result_;
  ForwardResult draft_result_;
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
                                                    bool fused_last_logits,
                                                    std::uint32_t draft_hidden) {
  auto session = std::make_unique<ExecutorSession>(executor, capacity, vocab,
                                                   fused_last_logits);
  if (draft_hidden != 0)
    session->EnableDraft(draft_hidden);
  return session;
}
#endif

struct Model::Impl {
  std::shared_ptr<const core::GgufReader> reader;
  std::unique_ptr<ModelWeights> weights;
#if defined(GLM53_ROCM_BUILD)
  std::unique_ptr<Executor> executor;
  bool fused_last_logits{true};
  bool mtp{false};  ///< The executor has the NextN block.
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
  bool CanDraft() const noexcept override { return inner_->CanDraft(); }
  bool Draft(TokenId anchor, TokenId* draft, std::string* error) override {
    return inner_->Draft(anchor, draft, error);
  }
  bool Verify(std::span<const TokenId> tokens, std::vector<float>* logits,
              std::string* error) override {
    return inner_->Verify(tokens, logits, error);
  }
  bool Accept(std::size_t keep, std::string* error) override {
    return inner_->Accept(keep, error);
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
  // MTP (NextN draft) when asked for and the GGUF has the block.
  const char* mtp_env = std::getenv("GUFO_GLM53_MTP");
  const bool want_mtp =
      options.mtp && config->nextn_layers > 0 &&
      !(mtp_env != nullptr && std::string_view(mtp_env) == "0");
  auto weights = ModelWeights::Bind(*reader, want_mtp, &detail);
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
  executor_options.nextn = want_mtp;
  executor_options.max_verify_tokens = want_mtp ? 2 : 1;
  model->impl_->mtp = want_mtp;
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
                          impl_->fused_last_logits,
                          impl_->mtp ? config().hidden_size : 0));
#else
  Fail(error, std::string(kLaneBMissingError));
  return nullptr;
#endif
}

}  // namespace gufo::models::glm53_flash
