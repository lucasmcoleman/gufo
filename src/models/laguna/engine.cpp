#include "src/models/laguna/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

#include "src/core/hip/heap_placement.hpp"

namespace gufo::models::laguna {
namespace {

constexpr std::uint32_t kMaxPrefillChunk = 2048;

void Assign(std::string* out, std::string message) {
  if (out != nullptr) {
    *out = std::move(message);
  }
}

/// Resolves the options against the artifact; false with a message when they
/// cannot be honoured.
bool ResolveOptions(const Config& config, const ModelOptions& options,
                    std::uint32_t* context, std::uint32_t* chunk,
                    std::string* error_msg) {
  *context =
      options.max_context != 0 ? options.max_context : config.context_length;
  if (*context < 2 || *context > config.context_length) {
    Assign(error_msg, "laguna: context " + std::to_string(*context) +
                          " is outside 2.." +
                          std::to_string(config.context_length));
    return false;
  }
  *chunk = options.prefill_chunk;
  if (*chunk == 0 || *chunk > kMaxPrefillChunk) {
    Assign(error_msg, "laguna: prefill chunk " + std::to_string(*chunk) +
                          " is outside 1.." + std::to_string(kMaxPrefillChunk));
    return false;
  }
  if (options.session_count == 0) {
    Assign(error_msg, "laguna: session count must be at least one");
    return false;
  }
  return true;
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(
    std::shared_ptr<const core::GgufReader> reader, const ModelOptions& options,
    std::string* error_msg) {
  if (reader == nullptr) {
    Assign(error_msg, "laguna: no GGUF reader");
    return nullptr;
  }
  std::string error;
  if (!ValidateGgufTemplate(*reader, &error)) {
    Assign(error_msg, "unsupported Laguna chat template: " + error);
    return nullptr;
  }
  auto bound = ModelWeights::Bind(*reader, &error);
  if (!bound) {
    Assign(error_msg, "Laguna weights: " + error);
    return nullptr;
  }
  auto weights = std::make_unique<ModelWeights>(std::move(*bound));
  std::uint32_t context = 0;
  std::uint32_t chunk = 0;
  if (!ResolveOptions(weights->config, options, &context, &chunk, error_msg)) {
    return nullptr;
  }
  auto tokenizer = Tokenizer::FromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    Assign(error_msg, "Laguna tokenizer: " + error);
    return nullptr;
  }
  if (tokenizer->vocab_size() != weights->config.vocab_size) {
    Assign(error_msg, "Laguna tokenizer has " +
                          std::to_string(tokenizer->vocab_size()) +
                          " tokens but the model's vocabulary is " +
                          std::to_string(weights->config.vocab_size));
    return nullptr;
  }

  // Weights and every session's KV, scratch and logits must fit together.
  // Check before the first byte is uploaded.
  const MemoryBudget budget =
      ComputeBudget(*weights, BudgetRequest{context, chunk, true});
  const std::uint64_t per_session = budget.kv() + budget.scratch();
  const std::uint64_t need =
      budget.weights + per_session * options.session_count;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess &&
      !hip::HeapFitsAboveReserve(free_bytes, need)) {
    Assign(error_msg,
           "Laguna weights plus " + std::to_string(options.session_count) +
               " session(s) at context " + std::to_string(context) + " need " +
               std::to_string(need) + " bytes but the device has " +
               std::to_string(free_bytes) + " free (driver reserve " +
               std::to_string(budget.reserve) +
               "); lower --context or --sessions\n" + budget.Format() +
               "  per session (kv + scratch): " + std::to_string(per_session) +
               " bytes");
    return nullptr;
  }

  auto device = rocm::DeviceModel::Upload(*weights, reader.get(), &error);
  if (device == nullptr) {
    Assign(error_msg, "Laguna weight upload: " + error);
    return nullptr;
  }
  std::string name(
      reader->GetMetadataString("general.name").value_or("Laguna-S-2.1"));
  std::shared_ptr<const void> keepalive = std::move(reader);
  return FromParts(std::move(keepalive), std::move(weights), std::move(device),
                   std::move(tokenizer), options, std::move(name), error_msg);
}

std::shared_ptr<Model> Model::FromParts(
    std::shared_ptr<const void> keepalive,
    std::unique_ptr<ModelWeights> weights,
    std::unique_ptr<rocm::DeviceModel> device,
    std::unique_ptr<Tokenizer> tokenizer, const ModelOptions& options,
    std::string name, std::string* error_msg) {
  if (weights == nullptr || device == nullptr || tokenizer == nullptr ||
      &device->weights() != weights.get()) {
    Assign(error_msg, "laguna: model parts do not belong together");
    return nullptr;
  }
  std::uint32_t context = 0;
  std::uint32_t chunk = 0;
  if (!ResolveOptions(weights->config, options, &context, &chunk, error_msg)) {
    return nullptr;
  }
  std::shared_ptr<Model> model(new Model());
  model->keepalive_ = std::move(keepalive);
  model->weights_ = std::move(weights);
  model->device_ = std::move(device);
  model->tokenizer_ = std::move(tokenizer);
  model->name_ = std::move(name);
  model->context_ = context;
  model->chunk_ = chunk;
  return model;
}

MemoryBudget Model::SessionBudget() const {
  return ComputeBudget(*weights_, BudgetRequest{context_, chunk_, true});
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text, bool add_bos,
                                          bool parse_special) const {
  const auto ids = tokenizer_->Encode(
      text, EncodeOptions{.add_bos = add_bos, .parse_special = parse_special});
  return {ids.begin(), ids.end()};
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<TokenId> ids;
  ids.reserve(tokens.size());
  for (const std::int32_t token : tokens) {
    if (token >= 0) {
      ids.push_back(static_cast<TokenId>(token));
    }
  }
  return tokenizer_->Decode(ids);
}

std::string Model::DecodeToken(std::int32_t token) const {
  return token < 0 ? std::string{}
                   : tokenizer_->DecodeToken(static_cast<TokenId>(token));
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return token >= 0 && tokenizer_->IsStopToken(static_cast<TokenId>(token));
}

std::vector<std::int32_t> Model::EncodeChat(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options) const {
  // The template starts with BOS's text, which the tokenizer matches as the
  // token itself, so the rendered prompt carries exactly one BOS.
  return Tokenize(RenderChat(messages, tools, options), /*add_bos=*/false,
                  /*parse_special=*/true);
}

void Model::InjectForwardFailureForTesting(std::uint32_t skip,
                                           std::string stage) const {
  const std::lock_guard<std::mutex> lock(fault_mutex_);
  fault_stage_ = std::move(stage);
  fault_countdown_.store(skip, std::memory_order_release);
}

bool Model::TakeInjectedFailure(std::string* stage) const {
  if (fault_countdown_.load(std::memory_order_acquire) < 0) {
    return false;
  }
  const std::lock_guard<std::mutex> lock(fault_mutex_);
  const std::int64_t left = fault_countdown_.load(std::memory_order_acquire);
  if (left < 0) {
    return false;
  }
  fault_countdown_.store(left - 1, std::memory_order_release);
  if (left > 0) {
    return false;
  }
  *stage = fault_stage_;
  return true;
}

std::unique_ptr<Session> Model::CreateSession(std::string* error_msg) const {
  rocm::ExecutorOptions options;
  options.context = context_;
  options.max_chunk = chunk_;
  auto executor = rocm::Executor::Create(*device_, options, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(executor)));
}

Session::Session(std::shared_ptr<const Model> model,
                 std::unique_ptr<rocm::Executor> executor)
    : model_(std::move(model)), executor_(std::move(executor)) {}

Session::~Session() = default;

bool Session::Recover(std::string* error_msg) {
  if (!invalid_) {
    return true;
  }
  std::string error;
  if (!executor_->Reset(&error)) {
    Assign(error_msg,
           "laguna session is invalid and could not be reset: " + error);
    return false;
  }
  logits_.clear();
  invalid_ = false;
  ++rebuilds_;
  return true;
}

bool Session::Fail(std::string* error_msg, const std::string& message) {
  ++failures_;
  invalid_ = true;
  logits_.clear();
  if (injected_) {
    executor_->attention_ops().SetFailAfterStage("");
    injected_ = false;
  }
  std::string text = message;
  std::string reset_error;
  if (!Recover(&reset_error)) {
    text += "; " + reset_error;
  }
  Assign(error_msg, std::move(text));
  return false;
}

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::string* error_msg) {
  logits_.resize(model_->VocabSize());
  std::string stage;
  if (model_->TakeInjectedFailure(&stage)) {
    InjectFailureForTesting(std::move(stage));
  }
  std::string error;
  if (!executor_->Forward(tokens, rocm::LogitsRows::kLast, logits_.data(),
                          &error)) {
    return Fail(error_msg,
                error.empty() ? "laguna forward pass failed" : error);
  }
  return true;
}

bool Session::Prefill(std::span<const std::int32_t> tokens,
                      std::string* error_msg) {
  if (!Recover(error_msg)) {
    return false;
  }
  if (tokens.empty()) {
    return true;
  }
  // Everything the pass could reject is checked before any work is enqueued,
  // so a rejection leaves the session untouched.
  const std::uint32_t vocab = model_->VocabSize();
  for (const std::int32_t token : tokens) {
    if (token < 0 || static_cast<std::uint32_t>(token) >= vocab) {
      Assign(error_msg, "laguna: token id outside the vocabulary");
      return false;
    }
  }
  if (std::uint64_t{Position()} + tokens.size() > Context()) {
    Assign(error_msg, "laguna: context of " + std::to_string(Context()) +
                          " tokens exhausted");
    return false;
  }
  const std::size_t chunk = model_->PrefillChunk();
  for (std::size_t offset = 0; offset < tokens.size(); offset += chunk) {
    if (!Forward(
            tokens.subspan(offset, std::min(chunk, tokens.size() - offset)),
            error_msg)) {
      return false;
    }
  }
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  return Prefill(std::span<const std::int32_t>(&token, 1), error_msg);
}

void Session::Invalidate() noexcept {
  try {
    logits_.clear();
    if (injected_) {
      executor_->attention_ops().SetFailAfterStage("");
      injected_ = false;
    }
    std::string error;
    // Clear() keeps the caches' memory when the executor is healthy and
    // rebuilds them when it is not; either way the session starts over.
    const bool was_invalid = invalid_;
    invalid_ = !executor_->Clear(&error);
    if (was_invalid && !invalid_) {
      ++rebuilds_;
    }
  } catch (...) {
    invalid_ = true;
  }
}

std::size_t Session::SnapshotBytes() const noexcept {
  return executor_->snapshot_bytes() + model_->VocabSize() * sizeof(float);
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(std::string* error_msg) {
  if (!Recover(error_msg)) {
    return nullptr;
  }
  if (Position() == 0 || logits_.size() != model_->VocabSize()) {
    Assign(error_msg, "laguna: nothing to snapshot at position 0");
    return nullptr;
  }
  auto snapshot = std::make_unique<SessionSnapshot>();
  if (!executor_->SaveSnapshot(&snapshot->kv, error_msg)) {
    return nullptr;
  }
  snapshot->logits = logits_;
  snapshot->context = Context();
  snapshot->chunk = model_->PrefillChunk();
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  if (!Recover(error_msg)) {
    return false;
  }
  if (snapshot.context != Context() ||
      snapshot.chunk != model_->PrefillChunk() ||
      snapshot.logits.size() != model_->VocabSize() ||
      snapshot.kv.position == 0) {
    Assign(error_msg, "laguna: snapshot does not fit this session");
    return false;
  }
  std::string error;
  if (!executor_->RestoreSnapshot(snapshot.kv, &error)) {
    return Fail(error_msg,
                error.empty() ? "laguna snapshot restore failed" : error);
  }
  logits_ = snapshot.logits;
  return true;
}

void Session::InjectFailureForTesting(std::string stage) {
  injected_ = !stage.empty();
  executor_->attention_ops().SetFailAfterStage(std::move(stage));
}

}  // namespace gufo::models::laguna
