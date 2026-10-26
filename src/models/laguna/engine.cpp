#include "src/models/laguna/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

#include "src/core/hip/heap_placement.hpp"
#include "src/models/laguna/kernels/dflash_session.hpp"
#include "src/models/laguna/speculative.hpp"

namespace gufo::models::laguna {
namespace {

constexpr std::uint32_t kMaxPrefillChunk = 2048;
/// Rows of a drafter injection (DFlashSession's vector-path limit); the ring
/// holds the window plus this many positions.
constexpr std::uint64_t kDrafterInjectRows = 32;
/// Drafter scratch of one session, measured at about 20 MiB for a block of
/// 16 on the real drafter; bounded generously.
constexpr std::uint64_t kDrafterScratchBound = std::uint64_t{96} << 20;

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
  model->sessions_ = options.session_count;
  return model;
}

bool Model::AttachDrafter(std::shared_ptr<const core::GgufReader> reader,
                          const DrafterOptions& options,
                          std::string* error_msg) {
  if (reader == nullptr) {
    Assign(error_msg, "laguna: no drafter GGUF reader");
    return false;
  }
  std::string error;
  const auto weights = DFlashWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    Assign(error_msg, "Laguna DFlash drafter: " + error);
    return false;
  }
  return AttachDrafterWeights(*weights, options, error_msg);
}

std::size_t Model::DrafterBytes() const noexcept {
  return drafter_ != nullptr ? drafter_->device_bytes() : 0;
}

std::size_t Model::DrafterSessionBytes() const noexcept {
  if (drafter_ == nullptr) {
    return 0;
  }
  const rocm::DrafterWeights& w = drafter_->weights();
  const std::uint64_t capture =
      std::uint64_t{chunk_} * w.taps * w.hidden * sizeof(float);
  const std::uint64_t ring =
      2ULL * w.layers.size() *
      (w.window + std::max<std::uint64_t>(kDrafterInjectRows, w.block_size)) *
      w.kv_heads * w.head_dim * 2;
  return static_cast<std::size_t>(capture + ring + kDrafterScratchBound);
}

bool Model::AttachDrafterWeights(const DFlashWeights& weights,
                                 const DrafterOptions& options,
                                 std::string* error_msg) {
  if (drafter_ != nullptr) {
    Assign(error_msg, "laguna: a drafter is already attached");
    return false;
  }
  if (options.block_rows < 2 || options.block_rows > 16 ||
      options.block_rows > weights.config.block_size) {
    Assign(error_msg, "laguna: draft block of " +
                          std::to_string(options.block_rows) +
                          " rows is outside 2.." +
                          std::to_string(std::min<std::uint32_t>(
                              16, weights.config.block_size)));
    return false;
  }
  // A verify of block_rows must fit one Forward chunk, and every rewind of a
  // rejected verify must stay within the rings' spare rows (DFLASH 6d).
  if (chunk_ < options.block_rows) {
    Assign(error_msg, "laguna: prefill chunk " + std::to_string(chunk_) +
                          " is smaller than the draft block");
    return false;
  }
  std::string error;
  if (!weights.config.ValidateAgainstTarget(weights_->config, &error)) {
    Assign(error_msg,
           "Laguna DFlash drafter does not fit the target: " + error);
    return false;
  }
  // The drafter's weights plus every session's drafter buffers must fit the
  // device memory the target left free. Checked before the upload.
  const std::uint64_t ring =
      2ULL * weights.config.num_layers *
      (weights.config.sliding_window +
       std::max<std::uint64_t>(kDrafterInjectRows, weights.config.block_size)) *
      weights.config.num_kv_heads * weights.config.head_dim * 2;
  const std::uint64_t per_session =
      std::uint64_t{chunk_} * weights.config.tap_count() *
          weights.config.hidden_size * sizeof(float) +
      ring + kDrafterScratchBound;
  const std::uint64_t need = weights.total_bytes + per_session * sessions_;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess &&
      !hip::HeapFitsAboveReserve(free_bytes, need)) {
    Assign(error_msg, "Laguna DFlash drafter (" +
                          std::to_string(weights.total_bytes) +
                          " bytes) plus " + std::to_string(sessions_) +
                          " session(s) of drafter buffers (" +
                          std::to_string(per_session) + " bytes each) need " +
                          std::to_string(need) + " bytes but the device has " +
                          std::to_string(free_bytes) + " free");
    return false;
  }
  auto uploaded = rocm::DeviceDrafter::Upload(weights, &error);
  if (!uploaded.has_value()) {
    Assign(error_msg, "Laguna DFlash drafter upload: " + error);
    return false;
  }
  std::vector<std::uint32_t> taps;
  for (std::size_t k = 0; k < weights.config.tap_count(); ++k) {
    taps.push_back(weights.config.tap_layer(k));
  }
  drafter_ = std::make_unique<rocm::DeviceDrafter>(std::move(*uploaded));
  drafter_taps_ = std::move(taps);
  drafter_mask_ = static_cast<std::int32_t>(weights.config.mask_token_id);
  draft_rows_ = options.block_rows;
  return true;
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

std::unique_ptr<Session> Model::CreateSession(std::string* error_msg,
                                              bool with_drafter) const {
  rocm::ExecutorOptions options;
  options.context = context_;
  options.max_chunk = chunk_;
  auto executor = rocm::Executor::Create(*device_, options, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  std::unique_ptr<Session> session(
      new Session(shared_from_this(), std::move(executor)));
  if (drafter_ != nullptr && with_drafter) {
    session->dflash_ =
        rocm::DFlashSession::Create(*session->executor_, drafter_->weights(),
                                    drafter_taps_, drafter_mask_, error_msg);
    if (session->dflash_ == nullptr) {
      return nullptr;
    }
    // The executor warmed up its batched path; a verify takes the vector
    // path, whose scratch would otherwise grow on the first request.
    std::string error;
    if (!session->dflash_->Warmup(draft_rows_, draft_rows_, &error)) {
      Assign(error_msg, "laguna drafter warm-up: " + error);
      return nullptr;
    }
    session->decoder_ = std::make_unique<SpeculativeDecoder>(
        *this, *session->dflash_, draft_rows_);
  }
  return session;
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
  RearmDrafter();
  return true;
}

void Session::RearmDrafter() noexcept {
  if (dflash_ == nullptr) {
    return;
  }
  try {
    // The executor is healthy here: Reset() clears it (a no-op at position 0)
    // and restarts the drafter with an empty ring, resuming a suspended one.
    std::string error;
    (void)dflash_->Reset(&error);
  } catch (...) {
  }
}

bool Session::ResumeDrafter(std::string* error_msg) {
  if (dflash_ == nullptr || !dflash_->suspended()) {
    return true;
  }
  return dflash_->Resume(error_msg);
}

bool Session::drafting() const noexcept {
  return dflash_ != nullptr && dflash_->capturing();
}

void Session::SuspendDrafter() noexcept {
  if (dflash_ != nullptr) {
    dflash_->Suspend();
  }
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
                      std::string* error_msg, std::uint32_t following) {
  logits_.resize(model_->VocabSize());
  std::string stage;
  if (model_->TakeInjectedFailure(&stage)) {
    InjectFailureForTesting(std::move(stage));
  }
  std::string error;
  bool ok = false;
  if (dflash_ != nullptr && !dflash_->suspended()) {
    // A single decode row is a one-row verify on the vector path, the path
    // Forward takes for it, so the logits are those of plain decode; its taps
    // reach the drafter. Prompt chunks keep the executor's own path.
    ok = tokens.size() == 1
             ? dflash_->Verify(tokens, logits_.data(), &error)
             : dflash_->Prefill(tokens, rocm::LogitsRows::kLast, logits_.data(),
                                &error, following);
  } else {
    ok = executor_->Forward(tokens, rocm::LogitsRows::kLast, logits_.data(),
                            &error);
  }
  if (!ok) {
    return Fail(error_msg,
                error.empty() ? "laguna forward pass failed" : error);
  }
  return true;
}

bool Session::Prefill(std::span<const std::int32_t> tokens,
                      std::string* error_msg, std::uint32_t following) {
  if (!Recover(error_msg)) {
    return false;
  }
  if (tokens.empty()) {
    return true;
  }
  if (tokens.size() > 1 && !ResumeDrafter(error_msg)) {
    return false;
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
    const std::size_t n = std::min(chunk, tokens.size() - offset);
    const std::uint64_t after = tokens.size() - offset - n + following;
    if (!Forward(tokens.subspan(offset, n), error_msg,
                 static_cast<std::uint32_t>(std::min<std::uint64_t>(
                     after, std::numeric_limits<std::uint32_t>::max())))) {
      return false;
    }
  }
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  return Prefill(std::span<const std::int32_t>(&token, 1), error_msg);
}

void Session::EvaluateBatch(std::span<BatchItem> items) {
  std::vector<BatchItem*> accepted;
  accepted.reserve(items.size());
  for (BatchItem& item : items) {
    item.completed = false;
    item.error.clear();
    Session* session = item.session;
    if (session == nullptr) {
      item.error = "laguna: no session";
      continue;
    }
    if (std::any_of(accepted.begin(), accepted.end(), [&](const BatchItem* o) {
          return o->session == session;
        })) {
      item.error = "laguna: session appears twice in a batch";
      continue;
    }
    // Everything a pass could reject is checked before any work is enqueued,
    // as Prefill does, so a rejection leaves the session untouched.
    if (!session->Recover(&item.error)) {
      continue;
    }
    if (item.token < 0 || static_cast<std::uint32_t>(item.token) >=
                              session->model_->VocabSize()) {
      item.error = "laguna: token id outside the vocabulary";
      continue;
    }
    if (std::uint64_t{session->Position()} + 1 > session->Context()) {
      item.error = "laguna: context of " + std::to_string(session->Context()) +
                   " tokens exhausted";
      continue;
    }
    // A session whose executor carries the drafter's taps (any session with
    // a drafter that is not suspended) decodes alone (DFLASH rule 6): the
    // shared pass neither captures its row nor takes a tapped executor.
    if (session->dflash_ != nullptr && !session->dflash_->suspended()) {
      item.completed = session->Evaluate(item.token, &item.error);
      continue;
    }
    accepted.push_back(&item);
  }
  if (accepted.size() == 1) {
    BatchItem& item = *accepted.front();
    item.completed = item.session->Evaluate(item.token, &item.error);
    return;
  }
  if (accepted.empty()) {
    return;
  }
  Session* lead = accepted.front()->session;
  const Model& model = *lead->model_;
  // Injected faults: a pending model-wide fault applies to every row, else
  // each session carries its own. DecodeBatch fails each group that holds
  // one, as a device error in that shared pass would, and no other group.
  std::string model_stage;
  const bool model_fault = model.TakeInjectedFailure(&model_stage);
  std::vector<rocm::DecodeItem> work(accepted.size());
  for (std::size_t i = 0; i < accepted.size(); ++i) {
    Session& session = *accepted[i]->session;
    session.logits_.resize(session.model_->VocabSize());
    work[i].executor = session.executor_.get();
    work[i].token = accepted[i]->token;
    work[i].logits = session.logits_.data();
    if (model_fault) {
      work[i].fail_stage = model_stage;
    } else if (session.injected_) {
      work[i].fail_stage = session.inject_stage_;
    }
  }
  rocm::Executor::DecodeBatch(work);
  for (std::size_t i = 0; i < accepted.size(); ++i) {
    BatchItem& item = *accepted[i];
    if (work[i].completed) {
      item.completed = true;
    } else {
      (void)item.session->Fail(&item.error, work[i].error.empty()
                                                ? "laguna forward pass failed"
                                                : work[i].error);
    }
  }
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
    if (!invalid_) {
      RearmDrafter();
    }
  } catch (...) {
    invalid_ = true;
  }
}

std::size_t Session::SnapshotBytes() const noexcept {
  return executor_->snapshot_bytes() + model_->VocabSize() * sizeof(float) +
         (drafting() ? dflash_->drafter_image_bytes() : 0);
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
  if (drafting()) {
    // Without an image (a drafter out of step, or a copy failure, which
    // drops it) a restore of this snapshot starts the drafter cold.
    rocm::DrafterImage image;
    if (dflash_->SaveDrafter(&image)) {
      snapshot->drafter = std::move(image);
    }
  }
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
  if (dflash_ != nullptr) {
    // The ring of the snapshot, or an empty (cold) ring at its position;
    // never the rows this session held before. A drafter that cannot take
    // the image is dropped; the target state stands.
    (void)dflash_->RestoreDrafter(
        snapshot.drafter ? &*snapshot.drafter : nullptr, &error);
  }
  return true;
}

bool Session::DecodeSpeculative(sampling::SamplerState& sampler,
                                std::uint32_t max_tokens,
                                std::uint32_t context_limit, bool stop_at_eos,
                                SpeculativeStepResult* out,
                                std::string* error_msg) {
  if (decoder_ == nullptr || out == nullptr) {
    Assign(error_msg, "laguna: the session has no drafter");
    return false;
  }
  if (sampler.config().uses_random_sampling() || max_tokens == 0) {
    Assign(error_msg,
           "laguna: speculative decoding takes deterministic requests and a "
           "positive budget");
    return false;
  }
  if (!Recover(error_msg)) {
    return false;
  }
  if (logits_.size() != model_->VocabSize()) {
    Assign(error_msg, "laguna: no frontier to decode from");
    return false;
  }
  if (!ResumeDrafter(error_msg)) {
    return false;
  }
  std::string stage;
  if (model_->TakeInjectedFailure(&stage)) {
    InjectFailureForTesting(std::move(stage));
  }
  std::vector<float> frontier;
  try {
    decoder_->Step(logits_, sampler, max_tokens, context_limit, stop_at_eos,
                   out, &frontier);
  } catch (const std::exception& e) {
    // A target failure poisoned the executor; any other failure may have left
    // the session between a verify and its commit. Start over either way.
    return Fail(error_msg, e.what());
  }
  if (!out->stop) {
    logits_ = std::move(frontier);
  }
  return true;
}

void Session::InjectFailureForTesting(std::string stage) {
  injected_ = !stage.empty();
  inject_stage_ = stage;
  executor_->attention_ops().SetFailAfterStage(std::move(stage));
}

}  // namespace gufo::models::laguna
