#ifndef GUFO_MODELS_LAGUNA_ENGINE_HPP_
#define GUFO_MODELS_LAGUNA_ENGINE_HPP_

// Laguna-S-2.1 serving engine: a Model (GGUF weights resident on the GPU, the
// tokenizer and the pinned chat template) and the Sessions that run on it.
//
// One Session owns one rocm::Executor: its own stream, its own full-layer KV
// caches and sliding-window rings, and its own operator scratch. Sessions
// share only the read-only DeviceModel, so any number of them can run on
// different threads (nothing is shared between executors, see executor.hpp).
//
// A Session keeps the logits of its last token (the "frontier"), so sampling
// the next token never needs a forward pass of its own. Every Forward asks for
// the last row, so even a state captured between two prefill chunks can pick
// the token that follows it.
//
// Failure contract: any failure after work was enqueued marks the session
// invalid and resets it at once (position 0, the executor's caches rebuilt).
// A session that could not be reset stays invalid and rejects every call
// until a later reset succeeds; nothing ever continues on a poisoned cache.
// Calls rejected up front (a token outside the vocabulary, a context overrun)
// touch nothing and leave the session as it was.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/budget.hpp"
#include "src/models/laguna/chat_template.hpp"
#include "src/models/laguna/kernels/device_model.hpp"
#include "src/models/laguna/kernels/executor.hpp"
#include "src/models/laguna/tokenizer.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna {

struct ModelOptions {
  /// Positions of every session (full-layer KV). Zero takes the GGUF's
  /// context_length (262144, about 12 GiB of KV per session).
  std::uint32_t max_context{0};
  /// Rows per forward pass of a prefill. The sliding rings hold the window
  /// plus this many positions.
  std::uint32_t prefill_chunk{512};
  /// Sessions the host will create. Only the load-time memory check uses it.
  std::uint32_t session_count{1};
};

/// Immutable host copy of a session's state at one position.
struct SessionSnapshot {
  rocm::ExecutorSnapshot kv;
  std::vector<float> logits;  ///< The frontier logits at `kv.position`.
  std::uint32_t context{0};
  std::uint32_t chunk{0};
  [[nodiscard]] std::uint32_t position() const noexcept { return kv.position; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return kv.bytes() + logits.size() * sizeof(float);
  }
};

class Session;

class Model : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// Validates the pinned chat template, binds the tensors, checks that the
  /// weights plus `session_count` sessions fit the free device memory (the
  /// message carries the budget), then uploads the weights. `reader` stays
  /// open for the model's lifetime (token embeddings are read from it).
  [[nodiscard]] static std::shared_ptr<Model> Load(
      std::shared_ptr<const core::GgufReader> reader,
      const ModelOptions& options, std::string* error_msg = nullptr);

  /// Assembles a model from parts already built (tests use synthetic
  /// weights). `device` must have been uploaded from `*weights`; `keepalive`
  /// owns whatever memory the weights' tensor references point into.
  [[nodiscard]] static std::shared_ptr<Model> FromParts(
      std::shared_ptr<const void> keepalive,
      std::unique_ptr<ModelWeights> weights,
      std::unique_ptr<rocm::DeviceModel> device,
      std::unique_ptr<Tokenizer> tokenizer, const ModelOptions& options,
      std::string name, std::string* error_msg = nullptr);

  [[nodiscard]] const std::string& ModelName() const noexcept { return name_; }
  [[nodiscard]] const Config& config() const noexcept {
    return weights_->config;
  }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept { return context_; }
  [[nodiscard]] std::uint32_t PrefillChunk() const noexcept { return chunk_; }
  [[nodiscard]] std::uint32_t VocabSize() const noexcept {
    return weights_->config.vocab_size;
  }
  [[nodiscard]] const rocm::DeviceModel& device() const noexcept {
    return *device_;
  }
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  /// What one session costs on the device at this context and chunk.
  [[nodiscard]] MemoryBudget SessionBudget() const;

  /// Raw text to ids. `add_bos` prepends BOS (id 2); `parse_special` matches
  /// control-token text such as `</assistant>`.
  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos = false,
      bool parse_special = true) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string DecodeToken(std::int32_t token) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;

  /// The conversation rendered by the pinned template and tokenized. The
  /// template writes BOS itself, so none is added.
  [[nodiscard]] std::vector<std::int32_t> EncodeChat(
      std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
      const ChatTemplateOptions& options) const;

  /// Test seam for the serving path: the forward pass that is `skip` passes
  /// from now (counted across every session of this model, so the order
  /// depends on scheduling when several run) fails once after attention
  /// stage `stage`, as a device error in the middle of a pass would.
  void InjectForwardFailureForTesting(std::uint32_t skip,
                                      std::string stage) const;
  /// Consumed by Session before each pass: true when this pass must fail.
  [[nodiscard]] bool TakeInjectedFailure(std::string* stage) const;

  /// A fresh session (its own executor). Nullptr with a message when the
  /// device cannot hold another one.
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::string* error_msg = nullptr) const;

private:
  Model() = default;

  std::shared_ptr<const void> keepalive_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::string name_;
  std::uint32_t context_{0};
  std::uint32_t chunk_{0};
  mutable std::atomic<std::int64_t> fault_countdown_{-1};
  mutable std::mutex fault_mutex_;
  mutable std::string fault_stage_;
};

class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Appends `tokens` (any number; run in chunks of Model::PrefillChunk) and
  /// leaves the frontier logits of the last one. Empty input is a no-op.
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens,
                             std::string* error_msg = nullptr);
  /// Appends one token (a decode step).
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);

  /// Logits [vocab] after the last token; empty at position 0 and after a
  /// failure.
  [[nodiscard]] const std::vector<float>& Logits() const noexcept {
    return logits_;
  }
  [[nodiscard]] std::uint32_t Position() const noexcept {
    return executor_->position();
  }
  [[nodiscard]] std::uint32_t Context() const noexcept {
    return executor_->context();
  }

  /// Back to position 0 for a new conversation. Cheap (the caches keep their
  /// memory) unless the executor failed, when they are rebuilt.
  void Invalidate() noexcept;

  /// True while a failure has not been recovered by a successful reset.
  [[nodiscard]] bool invalid() const noexcept { return invalid_; }
  /// Failed forward passes and the rebuilds they caused, since creation.
  [[nodiscard]] std::uint32_t failures() const noexcept { return failures_; }
  [[nodiscard]] std::uint32_t rebuilds() const noexcept { return rebuilds_; }
  [[nodiscard]] bool poisoned() const noexcept { return executor_->poisoned(); }

  /// Host bytes a snapshot of the current position takes.
  [[nodiscard]] std::size_t SnapshotBytes() const noexcept;
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr);
  /// Replaces the session's state with `snapshot` (frontier logits included).
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);

  /// Device bytes of this session's KV caches and of its other buffers.
  [[nodiscard]] std::size_t KvBytes() const noexcept {
    return executor_->kv_bytes();
  }

  /// Test seam: the next forward pass fails once, right after attention stage
  /// `stage` (see AttnOps::SetFailAfterStage), after that layer's K/V was
  /// appended, like a device error mid-pass.
  void InjectFailureForTesting(std::string stage);
  [[nodiscard]] rocm::Executor& executor() noexcept { return *executor_; }

private:
  friend class Model;
  Session(std::shared_ptr<const Model> model,
          std::unique_ptr<rocm::Executor> executor);
  [[nodiscard]] bool Recover(std::string* error_msg);
  bool Fail(std::string* error_msg, const std::string& message);
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::string* error_msg);

  std::shared_ptr<const Model> model_;
  std::unique_ptr<rocm::Executor> executor_;
  std::vector<float> logits_;
  bool invalid_{false};
  std::uint32_t failures_{0};
  std::uint32_t rebuilds_{0};
  bool injected_{false};
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_ENGINE_HPP_
