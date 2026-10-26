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
//
// Speculative decoding (DFLASH.md phase 6e): a Model with an attached DFlash
// drafter gives every session a rocm::DFlashSession over its executor and a
// SpeculativeDecoder. Prefill and Evaluate then capture the drafter's taps;
// DecodeSpeculative runs one greedy speculative step; snapshots carry the
// drafter ring. A session that serves a request which samples at random is
// suspended (SuspendDrafter) so it decodes plainly and can join
// EvaluateBatch; it never speculates such a request.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/laguna/budget.hpp"
#include "src/models/laguna/chat_template.hpp"
#include "src/models/laguna/dflash.hpp"
#include "src/models/laguna/kernels/device_model.hpp"
#include "src/models/laguna/kernels/dflash_ops.hpp"
#include "src/models/laguna/kernels/executor.hpp"
#include "src/models/laguna/tokenizer.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna {

namespace rocm {
class DFlashSession;
}
class SpeculativeDecoder;
struct SpeculativeStepResult;

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

/// Speculative decoding with the DFlash drafter (Model::AttachDrafter).
struct DrafterOptions {
  /// Rows per verify: the anchor plus block_rows - 1 drafts, 2 to 16. The
  /// measured default (DFLASH.md 6b-0 and 6d: a verify of 4 rows pays at every
  /// context depth, 8 only from about 8k, 16 not at short context).
  std::uint32_t block_rows{4};
};

/// Immutable host copy of a session's state at one position.
struct SessionSnapshot {
  rocm::ExecutorSnapshot kv;
  std::vector<float> logits;  ///< The frontier logits at `kv.position`.
  /// The drafter ring at `kv.position` when the session's drafter was in
  /// step; without it a restored session's drafter starts cold.
  std::optional<rocm::DrafterImage> drafter;
  std::uint32_t context{0};
  std::uint32_t chunk{0};
  [[nodiscard]] std::uint32_t position() const noexcept { return kv.position; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return kv.bytes() + logits.size() * sizeof(float) +
           (drafter ? drafter->bytes() : 0);
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
  /// device cannot hold another one. With a drafter attached the session also
  /// gets its DFlash session and decoder, warmed up for a verify of
  /// block_rows on the vector path, unless `with_drafter` is false (a plain
  /// session, as without a drafter).
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::string* error_msg = nullptr, bool with_drafter = true) const;

  /// Loads the DFlash drafter (`dflash` GGUF, decoder_arch laguna) for
  /// speculative decoding: binds and checks it against this model, checks
  /// that it and every session's drafter buffers fit the free device memory,
  /// and uploads it. Sessions created afterwards speculate. Call before
  /// creating sessions; refused when a drafter is already attached.
  [[nodiscard]] bool AttachDrafter(
      std::shared_ptr<const core::GgufReader> reader,
      const DrafterOptions& options, std::string* error_msg = nullptr);
  /// The same from bound weights (tests use synthetic drafters); the tensor
  /// data must stay valid for the call (the upload copies it).
  [[nodiscard]] bool AttachDrafterWeights(const DFlashWeights& weights,
                                          const DrafterOptions& options,
                                          std::string* error_msg = nullptr);
  [[nodiscard]] bool has_drafter() const noexcept {
    return drafter_ != nullptr;
  }
  [[nodiscard]] std::uint32_t draft_block_rows() const noexcept {
    return draft_rows_;
  }
  /// Device bytes of the uploaded drafter (0 without one).
  [[nodiscard]] std::size_t DrafterBytes() const noexcept;
  /// Upper bound of what one session's drafter side takes on the device
  /// (capture buffer, ring, scratch); 0 without a drafter.
  [[nodiscard]] std::size_t DrafterSessionBytes() const noexcept;

private:
  Model() = default;

  std::shared_ptr<const void> keepalive_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::string name_;
  std::uint32_t context_{0};
  std::uint32_t chunk_{0};
  std::uint32_t sessions_{1};
  std::unique_ptr<rocm::DeviceDrafter> drafter_;
  std::vector<std::uint32_t> drafter_taps_;
  std::int32_t drafter_mask_{0};
  std::uint32_t draft_rows_{0};
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
  /// `following`: prompt tokens known to come after these in a later call
  /// (the drafter skips rows no block can read).
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens,
                             std::string* error_msg = nullptr,
                             std::uint32_t following = 0);
  /// Appends one token (a decode step).
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);

  /// One session's next token in EvaluateBatch, and what became of it.
  struct BatchItem {
    Session* session{nullptr};
    std::int32_t token{0};
    bool completed{false};  ///< Output: the token was appended, Logits() set.
    std::string error;      ///< Output when !completed.
  };
  /// Appends one token to each session in shared forward passes (the
  /// projections, routing and experts run once over all rows, each row
  /// attends over its own session's caches); every session ends up exactly
  /// as Evaluate(token) leaves it, bit for bit (rows do not depend on their
  /// peers: see rocm::Executor::DecodeBatch). Sessions must be distinct and
  /// belong to one Model. A session that cannot take its token (invalid and
  /// not recoverable, token outside the vocabulary, context exhausted) is
  /// rejected on its own and nothing of it or its peers is touched. A device
  /// failure inside a shared pass invalidates and resets every session of
  /// that pass, as a failed Evaluate does its own session; sessions outside
  /// the batch are never touched.
  static void EvaluateBatch(std::span<BatchItem> items);

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

  // ---- speculative decoding (a Model with a drafter) ----------------------

  /// True when the session has a drafter (the model had one at creation).
  [[nodiscard]] bool speculative() const noexcept { return dflash_ != nullptr; }
  /// True while the drafter captures and drafts (not suspended or dropped).
  [[nodiscard]] bool drafting() const noexcept;
  /// Plain decode for a request that samples at random: the executor's taps
  /// are cleared so EvaluateBatch can take it. Undone by the next Prefill or
  /// DecodeSpeculative (cold ring) and by a snapshot restore or reset.
  void SuspendDrafter() noexcept;
  /// One greedy speculative step (SpeculativeDecoder::Step) from the
  /// frontier; `sampler` must not sample at random and is not advanced.
  /// Emits at most `max_tokens` tokens; the position stays within
  /// `context_limit`. A failure resets the session as a failed Evaluate does.
  [[nodiscard]] bool DecodeSpeculative(sampling::SamplerState& sampler,
                                       std::uint32_t max_tokens,
                                       std::uint32_t context_limit,
                                       bool stop_at_eos,
                                       SpeculativeStepResult* out,
                                       std::string* error_msg = nullptr);
  /// The session's DFlash session and decoder (null without a drafter).
  [[nodiscard]] rocm::DFlashSession* dflash() noexcept { return dflash_.get(); }
  [[nodiscard]] SpeculativeDecoder* decoder() noexcept {
    return decoder_.get();
  }

private:
  friend class Model;
  Session(std::shared_ptr<const Model> model,
          std::unique_ptr<rocm::Executor> executor);
  [[nodiscard]] bool Recover(std::string* error_msg);
  bool Fail(std::string* error_msg, const std::string& message);
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::string* error_msg,
                             std::uint32_t following = 0);
  /// Resets the drafter with the executor (after Recover or Invalidate).
  void RearmDrafter() noexcept;
  [[nodiscard]] bool ResumeDrafter(std::string* error_msg);

  std::shared_ptr<const Model> model_;
  std::unique_ptr<rocm::Executor> executor_;
  // Declared after the executor they borrow, so they are destroyed first.
  std::unique_ptr<rocm::DFlashSession> dflash_;
  std::unique_ptr<SpeculativeDecoder> decoder_;
  std::vector<float> logits_;
  bool invalid_{false};
  std::uint32_t failures_{0};
  std::uint32_t rebuilds_{0};
  bool injected_{false};
  std::string inject_stage_;
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_ENGINE_HPP_
