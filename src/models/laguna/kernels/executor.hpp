#ifndef GUFO_MODELS_LAGUNA_KERNELS_EXECUTOR_HPP_
#define GUFO_MODELS_LAGUNA_KERNELS_EXECUTOR_HPP_

// Full-model GPU executor for Laguna-S-2.1: one session, one stream, prefill
// chunks and single-token decode over all layers.
//
//   h = embed(tokens)
//   per layer: n = RMSNorm(h) attn_norm; h' = h + Attn(n)
//              n = RMSNorm(h') ffn_norm;  h  = h' + Ffn(n)   (dense L0, MoE 1+)
//   logits = RMSNorm(h) output_norm x output (Q6_K)
//
// Forward() runs one chunk of at most max_chunk rows at the session's current
// position and advances it. A chunk's rows do not depend on how the sequence
// was split provided every call takes the same projection Path: rows <= 8 use
// the vector (MMVQ) kernels under Path::kAuto and more rows the batched (MMQ)
// kernels, which round activations differently (see ffn.hpp), so decode
// (single rows) and a large prefill differ by that rounding. Pin the path in
// ExecutorOptions to compare splits bit for bit.
//
// All scratch is held per executor (and per stream inside AttnOps/FfnOps);
// nothing is shared between executors. A failed Forward may have appended some
// layers' K/V and not others, so the executor is poisoned: every later call
// fails until Reset(), which rebuilds the KV caches. Calls rejected up front
// (bad token, context exhausted) touch nothing and do not poison.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/laguna/kernels/attention.hpp"
#include "src/models/laguna/kernels/device_model.hpp"
#include "src/models/laguna/kernels/ffn.hpp"

namespace gufo::models::laguna::rocm {

struct ExecutorOptions {
  std::uint32_t context{0};    ///< Positions of the session (full-layer KV).
  std::uint32_t max_chunk{0};  ///< Rows per Forward call, at most 32 when
                               ///< path is kVector.
  Path path{Path::kAuto};
  /// Run one max_chunk pass at creation so every scratch buffer reaches its
  /// final size and the memory the operators really take is measured.
  bool warmup{true};
  /// Test seam for the executor's, and the operators', scratch allocations.
  DeviceAllocator allocator{};
};

enum class LogitsRows { kNone, kLast, kAll };

class Executor;

/// Host copy of one session's KV state: every layer's retained history (full
/// layers up to the position, sliding rings with their retained_begin) and
/// the position. Restoring it into an executor of the same geometry (context
/// and chunk) makes every later Forward behave as it would have in the
/// original session.
struct ExecutorSnapshot {
  std::uint32_t position{0};
  std::vector<KvImage> layers;
  [[nodiscard]] std::size_t bytes() const noexcept {
    std::size_t total = 0;
    for (const KvImage& layer : layers) {
      total += layer.bytes();
    }
    return total;
  }
};

/// One session's next token in a batched decode step (Executor::DecodeBatch).
struct DecodeItem {
  Executor* executor{nullptr};
  std::int32_t token{0};
  /// Receives the [vocab] logits of the token, as Forward(kLast) would.
  float* logits{nullptr};
  /// Test hook: an attention stage after which this row's pass fails.
  /// DecodeBatch applies the first one in each group to that group only.
  std::string fail_stage;
  /// Outputs. `completed` means the token was appended and `logits` filled;
  /// otherwise `error` says why (see DecodeBatch for what was touched).
  bool completed{false};
  std::string error;
};

class Executor {
public:
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// Checks the memory budget against the free device memory first and fails
  /// with the budget in the message when it does not fit. `model` must
  /// outlive the executor.
  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, const ExecutorOptions& options,
      std::string* error_msg = nullptr);

  /// Empties the KV caches (position 0) and clears a poisoned state. Fails,
  /// leaving the executor poisoned, when the caches cannot be rebuilt.
  [[nodiscard]] bool Reset(std::string* error_msg = nullptr);

  /// Empties the KV caches (position 0) without releasing their memory when
  /// the executor is healthy; a poisoned executor takes the full Reset().
  [[nodiscard]] bool Clear(std::string* error_msg = nullptr);

  /// Copies the KV state to the host. Fails on a poisoned executor.
  [[nodiscard]] bool SaveSnapshot(ExecutorSnapshot* snapshot,
                                  std::string* error_msg = nullptr);
  /// Host bytes SaveSnapshot would copy now (the retained K/V rows).
  [[nodiscard]] std::size_t snapshot_bytes() const noexcept;
  /// Replaces the KV state with `snapshot`. Fails (leaving the executor
  /// poisoned when a layer was already overwritten) on a geometry mismatch.
  /// Refuses a poisoned executor: Reset() it first.
  [[nodiscard]] bool RestoreSnapshot(const ExecutorSnapshot& snapshot,
                                     std::string* error_msg = nullptr);

  /// Runs `tokens` (at most max_chunk) at positions position()..+T-1. With
  /// kAll `logits` receives [T][vocab] floats, with kLast one [vocab] row (of
  /// the last token), with kNone nothing. Returns false with a message on a
  /// bad argument (nothing touched) or a device failure (poisoned).
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             LogitsRows rows, float* logits,
                             std::string* error_msg = nullptr);

  /// Appends one token to each of several executors of the same model in
  /// shared forward passes: the projections, norms, routing and experts run
  /// once over all rows, and each row attends over its own executor's caches
  /// at its own position. Every row is computed as Forward() computes the
  /// same single token alone (Path::kAuto or kVector), so an item's result
  /// does not depend on its peers; executors with another path, model or
  /// chunk geometry run alone. Items are processed in groups of at most
  /// min(32, the first executor's max_chunk); the first executor of a group
  /// lends its stream and scratch, and the others' streams stay idle.
  ///
  /// Failure: an item that cannot run (poisoned or invalid executor, token
  /// outside the vocabulary, context exhausted, duplicate executor, null
  /// logits) is rejected on its own before anything is enqueued and nothing
  /// of it, or of its peers, is touched. A device failure inside a shared pass
  /// may have appended K/V to any participant's layers, so every executor of
  /// that group is poisoned, exactly as a failed Forward() poisons its own.
  /// Executors outside the group are never touched.
  static void DecodeBatch(std::span<DecodeItem> items);

  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint32_t context() const noexcept {
    return options_.context;
  }
  [[nodiscard]] std::uint32_t max_chunk() const noexcept {
    return options_.max_chunk;
  }
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  /// One layer's KV cache, for inspection (tests compare the retained-history
  /// bookkeeping of a restored session with the original).
  [[nodiscard]] const KvCache& kv_cache(std::uint32_t layer) const {
    return kv_.at(layer);
  }
  [[nodiscard]] std::size_t kv_bytes() const noexcept;
  /// Executor buffers plus the FFN operator scratch held now (no KV, no
  /// weights; the attention scratch has no accessor, see measured_warmup).
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;
  /// Free device memory dropped by this much across the warm-up pass (0
  /// without a warm-up or when the query is unavailable); the qfn_mmq arena
  /// is included.
  [[nodiscard]] std::size_t measured_warmup_bytes() const noexcept {
    return measured_warmup_;
  }

  /// Debug tap: after every layer the stream is synchronized and `tap` gets
  /// the layer index and the hidden rows [T][hidden] (device pointer). Taps
  /// hide stream races, so checks also run untapped.
  using LayerTap = std::function<void(std::uint32_t layer, const float* h,
                                      std::size_t count)>;
  void SetLayerTap(LayerTap tap) { layer_tap_ = std::move(tap); }
  /// Operator taps for stage checks.
  [[nodiscard]] AttnOps& attention_ops() noexcept { return *attn_; }
  [[nodiscard]] FfnOps& ffn_ops() noexcept { return *ffn_; }

private:
  Executor(const DeviceModel& model, const ExecutorOptions& options)
      : model_(&model), options_(options) {}
  struct Buffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
  };
  bool Alloc(Buffer* b, std::size_t bytes, std::string* error_msg);
  void Free(Buffer* b) noexcept;
  bool BuildCaches(std::string* error_msg);
  bool Run(std::uint32_t tokens, std::uint32_t start, LogitsRows rows,
           std::string* error_msg, std::span<Executor* const> batch = {});
  static void RunGroup(std::span<DecodeItem* const> group);
  [[nodiscard]] bool BatchCompatible(const Executor& lead) const noexcept;
  [[nodiscard]] std::string RejectDecode(std::int32_t token,
                                         const float* logits) const;
  bool Fail(std::string* error_msg, const std::string& message);

  const DeviceModel* model_;
  ExecutorOptions options_;
  DeviceAllocator allocator_;
  hipStream_t stream_{nullptr};
  std::unique_ptr<AttnOps> attn_;
  std::unique_ptr<FfnOps> ffn_;
  std::vector<KvCache> kv_;
  Buffer h_, n_, x_, o_, logits_, zero_ids_, rows_;
  std::vector<float> host_embed_;
  std::uint32_t position_{0};
  bool poisoned_{false};
  std::size_t measured_warmup_{0};
  LayerTap layer_tap_;
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_EXECUTOR_HPP_
