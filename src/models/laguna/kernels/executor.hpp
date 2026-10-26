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

  /// Runs `tokens` (at most max_chunk) at positions position()..+T-1. With
  /// kAll `logits` receives [T][vocab] floats, with kLast one [vocab] row (of
  /// the last token), with kNone nothing. Returns false with a message on a
  /// bad argument (nothing touched) or a device failure (poisoned).
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             LogitsRows rows, float* logits,
                             std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint32_t context() const noexcept {
    return options_.context;
  }
  [[nodiscard]] std::uint32_t max_chunk() const noexcept {
    return options_.max_chunk;
  }
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
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
           std::string* error_msg);
  bool Fail(std::string* error_msg, const std::string& message);

  const DeviceModel* model_;
  ExecutorOptions options_;
  DeviceAllocator allocator_;
  hipStream_t stream_{nullptr};
  std::unique_ptr<AttnOps> attn_;
  std::unique_ptr<FfnOps> ffn_;
  std::vector<KvCache> kv_;
  Buffer h_, n_, x_, o_, logits_, zero_ids_;
  std::vector<float> host_embed_;
  std::uint32_t position_{0};
  bool poisoned_{false};
  std::size_t measured_warmup_{0};
  LayerTap layer_tap_;
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_EXECUTOR_HPP_
