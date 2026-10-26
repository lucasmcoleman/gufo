#ifndef GUFO_MODELS_LAGUNA_KERNELS_DFLASH_SESSION_HPP_
#define GUFO_MODELS_LAGUNA_KERNELS_DFLASH_SESSION_HPP_

// The target side of Laguna DFlash for one session (phase 6d of
// docs/models/laguna-s-2.1/DFLASH.md): a target Executor (borrowed) with its
// DrafterSession (owned), on the executor's stream.
//
//  * Taps: every prefill, decode and verify row's target layer outputs
//    (Executor::SetFeatureTaps, F32) are captured on the device into this
//    session's buffer and injected into the drafter's ring in stream order;
//    nothing synchronizes for them. Prefill injects only the rows a block at
//    the end of the prompt can read (the last window - 1).
//  * Verify runs on the pinned vector path (ForwardArgs::path), so its rows
//    equal the same tokens decoded alone bit for bit; prefill keeps the
//    executor's own path. Drafter injections are pinned to the vector path
//    too, so a drafter K/V row does not depend on the chunking or on which
//    call injected it. Drafter blocks are pinned to the batched path: it is
//    measured 9.2 to 9.8 ms per block of 4 to 16 rows against 8.0 to 29 ms on
//    the vector path (6d), and one pinned path keeps a shorter block the
//    prefix of a longer one.
//  * Commit is a rewind of both (rules 1 and 2): Executor::RewindTo and
//    DrafterKv::RewindTo, each through its retained-history validation.
//  * Failures (rule 4): a target call that fails after its K/V append poisons
//    the executor (until Reset) and drops the drafter with it. A drafter-side
//    failure (device error, rewind refused, a non-finite tap row, rule 8,
//    flagged on the device as it is injected) drops only the drafter: the
//    target's result stands and the session continues as plain decode. A
//    dropped drafter stays dropped until Reset().
//  * Not here (design notes for 6e): snapshots of the drafter ring (rule 5;
//    a target RestoreSnapshot desynchronizes the drafter, so it is dropped by
//    the next call that finds it out of step) and batched decode (rule 6:
//    Executor::DecodeBatch refuses an executor with feature taps).

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/laguna/kernels/dflash_ops.hpp"
#include "src/models/laguna/kernels/executor.hpp"

namespace gufo::models::laguna::rocm {

class DFlashSession {
public:
  /// `target` and `w` must outlive the session. `tap_layers` are the target
  /// layer outputs the drafter reads, in its feature order
  /// (DFlashConfig::tap_layer(0..taps-1)); they are set on the executor.
  /// `mask_token` is the drafter's MASK id. The drafter's head and embedding
  /// are the target model's. Allocates the capture buffer ([max_chunk] rows
  /// of taps x hidden floats) and the drafter's ring and buffers. Nullptr with
  /// a message on a mismatch or an allocation failure.
  [[nodiscard]] static std::unique_ptr<DFlashSession> Create(
      Executor& target, const DrafterWeights& w,
      std::span<const std::uint32_t> tap_layers, std::int32_t mask_token,
      std::string* error_msg = nullptr);
  ~DFlashSession();
  DFlashSession(const DFlashSession&) = delete;
  DFlashSession& operator=(const DFlashSession&) = delete;

  /// Appends `tokens` (any number, chunks of the executor's max_chunk on its
  /// own path) and injects their taps. With kAll `logits` gets [n][vocab],
  /// with kLast the last row. Refused up front, touching nothing, on a token
  /// outside the vocabulary or a context overflow. A target failure poisons
  /// the executor and drops the drafter.
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens,
                             LogitsRows rows, float* logits,
                             std::string* error_msg = nullptr);

  /// Runs `tokens` (1 to 32 rows: a decode token, or a verify block
  /// [anchor, drafts...]) at position() on the vector path, logits [T][vocab]
  /// to `logits`, and injects every row's taps (rejected rows too, as the
  /// fork does; Commit trims them). Failure as Prefill.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens, float* logits,
                            std::string* error_msg = nullptr);

  /// Keeps the rows before `position` (n + accepted + 1 after a verify at n)
  /// and gives up the rest, in the target and in the drafter. Refused,
  /// touching nothing, when the target refuses the rewind. A drafter that
  /// refuses is dropped (the target's rewind stands).
  [[nodiscard]] bool Commit(std::uint32_t position,
                            std::string* error_msg = nullptr);

  /// Drafts block_rows - 1 tokens after `anchor`, the token at position()
  /// (sampled, not yet given to the target), on the batched path. Fails when
  /// the drafter is dropped; a drafter failure here (including a non-finite
  /// draft row) drops it and leaves the target untouched.
  [[nodiscard]] bool Propose(std::int32_t anchor, std::uint32_t block_rows,
                             std::vector<std::int32_t>* drafts,
                             std::vector<float>* logits = nullptr,
                             std::string* error_msg = nullptr);

  /// A fresh request: resets the target (Reset() when poisoned, else Clear())
  /// and re-arms the drafter with an empty ring.
  [[nodiscard]] bool Reset(std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t position() const noexcept {
    return target_.position();
  }
  [[nodiscard]] bool drafter_active() const noexcept { return active_; }
  /// Why the drafter was dropped ("" while active).
  [[nodiscard]] const std::string& drafter_error() const noexcept {
    return drafter_error_;
  }
  [[nodiscard]] Executor& target() noexcept { return target_; }
  [[nodiscard]] DrafterSession& drafter() noexcept { return *drafter_; }
  /// The capture of the last Prefill chunk or Verify (rows x taps x hidden),
  /// device memory; read it on the executor's stream or after a synchronize.
  [[nodiscard]] const float* captured_features() const noexcept {
    return capture_;
  }

  /// Test hook: the next capture's first row is overwritten with NaN before
  /// it is injected (a non-finite tap, rule 8).
  void PoisonNextCaptureForTesting() { poison_next_ = true; }

private:
  DFlashSession(Executor& target, const DrafterWeights& w)
      : target_(target), w_(w) {}
  void Drop(const std::string& why);
  /// Injects rows [from, end) of a call whose rows start at `start` from the
  /// capture buffer; drops the drafter on failure.
  void InjectCaptured(std::uint32_t start, std::uint32_t from,
                      std::uint32_t end);

  Executor& target_;
  const DrafterWeights& w_;
  std::unique_ptr<DrafterSession> drafter_;
  float* capture_{nullptr};
  int* nonfinite_{nullptr};  ///< device flag: a non-finite tap was injected
  bool active_{true};
  bool poison_next_{false};
  std::string drafter_error_;
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_DFLASH_SESSION_HPP_
