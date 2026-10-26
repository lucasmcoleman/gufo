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
//  * Snapshots (rule 5, phase 6e): SaveDrafter/RestoreDrafter carry the
//    drafter ring with the target's snapshot. Restoring without an image
//    restarts the ring empty at the restored position (a cold drafter: it
//    drafts from the rows injected after that point), never from the rows
//    the ring held before the restore.
//  * Batched decode (rule 6): Executor::DecodeBatch refuses an executor with
//    feature taps. A session serving a request that does not speculate (a
//    sampled request, phase 6e) is Suspend()ed: its taps are cleared so it
//    can join batched decode, and Resume() sets them again with a cold ring.

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
  /// the executor and drops the drafter. `following` is how many prompt
  /// tokens are known to come after these (a later Prefill): rows that no
  /// block after them can read are not injected.
  [[nodiscard]] bool Prefill(std::span<const std::int32_t> tokens,
                             LogitsRows rows, float* logits,
                             std::string* error_msg = nullptr,
                             std::uint32_t following = 0);

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
  /// and re-arms the drafter with an empty ring (resuming a suspended one).
  [[nodiscard]] bool Reset(std::string* error_msg = nullptr);

  /// Stops speculating for now (a request that samples): clears the
  /// executor's feature taps, so DecodeBatch takes it, and parks the drafter.
  /// Prefill, Verify and Commit then neither capture nor inject and Propose
  /// is refused (nothing is dropped). No-op when already suspended.
  void Suspend();
  /// Undoes Suspend(): sets the taps again and restarts the ring empty at
  /// position() (cold: the rows decoded while suspended were not captured).
  /// A dropped drafter stays dropped. False, still suspended, when the taps
  /// cannot be set.
  [[nodiscard]] bool Resume(std::string* error_msg = nullptr);
  [[nodiscard]] bool suspended() const noexcept { return suspended_; }
  /// Captures and injects now: active, not suspended.
  [[nodiscard]] bool capturing() const noexcept {
    return active_ && !suspended_;
  }

  /// Copies the drafter ring to `image` when it is in step with the target
  /// (capturing and committed at position()); false otherwise or on a copy
  /// failure (which drops the drafter), leaving `image` untouched.
  [[nodiscard]] bool SaveDrafter(DrafterImage* image,
                                 std::string* error_msg = nullptr);
  /// After the target was restored to a snapshot: replaces the ring with
  /// `image` (which must have been taken at the target's position()), or,
  /// with no image, restarts it empty at position() (cold). Resumes a
  /// suspended drafter and re-arms a dropped one. A drafter that cannot take
  /// the image is dropped (false); the target is never touched.
  [[nodiscard]] bool RestoreDrafter(const DrafterImage* image,
                                    std::string* error_msg = nullptr);
  /// Bytes SaveDrafter copies.
  [[nodiscard]] std::size_t drafter_image_bytes() const noexcept {
    return drafter_->kv().device_bytes();
  }

  /// Grows the scratch a verify of `verify_rows` on the vector path and a
  /// block of `block_rows` need (DFLASH 6d open item 2: the executor's own
  /// warm-up runs the batched path only), then Reset()s. Call on a fresh
  /// session.
  [[nodiscard]] bool Warmup(std::uint32_t verify_rows, std::uint32_t block_rows,
                            std::string* error_msg = nullptr);
  /// Device bytes of the session's own buffers (capture buffer, flag) plus
  /// the drafter's ring and scratch held now.
  [[nodiscard]] std::size_t device_bytes() const noexcept;

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
  std::vector<std::uint32_t> taps_;
  float* capture_{nullptr};
  int* nonfinite_{nullptr};  ///< device flag: a non-finite tap was injected
  bool active_{true};
  bool suspended_{false};
  bool poison_next_{false};
  std::string drafter_error_;
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_DFLASH_SESSION_HPP_
