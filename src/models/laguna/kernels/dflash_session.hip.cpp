// Target side of Laguna DFlash for one session. See dflash_session.hpp.
#include "src/models/laguna/kernels/dflash_session.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace gufo::models::laguna::rocm {
namespace {

void Assign(std::string* out, const std::string& message) {
  if (out != nullptr) {
    *out = message;
  }
}

/// Rows per drafter injection: the vector path's limit.
constexpr std::uint32_t kInjectRows = 32;

/// Sets *flag when any of the `count` floats is not finite. The drafter's
/// activation quantization maps a NaN row to zeros, so a non-finite tap would
/// otherwise pass silently (rule 8 asks for the drafter to be dropped).
__global__ void FlagNonFiniteKernel(const float* __restrict__ x,
                                    std::size_t count, int* flag) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < count; i += std::size_t{gridDim.x} * blockDim.x) {
    if (!isfinite(x[i])) {
      *flag = 1;
    }
  }
}

}  // namespace

DFlashSession::~DFlashSession() {
  drafter_.reset();  // synchronizes the stream
  if (capture_ != nullptr) {
    (void)hipStreamSynchronize(target_.stream());
    (void)hipFree(capture_);
  }
  if (nonfinite_ != nullptr) {
    (void)hipFree(nonfinite_);
  }
  // The executor decodes as a plain session again (DecodeBatch takes it).
  (void)target_.SetFeatureTaps({});
}

std::unique_ptr<DFlashSession> DFlashSession::Create(
    Executor& target, const DrafterWeights& w,
    std::span<const std::uint32_t> tap_layers, std::int32_t mask_token,
    std::string* error_msg) {
  const DeviceModel& model = target.model();
  if (tap_layers.empty() || tap_layers.size() != w.taps ||
      model.config().hidden_size != w.hidden) {
    Assign(error_msg,
           "laguna dflash session: the drafter's taps or hidden size do not "
           "match the target");
    return nullptr;
  }
  if (target.poisoned()) {
    Assign(error_msg, "laguna dflash session: the target is poisoned");
    return nullptr;
  }
  std::unique_ptr<DFlashSession> s(new DFlashSession(target, w));
  // Everything is allocated before the executor's taps change, so a failure
  // leaves the executor as it was.
  s->drafter_ = DrafterSession::Create(w, model.output(),
                                       model.weights().token_embd, mask_token,
                                       kInjectRows, target.stream(), error_msg);
  if (s->drafter_ == nullptr) {
    return nullptr;
  }
  const std::size_t bytes = std::size_t{target.max_chunk()} *
                            tap_layers.size() * w.hidden * sizeof(float);
  if (const hipError_t e =
          hipMalloc(reinterpret_cast<void**>(&s->capture_), bytes);
      e != hipSuccess) {
    s->capture_ = nullptr;
    Assign(error_msg, std::string("laguna dflash session: capture buffer: ") +
                          hipGetErrorString(e));
    return nullptr;
  }
  if (const hipError_t e =
          hipMalloc(reinterpret_cast<void**>(&s->nonfinite_), sizeof(int));
      e != hipSuccess ||
      hipMemset(s->nonfinite_, 0, sizeof(int)) != hipSuccess) {
    if (e != hipSuccess) {
      s->nonfinite_ = nullptr;
    }
    Assign(error_msg, "laguna dflash session: flag allocation failed");
    return nullptr;
  }
  if (!target.SetFeatureTaps(tap_layers, error_msg)) {
    return nullptr;
  }
  s->taps_.assign(tap_layers.begin(), tap_layers.end());
  s->drafter_->kv().Reset(target.position());
  return s;
}

void DFlashSession::Drop(const std::string& why) {
  if (active_) {
    active_ = false;
    drafter_error_ = why;
  }
}

void DFlashSession::InjectCaptured(std::uint32_t start, std::uint32_t from,
                                   std::uint32_t end) {
  if (!capturing() || from >= end) {
    return;
  }
  const std::size_t width = target_.feature_width();
  if (poison_next_) {
    poison_next_ = false;
    // All-ones bytes are a NaN in every float.
    if (hipMemsetAsync(capture_ + std::size_t{from - start} * width, 0xFF,
                       width * sizeof(float), target_.stream()) != hipSuccess) {
      Drop("laguna dflash session: test poison failed");
      return;
    }
  }
  {
    const std::size_t count = std::size_t{end - from} * width;
    const auto blocks =
        static_cast<unsigned>(std::min<std::size_t>((count + 255) / 256, 1024));
    FlagNonFiniteKernel<<<blocks, 256, 0, target_.stream()>>>(
        capture_ + std::size_t{from - start} * width, count, nonfinite_);
    if (hipGetLastError() != hipSuccess) {
      Drop("laguna dflash session: tap check launch failed");
      return;
    }
  }
  std::string error;
  for (std::uint32_t p = from; p < end; p += kInjectRows) {
    const std::uint32_t rows = std::min(kInjectRows, end - p);
    if (!drafter_->InjectDevice(capture_ + std::size_t{p - start} * width, rows,
                                p, Path::kVector, &error)) {
      Drop("laguna dflash session: drafter injection failed: " + error);
      return;
    }
  }
}

bool DFlashSession::Prefill(std::span<const std::int32_t> tokens,
                            LogitsRows rows, float* logits,
                            std::string* error_msg, std::uint32_t following) {
  if (tokens.empty()) {
    return true;
  }
  const std::uint32_t P = target_.position();
  if (target_.poisoned()) {
    Assign(error_msg,
           "laguna executor: a previous call failed; Reset() is required");
    return false;
  }
  if (std::uint64_t{P} + tokens.size() > target_.context()) {
    Assign(error_msg, "laguna dflash session: context of " +
                          std::to_string(target_.context()) + " exhausted");
    return false;
  }
  const std::uint64_t vocab = target_.model().weights().token_embd.rows;
  for (const std::int32_t t : tokens) {
    if (t < 0 || static_cast<std::uint64_t>(t) >= vocab) {
      Assign(error_msg, "laguna executor: token id outside the vocabulary");
      return false;
    }
  }
  if (rows != LogitsRows::kNone && logits == nullptr) {
    Assign(error_msg, "laguna executor: logits output missing");
    return false;
  }
  const auto end = static_cast<std::uint32_t>(P + tokens.size());
  // A block at `end` (or, when `following` prompt tokens come after these,
  // at the end of the prompt) reads the window - 1 rows before it; older rows
  // need no injection. A ring out of step with the target (a dropped and
  // re-armed drafter, a restored target) starts over at the prompt rows it
  // can see. A ring with nothing to inject now restarts at `end`.
  const std::uint64_t horizon = std::uint64_t{end} + following;
  const std::uint32_t needed =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          horizon >= w_.window ? horizon - (w_.window - 1) : 0, end));
  std::uint32_t inject_from = P;
  if (capturing()) {
    DrafterKv& kv = drafter_->kv();
    if (!kv.valid() || kv.committed() != P || needed > P) {
      inject_from = std::max(P, needed);
      kv.Reset(inject_from);
    }
  }
  const std::uint32_t chunk = target_.max_chunk();
  const std::size_t V = target_.model().config().vocab_size;
  for (std::uint32_t s = P; s < end; s += chunk) {
    const std::uint32_t T = std::min(chunk, end - s);
    const bool last = s + T == end;
    LogitsRows r = LogitsRows::kNone;
    float* out = nullptr;
    if (rows == LogitsRows::kAll) {
      r = LogitsRows::kAll;
      out = logits + std::size_t{s - P} * V;
    } else if (rows == LogitsRows::kLast && last) {
      r = LogitsRows::kLast;
      out = logits;
    }
    // Rows below inject_from need no capture.
    const bool capture = capturing() && s + T > inject_from;
    ForwardArgs args;
    args.features = capture ? capture_ : nullptr;
    if (!target_.Forward(tokens.subspan(s - P, T), r, out, args, error_msg)) {
      if (target_.poisoned()) {
        Drop("laguna dflash session: the target failed");
      }
      return false;
    }
    if (capture) {
      InjectCaptured(s, std::max(s, inject_from), s + T);
    }
  }
  return true;
}

bool DFlashSession::Verify(std::span<const std::int32_t> tokens, float* logits,
                           std::string* error_msg) {
  const std::uint32_t P = target_.position();
  if (capturing() &&
      (!drafter_->kv().valid() || drafter_->kv().committed() != P)) {
    Drop("laguna dflash session: the drafter is out of step with the target");
  }
  ForwardArgs args;
  args.path = Path::kVector;
  args.features = capturing() ? capture_ : nullptr;
  if (!target_.Forward(tokens, LogitsRows::kAll, logits, args, error_msg)) {
    if (target_.poisoned()) {
      Drop("laguna dflash session: the target failed");
    }
    return false;
  }
  InjectCaptured(P, P, P + static_cast<std::uint32_t>(tokens.size()));
  return true;
}

bool DFlashSession::Commit(std::uint32_t position, std::string* error_msg) {
  if (!target_.RewindTo(position, error_msg)) {
    return false;
  }
  if (capturing()) {
    std::string error;
    if (!drafter_->kv().RewindTo(position, &error)) {
      Drop("laguna dflash session: drafter rewind refused: " + error);
    }
  }
  return true;
}

bool DFlashSession::Propose(std::int32_t anchor, std::uint32_t block_rows,
                            std::vector<std::int32_t>* drafts,
                            std::vector<float>* logits,
                            std::string* error_msg) {
  if (!active_) {
    Assign(error_msg,
           "laguna dflash session: no drafter (" + drafter_error_ + ")");
    return false;
  }
  if (suspended_) {
    Assign(error_msg, "laguna dflash session: the drafter is suspended");
    return false;
  }
  if (target_.poisoned()) {
    Assign(error_msg, "laguna dflash session: the target is poisoned");
    return false;
  }
  // Bad arguments are refused here, touching nothing; past this point any
  // failure is the drafter's.
  if (drafts == nullptr || block_rows < 2 || block_rows > w_.block_size ||
      anchor < 0 ||
      static_cast<std::uint64_t>(anchor) >=
          target_.model().weights().token_embd.rows) {
    Assign(error_msg, "laguna dflash session: bad anchor or block length");
    return false;
  }
  if (drafter_->kv().committed() != target_.position()) {
    Drop("laguna dflash session: the drafter is out of step with the target");
    Assign(error_msg, drafter_error_);
    return false;
  }
  std::string error;
  if (!drafter_->Propose(anchor, block_rows, drafts, logits, Path::kBatched,
                         &error)) {
    Drop("laguna dflash session: drafter failed: " + error);
    Assign(error_msg, drafter_error_);
    return false;
  }
  // A non-finite tap row injected since the last Reset (attention-sink
  // overflow, rule 8; flagged on the device, read after Propose's own
  // synchronize), or a block row with no finite maximum.
  int flag = 0;
  if (hipMemcpy(&flag, nonfinite_, sizeof(int), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    Drop("laguna dflash session: tap check read failed");
    Assign(error_msg, drafter_error_);
    return false;
  }
  if (flag != 0 || std::any_of(drafts->begin(), drafts->end(),
                               [](std::int32_t d) { return d < 0; })) {
    Drop("laguna dflash session: non-finite tap or draft logits");
    Assign(error_msg, drafter_error_);
    return false;
  }
  return true;
}

bool DFlashSession::Reset(std::string* error_msg) {
  const bool ok =
      target_.poisoned() ? target_.Reset(error_msg) : target_.Clear(error_msg);
  if (!ok) {
    Drop("laguna dflash session: the target could not be reset");
    return false;
  }
  if (suspended_) {
    if (!target_.SetFeatureTaps(taps_, error_msg)) {
      return false;
    }
    suspended_ = false;
  }
  drafter_->kv().Reset(0);
  if (hipMemset(nonfinite_, 0, sizeof(int)) != hipSuccess) {
    Drop("laguna dflash session: tap check reset failed");
    return false;
  }
  active_ = true;
  drafter_error_.clear();
  poison_next_ = false;
  return true;
}

}  // namespace gufo::models::laguna::rocm

namespace gufo::models::laguna::rocm {

void DFlashSession::Suspend() {
  if (suspended_) {
    return;
  }
  // Clearing the taps cannot fail (an empty list is always valid).
  (void)target_.SetFeatureTaps({});
  suspended_ = true;
}

bool DFlashSession::Resume(std::string* error_msg) {
  if (!suspended_) {
    return true;
  }
  if (!target_.SetFeatureTaps(taps_, error_msg)) {
    return false;
  }
  suspended_ = false;
  if (active_) {
    drafter_->kv().Reset(target_.position());
    if (hipMemset(nonfinite_, 0, sizeof(int)) != hipSuccess) {
      Drop("laguna dflash session: tap check reset failed");
    }
  }
  return true;
}

bool DFlashSession::SaveDrafter(DrafterImage* image, std::string* error_msg) {
  if (image == nullptr || !capturing() || !drafter_->kv().valid() ||
      drafter_->kv().committed() != target_.position()) {
    Assign(error_msg, "laguna dflash session: the drafter is not in step");
    return false;
  }
  // A pending non-finite tap would drop the drafter at the next block; an
  // image of that ring must not outlive the flag.
  int flag = 0;
  if (hipMemcpy(&flag, nonfinite_, sizeof(int), hipMemcpyDeviceToHost) !=
          hipSuccess ||
      flag != 0) {
    Drop("laguna dflash session: non-finite tap or tap check read failure");
    Assign(error_msg, drafter_error_);
    return false;
  }
  std::string error;
  if (!drafter_->kv().Export(image, target_.stream(), &error)) {
    Drop("laguna dflash session: drafter export failed: " + error);
    Assign(error_msg, drafter_error_);
    return false;
  }
  return true;
}

bool DFlashSession::RestoreDrafter(const DrafterImage* image,
                                   std::string* error_msg) {
  if (target_.poisoned()) {
    Assign(error_msg, "laguna dflash session: the target is poisoned");
    return false;
  }
  if (suspended_) {
    if (!target_.SetFeatureTaps(taps_, error_msg)) {
      return false;
    }
    suspended_ = false;
  }
  // Re-arm: whatever the ring held before belongs to another history.
  active_ = true;
  drafter_error_.clear();
  DrafterKv& kv = drafter_->kv();
  kv.Reset(target_.position());
  if (hipMemset(nonfinite_, 0, sizeof(int)) != hipSuccess) {
    Drop("laguna dflash session: tap check reset failed");
    Assign(error_msg, drafter_error_);
    return false;
  }
  if (image == nullptr) {
    return true;
  }
  if (image->committed != target_.position()) {
    Drop("laguna dflash session: the drafter image is of another position");
    Assign(error_msg, drafter_error_);
    return false;
  }
  std::string error;
  if (!kv.Import(*image, target_.stream(), &error)) {
    kv.Reset(target_.position());
    Drop("laguna dflash session: drafter import failed: " + error);
    Assign(error_msg, drafter_error_);
    return false;
  }
  return true;
}

bool DFlashSession::Warmup(std::uint32_t verify_rows, std::uint32_t block_rows,
                           std::string* error_msg) {
  if (target_.position() != 0 || verify_rows == 0 || verify_rows > 32 ||
      verify_rows > target_.max_chunk() || block_rows < 2 ||
      block_rows > w_.block_size) {
    Assign(error_msg, "laguna dflash session: bad warm-up");
    return false;
  }
  const std::size_t V = target_.model().config().vocab_size;
  std::vector<float> logits(std::size_t{verify_rows} * V);
  const std::vector<std::int32_t> tokens(verify_rows, 0);
  std::vector<std::int32_t> drafts;
  if (!Verify(tokens, logits.data(), error_msg)) {
    return false;
  }
  if (capturing() && !Propose(0, block_rows, &drafts, nullptr, error_msg)) {
    return false;
  }
  return Reset(error_msg);
}

std::size_t DFlashSession::device_bytes() const noexcept {
  return std::size_t{target_.max_chunk()} * taps_.size() * w_.hidden *
             sizeof(float) +
         sizeof(int) + drafter_->kv().device_bytes() +
         drafter_->ops().scratch_bytes();
}

}  // namespace gufo::models::laguna::rocm
