#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
void QwenGpuExecutor::CheckPrefillCancellation() const {
  if (cancellation_check_ && cancellation_check_())
    throw std::runtime_error("Qwen prefill cancelled");
}

void QwenGpuExecutor::PrefillLayerCheckpoint(std::uint32_t layer) {
  CheckPrefillCancellation();
  // Keep one four-layer group ahead of the GPU. Without backpressure, AR
  // can enqueue the whole prompt before the client disconnects; checking
  // cancellation on the host then cannot stop seconds of abandoned work.
  if (!cancellation_check_ || layer == 0 || layer % 4 != 0)
    return;
  for (auto& event : prefill_events_) {
    if (!event)
      HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  }
  const auto slot = (layer / 4 - 1) % prefill_events_.size();
  HIP_CHECK(hipEventRecord(prefill_events_[slot], arena_.stream));
  if (layer < 8)
    return;
  hipError_t status;
  while ((status = hipEventQuery(prefill_events_[1 - slot])) ==
         hipErrorNotReady) {
    CheckPrefillCancellation();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  HIP_CHECK(status);
}

tokenization::TokenId QwenGpuExecutor::ForwardSmallPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  // A few rows cost about as much as one prefill pass over every weight; the
  // verification forward reads each weight once per row group instead.
  const auto predictions =
      ForwardDecodeEquivalentVerificationChunk(prompt_tokens, start_pos, false);
  if (capture_prompt_hidden_)
    h_prompt_hidden_.insert(h_prompt_hidden_.end(),
                            h_verification_hidden_.begin(),
                            h_verification_hidden_.end());
  h_verification_hidden_.clear();
  last_verification_rows_ = 0;
  if (!compute_logits)
    return 0;
  const std::size_t vocab_size = weights_.config.vocab_size;
  auto scratch = arena_.GetScratchView();
  HIP_CHECK(hipMemcpyAsync(
      scratch.decode.logits.data(),
      d_verification_logits_ + (prompt_tokens.size() - 1) * vocab_size,
      vocab_size * sizeof(float), hipMemcpyDeviceToDevice, arena_.stream));
  return predictions.back();
}

tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  CheckPrefillCancellation();
  if (prompt_tokens.empty()) {
    return 0;
  }
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  if (capture_prompt_hidden_) {
    h_prompt_hidden_.clear();
    const std::size_t captured_layers =
        std::max<std::size_t>(arena_.GetTargetLayerCapture().size(), 1U);
    h_prompt_hidden_.reserve(prompt_tokens.size() * captured_layers *
                             weights_.config.hidden_size);
  }

  const std::size_t end_pos =
      static_cast<std::size_t>(start_pos) + prompt_tokens.size();
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }

  tokenization::TokenId next_token = 0;
  for (std::size_t offset = 0; offset < prompt_tokens.size();
       offset += arena_.GetMaxBatch()) {
    CheckPrefillCancellation();
    const std::size_t chunk_size = std::min<std::size_t>(
        arena_.GetMaxBatch(), prompt_tokens.size() - offset);
    const bool is_last = offset + chunk_size == prompt_tokens.size();
    const auto chunk = prompt_tokens.subspan(offset, chunk_size);
    const auto chunk_pos = start_pos + static_cast<std::uint32_t>(offset);
    next_token =
        chunk_size <= kSmallPromptChunkRows && weights_.config.IsMoE()
            ? ForwardSmallPromptChunk(chunk, chunk_pos,
                                      compute_logits && is_last)
            : ForwardPromptChunk(chunk, chunk_pos, compute_logits && is_last);
  }
  return next_token;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
