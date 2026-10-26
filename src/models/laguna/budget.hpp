#ifndef GUFO_MODELS_LAGUNA_BUDGET_HPP_
#define GUFO_MODELS_LAGUNA_BUDGET_HPP_

// Device memory budget of the Laguna-S-2.1 executor: weights, KV caches,
// scratch and logits for one session at a requested context and prefill chunk.
// Everything is computed from the tensor headers and the configuration, so it
// needs no GPU (laguna_logits --budget-only prints it without touching one).
//
// Weights and KV are exact (they equal DeviceModel::resident_bytes and the sum
// of KvCache::device_bytes). Scratch is an upper-bound model of what the
// operators and the executor allocate at the chunk size; the executor reports
// the allocation it actually measured after its warm-up pass next to it.

#include <cstddef>
#include <cstdint>
#include <string>

#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna {

struct BudgetRequest {
  std::uint32_t context{0};    ///< Positions of one session (full-layer KV).
  std::uint32_t max_chunk{0};  ///< Rows per Forward call (prefill chunk).
  /// Whether a call may request logits for every row of the chunk (else one).
  bool all_row_logits{true};
};

struct MemoryBudget {
  BudgetRequest request;
  std::size_t weights{0};      ///< Uploaded tensors incl. 512-col padding.
  std::size_t kv_full{0};      ///< 12 full layers x context, F16 K and V.
  std::size_t kv_swa{0};       ///< 36 rings of window + max_chunk.
  std::size_t activations{0};  ///< Residual/normed/branch rows + zero ids.
  std::size_t attention_scratch{0};
  std::size_t ffn_scratch{0};
  std::size_t logits{0};        ///< Device logits rows.
  std::size_t mmq_arena{0};     ///< qfn_mmq per-stream stack arena.
  std::size_t runtime{0};       ///< Fixed allowance: code objects, queues.
  std::size_t host_staging{0};  ///< Host: embeddings + logits staging.
  /// Kept free for driver command submission (gufo::hip::kDispatchReserve).
  std::size_t reserve{0};

  [[nodiscard]] std::size_t scratch() const noexcept {
    return activations + attention_scratch + ffn_scratch + logits + mmq_arena +
           runtime;
  }
  [[nodiscard]] std::size_t kv() const noexcept { return kv_full + kv_swa; }
  [[nodiscard]] std::size_t device_total() const noexcept {
    return weights + kv() + scratch();
  }
  /// Multi-line, human readable (GiB with two decimals, exact bytes).
  [[nodiscard]] std::string Format() const;
};

/// Device bytes one uploaded tensor occupies: payload plus the zero tail of
/// 512 columns after the last row that the MMQ tile loaders may read.
[[nodiscard]] std::size_t DeviceTensorBytes(const TensorRef& tensor) noexcept;

/// Bytes of every tensor the executor uploads (everything except
/// token_embd, which stays in the mapping and is gathered on the host).
[[nodiscard]] std::size_t WeightDeviceBytes(const ModelWeights& weights);

[[nodiscard]] MemoryBudget ComputeBudget(const ModelWeights& weights,
                                         const BudgetRequest& request);

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_BUDGET_HPP_
