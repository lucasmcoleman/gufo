#ifndef GUFO_MODELS_GLM53_FLASH_LAYER_OPS_HPP_
#define GUFO_MODELS_GLM53_FLASH_LAYER_OPS_HPP_

// Contract between the executor (Lane A: layer loop, hyper-connections, KDA,
// embedding/head) and the attention/FFN operators (Lane B: MLA + DSA indexer,
// dense FFN, MoE). The executor owns the layer loop, device weight upload and
// cache allocation; Lane B owns the math behind MlaForward / FfnForward and
// the layout behind MlaCache. Changing a signature here needs both owners.
//
// Conventions
// - Activations between sub-layers are F32, row-major [tokens x hidden].
// - `x` passed to an operator is already RMS-normalised by the executor
//   (attn_norm / ffn_norm applied after the hyper-connection pre-mix); the
//   operator returns the sub-layer output `y`, before the post-mix.
// - Tokens of one call are consecutive positions [pos0, pos0 + tokens) of a
//   single sequence (phase 1: one sequence, no batching across sequences).
// - All work is enqueued on `ctx.stream`; nothing synchronises the host.
// - Taps: when `ctx.tap` is set, operators report intermediate device
//   buffers under llama.cpp's cb() names (`q_resid`, `kv_cmpr`, `indexer_k`,
//   `indexer_top_k`, `kqv_out`, `ffn_moe_out`, `ffn_shexp`, ...) so V3-V5 can
//   diff them against eval-callback dumps. Taps may synchronise; they are off
//   in production.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/config.hpp"

#if defined(ENGINE_ENABLE_HIP) || defined(GLM53_ROCM_BUILD)
#include <hip/hip_runtime_api.h>
#endif

namespace gufo::models::glm53_flash {

#if defined(ENGINE_ENABLE_HIP) || defined(GLM53_ROCM_BUILD)
using Stream = hipStream_t;
#else
using Stream = void*;
#endif

/// A weight resident on the device, still in its GGUF encoding. `experts`
/// stacks whole [rows x cols] matrices (routed experts, per-head k_b/v_b).
struct DeviceMatrix {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};  ///< Reduction dimension (GGUF ne[0]).
  std::uint64_t rows{0};  ///< Output dimension (GGUF ne[1]).
  std::uint64_t experts{1};
  std::size_t row_bytes{0};
  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
};

/// Device weights of one MLA + DSA layer. F32 vectors are plain floats.
struct MlaLayerDevice {
  DeviceMatrix q_a, q_b, kv_a, k_b, v_b, out;
  const float* q_a_norm{nullptr};
  const float* kv_a_norm{nullptr};
  DeviceMatrix idx_q_b, idx_k, idx_proj, idx_gate;
  const float* idx_k_norm{nullptr};
  const float* idx_k_norm_bias{nullptr};
  const float* idx_ape{nullptr};  ///< [kpool x indexer_dim]
};

/// Device weights of one FFN sub-layer: dense SwiGLU or MoE + shared expert.
struct FfnLayerDevice {
  bool dense{false};
  float clamp_exp{0.0F};    ///< SwiGLU clamp for routed experts and dense FFN.
  float clamp_shexp{0.0F};  ///< SwiGLU clamp for the shared expert.
  DeviceMatrix gate, up, down;        ///< Dense FFN (dense == true).
  const float* router{nullptr};       ///< [num_experts x hidden], F32
  const float* router_bias{nullptr};  ///< [num_experts], selection only
  DeviceMatrix gate_exps, up_exps, down_exps;
  DeviceMatrix shexp_gate, shexp_up, shexp_down;
};

/// Per-layer attention state of one sequence. Lane B owns the layout; the
/// executor only allocates `MlaCacheBytes` per MLA layer and resets it.
struct MlaCache {
  void* data{nullptr};
  std::size_t bytes{0};
  std::uint32_t capacity{0};  ///< Max positions this cache can hold.
};

using TapFn = std::function<void(std::string_view name, std::uint32_t layer,
                                 const float* device, std::size_t count)>;

struct OpContext {
  Stream stream{};
  void* scratch{nullptr};  ///< Executor-owned, at least OpScratchBytes().
  std::size_t scratch_bytes{0};
  const TapFn* tap{nullptr};
};

// ---- Lane B: implemented in kernels/{mla,indexer,moe}.hip.cpp ----

/// Bytes of one MLA layer's cache (latent rows + indexer state) for
/// `capacity` positions.
[[nodiscard]] std::size_t MlaCacheBytes(const Config& config,
                                        std::uint32_t capacity) noexcept;

/// Scratch the operators need for a call of up to `max_tokens` tokens.
[[nodiscard]] std::size_t OpScratchBytes(const Config& config,
                                         std::uint32_t max_tokens) noexcept;

/// Absorbed MLA with DSA top-k selection. Appends rows for positions
/// [pos0, pos0 + tokens) to `cache` and writes `y` [tokens x hidden].
/// Returns false (and leaves `y` unspecified) on an unsupported weight
/// format or capacity overflow.
[[nodiscard]] bool MlaForward(const OpContext& ctx, const Config& config,
                              std::uint32_t layer, const MlaLayerDevice& w,
                              MlaCache& cache, const float* x,
                              std::uint32_t tokens, std::uint32_t pos0,
                              float* y);

/// Dense SwiGLU or sigmoid-routed MoE (+ shared expert), with the clamps
/// from `w`. Writes `y` [tokens x hidden].
[[nodiscard]] bool FfnForward(const OpContext& ctx, const Config& config,
                              std::uint32_t layer, const FfnLayerDevice& w,
                              const float* x, std::uint32_t tokens, float* y);

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_LAYER_OPS_HPP_
