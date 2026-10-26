#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_MOE_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_MOE_HPP_
#include "src/models/glm53_flash/layer_ops.hpp"
namespace gufo::models::glm53_flash {
// Caller-owned scratch, aligned to 256 bytes. Zero indicates invalid dimensions
// or size overflow. No allocation or host synchronization in FfnForward.
// FfnForward tap views borrow live output/scratch: callbacks must consume or
// enqueue a copy on ctx.stream before returning. Retaining the pointer is not
// a snapshot (the shared-expert addition updates ffn_moe_out in place).
// Negative or nonfinite clamps are rejected; zero disables clamping.
[[nodiscard]] std::size_t FfnScratchBytes(const Config& config,
                                          std::uint32_t max_tokens) noexcept;
// True if FfnForward on these weights and `tokens` only enqueues kernels on
// ctx.stream over caller scratch: no allocation, pool use or host sync, so a
// HIP graph may capture it (LayerOps::ffn_capture_safe). Dense SwiGLU and
// the shared expert always qualify at tokens <= 8; routed experts only on
// the expert_vec kernels (GUFO_GLM53_EXPERT_VEC=1), since the DS4 vendored
// path draws from its LIFO pool, whose arena may move when it grows.
[[nodiscard]] bool FfnCaptureSafe(const Config& config, std::uint32_t layer,
                                  const FfnLayerDevice& w,
                                  std::uint32_t tokens);
}  // namespace gufo::models::glm53_flash
#endif
