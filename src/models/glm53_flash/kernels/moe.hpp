#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_MOE_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_MOE_HPP_
#include "src/models/glm53_flash/layer_ops.hpp"
namespace gufo::models::glm53_flash {
// Caller-owned scratch, aligned to 256 bytes. Zero indicates invalid dimensions
// or size overflow. No allocation or host synchronization in FfnForward.
[[nodiscard]] std::size_t FfnScratchBytes(const Config& config,
                                          std::uint32_t max_tokens) noexcept;
}  // namespace gufo::models::glm53_flash
#endif
