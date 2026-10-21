#include <algorithm>

#include "src/models/glm53_flash/kernels/moe.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"

namespace gufo::models::glm53_flash {

std::size_t OpScratchBytes(const Config& config,
                           std::uint32_t max_tokens) noexcept {
  const auto ffn = FfnScratchBytes(config, max_tokens);
  const auto mla = MlaScratchBytes(config, max_tokens);
  // Operators reuse the same executor-owned allocation sequentially.
  // A failed size query must not be concealed by a successful other query.
  return ffn && mla ? std::max(ffn, mla) : 0;
}

}  // namespace gufo::models::glm53_flash
