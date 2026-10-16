#ifndef GUFO_MODELS_GLM53_FLASH_REFERENCE_MOE_HPP_
#define GUFO_MODELS_GLM53_FLASH_REFERENCE_MOE_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gufo::models::glm53_flash::reference {

// F32 scalar oracles. Matrices are contiguous [outputs][inputs]. Callers
// dequantize GGUF rows before use; these routines do not repack model weights.
enum class RouterDenominator { kLlamaClamp, kHfEpsilon };
struct Routing {
  std::vector<std::uint32_t> ids;
  std::vector<float> weights;
};
// Bias affects selection only. Ties use ascending expert ID (upstream top-k
// tie ordering is unspecified). Default is the llama.cpp production contract.
Routing Route(const float* logits, const float* bias, std::uint32_t experts,
              std::uint32_t used, float scale, bool normalize = true,
              RouterDenominator denominator = RouterDenominator::kLlamaClamp);
float ClampedSwiGlu(float gate, float up, float clamp);
void DenseFfn(const float* x, const float* gate, const float* up,
              const float* down, std::size_t hidden, std::size_t intermediate,
              float clamp, float* y);
// gate_up stacks [expert][2*intermediate][hidden], gate before up;
// down stacks [expert][hidden][intermediate]. Shared expert is added by caller.
void RoutedFfn(const float* x, const float* gate_up, const float* down,
               std::size_t hidden, std::size_t intermediate,
               const Routing& routing, float clamp, float* y);

}  // namespace gufo::models::glm53_flash::reference
#endif
