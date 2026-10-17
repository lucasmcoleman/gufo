#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_MMQ_IQ_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_MMQ_IQ_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm53_flash::rocm {

/// Asynchronous, allocation-free IQ2_XXS/IQ2_S/IQ3_XXS correctness primitives.
/// All operands are device-accessible and must remain alive through stream
/// completion. Outputs must not overlap inputs. No full dequantized weight
/// tensor or Q8 activation scratch is allocated by routed multiplication.
/// Host validation rejects null/misaligned pointers, unsupported types, zero
/// dimensions, overflow and cols not divisible by 256. A true return means
/// launch accepted, not completed; callers check completion and finite outputs.
/// Nonfinite inputs/weights propagate; they are never silently sanitized.
///
/// Dequantize contiguous packed rows into row-major F32 [rows][cols].
bool IqDequantize(core::GgmlType type, const void* weights, float* output,
                  std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

/// W=[experts][rows][cols] in canonical GGUF packed order,
/// expert_ids=[tokens][routes], output=[tokens][routes][rows], all contiguous.
/// Input is F32 [tokens][cols] when broadcast_input is true, otherwise
/// [tokens][routes][cols]. Each output is a dot product, with no route
/// weighting or reduction across routes. tokens=1 is routed matvec; larger
/// token counts implement routed GEMM with the same arithmetic. Invalid device
/// expert IDs write NaN for the affected output route, without reading out of
/// bounds.
bool IqRoutedMatmul(core::GgmlType type, const void* weights,
                    const float* input, const std::int32_t* expert_ids,
                    float* output, std::uint32_t tokens, std::uint32_t routes,
                    std::uint32_t experts, std::uint32_t rows,
                    std::uint32_t cols, bool broadcast_input,
                    hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_MMQ_IQ_HPP_
