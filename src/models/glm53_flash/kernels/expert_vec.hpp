#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_EXPERT_VEC_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_EXPERT_VEC_HPP_

// Few-token routed expert matvec for decode and MTP verify (tokens <= 8),
// with gate and up in one launch over one activation quantization.
// Idea ported from mikealanni/gufo c338778f, 5a394170, 8808f1a0 and
// 5b505adf MatvecExpertsPairTokens (MIT, Mike Alani); this is a GLM-local
// kernel over executor scratch, not a patch of his code. Block arithmetic:
// expert_vec_math.hpp.
//
// Switch: GUFO_GLM53_EXPERT_VEC = 0 (default, the DS4 vendored mmvq path),
// 1 / all (gate/up pair and down), gateup, down.

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/kernels/expert_vec_math.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"

namespace gufo::models::glm53_flash::expert_vec {

/// Largest token count of one launch.
inline constexpr std::uint32_t kMaxTokens = 8;

/// The process-wide mode (Mode, ParseMode: expert_vec_math.hpp), read once from
/// the environment.
[[nodiscard]] Mode EnvMode() noexcept;

/// True if routed matrices of `type` with `cols` reduction elements have a
/// kernel here: IQ2_XXS, IQ2_S or IQ3_XXS with cols 2048 or 4096.
[[nodiscard]] bool Supported(core::GgmlType type, std::uint64_t cols) noexcept;

/// Scratch for `rows` quantized activation rows of `cols` values (0 when
/// cols is not a multiple of 32).
[[nodiscard]] std::size_t ScratchBytes(std::uint32_t rows,
                                       std::uint64_t cols) noexcept;

/// out[(t * routes + r) * w.rows + row] = W[ids[t * routes + r]][row] . x
/// for t < tokens, r < routes, where x is row t of `x` ([tokens][w.cols])
/// or, with `per_route`, row t * routes + r ([tokens * routes][w.cols]).
/// With `w1` set (same shape and encoding as `w0`, per_route false) the same
/// launch also fills out1 from w1: the gate/up pair. Activations are
/// quantized once into `scratch` (ScratchBytes of the activation rows). An
/// expert ID outside [0, w.experts) writes NaN for that route. Asynchronous
/// on `stream`; no allocation, no host synchronization, so it is safe inside
/// HIP graph capture. False (nothing launched) on an unsupported shape.
[[nodiscard]] bool Launch(const DeviceMatrix& w0, const DeviceMatrix* w1,
                          const float* x, const std::int32_t* ids, float* out0,
                          float* out1, std::uint32_t tokens,
                          std::uint32_t routes, bool per_route, void* scratch,
                          std::size_t scratch_bytes, Stream stream);

}  // namespace gufo::models::glm53_flash::expert_vec

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_EXPERT_VEC_HPP_
