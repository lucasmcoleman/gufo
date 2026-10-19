#pragma once
// Prefill projections (see the crossover in PrefillQuantGemm) for quantized weights: Q8_1 activations
// into one WMMA GEMM that reads the weights once per tile column, instead of
// LaunchBatchedQuantGEMMFp32, which has no GLM-shape fast path past 8 tokens
// and re-streams the whole matrix once per 8 tokens (32 times per 256-token
// pass). This rounds the activations to 8 bits, like llama.cpp's MMQ;
// GUFO_GLM53_EXACT_PREFILL=1 keeps the exact FP32 path.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::models::glm53_flash {

inline bool ExactPrefill() {
  static const bool exact = [] {
    const char* v = std::getenv("GUFO_GLM53_EXACT_PREFILL");
    return v && *v && *v != '0';
  }();
  return exact;
}

/// Grow-only device buffer for the Q8_1 activations. GLM runs one stream, so
/// growing after a synchronize cannot race an earlier GEMM still reading it.
inline void* PrefillQ8Scratch(std::size_t bytes, hipStream_t stream) {
  static void* buffer = nullptr;
  static std::size_t capacity = 0;
  if (bytes > capacity) {
    if (hipStreamSynchronize(stream) != hipSuccess)
      return nullptr;
    if (buffer)
      (void)hipFree(buffer);
    buffer = nullptr;
    capacity = 0;
    if (hipMalloc(&buffer, bytes) != hipSuccess)
      return nullptr;
    capacity = bytes;
  }
  return buffer;
}

/// Y[tokens, rows] = X[tokens, cols] W^T through the prefill WMMA path when it
/// applies; returns false (nothing launched) when the caller should use its
/// exact path instead.
inline bool PrefillQuantGemm(core::GgmlType type, const void* w,
                             const float* x, float* y, std::size_t tokens,
                             std::size_t rows, std::size_t cols,
                             hipStream_t stream) {
  // Only the two dense types GLM's projections use, each of which the WMMA
  // path is known to handle (Qwen prefill runs both through it).
  // Crossover from glm53_flash_proj_bench on gfx1151 (GLM shapes, Q6_K and
  // Q8_0): the WMMA path wins for every shape from 64 tokens (>= 1.18x), and
  // from 24 tokens once the matrix has >= 8M elements (1.13-1.74x); below
  // that its quantize + launch overhead (~0.06-0.09 ms) loses to the FP32 path.
  const std::size_t elements = rows * cols;
  if (ExactPrefill() || cols % 256 != 0 ||
      (type != core::GgmlType::kQ8_0 && type != core::GgmlType::kQ6_K) ||
      !(tokens >= 64 || (tokens >= 24 && elements >= (std::size_t{8} << 20))))
    return false;
  void* q8 = PrefillQ8Scratch(
      ::gufo::hip::QuantizedActivationBytes(tokens, cols), stream);
  if (!q8)
    return false;
  ::gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(x, q8, tokens, cols,
                                                    stream);
  ::gufo::hip::LaunchBatchedQuantGEMMPreQuantized(type, w, q8, y, tokens, rows,
                                                  cols, stream);
  return true;
}

}  // namespace gufo::models::glm53_flash
