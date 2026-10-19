// PrefillQuantGemm (kernels/prefill_gemm.hpp): the Q8_1-activation WMMA path
// GLM's dense projections take in prefill, against an FP64 CPU product.
//
// The kernel rounds each 32-value activation block to int8 with
// d = amax / 127 (round to nearest, |error| <= d / 2, plus <= 127 * d * 2^-11
// if d is stored as fp16). So each output y_i = sum_k W_ik x_k may move by at
// most 0.6 * sum_k |W_ik| * d_block(k), plus F32 accumulation error. That
// bound is checked pointwise; relative L2 is printed so the bound is seen to
// be tight (~1e-2 of the output) rather than vacuous. Shapes are GLM's column
// widths and types with fewer rows, so the CPU reference stays fast.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/prefill_gemm.hpp"

using gufo::core::GgmlType;
namespace glm = gufo::models::glm53_flash;

namespace {
void Hip(hipError_t e) {
  if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}

void Case(GgmlType type, std::size_t rows, std::size_t cols,
          std::size_t tokens, std::mt19937& rng) {
  const bool q6 = type == GgmlType::kQ6_K;
  const std::size_t block = q6 ? 256 : 32, block_bytes = q6 ? 210 : 34;
  const std::size_t row_bytes = cols / block * block_bytes;
  std::vector<std::uint8_t> bytes(row_bytes * rows);
  for (auto& v : bytes) v = static_cast<std::uint8_t>(rng());
  for (std::size_t o = 0; o < bytes.size(); o += block_bytes) {
    const std::uint16_t scale = 0x1400 + rng() % 1024;  // small finite fp16
    std::memcpy(bytes.data() + o + (q6 ? block_bytes - 2 : 0), &scale, 2);
  }
  std::vector<float> w(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    if (q6)
      gufo::quant::DequantizeQ6_K(bytes.data() + r * row_bytes, w.data() + r * cols, cols);
    else
      gufo::quant::DequantizeQ8_0(bytes.data() + r * row_bytes, w.data() + r * cols, cols);
  }
  std::vector<float> x(tokens * cols);
  std::uniform_real_distribution<float> d(-1, 1);
  for (auto& v : x) v = d(rng) * (rng() % 8 == 0 ? 8.0F : 1.0F);  // some outliers
  void *dw, *dx, *dy;
  Hip(hipMalloc(&dw, bytes.size()));
  Hip(hipMemcpy(dw, bytes.data(), bytes.size(), hipMemcpyHostToDevice));
  Hip(hipMalloc(&dx, x.size() * 4));
  Hip(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  Hip(hipMalloc(&dy, tokens * rows * 4));
  Hip(hipMemset(dy, 0xff, tokens * rows * 4));
  if (!glm::PrefillQuantGemm(type, dw, static_cast<float*>(dx), static_cast<float*>(dy),
                             tokens, rows, cols, nullptr))
    throw std::runtime_error("PrefillQuantGemm declined a shape it must take");
  std::vector<float> y(tokens * rows);
  Hip(hipDeviceSynchronize());
  Hip(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
  const double u = 0x1p-24, gamma = (cols + 1) * u / (1 - (cols + 1) * u);
  double diff = 0, norm = 0, worst = 0;
  std::vector<float> dblk(cols / 32);
  for (std::size_t t = 0; t < tokens; ++t) {
    const float* xt = x.data() + t * cols;
    for (std::size_t b = 0; b < cols / 32; ++b) {
      float amax = 0;
      for (std::size_t i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xt[b * 32 + i]));
      dblk[b] = amax / 127.0F;
    }
    for (std::size_t r = 0; r < rows; ++r) {
      const float* wr = w.data() + r * cols;
      double ref = 0, abs_sum = 0, quant = 0;
      for (std::size_t k = 0; k < cols; ++k) {
        ref += double(wr[k]) * xt[k];
        abs_sum += std::fabs(double(wr[k]) * xt[k]);
        quant += std::fabs(double(wr[k])) * dblk[k / 32];
      }
      const double got = y[t * rows + r];
      if (!std::isfinite(got)) throw std::runtime_error("nonfinite output");
      const double bound = 0.6 * quant + gamma * (abs_sum + 0.6 * quant) + 1e-30;
      const double err = std::fabs(got - ref);
      worst = std::fmax(worst, err / bound);
      if (err > bound)
        throw std::runtime_error("out of bound at t=" + std::to_string(t) + " r=" +
                                 std::to_string(r) + " got=" + std::to_string(got) +
                                 " ref=" + std::to_string(ref) + " bound=" + std::to_string(bound));
      diff += err * err;
      norm += ref * ref;
    }
  }
  std::printf("type=%d rows=%zu cols=%zu tokens=%zu rel_l2=%.3e worst_err/bound=%.3f\n",
              static_cast<int>(type), rows, cols, tokens, std::sqrt(diff / norm), worst);
  (void)hipFree(dw); (void)hipFree(dx); (void)hipFree(dy);
}
}  // namespace

int main() {
  if (glm::ExactPrefill()) {  // the WMMA path is off by request
    std::printf("SKIP GUFO_GLM53_EXACT_PREFILL is set\n");
    return 77;
  }
  try {
    std::mt19937 rng(682);
    // >= 64 tokens: every GLM type/width. 77 exercises partial token tiles.
    for (std::size_t tokens : {64, 77, 256}) {
      Case(GgmlType::kQ6_K, 256, 4096, tokens, rng);   // qkv, q_a, kv_a, shexp, dense gate/up
      Case(GgmlType::kQ6_K, 160, 8192, tokens, rng);   // kda out
      Case(GgmlType::kQ6_K, 128, 1536, tokens, rng);   // mla q_b
      Case(GgmlType::kQ6_K, 128, 16384, tokens, rng);  // mla out
      Case(GgmlType::kQ8_0, 128, 4096, tokens, rng);   // f_a, g_a, indexer k/gate
      Case(GgmlType::kQ8_0, 64, 4096, tokens, rng);    // beta
      Case(GgmlType::kQ8_0, 136, 1536, tokens, rng);   // indexer q_b (rows not /128)
    }
    // 24 tokens: only matrices >= 8M elements take the WMMA path.
    Case(GgmlType::kQ6_K, 2048, 4096, 24, rng);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
