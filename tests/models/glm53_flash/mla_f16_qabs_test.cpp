// GPU test: mla::ProjectHeadsHalf (GUFO_GLM53_MLA_F16_QABS) stores exactly
// fp16(ProjectHeads(x) * scale), bit for bit, for the Q6_K one- and
// two-block head stacks (k_b is [256 -> 512] per head on GLM-5.3-Flash), at
// token counts across the 32-token tile, and rejects other encodings.
// Written in the source-only round (2026-10-26); first run is the GPU round.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/glm53_flash/kernels/mla.hpp"

namespace glm = gufo::models::glm53_flash;
using gufo::core::GgmlType;

namespace {

int g_failures = 0;

void Check(const std::string& what, bool ok) {
  if (!ok) {
    ++g_failures;
    std::cerr << "FAIL: " << what << "\n";
  }
}

void Hip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

void Run(std::uint32_t cols, std::uint32_t tokens, hipStream_t stream) {
  constexpr std::uint32_t kHeads = 4;
  constexpr std::uint32_t kRows = 72;  // not a multiple of the 8-row block
  const std::size_t row_bytes = (cols / 256) * 210;
  std::mt19937 rng(5 + cols + tokens);
  std::vector<std::uint8_t> w(std::size_t{kHeads} * kRows * row_bytes);
  for (auto& v : w) {
    v = static_cast<std::uint8_t>(rng());
  }
  // Q6_K d (last 2 bytes of each 210-byte block): a moderate half.
  for (std::size_t b = 0; b < w.size() / 210; ++b) {
    const auto d = static_cast<std::uint16_t>(0x1800 + (rng() % 1024));
    std::memcpy(w.data() + (b * 210) + 208, &d, sizeof(d));
  }
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(std::size_t{kHeads} * tokens * cols);
  for (auto& v : x) {
    v = normal(rng);
  }
  const std::size_t out_count = std::size_t{kHeads} * tokens * kRows;
  void *dw, *dx, *dy, *dh;
  Hip(hipMalloc(&dw, w.size()), "w");
  Hip(hipMalloc(&dx, x.size() * sizeof(float)), "x");
  Hip(hipMalloc(&dy, out_count * sizeof(float)), "y");
  Hip(hipMalloc(&dh, out_count * sizeof(__half)), "h");
  Hip(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice), "w up");
  Hip(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "x up");
  const glm::DeviceMatrix m{.data = dw,
                            .type = GgmlType::kQ6_K,
                            .cols = cols,
                            .rows = kRows,
                            .experts = kHeads,
                            .row_bytes = row_bytes};
  const float scale = 1.0F / 16.0F;
  const std::string name =
      "k" + std::to_string(cols) + " t" + std::to_string(tokens);
  Check(name + " f32 launch",
        glm::mla::ProjectHeads(m, kHeads, static_cast<const float*>(dx),
                               static_cast<float*>(dy), tokens, stream));
  Check(name + " f16 launch",
        glm::mla::ProjectHeadsHalf(m, kHeads, static_cast<const float*>(dx), dh,
                                   tokens, scale, stream));
  Hip(hipStreamSynchronize(stream), "sync");
  std::vector<float> y(out_count);
  std::vector<__half> h(out_count);
  Hip(hipMemcpy(y.data(), dy, out_count * sizeof(float), hipMemcpyDeviceToHost),
      "y down");
  Hip(hipMemcpy(h.data(), dh, out_count * sizeof(__half),
                hipMemcpyDeviceToHost),
      "h down");
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < out_count; ++i) {
    const __half want = __float2half(y[i] * scale);
    std::uint16_t a, b;
    std::memcpy(&a, &h[i], sizeof(a));
    std::memcpy(&b, &want, sizeof(b));
    mismatches += a != b ? 1 : 0;
  }
  Check(name + " bitwise fp16(f32 * scale), mismatches " +
            std::to_string(mismatches),
        mismatches == 0);
  glm::DeviceMatrix f32 = m;
  f32.type = GgmlType::kF32;
  f32.row_bytes = cols * sizeof(float);
  Check(name + " rejects F32",
        !glm::mla::ProjectHeadsHalf(f32, kHeads, static_cast<const float*>(dx),
                                    dh, tokens, scale, stream));
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(dy);
  (void)hipFree(dh);
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cerr << "no HIP device; skipping\n";
    return 77;
  }
  hipStream_t stream;
  Hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
  for (const std::uint32_t cols : {256U, 512U}) {
    for (const std::uint32_t tokens : {1U, 9U, 32U, 33U, 70U}) {
      Run(cols, tokens, stream);
    }
  }
  (void)hipStreamDestroy(stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "mla_f16_qabs: all checks passed\n";
  return 0;
}
