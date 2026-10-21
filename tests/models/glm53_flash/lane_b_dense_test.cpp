// GLM per-head Q6_K routes against scalar CPU block dequant + F64 dot.
// Distinct head weights, exact allocations and batch boundaries exercise
// dispatcher assumptions without loading a serving model (< 8 MiB device).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace {
void Hip(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
struct Buffer {
  void* p{};
  explicit Buffer(std::size_t bytes) { Hip(hipMalloc(&p, bytes)); }
  ~Buffer() { (void)hipFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  float* f() { return static_cast<float*>(p); }
};
void Run(std::size_t rows, std::size_t cols, std::size_t batch) {
  using gufo::core::GgmlType;
  constexpr std::size_t heads = 3;
  const auto row_bytes = gufo::quant::QuantizedRowBytes(GgmlType::kQ6_K, cols);
  const auto matrix_bytes = rows * row_bytes;
  std::mt19937 rng(static_cast<unsigned>(rows + batch));
  std::vector<std::uint8_t> packed(heads * matrix_bytes);
  // Random valid Q6_K payload; finite varying fp16 scales, signed subgroup
  // scales and both high/low packed bits. Not tied to a GPU quantizer.
  for (auto& v : packed)
    v = static_cast<std::uint8_t>(rng());
  const std::size_t block_bytes = row_bytes / (cols / 256);
  for (std::size_t b = 0; b < packed.size() / block_bytes; ++b) {
    const std::uint16_t scale =
        static_cast<std::uint16_t>(0x1800 + rng() % 2048);
    std::memcpy(packed.data() + (b + 1) * block_bytes - 2, &scale, 2);
  }
  std::vector<float> x(heads * batch * cols);
  std::uniform_real_distribution<float> dist(-0.7F, 0.7F);
  for (auto& v : x)
    v = dist(rng);
  std::vector<float> want(heads * batch * rows), got(want.size()),
      split(want.size());
  std::vector<float> row(cols);
  for (std::size_t h = 0; h < heads; ++h)
    for (std::size_t m = 0; m < rows; ++m) {
      gufo::quant::DequantizeQ6_K(
          packed.data() + h * matrix_bytes + m * row_bytes, row.data(), cols);
      for (std::size_t t = 0; t < batch; ++t) {
        double sum = 0;
        for (std::size_t k = 0; k < cols; ++k)
          sum += double(row[k]) * x[(h * batch + t) * cols + k];
        want[(h * batch + t) * rows + m] = static_cast<float>(sum);
      }
    }
  Buffer dw(packed.size()), dx(x.size() * sizeof(float)),
      dy(got.size() * sizeof(float));
  Hip(hipMemcpy(dw.p, packed.data(), packed.size(), hipMemcpyHostToDevice));
  Hip(hipMemcpy(dx.p, x.data(), x.size() * sizeof(float),
                hipMemcpyHostToDevice));
  for (std::size_t h = 0; h < heads; ++h)
    gufo::hip::LaunchBatchedQuantGEMMFp32(
        GgmlType::kQ6_K, static_cast<std::uint8_t*>(dw.p) + h * matrix_bytes,
        dx.f() + h * batch * cols, dy.f() + h * batch * rows, batch, rows,
        cols);
  Hip(hipGetLastError());
  Hip(hipMemcpy(got.data(), dy.p, got.size() * sizeof(float),
                hipMemcpyDeviceToHost));
  Hip(hipMemset(dy.p, 0xFF, got.size() * sizeof(float)));
  for (std::size_t h = 0; h < heads; ++h)
    for (std::size_t t = 0; t < batch; t += 6)
      gufo::hip::LaunchBatchedQuantGEMMFp32(
          GgmlType::kQ6_K, static_cast<std::uint8_t*>(dw.p) + h * matrix_bytes,
          dx.f() + (h * batch + t) * cols, dy.f() + (h * batch + t) * rows,
          std::min<std::size_t>(6, batch - t), rows, cols);
  Hip(hipGetLastError());
  Hip(hipMemcpy(split.data(), dy.p, split.size() * sizeof(float),
                hipMemcpyDeviceToHost));
  double diff = 0, norm = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(split[i]))
      throw std::runtime_error("nonfinite device output");
    diff += std::pow(double(got[i]) - want[i], 2);
    norm += double(want[i]) * want[i];
  }
  const double rel = std::sqrt(diff / norm);
  const bool exact =
      std::memcmp(got.data(), split.data(), got.size() * sizeof(float)) == 0;
  std::cout << "Q6_K rows=" << rows << " cols=" << cols << " batch=" << batch
            << " heads=" << heads << " rel_l2=" << rel
            << " chunk_exact=" << exact << '\n';
  if (!(rel <= 1e-5) || !exact)
    throw std::runtime_error("dense parity failed");
}
}  // namespace
int main() {
  try {
    for (auto batch : {1U, 8U, 9U, 17U}) {
      Run(512, 256, batch);  // k_b absorption
      Run(256, 512, batch);  // v_b projection
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
