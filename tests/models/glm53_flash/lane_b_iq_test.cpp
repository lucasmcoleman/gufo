// Independent packed CPU decoder + F64 dot oracle for routed IQ device ops.
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
#include "src/models/glm53_flash/kernels/mmq/iq.hpp"

namespace {
using gufo::core::GgmlType;
using namespace gufo::models::glm53_flash::rocm;
void Check(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
void Hip(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
struct Buffer {
  void* p{};
  explicit Buffer(std::size_t n) { Hip(hipMalloc(&p, n)); }
  ~Buffer() { (void)hipFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  float* f() { return static_cast<float*>(p); }
  std::int32_t* ids() { return static_cast<std::int32_t*>(p); }
};
struct Stream {
  hipStream_t s{};
  Stream() { Hip(hipStreamCreateWithFlags(&s, hipStreamNonBlocking)); }
  ~Stream() { (void)hipStreamDestroy(s); }
};
void Decode(GgmlType type, const void* p, float* out, std::size_t n) {
  switch (type) {
    case GgmlType::kIQ2_XXS:
      gufo::quant::DequantizeIQ2_XXS(p, out, n);
      break;
    case GgmlType::kIQ2_S:
      gufo::quant::DequantizeIQ2_S(p, out, n);
      break;
    case GgmlType::kIQ3_XXS:
      gufo::quant::DequantizeIQ3_XXS(p, out, n);
      break;
    default:
      throw std::runtime_error("unexpected quant type");
  }
}
void Run(GgmlType type, std::uint32_t tokens, std::uint32_t cols,
         bool broadcast) {
  constexpr std::uint32_t rows = 33, experts = 5, routes = 3;
  const auto row_bytes = gufo::quant::QuantizedRowBytes(type, cols);
  const auto block_bytes = row_bytes / (cols / 256);
  std::mt19937 rng(61 + tokens + cols);
  std::vector<std::uint8_t> packed(experts * rows * row_bytes);
  for (auto& v : packed)
    v = static_cast<std::uint8_t>(rng());
  // All packed indices/signs/scales are legal. Explicit half scale avoids NaN.
  for (std::size_t b = 0; b < packed.size() / block_bytes; ++b) {
    const std::uint16_t d = static_cast<std::uint16_t>(0x1400 + rng() % 2048);
    std::memcpy(packed.data() + b * block_bytes, &d, 2);
  }
  std::vector<float> weights(experts * rows * cols), actual(weights.size());
  Decode(type, packed.data(), weights.data(), weights.size());
  Buffer dw(packed.size()), dq(actual.size() * sizeof(float));
  Stream stream;
  Hip(hipMemcpyAsync(dw.p, packed.data(), packed.size(), hipMemcpyHostToDevice,
                     stream.s));
  Check(IqDequantize(type, dw.p, dq.f(), experts * rows, cols, stream.s),
        "dequant launch rejected");
  Hip(hipMemcpyAsync(actual.data(), dq.p, actual.size() * sizeof(float),
                     hipMemcpyDeviceToHost, stream.s));
  Hip(hipStreamSynchronize(stream.s));
  Check(std::memcmp(weights.data(), actual.data(),
                    actual.size() * sizeof(float)) == 0,
        "dequant not bit exact");
  std::vector<float> x(tokens * (broadcast ? 1 : routes) * cols);
  std::uniform_real_distribution<float> dist(-0.2F, 0.2F);
  for (auto& v : x)
    v = dist(rng);
  std::vector<std::int32_t> ids(tokens * routes);
  for (std::uint32_t t = 0; t < tokens; ++t)
    for (std::uint32_t r = 0; r < routes; ++r)
      ids[t * routes + r] = static_cast<std::int32_t>((t + r * 2) % experts);
  std::vector<float> got(tokens * routes * rows), want(got.size()),
      split(got.size());
  for (std::uint32_t t = 0; t < tokens; ++t)
    for (std::uint32_t r = 0; r < routes; ++r)
      for (std::uint32_t m = 0; m < rows; ++m) {
        double sum = 0;
        for (std::uint32_t k = 0; k < cols; ++k)
          sum += double(weights[(ids[t * routes + r] * rows + m) * cols + k]) *
                 x[(t * (broadcast ? 1 : routes) + (broadcast ? 0 : r)) * cols +
                   k];
        want[(t * routes + r) * rows + m] = static_cast<float>(sum);
      }
  Buffer dx(x.size() * sizeof(float)), di(ids.size() * sizeof(std::int32_t)),
      dy(got.size() * sizeof(float));
  Hip(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float),
                     hipMemcpyHostToDevice, stream.s));
  Hip(hipMemcpyAsync(di.p, ids.data(), ids.size() * sizeof(std::int32_t),
                     hipMemcpyHostToDevice, stream.s));
  Check(IqRoutedMatmul(type, dw.p, dx.f(), di.ids(), dy.f(), tokens, routes,
                       experts, rows, cols, broadcast, stream.s),
        "routed launch rejected");
  Hip(hipMemcpyAsync(got.data(), dy.p, got.size() * sizeof(float),
                     hipMemcpyDeviceToHost, stream.s));
  for (std::uint32_t t = 0; t < tokens; t += 6)
    Check(
        IqRoutedMatmul(type, dw.p, dx.f() + t * (broadcast ? 1 : routes) * cols,
                       di.ids() + t * routes, dy.f() + t * routes * rows,
                       std::min(6U, tokens - t), routes, experts, rows, cols,
                       broadcast, stream.s),
        "chunk launch rejected");
  Hip(hipMemcpyAsync(split.data(), dy.p, split.size() * sizeof(float),
                     hipMemcpyDeviceToHost, stream.s));
  Hip(hipStreamSynchronize(stream.s));
  double diff = 0, norm = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    Check(std::isfinite(got[i]), "nonfinite matmul result");
    diff += std::pow(double(got[i]) - want[i], 2);
    norm += double(want[i]) * want[i];
  }
  const auto rel = std::sqrt(diff / norm);
  Check(rel <= 1e-5, "routed F64 parity failed");
  Check(std::memcmp(got.data(), split.data(), got.size() * sizeof(float)) == 0,
        "chunk invariance failed");
  Check(!IqDequantize(GgmlType::kQ6_K, dw.p, dq.f(), rows, cols, stream.s),
        "unsupported type accepted");
  Check(!IqDequantize(type, dw.p, dq.f(), rows, cols - 1, stream.s),
        "invalid reduction accepted");
  Check(!IqRoutedMatmul(type, dw.p, dx.f(), di.ids(), dy.f(), tokens, 0,
                        experts, rows, cols, broadcast, stream.s),
        "zero routes accepted");
  ids[0] = -1;
  ids.back() = experts;
  Hip(hipMemcpyAsync(di.p, ids.data(), ids.size() * sizeof(std::int32_t),
                     hipMemcpyHostToDevice, stream.s));
  Check(IqRoutedMatmul(type, dw.p, dx.f(), di.ids(), dy.f(), tokens, routes,
                       experts, rows, cols, broadcast, stream.s),
        "invalid-ID launch rejected on host");
  Hip(hipMemcpyAsync(got.data(), dy.p, got.size() * sizeof(float),
                     hipMemcpyDeviceToHost, stream.s));
  Hip(hipStreamSynchronize(stream.s));
  for (std::size_t m = 0; m < rows; ++m) {
    Check(std::isnan(got[m]), "negative expert ID not poisoned");
    Check(std::isnan(got[got.size() - rows + m]),
          "oversized expert ID not poisoned");
  }
  std::cout << "type=" << static_cast<int>(type) << " tokens=" << tokens
            << " cols=" << cols << " broadcast=" << broadcast
            << " exact_dequant=1 rel_l2=" << rel
            << " chunk_exact=1 invalid_ids=PASS\n";
}
}  // namespace
int main() {
  try {
    for (auto type : {GgmlType::kIQ2_XXS, GgmlType::kIQ2_S, GgmlType::kIQ3_XXS})
      for (auto tokens : {1U, 8U, 9U, 33U})
        for (bool broadcast : {false, true})
          Run(type, tokens, tokens == 1 ? 256 : 768, broadcast);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
