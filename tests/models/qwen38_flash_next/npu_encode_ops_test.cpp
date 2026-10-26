// GPU check of the NPU block-float weight encoders (no NPU involved): the
// K-quant/IQ encoder NpuEncodeWeightsQuant (ported from mikealanni/gufo
// 4f346915) against the CPU dequantizers, and against the PR #568 Q8_0
// encoder bit for bit. Each 9-byte group holds one exponent byte and eight
// int8 mantissas, value = q * 2^(e - 127) / 64, laid out in the B tile order
// of tools/npu (64 x 128 tiles, column-major, 1 x 2 super blocks).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
using gufo::core::GgmlType;

namespace {

constexpr std::uint32_t kK = 2560;     // Flash-Next hidden size
constexpr std::uint32_t kCols = 1024;  // one NPU output slice
constexpr std::uint32_t kRows = 1100;  // tensor rows; the slice starts at 37
constexpr std::uint32_t kRow0 = 37;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

/// Byte offsets of the F16 super-block scales a random block must keep sane
/// (random bytes there would give NaN or huge values).
struct Layout {
  std::size_t block_bytes;
  std::size_t block_elems;
  std::vector<std::size_t> half_offsets;
};

Layout LayoutOf(GgmlType type) {
  switch (type) {
    case GgmlType::kQ8_0:
      return {34, 32, {0}};
    case GgmlType::kQ4_K:
      return {144, 256, {0, 2}};
    case GgmlType::kQ5_K:
      return {176, 256, {0, 2}};
    case GgmlType::kQ6_K:
      return {210, 256, {208}};
    case GgmlType::kQ3_K:
      return {110, 256, {108}};
    case GgmlType::kIQ4_NL:
      return {18, 32, {0}};
    case GgmlType::kIQ4_XS:
      return {136, 256, {0}};
    case GgmlType::kIQ3_S:
      return {110, 256, {0}};
    default:
      throw std::runtime_error("unsupported type in test");
  }
}

void Dequantize(GgmlType type, const void* row, float* out, std::size_t k) {
  namespace gq = gufo::quant;
  switch (type) {
    case GgmlType::kQ8_0:
      return gq::DequantizeQ8_0(row, out, k);
    case GgmlType::kQ4_K:
      return gq::DequantizeQ4_K(row, out, k);
    case GgmlType::kQ5_K:
      return gq::DequantizeQ5_K(row, out, k);
    case GgmlType::kQ6_K:
      return gq::DequantizeQ6_K(row, out, k);
    case GgmlType::kQ3_K:
      return gq::DequantizeQ3_K(row, out, k);
    case GgmlType::kIQ4_NL:
      return gq::DequantizeIQ4_NL(row, out, k);
    case GgmlType::kIQ4_XS:
      return gq::DequantizeIQ4_XS(row, out, k);
    case GgmlType::kIQ3_S:
      return gq::DequantizeIQ3_S(row, out, k);
    default:
      throw std::runtime_error("unsupported type in test");
  }
}

std::vector<std::uint8_t> RandomTensor(GgmlType type, std::uint32_t seed) {
  const Layout layout = LayoutOf(type);
  const std::size_t row_bytes = kK / layout.block_elems * layout.block_bytes;
  std::vector<std::uint8_t> bytes(row_bytes * kRows);
  for (auto& b : bytes) {
    b = static_cast<std::uint8_t>(NextRandom(&seed));
  }
  for (std::size_t block = 0; block < bytes.size() / layout.block_bytes;
       ++block) {
    for (const std::size_t off : layout.half_offsets) {
      // A positive scale (or min) between 2^-9 and 2^-6.
      const float mag = std::ldexp(
          1.0F + static_cast<float>(NextRandom(&seed) % 1000) / 1000.0F,
          -9 + static_cast<int>(NextRandom(&seed) % 3));
      const __half h = __float2half(mag);
      std::memcpy(bytes.data() + block * layout.block_bytes + off, &h, 2);
    }
  }
  return bytes;
}

/// The (k, n) of group g's first value in the B tile order.
void GroupOrigin(std::size_t g, std::uint32_t k, std::uint32_t* kk,
                 std::uint32_t* nn) {
  constexpr std::uint32_t kTileK = 64;
  constexpr std::uint32_t kTileN = 128;
  constexpr std::uint32_t kGroupsPerTile = kTileK * kTileN / 8;
  const std::size_t tile = g / kGroupsPerTile;
  const auto gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
  const std::uint32_t tiles_k = k / kTileK;
  const auto tile_col = static_cast<std::uint32_t>(tile / tiles_k);
  const auto tile_row = static_cast<std::uint32_t>(tile % tiles_k);
  const std::uint32_t cib = gi & 7;
  const std::uint32_t block = (gi >> 3) & 1;
  const std::uint32_t sr = (gi >> 4) & 7;
  const std::uint32_t sc2 = gi >> 7;
  *kk = tile_row * kTileK + sr * 8;
  *nn = tile_col * kTileN + (sc2 * 2 + block) * 8 + cib;
}

std::vector<std::uint8_t> Encode(GgmlType type,
                                 const std::vector<std::uint8_t>& tensor,
                                 bool q8_kernel, std::uint32_t row0) {
  void* d_w = nullptr;
  void* d_out = nullptr;
  const std::size_t out_bytes = std::size_t{kK} * kCols / 8 * 9;
  CheckHip(hipMalloc(&d_w, tensor.size() + 4096), "weights");
  CheckHip(hipMemcpy(d_w, tensor.data(), tensor.size(), hipMemcpyHostToDevice),
           "weights upload");
  CheckHip(hipMalloc(&d_out, out_bytes), "groups");
  if (q8_kernel) {
    q::NpuEncodeWeights(d_w, kK, row0, kCols, kK, 0, kK, d_out, nullptr);
  } else {
    q::NpuEncodeWeightsQuant(d_w, static_cast<std::uint32_t>(type), kK, row0,
                             kCols, kK, 0, kK, d_out, nullptr);
  }
  CheckHip(hipGetLastError(), "encode launch");
  CheckHip(hipDeviceSynchronize(), "encode");
  std::vector<std::uint8_t> out(out_bytes);
  CheckHip(hipMemcpy(out.data(), d_out, out_bytes, hipMemcpyDeviceToHost),
           "groups download");
  CheckHip(hipFree(d_w), "free");
  CheckHip(hipFree(d_out), "free");
  return out;
}

/// Largest error of the encoded groups against `reference` ([rows][kK],
/// starting at row `row0`), in units of each group's mantissa step; groups
/// whose step the reference cannot reach (all zero) must decode to zero.
double WorstSteps(const std::vector<std::uint8_t>& groups,
                  const std::vector<float>& reference, std::uint32_t row0) {
  double worst = 0.0;
  for (std::size_t g = 0; g < groups.size() / 9; ++g) {
    std::uint32_t kk = 0;
    std::uint32_t nn = 0;
    GroupOrigin(g, kK, &kk, &nn);
    const std::uint8_t* p = groups.data() + g * 9;
    const double step = std::ldexp(1.0, static_cast<int>(p[0]) - 127 - 6);
    for (int i = 0; i < 8; ++i) {
      const double value =
          static_cast<double>(static_cast<std::int8_t>(p[1 + i])) * step;
      const double ref = reference[(std::size_t{row0} + nn) * kK + kk +
                                   static_cast<std::size_t>(i)];
      const double err = std::abs(value - ref);
      worst =
          std::max(worst, p[0] == 0 ? (err == 0.0 ? 0.0 : 1e9) : err / step);
    }
  }
  return worst;
}

void CheckType(GgmlType type, std::uint32_t seed) {
  const auto tensor = RandomTensor(type, seed);
  const Layout layout = LayoutOf(type);
  const std::size_t row_bytes = kK / layout.block_elems * layout.block_bytes;
  std::vector<float> reference(std::size_t{kRows} * kK);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    Dequantize(type, tensor.data() + r * row_bytes,
               reference.data() + std::size_t{r} * kK, kK);
  }
  const auto groups = Encode(type, tensor, false, kRow0);
  const double worst = WorstSteps(groups, reference, kRow0);
  // Round to nearest gives half a mantissa step; a value that rounds to +128
  // saturates at 127, a full step. The slack covers the device decode's
  // rounding against the CPU dequantizer (a few ulps, far below a step).
  Require(
      worst <= 1.0 + 1e-3,
      std::string(gufo::core::ToString(type)) +
          " encoder error exceeds one mantissa step: " + std::to_string(worst));
  // Negative: the same groups against the slice one row lower must fail, so
  // a row-offset bug in the encoder cannot pass.
  Require(WorstSteps(groups, reference, kRow0 + 1) > 2.0,
          std::string(gufo::core::ToString(type)) +
              " negative check passed with a shifted reference");
  if (type == GgmlType::kQ8_0) {
    const auto q8 = Encode(type, tensor, true, kRow0);
    Require(q8 == groups,
            "Q8_0 groups from NpuEncodeWeightsQuant differ from "
            "NpuEncodeWeights");
  }
  std::cout << gufo::core::ToString(type) << ": worst " << worst
            << " mantissa steps\n";
}

}  // namespace

int main() {
  try {
    std::uint32_t seed = 0x5eed1234U;
    for (const GgmlType type :
         {GgmlType::kQ8_0, GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
          GgmlType::kQ3_K, GgmlType::kIQ4_NL, GgmlType::kIQ4_XS,
          GgmlType::kIQ3_S}) {
      CheckType(type, NextRandom(&seed));
    }
  } catch (const std::exception& e) {
    std::cerr << "npu_encode_ops_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "npu_encode_ops_test: PASS\n";
  return 0;
}
