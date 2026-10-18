#include "src/models/glm53_flash/kernels/mmq/iq.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::glm53_flash::rocm {
namespace {

#include "src/models/glm53_flash/kernels/mmq/iq_tables.hip.hpp"

using core::GgmlType;

// HIP __fadd_rn still permits reassociation under -ffast-math. Preserve the
// original reduction tree when moving its partials from LDS into registers.
__device__ __forceinline__ float OrderedAdd(float a, float b) {
#pragma clang fp reassociate(off) contract(off)
  return a + b;
}

__device__ std::uint32_t Read32(const std::uint8_t* p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8U) |
         (std::uint32_t(p[2]) << 16U) | (std::uint32_t(p[3]) << 24U);
}

// Scalar decoding follows the independent CPU oracle's operation order.
// Packed block fields are read bytewise where they need not be word-aligned.
template<GgmlType Type>
__device__ float Decode(const void* row, std::uint32_t col) {
  const std::uint32_t block = col / 256;
  const std::uint32_t i = col % 256;
  const std::uint32_t sub = i / 32;
  const std::uint32_t group = (i % 32) / 8;
  const std::uint32_t lane = i % 8;
  float scale;
  unsigned magnitude, signs;
  if constexpr (Type == GgmlType::kIQ2_XXS) {
    const auto& b = static_cast<const quant::block_iq2_xxs*>(row)[block];
    const auto* q = b.qs + sub * 8;
    const auto aux = Read32(q + 4);
    scale = __half2float(__ushort_as_half(b.d)) *
            (0.5F + static_cast<float>(aux >> 28U)) * 0.25F;
    magnitude = (kIq2xxsGrid[q[group]] >> (8 * lane)) & 255U;
    signs = kSignsIq2xs[(aux >> (7 * group)) & 127U];
  } else if constexpr (Type == GgmlType::kIQ3_XXS) {
    const auto& b = static_cast<const quant::block_iq3_xxs*>(row)[block];
    const auto aux = Read32(b.qs + 64 + sub * 4);
    scale = __half2float(__ushort_as_half(b.d)) *
            (0.5F + static_cast<float>(aux >> 28U)) * 0.5F;
    const unsigned index = b.qs[sub * 8 + group * 2 + lane / 4];
    magnitude = (kIq3xxsGrid[index] >> (8 * (lane % 4))) & 255U;
    signs = kSignsIq2xs[(aux >> (7 * group)) & 127U];
  } else {
    const auto& b = static_cast<const quant::block_iq2_s*>(row)[block];
    const unsigned index = b.qs[sub * 4 + group] |
                           ((unsigned(b.qh[sub]) << (8 - 2 * group)) & 768U);
    const unsigned nibble = (b.scales[sub] >> (4 * (group / 2))) & 15U;
    scale = __half2float(__ushort_as_half(b.d)) *
            (0.5F + static_cast<float>(nibble)) * 0.25F;
    magnitude = (kIq2sGrid[index] >> (8 * lane)) & 255U;
    signs = b.qs[32 + sub * 4 + group];
  }
  return (scale * static_cast<float>(magnitude)) *
         ((signs & (1U << lane)) ? -1.0F : 1.0F);
}

template<GgmlType Type>
__global__ void DequantKernel(const std::uint8_t* weights, float* output,
                              std::size_t count, std::uint32_t cols,
                              std::size_t row_bytes) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(gridDim.x) * blockDim.x) {
    output[i] = Decode<Type>(weights + (i / cols) * row_bytes, i % cols);
  }
}

template<GgmlType Type>
__global__ void RoutedKernel(const std::uint8_t* weights, const float* input,
                             const std::int32_t* ids, float* output,
                             std::size_t count, std::uint32_t routes,
                             std::uint32_t experts, std::uint32_t rows,
                             std::uint32_t cols, std::size_t row_bytes,
                             bool broadcast) {
  // Each wave preserves the original 256-lane dot/reduction tree in eight
  // register partials per lane. Independent output rows need no LDS barriers.
  // The gfx1151 target uses wave32. Each output occupies one complete wave.
  const unsigned lane = threadIdx.x % 32;
  const std::size_t wave =
      (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
  const std::size_t waves = std::size_t{gridDim.x} * blockDim.x / 32;
  for (std::size_t out = wave; out < count; out += waves) {
    const std::size_t route = out / rows;
    const std::int32_t expert = ids[route];
    float sums[8] = {};
    if (expert < 0 || std::uint32_t(expert) >= experts) {
      sums[0] = __int_as_float(0x7fc00000);
    } else {
      const auto* w =
          weights + (std::size_t(expert) * rows + out % rows) * row_bytes;
      const auto* x = input + (broadcast ? route / routes : route) * cols;
      for (std::size_t c = lane; c < cols; c += 256) {
#pragma unroll
        for (unsigned j = 0; j < 8; ++j)
          sums[j] =
              fmaf(Decode<Type>(w, static_cast<std::uint32_t>(c + j * 32)),
                   x[c + j * 32], sums[j]);
      }
    }
#pragma unroll
    for (unsigned j = 0; j < 4; ++j)
      sums[j] = OrderedAdd(sums[j], sums[j + 4]);
    sums[0] = OrderedAdd(sums[0], sums[2]);
    sums[1] = OrderedAdd(sums[1], sums[3]);
    float sum = OrderedAdd(sums[0], sums[1]);
#pragma unroll
    for (unsigned stride = 16; stride > 0; stride /= 2)
      sum = OrderedAdd(sum, __shfl_down(sum, stride, 32));
    if (lane == 0)
      output[out] = sum;
  }
}

std::size_t BlockBytes(GgmlType type) {
  switch (type) {
    case GgmlType::kIQ2_XXS:
      return sizeof(quant::block_iq2_xxs);
    case GgmlType::kIQ2_S:
      return sizeof(quant::block_iq2_s);
    case GgmlType::kIQ3_XXS:
      return sizeof(quant::block_iq3_xxs);
    default:
      return 0;
  }
}

bool Aligned(const void* p, std::size_t alignment) {
  return p && reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}

bool Product(std::size_t a, std::size_t b, std::size_t* out) {
  // Bound pointer differences as well as size_t multiplication.
  if (a == 0 || b == 0 || a > std::size_t(PTRDIFF_MAX) / b)
    return false;
  *out = a * b;
  return true;
}

unsigned Grid(std::size_t n) {
  return static_cast<unsigned>(n < 65535 ? n : 65535);
}

}  // namespace

bool IqDequantize(GgmlType type, const void* weights, float* output,
                  std::uint32_t rows, std::uint32_t cols, hipStream_t stream) {
  std::size_t count, bytes;
  const std::size_t block_bytes = BlockBytes(type);
  if (!block_bytes || !Aligned(weights, 2) || !Aligned(output, 4) ||
      cols % 256 || !Product(rows, cols, &count) ||
      !Product(count, sizeof(float), &bytes))
    return false;
  const auto row_bytes = std::size_t(cols / 256) * block_bytes;
  const unsigned grid = Grid((count + 255) / 256);
#define LAUNCH_DEQUANT(TYPE)                                                \
  DequantKernel<TYPE>                                                       \
      <<<grid, 256, 0, stream>>>(static_cast<const std::uint8_t*>(weights), \
                                 output, count, cols, row_bytes)
  switch (type) {
    case GgmlType::kIQ2_XXS:
      LAUNCH_DEQUANT(GgmlType::kIQ2_XXS);
      break;
    case GgmlType::kIQ2_S:
      LAUNCH_DEQUANT(GgmlType::kIQ2_S);
      break;
    case GgmlType::kIQ3_XXS:
      LAUNCH_DEQUANT(GgmlType::kIQ3_XXS);
      break;
    default:
      return false;
  }
#undef LAUNCH_DEQUANT
  return hipGetLastError() == hipSuccess;
}

bool IqRoutedMatmul(GgmlType type, const void* weights, const float* input,
                    const std::int32_t* expert_ids, float* output,
                    std::uint32_t tokens, std::uint32_t routes,
                    std::uint32_t experts, std::uint32_t rows,
                    std::uint32_t cols, bool broadcast_input,
                    hipStream_t stream) {
  const auto block_bytes = BlockBytes(type);
  std::size_t n_routes, count, bytes, weight_rows, input_elements;
  if (!block_bytes || !Aligned(weights, 2) || !Aligned(input, 4) ||
      !Aligned(expert_ids, 4) || !Aligned(output, 4) || !cols || cols % 256 ||
      !Product(tokens, routes, &n_routes) || !Product(n_routes, rows, &count) ||
      !Product(count, sizeof(float), &bytes) ||
      !Product(experts, rows, &weight_rows) ||
      !Product(weight_rows, std::size_t(cols / 256) * block_bytes, &bytes) ||
      !Product(broadcast_input ? tokens : n_routes, cols, &input_elements) ||
      !Product(input_elements, sizeof(float), &bytes))
    return false;
  const auto row_bytes = std::size_t(cols / 256) * block_bytes;
  const unsigned grid = Grid((count + 7) / 8);
#define LAUNCH_ROUTED(TYPE)                                                 \
  RoutedKernel<TYPE><<<grid, 256, 0, stream>>>(                             \
      static_cast<const std::uint8_t*>(weights), input, expert_ids, output, \
      count, routes, experts, rows, cols, row_bytes, broadcast_input)
  switch (type) {
    case GgmlType::kIQ2_XXS:
      LAUNCH_ROUTED(GgmlType::kIQ2_XXS);
      break;
    case GgmlType::kIQ2_S:
      LAUNCH_ROUTED(GgmlType::kIQ2_S);
      break;
    case GgmlType::kIQ3_XXS:
      LAUNCH_ROUTED(GgmlType::kIQ3_XXS);
      break;
    default:
      return false;
  }
#undef LAUNCH_ROUTED
  return hipGetLastError() == hipSuccess;
}

}  // namespace gufo::models::glm53_flash::rocm
