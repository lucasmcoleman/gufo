#include "src/models/glm53_flash/kernels/expert_vec.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "src/models/glm53_flash/kernels/expert_vec_math.hpp"

namespace gufo::models::glm53_flash::expert_vec {
namespace {

#include "src/models/glm53_flash/kernels/mmq/iq_tables.hip.hpp"

using core::GgmlType;

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;

// Activation quantization: one 32-lane group per 32-value block. Row r of
// the output is ActivationRowBytes(cols) bytes: cols int8 values, then
// cols / 32 F32 scales.
__global__ __launch_bounds__(kThreads) void QuantizeKernel(
    const float* __restrict__ x, std::uint8_t* __restrict__ out,
    std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t blocks_per_row = cols / 32;
  const std::size_t block =
      (std::size_t{blockIdx.x} * kWaves) + (threadIdx.x / 32);
  if (block >= std::size_t{rows} * blocks_per_row) {
    return;
  }
  const unsigned lane = threadIdx.x % 32;
  const std::size_t row = block / blocks_per_row;
  const std::uint32_t b = block % blocks_per_row;
  const float v = x[(row * cols) + (std::size_t{b} * 32) + lane];
  float amax = fabsf(v);
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, off, 32));
  }
  // fmaxf drops NaN; keep a nonfinite input visible through d.
  bool finite = isfinite(v);
  for (int off = 16; off > 0; off >>= 1) {
    finite = finite && __shfl_xor(static_cast<int>(finite), off, 32) != 0;
  }
  const float d = finite ? amax / 127.0F : NAN;
  std::uint8_t* r = out + (row * ActivationRowBytes(cols));
  const int q = finite ? QuantizeValue(v, d) : 0;
  r[(std::size_t{b} * 32) + lane] =
      static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
  if (lane == 0) {
    reinterpret_cast<float*>(r + cols)[b] = d;
  }
}

using Unaligned16 = uint4 __attribute__((aligned(2)));

template<Kind K, int KB>
__global__ __launch_bounds__(kThreads) void ExpertVecKernel(
    const std::uint8_t* __restrict__ w0, const std::uint8_t* __restrict__ w1,
    std::size_t expert_bytes, std::size_t row_bytes,
    const std::int32_t* __restrict__ ids, int experts,
    const std::uint8_t* __restrict__ act, int routes, int slots, bool per_route,
    int m, float* __restrict__ out0, float* __restrict__ out1) {
  constexpr int kPerWave = 32 / KB;
  constexpr int kPerBlock = kWaves * kPerWave;
  constexpr int kCols = KB * 256;
  __shared__ int yq[KB * 64];
  __shared__ float yd[KB * 8];
  const bool second = static_cast<int>(blockIdx.y) >= slots;
  const int slot = second ? static_cast<int>(blockIdx.y) - slots
                          : static_cast<int>(blockIdx.y);
  const std::uint8_t* __restrict__ w = second ? w1 : w0;
  float* __restrict__ out = second ? out1 : out0;
  const int yrow = per_route ? slot : slot / routes;
  const std::uint8_t* a =
      act + (static_cast<std::size_t>(yrow) * ActivationRowBytes(kCols));
  const int* aq = reinterpret_cast<const int*>(a);
  const float* ad = reinterpret_cast<const float*>(a + kCols);
  // Word idx of the row is word (idx & 7) of 32-value block idx >> 3, which
  // is sub-block (idx >> 3) & 7 of superblock idx >> 6.
  for (int idx = static_cast<int>(threadIdx.x); idx < KB * 64;
       idx += kThreads) {
    yq[LdsWord<KB>(idx >> 6, (idx >> 3) & 7, idx & 7)] = aq[idx];
  }
  for (int idx = static_cast<int>(threadIdx.x); idx < KB * 8; idx += kThreads) {
    yd[LdsScale<KB>(idx >> 3, idx & 7)] = ad[idx];
  }
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  const int b = lane % KB;
  const int row = (static_cast<int>(blockIdx.x) * kPerBlock) +
                  (wave * kPerWave) + (lane / KB);
  const int expert = ids[slot];
  const bool valid = expert >= 0 && expert < experts;
  const std::uint8_t* p =
      w + (static_cast<std::size_t>(valid ? expert : 0) * expert_bytes) +
      (static_cast<std::size_t>(row < m ? row : m - 1) * row_bytes) +
      (static_cast<std::size_t>(b) * BlockBytes(K));
  BlockWords words;
  words.d = HalfBits(*reinterpret_cast<const std::uint16_t*>(p));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = *reinterpret_cast<const Unaligned16*>(p + 2 + (16 * i));
    words.q[(4 * i) + 0] = v.x;
    words.q[(4 * i) + 1] = v.y;
    words.q[(4 * i) + 2] = v.z;
    words.q[(4 * i) + 3] = v.w;
  }
  if constexpr (K != Kind::kIq2xxs) {
    constexpr int kTailLoads = K == Kind::kIq2s ? 1 : 2;
#pragma unroll
    for (int i = 0; i < kTailLoads; ++i) {
      const uint4 v = *reinterpret_cast<const Unaligned16*>(p + 66 + (16 * i));
      words.tail[(4 * i) + 0] = v.x;
      words.tail[(4 * i) + 1] = v.y;
      words.tail[(4 * i) + 2] = v.z;
      words.tail[(4 * i) + 3] = v.w;
    }
  }
  const Grids grids{kIq2xxsGrid, kIq2sGrid, kIq3xxsGrid};
  float v = BlockDot<K, KB>(words, grids, yq, yd, b);
#pragma unroll
  for (int off = KB / 2; off > 0; off >>= 1) {
    v += __shfl_xor(v, off, 32);
  }
  if (b == 0 && row < m) {
    out[(static_cast<std::size_t>(slot) * m) + row] = valid ? v : NAN;
  }
}

template<Kind K, int KB>
void LaunchKernel(const DeviceMatrix& w0, const DeviceMatrix* w1,
                  const std::int32_t* ids, const std::uint8_t* act, int routes,
                  int slots, bool per_route, float* out0, float* out1,
                  Stream stream) {
  constexpr int kPerBlock = kWaves * (32 / KB);
  const int m = static_cast<int>(w0.rows);
  const dim3 grid(static_cast<unsigned>((m + kPerBlock - 1) / kPerBlock),
                  static_cast<unsigned>(slots * (w1 != nullptr ? 2 : 1)));
  ExpertVecKernel<K, KB><<<grid, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(w0.data),
      static_cast<const std::uint8_t*>(w1 != nullptr ? w1->data : w0.data),
      static_cast<std::size_t>(w0.rows) * w0.row_bytes, w0.row_bytes, ids,
      static_cast<int>(w0.experts), act, routes, slots, per_route, m, out0,
      w1 != nullptr ? out1 : out0);
}

template<Kind K>
void LaunchCols(const DeviceMatrix& w0, const DeviceMatrix* w1,
                const std::int32_t* ids, const std::uint8_t* act, int routes,
                int slots, bool per_route, float* out0, float* out1,
                Stream stream) {
  if (w0.cols == 4096) {
    LaunchKernel<K, 16>(w0, w1, ids, act, routes, slots, per_route, out0, out1,
                        stream);
  } else {
    LaunchKernel<K, 8>(w0, w1, ids, act, routes, slots, per_route, out0, out1,
                       stream);
  }
}

}  // namespace

Mode EnvMode() noexcept {
  static const Mode mode = ParseMode(std::getenv("GUFO_GLM53_EXPERT_VEC"));
  return mode;
}

bool Supported(GgmlType type, std::uint64_t cols) noexcept {
  return (type == GgmlType::kIQ2_XXS || type == GgmlType::kIQ2_S ||
          type == GgmlType::kIQ3_XXS) &&
         (cols == 2048 || cols == 4096);
}

std::size_t ScratchBytes(std::uint32_t rows, std::uint64_t cols) noexcept {
  if (cols == 0 || cols % 32 != 0) {
    return 0;
  }
  return std::size_t{rows} * ActivationRowBytes(cols);
}

bool Launch(const DeviceMatrix& w0, const DeviceMatrix* w1, const float* x,
            const std::int32_t* ids, float* out0, float* out1,
            std::uint32_t tokens, std::uint32_t routes, bool per_route,
            void* scratch, std::size_t scratch_bytes, Stream stream) {
  const std::uint32_t act_rows = per_route ? tokens * routes : tokens;
  if (!Supported(w0.type, w0.cols) || w0.empty() || x == nullptr ||
      ids == nullptr || out0 == nullptr || scratch == nullptr || tokens == 0 ||
      tokens > kMaxTokens || routes == 0 || routes > 64 || w0.rows == 0 ||
      w0.rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      w0.experts == 0 ||
      w0.experts >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      w0.row_bytes != (w0.cols / 256) *
                          static_cast<std::size_t>(BlockBytes(
                              w0.type == GgmlType::kIQ2_XXS ? Kind::kIq2xxs
                              : w0.type == GgmlType::kIQ2_S ? Kind::kIq2s
                                                            : Kind::kIq3xxs)) ||
      reinterpret_cast<std::uintptr_t>(w0.data) % 2 != 0 ||
      scratch_bytes < ScratchBytes(act_rows, w0.cols) ||
      reinterpret_cast<std::uintptr_t>(scratch) % 16 != 0) {
    return false;
  }
  if (w1 != nullptr &&
      (per_route || out1 == nullptr || w1->empty() || w1->type != w0.type ||
       w1->cols != w0.cols || w1->rows != w0.rows ||
       w1->experts != w0.experts || w1->row_bytes != w0.row_bytes ||
       reinterpret_cast<std::uintptr_t>(w1->data) % 2 != 0)) {
    return false;
  }
  const auto cols = static_cast<std::uint32_t>(w0.cols);
  auto* act = static_cast<std::uint8_t*>(scratch);
  const std::size_t blocks = std::size_t{act_rows} * (cols / 32);
  QuantizeKernel<<<static_cast<unsigned>((blocks + kWaves - 1) / kWaves),
                   kThreads, 0, stream>>>(x, act, act_rows, cols);
  const int slots = static_cast<int>(tokens * routes);
  switch (w0.type) {
    case GgmlType::kIQ2_XXS:
      LaunchCols<Kind::kIq2xxs>(w0, w1, ids, act, static_cast<int>(routes),
                                slots, per_route, out0, out1, stream);
      break;
    case GgmlType::kIQ2_S:
      LaunchCols<Kind::kIq2s>(w0, w1, ids, act, static_cast<int>(routes), slots,
                              per_route, out0, out1, stream);
      break;
    default:
      LaunchCols<Kind::kIq3xxs>(w0, w1, ids, act, static_cast<int>(routes),
                                slots, per_route, out0, out1, stream);
      break;
  }
  return hipPeekAtLastError() == hipSuccess;
}

}  // namespace gufo::models::glm53_flash::expert_vec
