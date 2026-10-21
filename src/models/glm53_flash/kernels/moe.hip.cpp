#include "src/models/glm53_flash/kernels/moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include <optional>
#include <type_traits>

#include "ds4_mmq.h"
#include "ds4_pool.h"
#include "src/models/glm53_flash/kernels/mmq/iq.hpp"
#include "src/models/glm53_flash/kernels/prefill_gemm.hpp"
#include "src/models/glm53_flash/profile.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::models::glm53_flash {
namespace {
using core::GgmlType;
constexpr std::size_t kAlign = 256;

struct Layout {
  std::size_t bytes = 0;
  bool valid = true;
  std::size_t Take(std::size_t rows, std::size_t cols) {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (!valid || (cols && rows > (max - kAlign) / sizeof(float) / cols)) {
      valid = false;
      return 0;
    }
    const auto n = (rows * cols * sizeof(float) + kAlign - 1) & ~(kAlign - 1);
    const auto start = bytes;
    if (n > max - bytes) {
      valid = false;
      return 0;
    }
    bytes += n;
    return start;
  }
};
struct Buffers {
  Layout layout;
  std::size_t gate, up, down, logits, probs, ids, weights, raw_weights,
      norm_weights, shared;
  Buffers(const Config& c, std::uint32_t tokens) {
    const std::size_t routed = std::size_t{c.num_experts_used} * c.expert_ff;
    const auto ff = std::max(
        {std::size_t{c.dense_ff}, routed, std::size_t{c.shared_expert_ff}});
    gate = layout.Take(tokens, ff);
    up = layout.Take(tokens, ff);
    down = layout.Take(tokens, std::size_t{c.num_experts_used} * c.hidden_size);
    logits = layout.Take(tokens, c.num_experts);
    probs = layout.Take(tokens, c.num_experts);
    ids = layout.Take(tokens, c.num_experts_used);
    weights = layout.Take(tokens, c.num_experts_used);
    raw_weights = layout.Take(tokens, c.num_experts_used);
    norm_weights = layout.Take(tokens, c.num_experts_used);
    shared = layout.Take(tokens, c.hidden_size);
  }
};
bool IsIq(GgmlType t) {
  return t == GgmlType::kIQ2_XXS || t == GgmlType::kIQ2_S ||
         t == GgmlType::kIQ3_XXS;
}
bool Ds4Enabled();
bool Matrix(const DeviceMatrix& w, std::uint32_t rows, std::uint32_t cols,
            std::uint32_t experts, bool routed) {
  if (!rows || !cols || !experts || w.empty() || w.rows != rows ||
      w.cols != cols || w.experts != experts)
    return false;
  std::size_t bytes = 0;
  switch (w.type) {
    case GgmlType::kF32:
      bytes = std::size_t{cols} * 4;
      break;
    case GgmlType::kBF16:
      bytes = std::size_t{cols} * 2;
      break;
    case GgmlType::kQ6_K:
      if (routed || cols % 256)
        return false;
      bytes = std::size_t{cols / 256} * 210;
      break;
    case GgmlType::kQ8_0:
      if (routed || cols % 32)
        return false;
      bytes = std::size_t{cols / 32} * 34;
      break;
    case GgmlType::kQ2_K:
    case GgmlType::kQ3_K:
      // NextN routed experts; DS4 kernels only (no scalar routed path).
      if (!routed || cols % 256 || !Ds4Enabled())
        return false;
      bytes = std::size_t{cols / 256} * (w.type == GgmlType::kQ2_K ? 84 : 110);
      break;
    case GgmlType::kIQ2_XXS:
    case GgmlType::kIQ2_S:
    case GgmlType::kIQ3_XXS:
      if (!routed || cols % 256)
        return false;
      bytes = std::size_t{cols / 256} * (w.type == GgmlType::kIQ2_XXS ? 66
                                         : w.type == GgmlType::kIQ2_S ? 82
                                                                      : 98);
      break;
    default:
      return false;
  }
  const auto alignment = w.type == GgmlType::kF32 ? 4U : 2U;
  return w.row_bytes == bytes &&
         reinterpret_cast<std::uintptr_t>(w.data) % alignment == 0 &&
         bytes <= std::numeric_limits<std::size_t>::max() / rows / experts;
}
__device__ float Sigmoid(float x) {
  const float e = expf(-fabsf(x));
  return x >= 0 ? 1.0F / (1.0F + e) : e / (1.0F + e);
}
// Fixture/unquantized route also supports ragged dimensions without vector
// loads.
__global__ void FloatProject(const void* w, bool bf16, const float* x,
                             const std::int32_t* ids, float* y,
                             std::size_t count, std::uint32_t routes,
                             std::uint32_t rows, std::uint32_t cols,
                             bool broadcast) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < count; i += std::size_t{gridDim.x} * blockDim.x) {
    const auto tr = i / rows;
    // Only Route supplies IDs here; it writes [0, num_experts) even when
    // poisoning nonfinite routing. External IDs use IqRoutedMatmul checks.
    const auto expert = ids ? ids[tr] : 0;
    const auto offset =
        (static_cast<std::size_t>(expert) * rows + i % rows) * cols;
    const float* in = x + (broadcast ? tr / routes : tr) * cols;
    // Preserve cancellation through the unquantized fallback. Sequential
    // FP32 accumulation drifts enough to fail near-zero routed/shared sums.
    double sum = 0;
    for (std::uint32_t k = 0; k < cols; ++k) {
      const float v =
          bf16
              ? __uint_as_float(std::uint32_t{static_cast<const std::uint16_t*>(
                                    w)[offset + k]}
                                << 16)
              : static_cast<const float*>(w)[offset + k];
      sum += static_cast<double>(v) * in[k];
    }
    y[i] = static_cast<float>(sum);
  }
}
// One wave cooperates on each output row. FP64 partial sums retain the
// cancellation accuracy of FloatProject while consecutive lanes coalesce
// weight/input loads and shorten the serial dependency chain.
// Acc = double is the exact path (decode, GUFO_GLM53_EXACT_PREFILL). Prefill
// uses float: gfx1151 runs FP64 at a small fraction of the FP32 rate, which
// made the 4096 x 288 router cost ~4 ms per MoE layer per 256-token pass.
template <typename Acc>
__global__ void FloatProjectWave(const void* w, bool bf16, const float* x,
                                 const std::int32_t* ids, float* y,
                                 std::size_t count, std::uint32_t routes,
                                 std::uint32_t rows, std::uint32_t cols,
                                 bool broadcast) {
  // gfx1151 builds use wave32; shuffle width and output mapping agree.
  constexpr unsigned kWave = 32;
  const unsigned lane = threadIdx.x % kWave;
  for (std::size_t i =
           std::size_t{blockIdx.x} * (blockDim.x / kWave) + threadIdx.x / kWave;
       i < count; i += std::size_t{gridDim.x} * (blockDim.x / kWave)) {
    const auto tr = i / rows;
    const auto expert = ids ? ids[tr] : 0;
    const auto offset =
        (static_cast<std::size_t>(expert) * rows + i % rows) * cols;
    const float* in = x + (broadcast ? tr / routes : tr) * cols;
    // Float: Kahan-compensated per lane (error ~2u, near the double path's
    // for these lengths), then the lane partials reduce in double.
    Acc sum = 0, carry = 0;
    // Issue kBatch independent weight/input loads before accumulating them in
    // the original k order. Decode has one wave per router row (288 waves),
    // too few to hide memory latency behind a load -> FMA chain.
    constexpr unsigned kBatch = 8;
    for (std::size_t k0 = lane; k0 < cols; k0 += kWave * kBatch) {
      float vs[kBatch], xs[kBatch];
#pragma unroll
      for (unsigned b = 0; b < kBatch; ++b) {
        const std::size_t k = k0 + std::size_t{b} * kWave;
        vs[b] = 0.0f;
        xs[b] = 0.0f;
        if (k < cols) {
          vs[b] = bf16 ? __uint_as_float(
                             std::uint32_t{static_cast<const std::uint16_t*>(
                                 w)[offset + k]}
                             << 16)
                       : static_cast<const float*>(w)[offset + k];
          xs[b] = in[k];
        }
      }
#pragma unroll
      for (unsigned b = 0; b < kBatch; ++b) {
        if (k0 + std::size_t{b} * kWave >= cols)
          break;
        const float v = vs[b];
        if constexpr (std::is_same_v<Acc, float>) {
          // This library builds with -ffast-math, whose reassociation folds
          // the compensation to carry = term - term (codexms #690). Keep the
          // Kahan update exactly as written.
#pragma clang fp reassociate(off) contract(off)
          const float term = fmaf(v, xs[b], -carry);
          const float next = sum + term;
          carry = (next - sum) - term;
          sum = next;
        } else {
          sum += static_cast<Acc>(v) * xs[b];
        }
      }
    }
    double total = static_cast<double>(sum) - static_cast<double>(carry);
    for (unsigned delta = kWave / 2; delta; delta /= 2)
      total += __shfl_down(total, delta, kWave);
    if (lane == 0)
      y[i] = static_cast<float>(total);
  }
}
unsigned Grid(std::size_t n) {
  return static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 65535));
}
// Routed IQ projections through the vendored llama.cpp kernels (MMVQ for
// decode, expert-grouped MMQ for prefill) with Q8_1 activations. Set
// GUFO_GLM53_SCALAR_MOE=1 to keep the scalar IqRoutedMatmul path instead.
int Ds4TileCols() {
  static const int cols = [] {
    const char* v = std::getenv("GUFO_GLM53_MOE_TILE_COLS");
    return v && *v ? std::atoi(v) : 16;
  }();
  return cols;
}

bool Ds4Enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("GUFO_GLM53_SCALAR_MOE");
    // GLM runs one stream without graph capture, so the LIFO arena replaces
    // a hipMallocAsync/hipFreeAsync pair per call.
    ds4_pool_use_stack_arena(1);
    // Prefill gate/up column-tile width. DS4's default rule picks 80 columns,
    // but a 256-token GLM pass gives each of the 288 experts ~7 rows, so most
    // of every tile is padding. glm53_flash_moe_bench (GLM shapes, gfx1151):
    // 256 tokens 34.5 -> 13.3 ms with 16 and the token bound, 64 tokens
    // 27.7 -> 10.7 ms. The width changes the per-element summation order (max
    // |delta| 5.7e-6 on outputs up to 17), not the arithmetic. 0 restores the
    // default rule.
    ds4_mmq_set_routed_tile_cols(Ds4TileCols());
    return !(v && *v && *v != '0');
  }();
  return enabled;
}

bool Ds4Shape(const DeviceMatrix& w, const std::int32_t* ids) {
  return Ds4Enabled() && ids && w.experts > 1 && w.experts <= INT32_MAX &&
         w.rows <= INT32_MAX && w.cols <= INT32_MAX;
}

bool UseDs4(const DeviceMatrix& gate, const DeviceMatrix& up,
            const std::int32_t* ids) {
  return Ds4Shape(gate, ids) && gate.type == up.type &&
         (gate.type == GgmlType::kIQ2_XXS || gate.type == GgmlType::kQ2_K);
}

constexpr std::uint32_t kVecTokens = 8;

bool Ds4GateUp(const DeviceMatrix& gate, const DeviceMatrix& up,
               const float* x, const std::int32_t* ids, float* g, float* u,
               std::uint32_t tokens, std::uint32_t routes, Stream stream) {
  const int m = static_cast<int>(gate.rows), k = static_cast<int>(gate.cols),
            t = static_cast<int>(tokens), e = static_cast<int>(gate.experts),
            r = static_cast<int>(routes);
  if (gate.type == GgmlType::kQ2_K) {
    // Q2_K (the NextN experts) runs the vec kernel over 8-token chunks at
    // every size: ds4_mmq_q2_K_moe is far slower on these shapes.
    // glm53_flash_q2k_moe_bench (gfx1151): 30 tokens 272 -> 6.4 ms, 256
    // tokens 561 -> 66 ms, and closer to an fp64 reference (rel L2 3.6e-3
    // vs 4.1e-3).
    for (int t0 = 0; t0 < t; t0 += static_cast<int>(kVecTokens)) {
      const int n = std::min(static_cast<int>(kVecTokens), t - t0);
      const float* xc = x + (std::size_t(t0) * k);
      const std::int32_t* ic = ids + (std::size_t(t0) * r);
      const std::size_t out = std::size_t(t0) * r * m;
      if (ds4_mmq_q2_K_moe_vec(gate.data, xc, ic, g + out, m, k, n, e, r,
                               stream) != 0 ||
          ds4_mmq_q2_K_moe_vec(up.data, xc, ic, u + out, m, k, n, e, r,
                               stream) != 0)
        return false;
    }
    return true;
  }
  if (tokens <= kVecTokens)
    return ds4_mmq_iq2_xxs_moe_vec(gate.data, x, ids, g, m, k, t, e, r,
                                   stream) == 0 &&
           ds4_mmq_iq2_xxs_moe_vec(up.data, x, ids, u, m, k, t, e, r,
                                   stream) == 0;
  // The paired MMQ shares one activation quantization and expert map
  // between gate and up. Token-bound: a token's top-k experts are distinct,
  // so no expert gets more rows than there are tokens, and the launch grid is
  // sized by tokens instead of tokens * routes (empty blocks only; same
  // results).
  return ds4_mmq_iq2_xxs_moe_pair_token_bound(gate.data, up.data, x, ids, g, u,
                                              m, k, t, e, r, stream) == 0;
}

bool UseDs4Down(const DeviceMatrix& down, const std::int32_t* ids) {
  return Ds4Shape(down, ids) &&
         (down.type == GgmlType::kIQ2_S || down.type == GgmlType::kIQ3_XXS ||
          down.type == GgmlType::kQ3_K);
}

// Each route's down input differs, so the assignments are flattened to
// tokens * routes rows with one expert each.
bool Ds4Down(const DeviceMatrix& down, const float* x, const std::int32_t* ids,
             float* y, std::uint32_t tokens, std::uint32_t routes,
             Stream stream) {
  const auto rows = std::size_t{tokens} * routes;
  if (rows > INT32_MAX)
    return false;
  const int m = static_cast<int>(down.rows), k = static_cast<int>(down.cols),
            n = static_cast<int>(rows), e = static_cast<int>(down.experts);
  const bool vec = tokens <= kVecTokens;
  // Prefill tiling, as for gate/up (Ds4Enabled): a token's routes go to
  // distinct experts, so no expert gets more than `tokens` rows, and narrow
  // tiles fit the ~7 rows per expert. glm53_flash_moe_bench, 256 tokens,
  // IQ3_XXS: 20.9 -> 11.4 ms (order-only differences, max 1.1e-5).
  if (!vec)
    ds4_mmq_set_moe_down_bounds(Ds4TileCols(), static_cast<int>(tokens));
  const auto project =
      down.type == GgmlType::kIQ2_S
          ? (vec ? ds4_mmq_iq2_s_moe_vec : ds4_mmq_iq2_s_moe)
      : down.type == GgmlType::kQ3_K
          ? (vec ? ds4_mmq_q3_K_moe_vec : ds4_mmq_q3_K_moe)
          : (vec ? ds4_mmq_iq3_xxs_moe_vec : ds4_mmq_iq3_xxs_moe);
  const bool ok = project(down.data, x, ids, y, m, k, n, e, 1, stream) == 0;
  if (!vec)
    ds4_mmq_set_moe_down_bounds(0, 0);
  return ok;
}

bool Project(const DeviceMatrix& w, const float* x, const std::int32_t* ids,
             float* y, std::uint32_t tokens, std::uint32_t routes,
             bool broadcast, Stream stream) {
  if (IsIq(w.type))
    return rocm::IqRoutedMatmul(w.type, w.data, x, ids, y, tokens, routes,
                                w.experts, w.rows, w.cols, broadcast, stream);
  if (w.type == GgmlType::kF32 || w.type == GgmlType::kBF16) {
    const auto n = std::size_t{tokens} * routes * w.rows;
    if (w.cols < 64) {
      FloatProject<<<Grid(n), 256, 0, stream>>>(
          w.data, w.type == GgmlType::kBF16, x, ids, y, n, routes, w.rows,
          w.cols, broadcast);
    } else {
      const auto grid =
          static_cast<unsigned>(std::min<std::size_t>((n + 7) / 8, 65535));
      (tokens > 8 && !ExactPrefill() ? FloatProjectWave<float>
                                     : FloatProjectWave<double>)<<<grid, 256, 0,
                                                                   stream>>>(
          w.data, w.type == GgmlType::kBF16, x, ids, y, n, routes, w.rows,
          w.cols, broadcast);
    }
  } else if (ids || routes != 1 || w.experts != 1 ||
             !PrefillQuantGemm(w.type, w.data, x, y, tokens, w.rows, w.cols,
                               stream)) {
    // Dense and shared-expert projections take the prefill WMMA path when it
    // applies (prefill_gemm.hpp); everything else stays on the exact path.
    ::gufo::hip::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, y, tokens,
                                            w.rows, w.cols, stream);
  }
  return hipGetLastError() == hipSuccess;
}
__global__ void Activate(float* gate, const float* up, std::size_t n,
                         float clamp) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    float g = gate[i], u = up[i];
    if (!isfinite(g) || !isfinite(u)) {
      gate[i] = NAN;
      continue;
    }
    if (clamp > 0) {
      g = fminf(g, clamp);
      u = fmaxf(-clamp, fminf(u, clamp));
    }
    gate[i] = g * Sigmoid(g) * u;
  }
}
__global__ void Route(const float* logits, const float* bias, float* probs,
                      std::int32_t* ids, float* weights, float* raw,
                      float* norm, std::uint32_t tokens, std::uint32_t experts,
                      std::uint32_t used, float scale, bool normalize) {
  const auto t = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (t >= tokens)
    return;
  probs += t * experts;
  logits += t * experts;
  ids += t * used;
  weights += t * used;
  raw += t * used;
  norm += t * used;
  bool finite = true;
  for (std::uint32_t e = 0; e < experts; ++e)
    finite &= isfinite(logits[e]) && (!bias || isfinite(bias[e]));
  if (!finite) {
    for (std::uint32_t e = 0; e < experts; ++e)
      probs[e] = NAN;
    for (std::uint32_t s = 0; s < used; ++s) {
      ids[s] = s;
      weights[s] = raw[s] = norm[s] = NAN;
    }
    return;
  }
  for (std::uint32_t e = 0; e < experts; ++e)
    probs[e] = Sigmoid(logits[e]);
  float sum = 0;
  for (std::uint32_t s = 0; s < used; ++s) {
    std::int32_t best = -1;
    float best_score = -INFINITY;
    for (std::uint32_t e = 0; e < experts; ++e) {
      bool selected = false;
      for (std::uint32_t j = 0; j < s; ++j)
        selected |= ids[j] == e;
      const float score = probs[e] + (bias ? bias[e] : 0);
      if (!selected && (best < 0 || score > best_score)) {
        best = e;
        best_score = score;
      }
    }
    ids[s] = best;
    sum += probs[best];
  }
  const float denom = normalize ? fmaxf(sum, 0x1p-14F) : 1.0F;
  for (std::uint32_t s = 0; s < used; ++s) {
    raw[s] = probs[ids[s]];
    norm[s] = raw[s] / denom;
    weights[s] = norm[s] * scale;
  }
}
// Route, one block per token: the same scores (sigmoid + bias in F32), the
// same selection rule (strictly greater wins, so a tie goes to the lower
// expert index) and the same sum order, so it is bit-identical to Route. The
// one-thread-per-token Route spent ~0.5 ms per MoE layer in decode looping
// used x experts serially. Shared memory: experts scores + selected flags.
constexpr std::uint32_t kRouteThreads = 256;
constexpr std::uint32_t kRouteBlockMaxExperts = 4096;
__global__ void __launch_bounds__(kRouteThreads)
    RouteBlock(const float* logits, const float* bias, float* probs,
               std::int32_t* ids, float* weights, float* raw, float* norm,
               std::uint32_t experts, std::uint32_t used, float scale,
               bool normalize) {
  extern __shared__ float route_shared[];
  float* score = route_shared;
  auto* taken = reinterpret_cast<unsigned char*>(score + experts);
  __shared__ float best_score[kRouteThreads];
  __shared__ std::int32_t best_id[kRouteThreads];
  __shared__ int nonfinite;
  const auto t = std::size_t{blockIdx.x};
  const unsigned tid = threadIdx.x;
  probs += t * experts;
  logits += t * experts;
  ids += t * used;
  weights += t * used;
  raw += t * used;
  norm += t * used;
  if (tid == 0)
    nonfinite = 0;
  __syncthreads();
  for (std::uint32_t e = tid; e < experts; e += kRouteThreads)
    if (!isfinite(logits[e]) || (bias && !isfinite(bias[e])))
      nonfinite = 1;
  __syncthreads();
  if (nonfinite) {
    for (std::uint32_t e = tid; e < experts; e += kRouteThreads)
      probs[e] = NAN;
    for (std::uint32_t s = tid; s < used; s += kRouteThreads) {
      ids[s] = s;
      weights[s] = raw[s] = norm[s] = NAN;
    }
    return;
  }
  for (std::uint32_t e = tid; e < experts; e += kRouteThreads) {
    const float p = Sigmoid(logits[e]);
    probs[e] = p;
    score[e] = p + (bias ? bias[e] : 0);
    taken[e] = 0;
  }
  __syncthreads();
  for (std::uint32_t s = 0; s < used; ++s) {
    // Each thread scans its experts in ascending order with strict >, so
    // its candidate is its lowest-index maximum; the tree then prefers the
    // higher score and, on equal scores, the lower index.
    std::int32_t best = -1;
    float best_value = -INFINITY;
    for (std::uint32_t e = tid; e < experts; e += kRouteThreads)
      if (!taken[e] && (best < 0 || score[e] > best_value)) {
        best = static_cast<std::int32_t>(e);
        best_value = score[e];
      }
    best_id[tid] = best;
    best_score[tid] = best_value;
    __syncthreads();
    for (unsigned stride = kRouteThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const std::int32_t o = best_id[tid + stride];
        const float ov = best_score[tid + stride];
        const std::int32_t m = best_id[tid];
        if (o >= 0 && (m < 0 || ov > best_score[tid] ||
                       (ov == best_score[tid] && o < m))) {
          best_id[tid] = o;
          best_score[tid] = ov;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      ids[s] = best_id[0];
      taken[best_id[0]] = 1;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float sum = 0;
    for (std::uint32_t s = 0; s < used; ++s)
      sum += probs[ids[s]];
    const float denom = normalize ? fmaxf(sum, 0x1p-14F) : 1.0F;
    for (std::uint32_t s = 0; s < used; ++s) {
      raw[s] = probs[ids[s]];
      norm[s] = raw[s] / denom;
      weights[s] = norm[s] * scale;
    }
  }
}
__global__ void Gather(const float* down, const float* weights, float* y,
                       std::size_t n, std::uint32_t hidden,
                       std::uint32_t used) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    const auto t = i / hidden, d = i % hidden;
    float sum = 0;
    for (std::uint32_t s = 0; s < used; ++s)
      sum = fmaf(weights[t * used + s], down[(t * used + s) * hidden + d], sum);
    y[i] = sum;
  }
}
__global__ void Add(float* y, const float* shared, std::size_t n) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x)
    y[i] += shared[i];
}
}  // namespace

std::size_t FfnScratchBytes(const Config& c, std::uint32_t tokens) noexcept {
  if (!tokens || !c.hidden_size)
    return 0;
  const Buffers b(c, tokens);
  return b.layout.valid ? b.layout.bytes : 0;
}

bool FfnForward(const OpContext& ctx, const Config& c, std::uint32_t layer,
                const FfnLayerDevice& w, const float* x, std::uint32_t tokens,
                float* y) {
  if (!tokens)
    return true;
  const Buffers b(c, tokens);
  if (!x || !y || x == y || !c.hidden_size || !ctx.scratch || !b.layout.valid ||
      ctx.scratch_bytes < b.layout.bytes ||
      reinterpret_cast<std::uintptr_t>(ctx.scratch) % kAlign ||
      reinterpret_cast<std::uintptr_t>(x) % alignof(float) ||
      reinterpret_cast<std::uintptr_t>(y) % alignof(float) ||
      !std::isfinite(w.clamp_exp) || !std::isfinite(w.clamp_shexp) ||
      w.clamp_exp < 0 || w.clamp_shexp < 0)
    return false;
  const auto h = c.hidden_size;
  const auto valid_ff = [&](const DeviceMatrix& g, const DeviceMatrix& u,
                            const DeviceMatrix& d, std::uint32_t ff,
                            std::uint32_t experts, bool routed) {
    return Matrix(g, ff, h, experts, routed) &&
           Matrix(u, ff, h, experts, routed) &&
           Matrix(d, h, ff, experts, routed);
  };
  const bool shared = c.shared_expert_ff != 0;
  if (w.dense) {
    if (!valid_ff(w.gate, w.up, w.down, c.dense_ff, 1, false))
      return false;
  } else {
    if (!w.router || reinterpret_cast<std::uintptr_t>(w.router) % 4 ||
        (w.router_bias &&
         reinterpret_cast<std::uintptr_t>(w.router_bias) % 4) ||
        !c.num_experts || c.num_experts > INT32_MAX || !c.num_experts_used ||
        c.num_experts_used > c.num_experts ||
        !std::isfinite(c.expert_weights_scale) ||
        !valid_ff(w.gate_exps, w.up_exps, w.down_exps, c.expert_ff,
                  c.num_experts, true) ||
        (shared && (c.num_shared_experts != 1 ||
                    !valid_ff(w.shexp_gate, w.shexp_up, w.shexp_down,
                              c.shared_expert_ff, 1, false))) ||
        (!shared && c.num_shared_experts))
      return false;
  }
  const auto ptr = [&](std::size_t offset) {
    return reinterpret_cast<float*>(static_cast<char*>(ctx.scratch) + offset);
  };
  auto* g = ptr(b.gate);
  auto* u = ptr(b.up);
  auto* ids = reinterpret_cast<std::int32_t*>(ptr(b.ids));
  const auto tap = [&](std::string_view name, const void* p, std::uint64_t ne0,
                       std::uint64_t ne1,
                       TapView::Dtype type = TapView::Dtype::kF32,
                       std::uint64_t ne2 = 1) {
    if (ctx.tap && *ctx.tap)
      (*ctx.tap)(name, layer, {p, type, ne0, ne1, ne2});
  };
  // GUFO_GLM53_PROFILE sub-op timers (no events or syncs when it is off).
  std::optional<OpScope> scope;
  const auto ff = [&](const DeviceMatrix& gate, const DeviceMatrix& up,
                      const DeviceMatrix& down, std::uint32_t routes,
                      const std::int32_t* selected, float clamp, float* out) {
    const bool routed = selected != nullptr;
    scope.emplace(ctx.profiler, routed ? "ffn_moe/gate_up" : "ffn/gate_up",
                  ctx.stream);
    if (UseDs4(gate, up, selected)) {
      if (!Ds4GateUp(gate, up, x, selected, g, u, tokens, routes, ctx.stream))
        return false;
    } else if (!Project(gate, x, selected, g, tokens, routes, true,
                        ctx.stream) ||
               !Project(up, x, selected, u, tokens, routes, true, ctx.stream)) {
      return false;
    }
    scope.emplace(ctx.profiler, routed ? "ffn_moe/act" : "ffn/act", ctx.stream);
    const auto n = std::size_t{tokens} * routes * gate.rows;
    Activate<<<Grid(n), 256, 0, ctx.stream>>>(g, u, n, clamp);
    if (hipGetLastError() != hipSuccess)
      return false;
    // The down projection's input: lets a test round exactly what the
    // quantized down kernels see ([tokens * routes][rows]).
    tap(selected ? "ffn_moe_act" : "ffn_act", g, gate.rows,
        std::uint64_t{tokens} * routes);
    scope.emplace(ctx.profiler, routed ? "ffn_moe/down" : "ffn/down",
                  ctx.stream);
    const bool ok =
        UseDs4Down(down, selected)
            ? Ds4Down(down, g, selected, out, tokens, routes, ctx.stream)
            : Project(down, g, selected, out, tokens, routes, false,
                      ctx.stream);
    scope.reset();
    return ok;
  };
  if (w.dense) {
    if (!ff(w.gate, w.up, w.down, 1, nullptr, w.clamp_exp, y))
      return false;
  } else {
    const DeviceMatrix router{w.router, GgmlType::kF32,
                              h,        c.num_experts,
                              1,        std::size_t{h} * sizeof(float)};
    scope.emplace(ctx.profiler, "ffn_moe/router", ctx.stream);
    if (!Project(router, x, nullptr, ptr(b.logits), tokens, 1, true,
                 ctx.stream))
      return false;
    tap("ffn_moe_logits", ptr(b.logits), c.num_experts, tokens);
    scope.emplace(ctx.profiler, "ffn_moe/topk", ctx.stream);
    if (c.num_experts <= kRouteBlockMaxExperts && tokens <= 0x7fffffffU)
      RouteBlock<<<tokens, kRouteThreads,
                   std::size_t{c.num_experts} * (sizeof(float) + 1),
                   ctx.stream>>>(
          ptr(b.logits), w.router_bias, ptr(b.probs), ids, ptr(b.weights),
          ptr(b.raw_weights), ptr(b.norm_weights), c.num_experts,
          c.num_experts_used, c.expert_weights_scale, c.expert_weights_norm);
    else
      Route<<<static_cast<unsigned>((std::size_t{tokens} + 255) / 256), 256,
              0, ctx.stream>>>(
          ptr(b.logits), w.router_bias, ptr(b.probs), ids, ptr(b.weights),
          ptr(b.raw_weights), ptr(b.norm_weights), tokens, c.num_experts,
          c.num_experts_used, c.expert_weights_scale, c.expert_weights_norm);
    if (hipGetLastError() != hipSuccess)
      return false;
    tap("ffn_moe_probs", ptr(b.probs), c.num_experts, tokens);
    tap("ffn_moe_topk", ids, c.num_experts_used, tokens, TapView::Dtype::kI32);
    tap("ffn_moe_weights", ptr(b.raw_weights), 1, c.num_experts_used,
        TapView::Dtype::kF32, tokens);
    if (c.expert_weights_norm)
      tap("ffn_moe_weights_norm", ptr(b.norm_weights), c.num_experts_used,
          tokens);
    tap("ffn_moe_weights_scaled", ptr(b.weights), 1, c.num_experts_used,
        TapView::Dtype::kF32, tokens);
    if (!ff(w.gate_exps, w.up_exps, w.down_exps, c.num_experts_used, ids,
            w.clamp_exp, ptr(b.down)))
      return false;
    const auto n = std::size_t{tokens} * h;
    scope.emplace(ctx.profiler, "ffn_moe/gather", ctx.stream);
    Gather<<<Grid(n), 256, 0, ctx.stream>>>(ptr(b.down), ptr(b.weights), y, n,
                                            h, c.num_experts_used);
    scope.reset();
    if (hipGetLastError() != hipSuccess)
      return false;
    tap("ffn_moe_out", y, h, tokens);
    if (shared) {
      if (!ff(w.shexp_gate, w.shexp_up, w.shexp_down, 1, nullptr, w.clamp_shexp,
              ptr(b.shared)))
        return false;
      tap("ffn_shexp", ptr(b.shared), h, tokens);
      Add<<<Grid(n), 256, 0, ctx.stream>>>(y, ptr(b.shared), n);
      if (hipGetLastError() != hipSuccess)
        return false;
    }
  }
  tap("ffn_out", y, h, tokens);
  return true;
}
}  // namespace gufo::models::glm53_flash
