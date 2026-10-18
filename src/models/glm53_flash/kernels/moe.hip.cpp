#include "src/models/glm53_flash/kernels/moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "src/models/glm53_flash/kernels/mmq/iq.hpp"
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
    double sum = 0;
    for (std::size_t k = lane; k < cols; k += kWave) {
      const float v =
          bf16
              ? __uint_as_float(std::uint32_t{static_cast<const std::uint16_t*>(
                                    w)[offset + k]}
                                << 16)
              : static_cast<const float*>(w)[offset + k];
      sum += static_cast<double>(v) * in[k];
    }
    for (unsigned delta = kWave / 2; delta; delta /= 2)
      sum += __shfl_down(sum, delta, kWave);
    if (lane == 0)
      y[i] = static_cast<float>(sum);
  }
}
unsigned Grid(std::size_t n) {
  return static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 65535));
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
      FloatProjectWave<<<grid, 256, 0, stream>>>(
          w.data, w.type == GgmlType::kBF16, x, ids, y, n, routes, w.rows,
          w.cols, broadcast);
    }
  } else {
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
  const auto ff = [&](const DeviceMatrix& gate, const DeviceMatrix& up,
                      const DeviceMatrix& down, std::uint32_t routes,
                      const std::int32_t* selected, float clamp, float* out) {
    if (!Project(gate, x, selected, g, tokens, routes, true, ctx.stream) ||
        !Project(up, x, selected, u, tokens, routes, true, ctx.stream))
      return false;
    const auto n = std::size_t{tokens} * routes * gate.rows;
    Activate<<<Grid(n), 256, 0, ctx.stream>>>(g, u, n, clamp);
    if (hipGetLastError() != hipSuccess)
      return false;
    return Project(down, g, selected, out, tokens, routes, false, ctx.stream);
  };
  if (w.dense) {
    if (!ff(w.gate, w.up, w.down, 1, nullptr, w.clamp_exp, y))
      return false;
  } else {
    const DeviceMatrix router{w.router, GgmlType::kF32,
                              h,        c.num_experts,
                              1,        std::size_t{h} * sizeof(float)};
    if (!Project(router, x, nullptr, ptr(b.logits), tokens, 1, true,
                 ctx.stream))
      return false;
    tap("ffn_moe_logits", ptr(b.logits), c.num_experts, tokens);
    Route<<<static_cast<unsigned>((std::size_t{tokens} + 255) / 256), 256, 0,
            ctx.stream>>>(
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
    Gather<<<Grid(n), 256, 0, ctx.stream>>>(ptr(b.down), ptr(b.weights), y, n,
                                            h, c.num_experts_used);
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
