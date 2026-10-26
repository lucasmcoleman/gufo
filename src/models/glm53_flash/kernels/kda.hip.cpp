#include "src/models/glm53_flash/kernels/kda.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

// Forked from src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp
// (SsmConv4Kernel, HistoryShiftKernel, GdnPrepKernel, GdnPrepAbKernel,
// GdnKernel, GdnEpilogueKernel) without the batch, rollback-snapshot,
// checkpoint and quantized-output paths. Math follows HF
// modeling_glm5_next.py:353-369 (gate), 409-459 (conv, l2norm), 497-511
// (recurrence) and 381-393 (gated RMSNorm); see reference_kda.cpp.

namespace gufo::models::glm53_flash::rocm {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWave = 32;  // row kernels use 32-lane groups
constexpr std::uint32_t kD = kKdaHeadDim;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SoftplusF(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}

/// Sum over a 32-lane group; every lane of the group receives the total.
__device__ __forceinline__ float GroupSum32(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, kWave);
  }
  return v;
}

/// QFN SsmConv4Kernel: one thread per (channel, kTokensPerThread tokens);
/// the taps stay in one float4 and the window slides in registers. With a
/// single token tile (n_tokens <= kTokensPerThread) the thread also owns
/// its channel's history update.
constexpr std::uint32_t kConvTokensPerThread = 8;
template<bool kSaveHistory>
__global__ void KdaConv4Kernel(const float* qkv, std::uint32_t qkv_stride,
                               const float* w, float* conv_state, float* out,
                               std::uint32_t n_tokens, std::uint32_t channels,
                               float* checkpoint) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  const std::uint32_t t0 = blockIdx.y * kConvTokensPerThread;
  const float4 taps = *reinterpret_cast<const float4*>(w + c * kKdaConvTaps);
  // window[i] holds token t0 - 3 + i.
  float window[kKdaConvTaps - 1];
#pragma unroll
  for (std::uint32_t i = 0; i < kKdaConvTaps - 1; ++i) {
    const std::int32_t src_t =
        static_cast<std::int32_t>(t0) - 3 + static_cast<std::int32_t>(i);
    window[i] =
        src_t >= 0
            ? qkv[static_cast<std::size_t>(src_t) * qkv_stride + c]
            : conv_state[static_cast<std::size_t>(kKdaConvTaps - 1 + src_t) *
                             channels +
                         c];
  }
#pragma unroll
  for (std::uint32_t i = 0; i < kConvTokensPerThread; ++i) {
    const std::uint32_t t = t0 + i;
    if (t >= n_tokens) {
      break;
    }
    const float v = qkv[static_cast<std::size_t>(t) * qkv_stride + c];
    const float acc = taps.x * window[0] + taps.y * window[1] +
                      taps.z * window[2] + taps.w * v;
    out[static_cast<std::size_t>(t) * channels + c] = SiluF(acc);
    window[0] = window[1];
    window[1] = window[2];
    window[2] = v;
    if constexpr (kSaveHistory) {
      // The history after token t, for a rollback that keeps t + 1 tokens.
      if (checkpoint != nullptr && t + 1 < n_tokens) {
        float* ck = checkpoint + (std::size_t{t} * (kKdaConvTaps - 1) * channels);
#pragma unroll
        for (std::uint32_t j = 0; j < kKdaConvTaps - 1; ++j)
          ck[std::size_t{j} * channels + c] = window[j];
      }
    }
  }
  if constexpr (kSaveHistory) {
#pragma unroll
    for (std::uint32_t j = 0; j < kKdaConvTaps - 1; ++j)
      conv_state[std::size_t{j} * channels + c] = window[j];
  }
}

/// History update after a multi-tile conv: the new history is the last
/// three positions of [conv_state ; qkv]. One thread per channel walks the
/// rows in ascending order, so an in-place read is always ahead of the
/// row being written.
__global__ void KdaConvHistoryKernel(const float* qkv, std::uint32_t qkv_stride,
                                     float* conv_state, std::uint32_t n_tokens,
                                     std::uint32_t channels) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  constexpr std::uint32_t hist = kKdaConvTaps - 1;
#pragma unroll
  for (std::uint32_t r = 0; r < hist; ++r) {
    const std::uint32_t p = n_tokens + r;  // position in [history ; qkv]
    conv_state[std::size_t{r} * channels + c] =
        p < hist ? conv_state[std::size_t{p} * channels + c]
                 : qkv[static_cast<std::size_t>(p - hist) * qkv_stride + c];
  }
}

/// QFN GdnPrepAbKernel's role with KDA's per-channel gate: one thread per
/// (token, head, key channel).
__global__ void KdaGateKernel(const float* f, std::uint32_t f_stride,
                              const float* dt_bias, const float* a,
                              float lower_bound, bool bounded, float* g,
                              float* decay, std::size_t count,
                              std::uint32_t width) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const std::size_t t = idx / width;
  const std::uint32_t i = idx % width;
  const float a_h = a[i / kD];  // -exp(A_log)
  const float z = f[t * f_stride + i] + dt_bias[i];
  const float log_decay =
      bounded ? lower_bound * SigmoidF(-a_h * z) : a_h * SoftplusF(z);
  if (g != nullptr) {
    g[idx] = log_decay;
  }
  decay[idx] = __expf(log_decay);
}

/// QFN GdnPrepKernel plus the beta sigmoid: one 32-lane group per (token,
/// head) row normalizes q and k (HF l2norm: x / sqrt(sum x^2 + eps)) and
/// folds the 1/sqrt(D) query scale in.
__global__ void KdaPrepKernel(const float* conv_out, std::uint32_t conv_stride,
                              const float* beta_in, std::uint32_t beta_stride,
                              float* qn, float* kn, float* beta,
                              std::uint32_t n_rows, std::uint32_t heads,
                              float eps) {
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave;
  if (row >= n_rows) {
    return;
  }
  const std::uint32_t t = row / heads;
  const std::uint32_t h = row % heads;
  const float* src = conv_out + static_cast<std::size_t>(t) * conv_stride;
  constexpr std::uint32_t per_lane = kD / kWave;
  float qv[per_lane];
  float kv[per_lane];
  float qs = 0.0f;
  float ks = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * kWave + lane;
    qv[r] = src[h * kD + i];
    kv[r] = src[heads * kD + h * kD + i];
    qs += qv[r] * qv[r];
    ks += kv[r] * kv[r];
  }
  qs = rsqrtf(GroupSum32(qs) + eps) * rsqrtf(static_cast<float>(kD));
  ks = rsqrtf(GroupSum32(ks) + eps);
  float* q_out = qn + static_cast<std::size_t>(row) * kD;
  float* k_out = kn + static_cast<std::size_t>(row) * kD;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * kWave + lane;
    q_out[i] = qv[r] * qs;
    k_out[i] = kv[r] * ks;
  }
  if (lane == 0) {
    beta[row] =
        SigmoidF(beta_in[static_cast<std::size_t>(t) * beta_stride + h]);
  }
}

/// QFN GdnKernel with a per-key decay vector. grid (heads, D / kRows):
/// each block owns kRows value rows of one head's (transposed) state, kLanes
/// threads per row, each lane a contiguous 32-key slice. Rows of the delta
/// rule are independent, so the token loop is serial and barrier-free; the
/// only reductions are inside a lane group.
constexpr std::uint32_t kRecLanes = 4;
constexpr std::uint32_t kRecRows = 32;
__global__ void __launch_bounds__(kRecRows* kRecLanes)
    KdaRecurrentKernel(const float* qn, const float* kn, const float* v,
                       std::uint32_t v_stride, const float* decay,
                       const float* beta, float* state, float* o,
                       std::uint32_t n_tokens, std::uint32_t heads,
                       float* checkpoint) {
  constexpr std::uint32_t slice = kD / kRecLanes;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t j = blockIdx.y * kRecRows + threadIdx.x / kRecLanes;
  const std::uint32_t lane = threadIdx.x % kRecLanes;
  const std::uint32_t i0 = lane * slice;
  float* S = state + static_cast<std::size_t>(h) * kD * kD + j * kD + i0;
  float row[slice];
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    row[i] = S[i];
  }
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::size_t th = static_cast<std::size_t>(t) * heads + h;
    const float* q = qn + th * kD + i0;
    const float* k = kn + th * kD + i0;
    const float* dec = decay + th * kD + i0;
    const float vv = v[static_cast<std::size_t>(t) * v_stride + h * kD + j];
    const float b = beta[th];
    float kr[slice];
    float u = 0.0f;
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      kr[i] = k[i];
      // S <- diag(exp g) S: key i (a column here) scales by decay[i].
      row[i] *= dec[i];
      u += row[i] * kr[i];
    }
#pragma unroll
    for (unsigned off = kRecLanes / 2; off > 0; off >>= 1) {
      u += __shfl_xor(u, off, kRecLanes);
    }
    const float delta = (vv - u) * b;
    float acc = 0.0f;
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      row[i] = __builtin_fmaf(delta, kr[i], row[i]);
      acc += row[i] * q[i];
    }
#pragma unroll
    for (unsigned off = kRecLanes / 2; off > 0; off >>= 1) {
      acc += __shfl_xor(acc, off, kRecLanes);
    }
    if (lane == 0) {
      o[th * kD + j] = acc;
    }
    // The state after token t, for a rollback that keeps t + 1 tokens.
    if (checkpoint != nullptr && t + 1 < n_tokens) {
      float* ck = checkpoint + (std::size_t{t} * heads * kD * kD) +
                  (static_cast<std::size_t>(h) * kD * kD) + (j * kD) + i0;
#pragma unroll
      for (std::uint32_t i = 0; i < slice; ++i) {
        ck[i] = row[i];
      }
    }
  }
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    S[i] = row[i];
  }
}

/// QFN GdnEpilogueKernel (F32 output path): per-head RMSNorm of the raw
/// recurrence rows and the sigmoid output gate, one 32-lane group per
/// (token, head) row.
__global__ void KdaOutputKernel(const float* o, const float* gate,
                                std::uint32_t gate_stride, const float* norm_w,
                                float* out, std::uint32_t n_rows,
                                std::uint32_t heads, float eps) {
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::size_t row =
      blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave;
  if (row >= n_rows) {
    return;
  }
  const float* zrow = gate + (row / heads) * gate_stride + (row % heads) * kD;
  constexpr std::uint32_t per_lane = kD / kWave;
  const float* src = o + row * kD;
  float v[per_lane];
  float ss = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    v[r] = src[r * kWave + lane];
    ss += v[r] * v[r];
  }
  const float scale = rsqrtf(GroupSum32(ss) / static_cast<float>(kD) + eps);
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * kWave + lane;
    out[row * kD + i] = v[r] * scale * norm_w[i] * SigmoidF(zrow[i]);
  }
}

bool Aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0u;
}

unsigned Blocks(std::uint64_t n, unsigned threads) {
  return static_cast<unsigned>((n + threads - 1) / threads);
}

}  // namespace

bool KdaConv(const float* qkv, std::uint32_t qkv_stride, const float* conv_w,
             float* conv_state, float* conv_out, std::uint32_t n_tokens,
             std::uint32_t channels, hipStream_t stream, float* checkpoint) {
  if (qkv == nullptr || conv_w == nullptr || conv_state == nullptr ||
      conv_out == nullptr || channels == 0 || qkv_stride < channels ||
      !Aligned16(conv_w)) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  if (checkpoint != nullptr && n_tokens > kConvTokensPerThread)
    return false;
  if (n_tokens <= kConvTokensPerThread) {
    KdaConv4Kernel<true><<<Blocks(channels, kThreads), kThreads, 0, stream>>>(
        qkv, qkv_stride, conv_w, conv_state, conv_out, n_tokens, channels,
        checkpoint);
  } else {
    const dim3 grid(Blocks(channels, kThreads),
                    Blocks(n_tokens, kConvTokensPerThread));
    KdaConv4Kernel<false><<<grid, kThreads, 0, stream>>>(
        qkv, qkv_stride, conv_w, conv_state, conv_out, n_tokens, channels,
        nullptr);
    // Ordered after every tile has read the old history.
    KdaConvHistoryKernel<<<Blocks(channels, kThreads), kThreads, 0, stream>>>(
        qkv, qkv_stride, conv_state, n_tokens, channels);
  }
  return hipGetLastError() == hipSuccess;
}

bool KdaGate(const float* f, std::uint32_t f_stride, const float* dt_bias,
             const float* a, float lower_bound, float* g, float* decay,
             std::uint32_t n_tokens, std::uint32_t heads, hipStream_t stream) {
  const std::uint32_t width = heads * kD;
  if (f == nullptr || dt_bias == nullptr || a == nullptr || decay == nullptr ||
      heads == 0 || f_stride < width || std::isnan(lower_bound)) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * width;
  KdaGateKernel<<<Blocks(count, kThreads), kThreads, 0, stream>>>(
      f, f_stride, dt_bias, a, lower_bound, std::isfinite(lower_bound), g,
      decay, count, width);
  return hipGetLastError() == hipSuccess;
}

bool KdaPrep(const float* conv_out, std::uint32_t conv_stride,
             const float* beta_in, std::uint32_t beta_stride, float* qn,
             float* kn, float* beta, std::uint32_t n_tokens,
             std::uint32_t heads, float eps, hipStream_t stream) {
  if (conv_out == nullptr || beta_in == nullptr || qn == nullptr ||
      kn == nullptr || beta == nullptr || heads == 0 ||
      conv_stride < 2 * heads * kD || beta_stride < heads) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  const std::uint32_t rows = n_tokens * heads;
  constexpr unsigned groups = kThreads / kWave;
  KdaPrepKernel<<<Blocks(rows, groups), kThreads, 0, stream>>>(
      conv_out, conv_stride, beta_in, beta_stride, qn, kn, beta, rows, heads,
      eps);
  return hipGetLastError() == hipSuccess;
}

bool KdaRecurrent(const float* qn, const float* kn, const float* v,
                  std::uint32_t v_stride, const float* decay, const float* beta,
                  float* state, float* o, std::uint32_t n_tokens,
                  std::uint32_t heads, hipStream_t stream, float* checkpoint) {
  if (qn == nullptr || kn == nullptr || v == nullptr || decay == nullptr ||
      beta == nullptr || state == nullptr || o == nullptr || heads == 0 ||
      v_stride < heads * kD) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  KdaRecurrentKernel<<<dim3(heads, kD / kRecRows), kRecRows * kRecLanes, 0,
                       stream>>>(qn, kn, v, v_stride, decay, beta, state, o,
                                 n_tokens, heads, checkpoint);
  return hipGetLastError() == hipSuccess;
}

bool KdaOutput(const float* o, const float* gate, std::uint32_t gate_stride,
               const float* norm_w, float eps, float* out,
               std::uint32_t n_tokens, std::uint32_t heads,
               hipStream_t stream) {
  if (o == nullptr || gate == nullptr || norm_w == nullptr || out == nullptr ||
      heads == 0 || gate_stride < heads * kD) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  const std::uint32_t rows = n_tokens * heads;
  constexpr unsigned groups = kThreads / kWave;
  KdaOutputKernel<<<Blocks(rows, groups), kThreads, 0, stream>>>(
      o, gate, gate_stride, norm_w, out, rows, heads, eps);
  return hipGetLastError() == hipSuccess;
}

namespace {

constexpr std::uint32_t kLowRankCols = 128;
constexpr std::uint32_t kLowRankRows = 64;    // weight rows per block
constexpr std::uint32_t kLowRankTokens = 32;  // tokens per block
constexpr std::uint32_t kLowRankPad = kLowRankCols + 1;

// Thread (r, g) = (tid % 64, tid / 64) owns row r of the tile and tokens
// g * 8 .. g * 8 + 7. LDS rows are padded to 129 floats: lanes of a wave
// read consecutive rows at the same k without bank conflicts, and the token
// operand is a broadcast.
__global__ void __launch_bounds__(kThreads)
    LowRankQ8Kernel(const std::uint8_t* __restrict__ w,
                    const float* __restrict__ x, float* __restrict__ y,
                    std::uint32_t rows, std::uint32_t n_tokens) {
  __shared__ float wl[kLowRankRows * kLowRankPad];
  __shared__ float xl[kLowRankTokens * kLowRankPad];
  constexpr std::uint32_t kRowBytes = (kLowRankCols / 32) * 34;
  const std::uint32_t r0 = blockIdx.x * kLowRankRows;
  const std::uint32_t t0 = blockIdx.y * kLowRankTokens;
  for (std::uint32_t i = threadIdx.x; i < kLowRankRows * kLowRankCols;
       i += kThreads) {
    const std::uint32_t r = i / kLowRankCols;
    const std::uint32_t k = i % kLowRankCols;
    float v = 0.0f;
    if (r0 + r < rows) {
      const std::uint8_t* blk =
          w + (std::size_t{r0 + r} * kRowBytes) + ((k / 32) * 34);
      const float d = __half2float(
          __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(blk)));
      v = d * static_cast<float>(static_cast<std::int8_t>(blk[2 + (k % 32)]));
    }
    wl[(r * kLowRankPad) + k] = v;
  }
  for (std::uint32_t i = threadIdx.x; i < kLowRankTokens * kLowRankCols;
       i += kThreads) {
    const std::uint32_t t = i / kLowRankCols;
    const std::uint32_t k = i % kLowRankCols;
    xl[(t * kLowRankPad) + k] =
        t0 + t < n_tokens ? x[(std::size_t{t0 + t} * kLowRankCols) + k] : 0.0f;
  }
  __syncthreads();
  const std::uint32_t r = threadIdx.x % kLowRankRows;
  const std::uint32_t g = threadIdx.x / kLowRankRows;
  constexpr std::uint32_t kPerThread =
      kLowRankTokens / (kThreads / kLowRankRows);
  float acc[kPerThread];
#pragma unroll
  for (std::uint32_t j = 0; j < kPerThread; ++j) {
    acc[j] = 0.0f;
  }
  const float* wr = wl + (r * kLowRankPad);
  const float* xg = xl + (g * kPerThread * kLowRankPad);
  for (std::uint32_t k = 0; k < kLowRankCols; ++k) {
    const float wv = wr[k];
#pragma unroll
    for (std::uint32_t j = 0; j < kPerThread; ++j) {
      acc[j] = fmaf(wv, xg[(j * kLowRankPad) + k], acc[j]);
    }
  }
  if (r0 + r >= rows) {
    return;
  }
#pragma unroll
  for (std::uint32_t j = 0; j < kPerThread; ++j) {
    const std::uint32_t t = t0 + (g * kPerThread) + j;
    if (t < n_tokens) {
      y[(std::size_t{t} * rows) + r0 + r] = acc[j];
    }
  }
}

}  // namespace

bool KdaLowRankQ8(const void* w, std::uint32_t rows, std::uint32_t cols,
                  const float* x, float* y, std::uint32_t n_tokens,
                  hipStream_t stream) {
  if (n_tokens == 0) {
    return true;
  }
  if (w == nullptr || x == nullptr || y == nullptr || rows == 0 ||
      cols != kLowRankCols || reinterpret_cast<std::uintptr_t>(w) % 2 != 0) {
    return false;
  }
  const dim3 grid((rows + kLowRankRows - 1) / kLowRankRows,
                  (n_tokens + kLowRankTokens - 1) / kLowRankTokens);
  LowRankQ8Kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(w), x, y, rows, n_tokens);
  return hipPeekAtLastError() == hipSuccess;
}

}  // namespace gufo::models::glm53_flash::rocm
