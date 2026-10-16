#include "src/models/glm53_flash/kernels/hc.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

// Forked from src/models/deepseek_v4_flash/kernels/rocm/detail/
// ds4_rocm_hc.hip.hpp. Differences: F32 mix weights (DS4 reads F16), epsilons
// from Config, multi-token launchers with model-private names. Math follows
// HF modeling_glm5_next.py:271-329 and 1336-1338; see reference_hc.cpp.

namespace gufo::models::glm53_flash::rocm {
namespace {

constexpr std::uint32_t kMix = kHcMixDim;

__device__ __forceinline__ float WarpSum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset, 32);
  }
  return v;
}

/// ds4 hc4_split_one: pre sigma + eps, post 2 sigma, row softmax + eps, one
/// column normalisation, then sinkhorn_iters - 1 row/column passes.
__device__ void Hc4Split(float* out, const float* mix, const float* scale,
                         const float* base, std::uint32_t sinkhorn_iters,
                         float eps) {
  const float pre_scale = scale[0];
  const float post_scale = scale[1];
  const float comb_scale = scale[2];
  for (int i = 0; i < 4; i++) {
    const float z = (mix[i] * pre_scale) + base[i];
    out[i] = (1.0f / (1.0f + expf(-z))) + eps;
  }
  for (int i = 0; i < 4; i++) {
    const float z = (mix[4 + i] * post_scale) + base[4 + i];
    out[4 + i] = 2.0f / (1.0f + expf(-z));
  }
  float c[16];
  for (int r = 0; r < 4; r++) {
    float m = -INFINITY;
    for (int col = 0; col < 4; col++) {
      const float v =
          (mix[8 + (r * 4) + col] * comb_scale) + base[8 + (r * 4) + col];
      c[(r * 4) + col] = v;
      m = fmaxf(m, v);
    }
    float s = 0.0f;
    for (int col = 0; col < 4; col++) {
      const float v = expf(c[(r * 4) + col] - m);
      c[(r * 4) + col] = v;
      s += v;
    }
    for (int col = 0; col < 4; col++) {
      c[(r * 4) + col] = (c[(r * 4) + col] / s) + eps;
    }
  }
  for (int col = 0; col < 4; col++) {
    float s = eps;
    for (int r = 0; r < 4; r++)
      s += c[(r * 4) + col];
    for (int r = 0; r < 4; r++)
      c[(r * 4) + col] /= s;
  }
  for (std::uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
    for (int r = 0; r < 4; r++) {
      float s = eps;
      for (int col = 0; col < 4; col++)
        s += c[(r * 4) + col];
      for (int col = 0; col < 4; col++)
        c[(r * 4) + col] /= s;
    }
    for (int col = 0; col < 4; col++) {
      float s = eps;
      for (int r = 0; r < 4; r++)
        s += c[(r * 4) + col];
      for (int r = 0; r < 4; r++)
        c[(r * 4) + col] /= s;
    }
  }
  for (int i = 0; i < 16; i++)
    out[8 + i] = c[i];
}

/// ds4 hc4_norm_mix_split_weighted_sum_kernel with F32 weights. TOK tokens
/// per workgroup, TPT threads per token, each owning COLS contiguous columns
/// of all four streams in registers, so the collapse re-reads nothing. The
/// projection uses the raw row and applies the RMS scale to the 24 sums,
/// which equals HF's norm-then-linear up to rounding.
template<std::uint32_t TOK, std::uint32_t TPT, std::uint32_t COLS>
__global__ void __launch_bounds__(TOK* TPT)
    Hc4PreFusedKernel(float* out, float* split_out, float* logits_out,
                      const float* residual_hc, const float* mix_w,
                      const float* scale, const float* base,
                      std::uint32_t n_embd, std::uint32_t n_rows,
                      std::uint32_t sinkhorn_iters, float eps, float norm_eps) {
  constexpr std::uint32_t kWaves = TPT / 32u;
  const std::uint32_t tl = threadIdx.x / TPT;
  const std::uint32_t lane = threadIdx.x - (tl * TPT);
  const std::uint32_t wave = lane >> 5u;
  const std::uint32_t wlane = lane & 31u;
  const std::uint32_t row = (blockIdx.x * TOK) + tl;
  const bool active = row < n_rows;
  const std::uint32_t t = active ? row : 0u;
  const std::uint64_t hc_dim = 4ull * n_embd;
  const std::uint32_t col0 = lane * COLS;

  __shared__ float partial[TOK][kWaves];
  __shared__ float wavered[TOK][kWaves][kMix];
  __shared__ float mixbuf[TOK][kMix];
  __shared__ float splitbuf[TOK][kMix];

  float v[4][COLS];
  const float* rrow = residual_hc + (static_cast<std::uint64_t>(t) * hc_dim);
#pragma unroll
  for (std::uint32_t h = 0; h < 4u; h++) {
    const float* src = rrow + (static_cast<std::uint64_t>(h) * n_embd) + col0;
#pragma unroll
    for (std::uint32_t c = 0; c < COLS; c += 4u) {
      const float4 q = *reinterpret_cast<const float4*>(src + c);
      v[h][c + 0u] = q.x;
      v[h][c + 1u] = q.y;
      v[h][c + 2u] = q.z;
      v[h][c + 3u] = q.w;
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (std::uint32_t h = 0; h < 4u; h++) {
#pragma unroll
    for (std::uint32_t c = 0; c < COLS; c++)
      sum += v[h][c] * v[h][c];
  }
  sum = WarpSum(sum);
  if (wlane == 0u)
    partial[tl][wave] = sum;
  __syncthreads();
  float rowsum = 0.0f;
#pragma unroll
  for (std::uint32_t w = 0; w < kWaves; w++)
    rowsum += partial[tl][w];
  const float nscale = rsqrtf((rowsum / static_cast<float>(hc_dim)) + norm_eps);

  for (std::uint32_t j = 0; j < kMix; j++) {
    const float* wrow = mix_w + (static_cast<std::uint64_t>(j) * hc_dim);
    float a = 0.0f;
#pragma unroll
    for (std::uint32_t h = 0; h < 4u; h++) {
      const float* wsrc =
          wrow + (static_cast<std::uint64_t>(h) * n_embd) + col0;
#pragma unroll
      for (std::uint32_t c = 0; c < COLS; c += 4u) {
        const float4 w4 = *reinterpret_cast<const float4*>(wsrc + c);
        a += w4.x * v[h][c + 0u];
        a += w4.y * v[h][c + 1u];
        a += w4.z * v[h][c + 2u];
        a += w4.w * v[h][c + 3u];
      }
    }
    const float s = WarpSum(a);
    if (wlane == 0u)
      wavered[tl][wave][j] = s;
  }
  __syncthreads();
  if (lane < kMix) {
    float s = 0.0f;
#pragma unroll
    for (std::uint32_t w = 0; w < kWaves; w++)
      s += wavered[tl][w][lane];
    s *= nscale;
    mixbuf[tl][lane] = s;
    if (active && logits_out != nullptr) {
      logits_out[(static_cast<std::uint64_t>(t) * kMix) + lane] = s;
    }
  }
  __syncthreads();
  if (lane == 0u) {
    Hc4Split(splitbuf[tl], mixbuf[tl], scale, base, sinkhorn_iters, eps);
  }
  __syncthreads();
  if (lane < kMix && active) {
    split_out[(static_cast<std::uint64_t>(t) * kMix) + lane] =
        splitbuf[tl][lane];
  }

  if (!active)
    return;
  const float s0 = splitbuf[tl][0];
  const float s1 = splitbuf[tl][1];
  const float s2 = splitbuf[tl][2];
  const float s3 = splitbuf[tl][3];
  float* orow = out + (static_cast<std::uint64_t>(t) * n_embd) + col0;
#pragma unroll
  for (std::uint32_t c = 0; c < COLS; c += 4u) {
    float4 q;
    q.x = v[0][c + 0u] * s0;
    q.x += v[1][c + 0u] * s1;
    q.x += v[2][c + 0u] * s2;
    q.x += v[3][c + 0u] * s3;
    q.y = v[0][c + 1u] * s0;
    q.y += v[1][c + 1u] * s1;
    q.y += v[2][c + 1u] * s2;
    q.y += v[3][c + 1u] * s3;
    q.z = v[0][c + 2u] * s0;
    q.z += v[1][c + 2u] * s1;
    q.z += v[2][c + 2u] * s2;
    q.z += v[3][c + 2u] * s3;
    q.w = v[0][c + 3u] * s0;
    q.w += v[1][c + 3u] * s1;
    q.w += v[2][c + 3u] * s2;
    q.w += v[3][c + 3u] * s3;
    *reinterpret_cast<float4*>(orow + c) = q;
  }
}

/// Any hidden size: one 256-thread workgroup per token, strided columns.
constexpr std::uint32_t kGenericThreads = 256;

__global__ void __launch_bounds__(kGenericThreads)
    Hc4PreGenericKernel(float* out, float* split_out, float* logits_out,
                        const float* residual_hc, const float* mix_w,
                        const float* scale, const float* base,
                        std::uint32_t n_embd, std::uint32_t sinkhorn_iters,
                        float eps, float norm_eps) {
  constexpr std::uint32_t kWaves = kGenericThreads / 32u;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t wlane = tid & 31u;
  const std::uint64_t hc_dim = 4ull * n_embd;
  const float* rrow = residual_hc + (static_cast<std::uint64_t>(t) * hc_dim);

  __shared__ float wavered[kWaves][kMix + 1u];
  __shared__ float splitbuf[kMix];
  __shared__ float mixbuf[kMix];

  float acc[kMix + 1u];
#pragma unroll
  for (std::uint32_t j = 0; j <= kMix; j++)
    acc[j] = 0.0f;
  for (std::uint64_t k = tid; k < hc_dim; k += kGenericThreads) {
    const float x = rrow[k];
    acc[kMix] += x * x;
#pragma unroll
    for (std::uint32_t j = 0; j < kMix; j++) {
      acc[j] += mix_w[(static_cast<std::uint64_t>(j) * hc_dim) + k] * x;
    }
  }
#pragma unroll
  for (std::uint32_t j = 0; j <= kMix; j++) {
    const float s = WarpSum(acc[j]);
    if (wlane == 0u)
      wavered[wave][j] = s;
  }
  __syncthreads();
  if (tid <= kMix) {
    float s = 0.0f;
#pragma unroll
    for (std::uint32_t w = 0; w < kWaves; w++)
      s += wavered[w][tid];
    wavered[0][tid] = s;  // Only wave 0's slot is reused; read below.
  }
  __syncthreads();
  if (tid < kMix) {
    const float nscale =
        rsqrtf((wavered[0][kMix] / static_cast<float>(hc_dim)) + norm_eps);
    const float s = wavered[0][tid] * nscale;
    mixbuf[tid] = s;
    if (logits_out != nullptr) {
      logits_out[(static_cast<std::uint64_t>(t) * kMix) + tid] = s;
    }
  }
  __syncthreads();
  if (tid == 0u)
    Hc4Split(splitbuf, mixbuf, scale, base, sinkhorn_iters, eps);
  __syncthreads();
  if (tid < kMix) {
    split_out[(static_cast<std::uint64_t>(t) * kMix) + tid] = splitbuf[tid];
  }
  float* orow = out + (static_cast<std::uint64_t>(t) * n_embd);
  for (std::uint32_t d = tid; d < n_embd; d += kGenericThreads) {
    float q = rrow[d] * splitbuf[0];
    q += rrow[n_embd + d] * splitbuf[1];
    q += rrow[(2ull * n_embd) + d] * splitbuf[2];
    q += rrow[(3ull * n_embd) + d] * splitbuf[3];
    orow[d] = q;
  }
}

/// ds4 hc_expand4_vec4_kernel: four adjacent columns per thread.
__global__ void Hc4PostVec4Kernel(float* out_hc, const float* block_out,
                                  const float* residual_hc, const float* split,
                                  std::uint32_t n_embd,
                                  std::uint32_t n_tokens) {
  const std::uint32_t n_vec = n_embd >> 2u;
  const std::uint64_t gid =
      (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (gid >= static_cast<std::uint64_t>(n_tokens) * n_vec)
    return;
  const auto dv = static_cast<std::uint32_t>(gid % n_vec);
  const auto t = static_cast<std::uint32_t>(gid / n_vec);
  const std::uint64_t td =
      (static_cast<std::uint64_t>(t) * n_embd) + (dv * 4ull);
  const std::uint64_t hc_base =
      (static_cast<std::uint64_t>(t) * 4u * n_embd) + (dv * 4ull);
  const std::uint64_t e = n_embd;
  const float4 bv = *reinterpret_cast<const float4*>(block_out + td);
  const float4 r0 = *reinterpret_cast<const float4*>(residual_hc + hc_base);
  const float4 r1 = *reinterpret_cast<const float4*>(residual_hc + hc_base + e);
  const float4 r2 =
      *reinterpret_cast<const float4*>(residual_hc + hc_base + (2u * e));
  const float4 r3 =
      *reinterpret_cast<const float4*>(residual_hc + hc_base + (3u * e));
  const float* sp = split + (static_cast<std::uint64_t>(t) * kMix);
  const float* post = sp + 4u;
  const float* comb = sp + 8u;
#pragma unroll
  for (std::uint32_t dst = 0; dst < 4u; dst++) {
    const float p = post[dst];
    const float c0 = comb[0u + dst];
    const float c1 = comb[4u + dst];
    const float c2 = comb[8u + dst];
    const float c3 = comb[12u + dst];
    float4 acc;
    acc.x = bv.x * p;
    acc.x += c0 * r0.x;
    acc.x += c1 * r1.x;
    acc.x += c2 * r2.x;
    acc.x += c3 * r3.x;
    acc.y = bv.y * p;
    acc.y += c0 * r0.y;
    acc.y += c1 * r1.y;
    acc.y += c2 * r2.y;
    acc.y += c3 * r3.y;
    acc.z = bv.z * p;
    acc.z += c0 * r0.z;
    acc.z += c1 * r1.z;
    acc.z += c2 * r2.z;
    acc.z += c3 * r3.z;
    acc.w = bv.w * p;
    acc.w += c0 * r0.w;
    acc.w += c1 * r1.w;
    acc.w += c2 * r2.w;
    acc.w += c3 * r3.w;
    *reinterpret_cast<float4*>(out_hc + hc_base + (dst * e)) = acc;
  }
}

/// ds4 hc_expand4_kernel: scalar form for hidden sizes not divisible by 4.
__global__ void Hc4PostKernel(float* out_hc, const float* block_out,
                              const float* residual_hc, const float* split,
                              std::uint32_t n_embd, std::uint32_t n_tokens) {
  const std::uint64_t gid =
      (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (gid >= static_cast<std::uint64_t>(n_tokens) * n_embd)
    return;
  const auto d = static_cast<std::uint32_t>(gid % n_embd);
  const auto t = static_cast<std::uint32_t>(gid / n_embd);
  const std::uint64_t e = n_embd;
  const std::uint64_t hc_base = (static_cast<std::uint64_t>(t) * 4u * e) + d;
  const float bv = block_out[(static_cast<std::uint64_t>(t) * e) + d];
  const float r0 = residual_hc[hc_base];
  const float r1 = residual_hc[hc_base + e];
  const float r2 = residual_hc[hc_base + (2u * e)];
  const float r3 = residual_hc[hc_base + (3u * e)];
  const float* sp = split + (static_cast<std::uint64_t>(t) * kMix);
  const float* post = sp + 4u;
  const float* comb = sp + 8u;
#pragma unroll
  for (std::uint32_t dst = 0; dst < 4u; dst++) {
    float acc = bv * post[dst];
    acc += comb[0u + dst] * r0;
    acc += comb[4u + dst] * r1;
    acc += comb[8u + dst] * r2;
    acc += comb[12u + dst] * r3;
    out_hc[hc_base + (dst * e)] = acc;
  }
}

__global__ void Hc4HeadMeanKernel(float* out, const float* hc,
                                  std::uint32_t n_embd,
                                  std::uint32_t n_tokens) {
  const std::uint64_t gid =
      (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (gid >= static_cast<std::uint64_t>(n_tokens) * n_embd)
    return;
  const auto d = static_cast<std::uint32_t>(gid % n_embd);
  const auto t = static_cast<std::uint32_t>(gid / n_embd);
  const std::uint64_t e = n_embd;
  const float* row = hc + (static_cast<std::uint64_t>(t) * 4u * e) + d;
  float acc = row[0];
  acc += row[e];
  acc += row[2u * e];
  acc += row[3u * e];
  out[(static_cast<std::uint64_t>(t) * e) + d] = acc * 0.25f;
}

bool Aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0u;
}

unsigned Blocks(std::uint64_t n, unsigned threads) {
  return static_cast<unsigned>((n + threads - 1) / threads);
}

}  // namespace

bool HcPre(const float* H, const float* fn, const float* base,
           const float* scale, float* x, float* split, float* logits,
           std::uint32_t n_tokens, std::uint32_t hidden,
           std::uint32_t sinkhorn_iters, float hc_eps, float rms_eps,
           hipStream_t stream) {
  if (H == nullptr || fn == nullptr || base == nullptr || scale == nullptr ||
      x == nullptr || split == nullptr || hidden == 0 || sinkhorn_iters == 0) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  constexpr std::uint32_t kTok = 4u;
  constexpr std::uint32_t kTpt = 256u;
  constexpr std::uint32_t kCols = 16u;
  if (hidden == kTpt * kCols && Aligned16(H) && Aligned16(fn) && Aligned16(x)) {
    Hc4PreFusedKernel<kTok, kTpt, kCols>
        <<<(n_tokens + kTok - 1u) / kTok, kTok * kTpt, 0, stream>>>(
            x, split, logits, H, fn, scale, base, hidden, n_tokens,
            sinkhorn_iters, hc_eps, rms_eps);
  } else {
    Hc4PreGenericKernel<<<n_tokens, kGenericThreads, 0, stream>>>(
        x, split, logits, H, fn, scale, base, hidden, sinkhorn_iters, hc_eps,
        rms_eps);
  }
  return hipGetLastError() == hipSuccess;
}

bool HcPost(const float* a, const float* H_old, const float* split,
            float* H_new, std::uint32_t n_tokens, std::uint32_t hidden,
            hipStream_t stream) {
  if (a == nullptr || H_old == nullptr || split == nullptr ||
      H_new == nullptr || hidden == 0) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  constexpr unsigned kThreads = 256;
  if ((hidden & 3u) == 0u && Aligned16(a) && Aligned16(H_old) &&
      Aligned16(H_new)) {
    const std::uint64_t n = static_cast<std::uint64_t>(n_tokens) * (hidden / 4);
    Hc4PostVec4Kernel<<<Blocks(n, kThreads), kThreads, 0, stream>>>(
        H_new, a, H_old, split, hidden, n_tokens);
  } else {
    const std::uint64_t n = static_cast<std::uint64_t>(n_tokens) * hidden;
    Hc4PostKernel<<<Blocks(n, kThreads), kThreads, 0, stream>>>(
        H_new, a, H_old, split, hidden, n_tokens);
  }
  return hipGetLastError() == hipSuccess;
}

bool HcHeadMean(const float* H, float* h, std::uint32_t n_tokens,
                std::uint32_t hidden, hipStream_t stream) {
  if (H == nullptr || h == nullptr || hidden == 0)
    return false;
  if (n_tokens == 0)
    return true;
  constexpr unsigned kThreads = 256;
  const std::uint64_t n = static_cast<std::uint64_t>(n_tokens) * hidden;
  Hc4HeadMeanKernel<<<Blocks(n, kThreads), kThreads, 0, stream>>>(h, H, hidden,
                                                                  n_tokens);
  return hipGetLastError() == hipSuccess;
}

}  // namespace gufo::models::glm53_flash::rocm
