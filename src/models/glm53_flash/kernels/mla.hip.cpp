// Absorbed NoPE MLA with the pooled DSA indexer for GLM-5.3-Flash (HF
// modeling_glm5_next.py 1174-1235, llama.cpp glm5-next.cpp build_dsa_layer).
// The scalar contract is ReferenceModel::Mla (reference.cpp): q_resid =
// RMSNorm(q_a x), q = q_b q_resid, latent = RMSNorm(kv_a x) (the cache row,
// K = V), q absorbed through k_b per head, softmax(q_abs . latent /
// sqrt(qk_dim)) over the indexer's selection (top pools + raw tail, causal),
// v_b per head, then attn_output. Layouts are in kernels/mla.hpp.

#include "src/models/glm53_flash/kernels/mla.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <string_view>
#include <type_traits>

#include <hipblas/hipblas.h>

#include "src/core/hip/managed_alloc.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/kernels/prefill_gemm.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"
#include "src/models/glm53_flash/profile.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::models::glm53_flash {
namespace mla {
namespace {

using core::GgmlType;

constexpr unsigned kWave = 32;
constexpr unsigned kThreads = 256;
constexpr std::size_t kAlign = 256;
constexpr std::size_t kMaxLds = 64 * 1024;

constexpr std::size_t AlignUp(std::size_t n) noexcept {
  return (n + kAlign - 1) / kAlign * kAlign;
}

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, kWave);
  }
  return v;
}

/// Sum (or max) of one value per thread over a kThreads block, fixed tree
/// order. `red` holds kThreads floats; every thread gets the result.
template<bool kMax>
__device__ float BlockReduce(float v, float* red) {
  red[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float o = red[threadIdx.x + s];
      red[threadIdx.x] =
          kMax ? fmaxf(red[threadIdx.x], o) : red[threadIdx.x] + o;
    }
    __syncthreads();
  }
  const float r = red[0];
  __syncthreads();
  return r;
}

/// Y[t][r] = sum_c W[r][c] X[t][c], F32 weights, a wave per output.
__global__ void MatmulF32Kernel(const float* w, const float* x, float* y,
                                std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t r =
      (blockIdx.x * (blockDim.x / kWave)) + (threadIdx.x / kWave);
  const std::uint32_t t = blockIdx.y;
  if (r >= rows) {
    return;
  }
  const float* wr = w + (std::size_t{r} * cols);
  const float* xt = x + (std::size_t{t} * cols);
  float acc = 0.0F;
  for (std::uint32_t c = lane; c < cols; c += kWave) {
    acc += wr[c] * xt[c];
  }
  acc = WaveSum(acc);
  if (lane == 0) {
    y[(std::size_t{t} * rows) + r] = acc;
  }
}

/// MatmulF32Kernel for every head at once: blockIdx.z is the head. Weights
/// are [heads][rows][cols], X [heads][tokens][cols], Y [heads][tokens][rows].
/// Per-output arithmetic (lane-strided accumulation, WaveSum) is identical to
/// MatmulF32Kernel, so the result is bit-identical to a launch per head.
__global__ void MatmulF32HeadsKernel(const float* w, const float* x, float* y,
                                     std::uint32_t rows, std::uint32_t cols,
                                     std::uint32_t tokens) {
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t r =
      (blockIdx.x * (blockDim.x / kWave)) + (threadIdx.x / kWave);
  const std::uint32_t t = blockIdx.y;
  const std::size_t head = blockIdx.z;
  if (r >= rows) {
    return;
  }
  const float* wr = w + (((head * rows) + r) * cols);
  const float* xt = x + (((head * tokens) + t) * cols);
  float acc = 0.0F;
  for (std::uint32_t c = lane; c < cols; c += kWave) {
    acc += wr[c] * xt[c];
  }
  acc = WaveSum(acc);
  if (lane == 0) {
    y[(((head * tokens) + t) * rows) + r] = acc;
  }
}

constexpr unsigned kQ6KBlockBytes = 210;  ///< ql[128] qh[64] scales[16] d
constexpr unsigned kQ6KBlockElems = 256;
constexpr unsigned kHeadsTokenTile = 32;  ///< tokens per block in z

/// Q6_K weights for every head at once. A wave owns one weight row of
/// kBlocks Q6_K blocks (cols = 256 * kBlocks), dequantises it once into
/// registers (d * scale * (q - 32), ggml's DequantizeQ6_K), then dots it with
/// each token of its tile, summing the lane partials with WaveSum. Lane l
/// holds elements blk * 256 + half * 128 + quad * 32 + l, so weight and
/// activation loads are coalesced. The reduction order differs from gufo's
/// Q6_K GEMMs (sequential per lane, then a wave tree); results agree to
/// F32 rounding.
template<unsigned kBlocks>
__global__ void MatmulQ6KHeadsKernel(const std::uint8_t* w, const float* x,
                                     float* y, std::uint32_t rows,
                                     std::uint32_t tokens) {
  constexpr std::uint32_t kCols = kBlocks * kQ6KBlockElems;
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t r =
      (blockIdx.x * (blockDim.x / kWave)) + (threadIdx.x / kWave);
  const std::size_t head = blockIdx.y;
  if (r >= rows) {
    return;
  }
  const std::uint8_t* row =
      w + (((head * rows) + r) * (std::size_t{kBlocks} * kQ6KBlockBytes));
  float wv[kBlocks * 8];
#pragma unroll
  for (unsigned b = 0; b < kBlocks; ++b) {
    const std::uint8_t* blk = row + (b * kQ6KBlockBytes);
    const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
    const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
#pragma unroll
    for (unsigned hh = 0; hh < 2; ++hh) {
      const std::uint8_t qh = blk[128 + (hh * 32) + lane];
#pragma unroll
      for (unsigned qd = 0; qd < 4; ++qd) {
        const std::uint8_t ql = blk[(hh * 64) + ((qd & 1) * 32) + lane];
        const int q = ((qd >> 1) != 0 ? (ql >> 4) : (ql & 0xF)) |
                      (((qh >> (2 * qd)) & 3) << 4);
        const int s = sc[(hh * 8) + (qd * 2) + (lane / 16)];
        wv[(b * 8) + (hh * 4) + qd] =
            d * static_cast<float>(s) * static_cast<float>(q - 32);
      }
    }
  }
  const std::uint32_t t0 = blockIdx.z * kHeadsTokenTile;
  const std::uint32_t t1 = min(t0 + kHeadsTokenTile, tokens);
  for (std::uint32_t t = t0; t < t1; ++t) {
    const float* xt = x + (((head * tokens) + t) * kCols);
    float acc = 0.0F;
#pragma unroll
    for (unsigned b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        // i = half * 4 + quad -> element half * 128 + quad * 32 + lane
        acc +=
            wv[(b * 8) + i] *
            xt[(b * kQ6KBlockElems) + ((i >> 2) * 128) + ((i & 3) * 32) + lane];
      }
    }
    acc = WaveSum(acc);
    if (lane == 0) {
      y[(((head * tokens) + t) * rows) + r] = acc;
    }
  }
}

/// RMSNorm (bias == nullptr) or LayerNorm of one row per block.
__global__ void NormKernel(const float* src, const float* weight,
                           const float* bias, float* dst, std::uint32_t width,
                           float eps) {
  __shared__ float red[kThreads];
  const float* x = src + (std::size_t{blockIdx.x} * width);
  float* y = dst + (std::size_t{blockIdx.x} * width);
  float mean = 0.0F;
  if (bias != nullptr) {
    float s = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
      s += x[i];
    }
    mean = BlockReduce<false>(s, red) / static_cast<float>(width);
  }
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = x[i] - mean;
    ss += d * d;
  }
  const float inv =
      rsqrtf((BlockReduce<false>(ss, red) / static_cast<float>(width)) + eps);
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = (x[i] - mean) * inv * weight[i];
    y[i] = bias != nullptr ? v + bias[i] : v;
  }
}

/// dst[b][a][d] = src[a][b][d] for src [A][B][D].
__global__ void SwapKernel(const float* src, float* dst, std::uint32_t a_dim,
                           std::uint32_t b_dim, std::uint32_t d_dim) {
  const std::uint32_t a = blockIdx.x;
  const std::uint32_t b = blockIdx.y;
  const float* s = src + (((std::size_t{a} * b_dim) + b) * d_dim);
  float* o = dst + (((std::size_t{b} * a_dim) + a) * d_dim);
  for (std::uint32_t d = threadIdx.x; d < d_dim; d += blockDim.x) {
    o[d] = s[d];
  }
}

/// Latent attention of token blockIdx.x for heads [blockIdx.y * G, + G).
/// Rows: the selected pools (ascending, kpool rows each), then the raw tail
/// up to the token's own position. LDS: q [G][L], scores [G][smax], row
/// ids [smax], reduction [kThreads].
template<int G>
__global__ void __launch_bounds__(kThreads)
    LatentAttentionKernel(const float* qabs, const float* latent,
                          const std::int32_t* sel, float* ctx, std::uint32_t n,
                          std::uint32_t pos0, std::uint32_t L,
                          std::uint32_t kpool, std::uint32_t budget,
                          std::uint32_t smax, float scale) {
  extern __shared__ float lds[];
  float* q_s = lds;
  float* s_s = q_s + (std::size_t{G} * L);
  auto* row_s = reinterpret_cast<std::int32_t*>(s_s + (std::size_t{G} * smax));
  float* red = reinterpret_cast<float*>(row_s + smax);
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t h0 = blockIdx.y * G;
  const std::uint32_t pos = pos0 + t;
  const std::uint32_t pools = (pos + 1) / kpool;
  const std::uint32_t count = min(pools, budget);
  const std::uint32_t pooled_rows = count * kpool;
  const std::uint32_t rows = pooled_rows + (pos + 1 - (pools * kpool));
  for (std::uint32_t i = threadIdx.x; i < G * L; i += blockDim.x) {
    const std::uint32_t g = i / L;
    q_s[i] = qabs[(((std::size_t{h0 + g} * n) + t) * L) + (i % L)];
  }
  const std::int32_t* st = sel + (std::size_t{t} * budget);
  for (std::uint32_t i = threadIdx.x; i < rows; i += blockDim.x) {
    row_s[i] =
        i < pooled_rows
            ? (st[i / kpool] * static_cast<std::int32_t>(kpool)) +
                  static_cast<std::int32_t>(i % kpool)
            : static_cast<std::int32_t>((pools * kpool) + (i - pooled_rows));
  }
  __syncthreads();
  const std::uint32_t lane = threadIdx.x % kWave;
  for (std::uint32_t i = threadIdx.x / kWave; i < rows;
       i += blockDim.x / kWave) {
    const float* lr = latent + (static_cast<std::size_t>(row_s[i]) * L);
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      acc[g] = 0.0F;
    }
    for (std::uint32_t d = lane; d < L; d += kWave) {
      const float v = lr[d];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        acc[g] += q_s[(g * L) + d] * v;
      }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      acc[g] = WaveSum(acc[g]);
    }
    if (lane == 0) {
#pragma unroll
      for (int g = 0; g < G; ++g) {
        s_s[(g * smax) + i] = acc[g] * scale;
      }
    }
  }
  __syncthreads();
  float inv_sum[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float* sg = s_s + (g * smax);
    float m = -INFINITY;
    for (std::uint32_t i = threadIdx.x; i < rows; i += blockDim.x) {
      m = fmaxf(m, sg[i]);
    }
    m = BlockReduce<true>(m, red);
    float sum = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < rows; i += blockDim.x) {
      const float e = expf(sg[i] - m);
      sg[i] = e;
      sum += e;
    }
    inv_sum[g] = 1.0F / BlockReduce<false>(sum, red);
  }
  for (std::uint32_t d = threadIdx.x; d < L; d += blockDim.x) {
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      acc[g] = 0.0F;
    }
    for (std::uint32_t i = 0; i < rows; ++i) {
      const float v = latent[(static_cast<std::size_t>(row_s[i]) * L) + d];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        acc[g] += s_s[(g * smax) + i] * v;
      }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      ctx[(((std::size_t{h0 + g} * n) + t) * L) + d] = acc[g] * inv_sum[g];
    }
  }
}

/// Dense prefill attention (DenseLatentAttention): row i = h * n + t of
/// S [heads * n][r_len] holds token t's scaled scores against latent rows
/// 0..r_len-1; P (fp16, same shape) gets the softmax over its causal rows
/// 0..pos0 + t and 0 past them, so the P x latent GEMM sees only those rows.
__global__ void __launch_bounds__(kThreads)
    CausalSoftmaxKernel(const float* s, __half* p, std::uint32_t n,
                        std::uint32_t r_len, std::uint32_t pos0) {
  __shared__ float red[kThreads];
  const std::size_t i = blockIdx.x;
  const std::uint32_t valid = pos0 + static_cast<std::uint32_t>(i % n) + 1;
  const float* row = s + (i * r_len);
  __half* out = p + (i * r_len);
  float m = -INFINITY;
  for (std::uint32_t j = threadIdx.x; j < valid; j += blockDim.x) {
    m = fmaxf(m, row[j]);
  }
  m = BlockReduce<true>(m, red);
  float sum = 0.0F;
  for (std::uint32_t j = threadIdx.x; j < valid; j += blockDim.x) {
    sum += expf(row[j] - m);
  }
  const float inv = 1.0F / BlockReduce<false>(sum, red);
  for (std::uint32_t j = threadIdx.x; j < r_len; j += blockDim.x) {
    out[j] = __float2half(j < valid ? expf(row[j] - m) * inv : 0.0F);
  }
}

/// y = fp16(x * scale), x and y [count].
__global__ void ToHalfKernel(const float* x, __half* y, std::size_t count,
                             float scale) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < count; i += std::size_t{gridDim.x} * blockDim.x) {
    y[i] = __float2half(x[i] * scale);
  }
}

/// One hipBLAS handle per stream: hipblasSetStream changes handle state, so
/// a shared handle would race between streams. Null if creation fails.
hipblasHandle_t Blas(hipStream_t stream) {
  static std::mutex mutex;
  static std::map<hipStream_t, hipblasHandle_t> handles;
  const std::lock_guard<std::mutex> lock(mutex);
  auto it = handles.find(stream);
  if (it == handles.end()) {
    hipblasHandle_t h = nullptr;
    if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS ||
        hipblasSetStream(h, stream) != HIPBLAS_STATUS_SUCCESS)
      return nullptr;
    it = handles.emplace(stream, h).first;
  }
  return it->second;
}

/// Prefill latent attention as two fp16 GEMMs with F32 accumulation, when
/// every token attends to all of its causal rows, which holds while the
/// indexer keeps every pool (SelectKernel: pools <= budget selects
/// 0..pools-1, plus the raw tail):
///   S = (scale Q) L^T   (Q = qabs [heads * n][L], L = latent rows 0..R-1)
///   P = causal softmax of S (CausalSoftmaxKernel, fp16 out)
///   ctx = P L           ([heads][n][L], as LatentAttentionKernel writes it)
/// All heads share the latent rows, so the GEMMs read them as one dense
/// matrix instead of once per (token, 4 heads) like LatentAttentionKernel.
/// The scale is folded into Q before the fp16 cast to keep it in range; Q,
/// latent and P are rounded to fp16, products accumulate in F32. False
/// (nothing written) if the BLAS handle or scratch is unavailable.
bool DenseLatentAttention(const float* qabs, const float* latent, float* ctx,
                          std::uint32_t heads, std::uint32_t n,
                          std::uint32_t pos0, std::uint32_t L, float scale,
                          Stream stream) {
  const hipblasHandle_t h = Blas(stream);
  if (h == nullptr) {
    return false;
  }
  const std::uint32_t r = pos0 + n;
  const std::size_t m = std::size_t{heads} * n;
  if (m > 0x7fffffffU || r > 0x7fffffffU) {
    return false;
  }
  const auto align = [](std::size_t b) { return (b + 255) & ~std::size_t{255}; };
  const std::size_t s_bytes = align(m * r * sizeof(float));
  const std::size_t p_bytes = align(m * r * sizeof(__half));
  const std::size_t q_bytes = align(m * L * sizeof(__half));
  const std::size_t l_bytes = align(std::size_t{r} * L * sizeof(__half));
  auto* base = static_cast<std::uint8_t*>(StreamScratch(
      kScratchDenseAttention, s_bytes + p_bytes + q_bytes + l_bytes, stream));
  if (base == nullptr) {
    return false;
  }
  auto* scores = reinterpret_cast<float*>(base);
  auto* p16 = reinterpret_cast<__half*>(base + s_bytes);
  auto* q16 = reinterpret_cast<__half*>(base + s_bytes + p_bytes);
  auto* l16 = reinterpret_cast<__half*>(base + s_bytes + p_bytes + q_bytes);
  const auto grid = [](std::size_t count) {
    return static_cast<unsigned>(std::min<std::size_t>((count + 255) / 256, 65535));
  };
  ToHalfKernel<<<grid(m * L), 256, 0, stream>>>(qabs, q16, m * L, scale);
  ToHalfKernel<<<grid(std::size_t{r} * L), 256, 0, stream>>>(
      latent, l16, std::size_t{r} * L, 1.0F);
  const float zero = 0.0F, one = 1.0F;
  const int mi = static_cast<int>(m), ri = static_cast<int>(r),
            li = static_cast<int>(L);
  // Row-major S[m][r] = Q[m][L] latent[r][L]^T is column-major
  // S^T (r x m) = latent (op T of the L x r column-major view) Q^T (L x m).
  if (hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, ri, mi, li, &one, l16,
                    HIP_R_16F, li, q16, HIP_R_16F, li, &zero, scores, HIP_R_32F,
                    ri, HIPBLAS_COMPUTE_32F,
                    HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  CausalSoftmaxKernel<<<static_cast<unsigned>(m), kThreads, 0, stream>>>(
      scores, p16, n, r, pos0);
  // Row-major ctx[m][L] = P[m][r] latent[r][L] is column-major
  // ctx^T (L x m) = latent^T (L x r) P^T (r x m).
  if (hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, li, mi, ri, &one, l16,
                    HIP_R_16F, li, p16, HIP_R_16F, ri, &zero, ctx, HIP_R_32F,
                    li, HIPBLAS_COMPUTE_32F,
                    HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  return hipGetLastError() == hipSuccess;
}

std::uint32_t SelectionRows(const Config& c) {
  return (c.IndexerPoolBudget() * c.indexer_kpool) + c.indexer_kpool - 1;
}

std::size_t AttentionLds(const Config& c, int g) {
  const std::size_t smax = SelectionRows(c);
  return ((std::size_t(g) * (c.kv_lora_rank + smax)) + smax + kThreads) *
         sizeof(float);
}

/// Heads per attention block: the largest of 4, 2, 1 that divides the head
/// count and fits the LDS; 0 if none.
int AttentionGroup(const Config& c) {
  for (const int g : {4, 2, 1}) {
    if (c.num_heads % g == 0 && AttentionLds(c, g) <= kMaxLds) {
      return g;
    }
  }
  return 0;
}

/// Scratch regions of one MlaForward pass of `n` tokens.
struct Scratch {
  float* qr{nullptr};            ///< [n][q_lora]
  float* q{nullptr};             ///< [n][H][nope]
  float* qh{nullptr};            ///< [H][n][nope]
  float* qabs{nullptr};          ///< [H][n][L]
  float* ctx{nullptr};           ///< [H][n][L]
  float* vo{nullptr};            ///< [H][n][v]
  float* attn{nullptr};          ///< [n][H][v] (kqv_out)
  float* kv{nullptr};            ///< [n][L] before the norm
  float* iq{nullptr};            ///< [n][ih][idim]
  float* ik{nullptr};            ///< [n][idim]
  float* ig{nullptr};            ///< [n][idim]
  float* iw{nullptr};            ///< [n][ih]
  std::int32_t* sel{nullptr};    ///< [n][budget]
  std::int32_t* top_k{nullptr};  ///< [n][budget]
};

std::size_t LayoutScratch(const Config& c, std::uint32_t tokens, void* base,
                          Scratch* s) {
  const std::size_t n = tokens;
  const std::size_t h = c.num_heads;
  const std::size_t l = c.kv_lora_rank;
  std::size_t offset = 0;
  const auto take = [&](auto*& region, std::size_t elems) {
    if (base != nullptr) {
      region = reinterpret_cast<std::remove_reference_t<decltype(region)>>(
          static_cast<std::uint8_t*>(base) + offset);
    }
    offset += AlignUp(elems * 4);
  };
  Scratch unused;
  Scratch& r = s != nullptr ? *s : unused;
  take(r.qr, n * c.q_lora_rank);
  take(r.q, n * h * c.mla_head_dim);
  take(r.qh, n * h * c.mla_head_dim);
  take(r.qabs, n * h * l);
  take(r.ctx, n * h * l);
  take(r.vo, n * h * c.mla_value_dim);
  take(r.attn, n * h * c.mla_value_dim);
  take(r.kv, n * l);
  take(r.iq, n * c.indexer_heads * c.indexer_head_dim);
  take(r.ik, n * c.indexer_head_dim);
  take(r.ig, n * c.indexer_head_dim);
  take(r.iw, n * c.indexer_heads);
  take(r.sel, n * c.IndexerPoolBudget());
  take(r.top_k, n * c.IndexerPoolBudget());
  return offset;
}

bool Supported(const Config& c) {
  return c.num_heads > 0 && c.q_lora_rank > 0 && c.kv_lora_rank > 0 &&
         c.mla_head_dim > 0 && c.mla_value_dim > 0 && IndexerSupported(c) &&
         AttentionGroup(c) > 0;
}

/// Reports a borrowed device view to the tap (layer_ops.hpp), no sync.
void Tap(const OpContext& ctx, std::string_view name, std::uint32_t layer,
         const void* data, TapView::Dtype dtype, std::uint64_t ne0,
         std::uint64_t ne1, std::uint64_t ne2 = 1) {
  if (ctx.tap != nullptr && *ctx.tap) {
    (*ctx.tap)(name, layer,
               TapView{.device = data,
                       .dtype = dtype,
                       .ne0 = ne0,
                       .ne1 = ne1,
                       .ne2 = ne2});
  }
}

/// One pass whose scratch fits.
bool Pass(const OpContext& ctx, const Config& c, std::uint32_t layer,
          const MlaLayerDevice& w, const CacheLayout& cache, const float* x,
          std::uint32_t n, std::uint32_t pos0, float* y) {
  const Stream stream = ctx.stream;
  const std::uint32_t heads = c.num_heads;
  const std::uint32_t nope = c.mla_head_dim;
  const std::uint32_t L = c.kv_lora_rank;
  const std::uint32_t v_dim = c.mla_value_dim;
  const bool tapping = ctx.tap != nullptr && *ctx.tap;
  Scratch s;
  (void)LayoutScratch(c, n, ctx.scratch, &s);
  float* latent = cache.latent + (std::size_t{pos0} * L);

  // Query and latent (HF:1181-1186).
  std::optional<OpScope> scope;
  scope.emplace(ctx.profiler, "mla/q_kv_proj", stream);
  bool ok = Project(w.q_a, 0, x, s.qr, n, stream);
  ok = ok && NormRows(s.qr, w.q_a_norm, nullptr, s.qr, n, c.q_lora_rank,
                      c.rms_eps, stream);
  ok = ok && Project(w.q_b, 0, s.qr, s.q, n, stream);
  ok = ok && Project(w.kv_a, 0, x, s.kv, n, stream);
  ok = ok &&
       NormRows(s.kv, w.kv_a_norm, nullptr, latent, n, L, c.rms_eps, stream);
  scope.reset();
  if (!ok) {
    return false;
  }
  Tap(ctx, "q_resid", layer, s.qr, TapView::Dtype::kF32, c.q_lora_rank, n);
  Tap(ctx, "kv_cmpr", layer, latent, TapView::Dtype::kF32, L, 1, n);

  // Indexer selection (HF:829-905, 942-1060).
  const IndexerBuffers ib{.x = x,
                          .q_resid = s.qr,
                          .iq = s.iq,
                          .ik = s.ik,
                          .ig = s.ig,
                          .iw = s.iw,
                          .sel = s.sel,
                          .top_k = tapping ? s.top_k : nullptr};
  scope.emplace(ctx.profiler, "mla/indexer", stream);
  if (!IndexerForward(c, w, cache, ib, n, pos0, stream)) {
    return false;
  }
  scope.reset();
  if (tapping) {
    const std::uint32_t ih = c.indexer_heads;
    const std::uint32_t idim = c.indexer_head_dim;
    const std::uint32_t pools = (pos0 + n) / c.indexer_kpool;
    Tap(ctx, "indexer_q", layer, s.iq, TapView::Dtype::kF32, idim, ih, n);
    Tap(ctx, "indexer_k", layer, s.ik, TapView::Dtype::kF32, idim, n);
    Tap(ctx, "indexer_gate", layer, s.ig, TapView::Dtype::kF32, idim, n);
    Tap(ctx, "indexer_weights", layer, s.iw, TapView::Dtype::kF32, ih, n);
    if (pools > 0) {
      Tap(ctx, "indexer_pool_k", layer, cache.pooled, TapView::Dtype::kF32,
          idim, 1, pools);
    }
    // Row stride is the budget; entries past min(pools, budget) are -1.
    Tap(ctx, "indexer_top_k", layer, s.top_k, TapView::Dtype::kI32,
        c.IndexerPoolBudget(), n);
  }

  // Absorb k_b per head, attend over the latent rows, v_b per head.
  scope.emplace(ctx.profiler, "mla/k_b", stream);
  SwapKernel<<<dim3(n, heads), 64, 0, stream>>>(s.q, s.qh, n, heads, nope);
  ok = ProjectHeads(w.k_b, heads, s.qh, s.qabs, n, stream);
  scope.reset();
  if (!ok) {
    return false;
  }
  scope.emplace(ctx.profiler, "mla/attention", stream);
  const int g = AttentionGroup(c);
  const std::uint32_t smax = SelectionRows(c);
  const float scale = 1.0F / std::sqrt(static_cast<float>(nope));
  const dim3 grid(n, heads / static_cast<std::uint32_t>(g));
  const std::size_t lds = AttentionLds(c, g);
  const std::uint32_t kpool = c.indexer_kpool;
  const std::uint32_t budget = c.IndexerPoolBudget();
  // Prefill while every token still sees all its rows: two SGEMMs.
  static const bool dense_enabled = [] {
    const char* v = std::getenv("GUFO_GLM53_MLA_DENSE");
    return !(v && *v == '0');
  }();
  const bool dense = dense_enabled && n > 8 && !ExactPrefill() && kpool > 0 &&
                     (pos0 + n) / kpool <= budget;
  if (dense && DenseLatentAttention(s.qabs, cache.latent, s.ctx, heads, n,
                                    pos0, L, scale, stream)) {
  } else if (g == 4) {
    LatentAttentionKernel<4>
        <<<grid, kThreads, lds, stream>>>(s.qabs, cache.latent, s.sel, s.ctx, n,
                                          pos0, L, kpool, budget, smax, scale);
  } else if (g == 2) {
    LatentAttentionKernel<2>
        <<<grid, kThreads, lds, stream>>>(s.qabs, cache.latent, s.sel, s.ctx, n,
                                          pos0, L, kpool, budget, smax, scale);
  } else {
    LatentAttentionKernel<1>
        <<<grid, kThreads, lds, stream>>>(s.qabs, cache.latent, s.sel, s.ctx, n,
                                          pos0, L, kpool, budget, smax, scale);
  }
  scope.reset();
  scope.emplace(ctx.profiler, "mla/v_b", stream);
  ok = ProjectHeads(w.v_b, heads, s.ctx, s.vo, n, stream);
  scope.reset();
  if (!ok) {
    return false;
  }
  scope.emplace(ctx.profiler, "mla/out_proj", stream);
  SwapKernel<<<dim3(heads, n), 64, 0, stream>>>(s.vo, s.attn, heads, n, v_dim);
  Tap(ctx, "kqv_out", layer, s.attn, TapView::Dtype::kF32,
      std::uint64_t{heads} * v_dim, n);
  ok = Project(w.out, 0, s.attn, y, n, stream);
  scope.reset();
  Tap(ctx, "attn_out", layer, y, TapView::Dtype::kF32, c.hidden_size, n);
  return ok && hipPeekAtLastError() == hipSuccess;
}

}  // namespace

bool CanProject(const DeviceMatrix& w) noexcept {
  if (w.empty() || w.cols == 0 || w.rows == 0) {
    return false;
  }
  return w.type == GgmlType::kF32 || SupportsProjection(w.type, w.cols);
}

bool Project(const DeviceMatrix& w, std::uint64_t expert, const float* x,
             float* y, std::uint32_t tokens, Stream stream) {
  if (!CanProject(w) || expert >= w.experts || x == nullptr || y == nullptr) {
    return false;
  }
  if (tokens == 0) {
    return true;
  }
  const std::size_t rows = w.rows;
  const std::size_t cols = w.cols;
  const auto* base =
      static_cast<const std::uint8_t*>(w.data) + (expert * rows * w.row_bytes);
  if (w.type == GgmlType::kF32) {
    constexpr unsigned kWaves = kThreads / kWave;
    const dim3 grid(static_cast<unsigned>((rows + kWaves - 1) / kWaves),
                    tokens);
    MatmulF32Kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const float*>(base), x, y,
        static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols));
    return true;
  }
  if (w.type == GgmlType::kBF16) {
    for (std::uint32_t t = 0; t < tokens; ++t) {
      ::gufo::hip::LaunchGEMV(base, w.type, x + (std::size_t{t} * cols),
                              y + (std::size_t{t} * rows), rows, cols, stream);
    }
    return true;
  }
  if (PrefillQuantGemm(w.type, base, x, y, tokens, rows, cols, stream))
    return true;
  ::gufo::hip::LaunchBatchedQuantGEMMFp32(w.type, base, x, y, tokens, rows,
                                          cols, stream);
  return true;
}

bool ProjectHeads(const DeviceMatrix& w, std::uint32_t heads, const float* x,
                  float* y, std::uint32_t tokens, Stream stream) {
  if (!CanProject(w) || w.experts != heads || x == nullptr || y == nullptr) {
    return false;
  }
  if (tokens == 0 || heads == 0) {
    return true;
  }
  const std::uint32_t rows = static_cast<std::uint32_t>(w.rows);
  constexpr unsigned kWaves = kThreads / kWave;
  const unsigned row_blocks = (rows + kWaves - 1) / kWaves;
  if (w.type == GgmlType::kF32) {
    MatmulF32HeadsKernel<<<dim3(row_blocks, tokens, heads), kThreads, 0,
                           stream>>>(static_cast<const float*>(w.data), x, y,
                                     rows, static_cast<std::uint32_t>(w.cols),
                                     tokens);
    return true;
  }
  const unsigned tiles = (tokens + kHeadsTokenTile - 1) / kHeadsTokenTile;
  if (w.type == GgmlType::kQ6_K && w.cols % kQ6KBlockElems == 0 &&
      w.row_bytes == (w.cols / kQ6KBlockElems) * kQ6KBlockBytes) {
    const auto* base = static_cast<const std::uint8_t*>(w.data);
    if (w.cols == kQ6KBlockElems) {
      MatmulQ6KHeadsKernel<1>
          <<<dim3(row_blocks, heads, tiles), kThreads, 0, stream>>>(
              base, x, y, rows, tokens);
      return true;
    }
    if (w.cols == 2 * kQ6KBlockElems) {
      MatmulQ6KHeadsKernel<2>
          <<<dim3(row_blocks, heads, tiles), kThreads, 0, stream>>>(
              base, x, y, rows, tokens);
      return true;
    }
  }
  // Other encodings: a GEMM per head.
  for (std::uint32_t h = 0; h < heads; ++h) {
    if (!Project(w, h, x + (std::size_t{h} * tokens * w.cols),
                 y + (std::size_t{h} * tokens * w.rows), tokens, stream)) {
      return false;
    }
  }
  return true;
}

bool NormRows(const float* src, const float* weight, const float* bias,
              float* dst, std::uint32_t rows, std::uint32_t width, float eps,
              Stream stream) {
  if (src == nullptr || weight == nullptr || dst == nullptr || width == 0) {
    return false;
  }
  if (rows > 0) {
    NormKernel<<<rows, kThreads, 0, stream>>>(src, weight, bias, dst, width,
                                              eps);
  }
  return true;
}

std::size_t LayoutCache(const Config& c, std::uint32_t capacity, void* base,
                        CacheLayout* out) noexcept {
  const std::size_t pools =
      c.indexer_kpool == 0 ? 0 : capacity / c.indexer_kpool;
  const std::size_t idim = c.indexer_head_dim;
  std::size_t offset = 0;
  CacheLayout unused;
  CacheLayout& r = out != nullptr ? *out : unused;
  const auto take = [&](float*& region, std::size_t floats) {
    if (base != nullptr) {
      region =
          reinterpret_cast<float*>(static_cast<std::uint8_t*>(base) + offset);
    }
    offset += AlignUp(floats * sizeof(float));
  };
  take(r.latent, std::size_t{capacity} * c.kv_lora_rank);
  take(r.pooled, pools * idim);
  take(r.ring_key, std::size_t{c.indexer_kpool} * idim);
  take(r.ring_gate, std::size_t{c.indexer_kpool} * idim);
  take(r.scores, std::size_t{kSelectTokens} * pools);
  r.pool_capacity = static_cast<std::uint32_t>(pools);
  return offset;
}

}  // namespace mla

std::size_t MlaCacheBytes(const Config& config,
                          std::uint32_t capacity) noexcept {
  return mla::LayoutCache(config, capacity, nullptr, nullptr);
}

std::size_t MlaScratchBytes(const Config& config,
                            std::uint32_t max_tokens) noexcept {
  return mla::LayoutScratch(config, std::max<std::uint32_t>(max_tokens, 1),
                            nullptr, nullptr);
}

bool MlaForward(const OpContext& ctx, const Config& config, std::uint32_t layer,
                const MlaLayerDevice& w, MlaCache& cache, const float* x,
                std::uint32_t tokens, std::uint32_t pos0, float* y) {
  if (tokens == 0) {
    return true;
  }
  if (!config.IsMlaLayer(layer) || !mla::Supported(config) || x == nullptr ||
      y == nullptr || ctx.scratch == nullptr || cache.data == nullptr ||
      w.q_a_norm == nullptr || w.kv_a_norm == nullptr ||
      w.idx_k_norm == nullptr || w.idx_k_norm_bias == nullptr ||
      w.idx_ape == nullptr) {
    return false;
  }
  for (const DeviceMatrix* m : {&w.q_a, &w.q_b, &w.kv_a, &w.out, &w.idx_q_b,
                                &w.idx_k, &w.idx_proj, &w.idx_gate}) {
    if (!mla::CanProject(*m) || m->experts != 1) {
      return false;
    }
  }
  if (!mla::CanProject(w.k_b) || !mla::CanProject(w.v_b) ||
      w.k_b.experts != config.num_heads || w.v_b.experts != config.num_heads ||
      w.k_b.cols != config.mla_head_dim || w.k_b.rows != config.kv_lora_rank ||
      w.v_b.cols != config.kv_lora_rank || w.v_b.rows != config.mla_value_dim) {
    return false;
  }
  if (std::uint64_t{pos0} + tokens > cache.capacity ||
      cache.bytes < MlaCacheBytes(config, cache.capacity)) {
    return false;
  }
  mla::CacheLayout layout;
  (void)mla::LayoutCache(config, cache.capacity, cache.data, &layout);
  // Largest pass whose scratch fits (scratch is linear in tokens), as
  // KdaMixerForward does; passes are exact because the cache carries state.
  const std::size_t one = MlaScratchBytes(config, 1);
  if (ctx.scratch_bytes < one) {
    return false;
  }
  std::uint32_t lo = static_cast<std::uint32_t>(
      std::min<std::size_t>(tokens, ctx.scratch_bytes / one));
  std::uint32_t hi = tokens;
  while (lo < hi) {
    const std::uint32_t mid = lo + ((hi - lo + 1) / 2);
    if (MlaScratchBytes(config, mid) <= ctx.scratch_bytes) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const std::uint32_t chunk = lo;
  const std::size_t hidden = config.hidden_size;
  for (std::uint32_t t0 = 0; t0 < tokens; t0 += chunk) {
    const std::uint32_t n = std::min(chunk, tokens - t0);
    if (!mla::Pass(ctx, config, layer, w, layout, x + (t0 * hidden), n,
                   pos0 + t0, y + (t0 * hidden))) {
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::glm53_flash
