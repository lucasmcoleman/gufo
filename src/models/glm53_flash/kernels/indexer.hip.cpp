// DSA k-pool indexer of GLM-5.3-Flash MLA layers (HF modeling_glm5_next.py
// 810-1060, llama.cpp glm5-next.cpp build_kpool_select). The scalar contract
// is ReferenceModel::Mla (reference.cpp) on reference_mla.hpp: LayerNorm keys,
// PoolKeys, IndexerScores, SelectPools with the tail always appended.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/glm53_flash/kernels/mla.hpp"

namespace gufo::models::glm53_flash::mla {
namespace {

constexpr unsigned kWave = 32;
constexpr unsigned kPoolThreads = 128;
constexpr unsigned kScoreThreads = 256;
constexpr unsigned kPoolsPerScoreBlock = 64;
constexpr unsigned kSelectThreads = 1024;
constexpr std::size_t kMaxLds = 64 * 1024;

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, kWave);
  }
  return v;
}

/// Pools [first, first + gridDim.x): softmax over the pool's tokens of
/// gate + ape per channel, weighting the LayerNorm keys (PoolKeys). Tokens
/// before pos0 come from the ring, the others from this pass's rows.
__global__ void PoolKernel(const float* ik, const float* ig,
                           const float* ring_key, const float* ring_gate,
                           const float* ape, float* pooled, std::uint32_t pos0,
                           std::uint32_t first, std::uint32_t kpool,
                           std::uint32_t idim) {
  const std::uint32_t p = first + blockIdx.x;
  for (std::uint32_t d = threadIdx.x; d < idim; d += blockDim.x) {
    const auto key_at = [&](std::uint32_t i) {
      const std::uint32_t pos = (p * kpool) + i;
      return pos < pos0 ? ring_key[(std::size_t{pos % kpool} * idim) + d]
                        : ik[(std::size_t{pos - pos0} * idim) + d];
    };
    const auto logit_at = [&](std::uint32_t i) {
      const std::uint32_t pos = (p * kpool) + i;
      const float g = pos < pos0
                          ? ring_gate[(std::size_t{pos % kpool} * idim) + d]
                          : ig[(std::size_t{pos - pos0} * idim) + d];
      return g + ape[(std::size_t{i} * idim) + d];
    };
    float maximum = -INFINITY;
    for (std::uint32_t i = 0; i < kpool; ++i) {
      maximum = fmaxf(maximum, logit_at(i));
    }
    float sum = 0.0F;
    float value = 0.0F;
    for (std::uint32_t i = 0; i < kpool; ++i) {
      const float e = expf(logit_at(i) - maximum);
      sum += e;
      value += e * key_at(i);
    }
    pooled[(std::size_t{p} * idim) + d] = value / sum;
  }
}

/// Copies rows [first, first + gridDim.x) of this pass into ring slots.
__global__ void RingKernel(const float* ik, const float* ig, float* ring_key,
                           float* ring_gate, std::uint32_t pos0,
                           std::uint32_t first, std::uint32_t kpool,
                           std::uint32_t idim) {
  const std::uint32_t pos = first + blockIdx.x;
  const std::size_t src = std::size_t{pos - pos0} * idim;
  const std::size_t dst = std::size_t{pos % kpool} * idim;
  for (std::uint32_t d = threadIdx.x; d < idim; d += blockDim.x) {
    ring_key[dst + d] = ik[src + d];
    ring_gate[dst + d] = ig[src + d];
  }
}

__global__ void ScaleKernel(float* v, std::size_t count, float scale) {
  const std::size_t i = (std::size_t{blockIdx.x} * blockDim.x) + threadIdx.x;
  if (i < count) {
    v[i] *= scale;
  }
}

/// scores[ti][p] = sum_h w[h] * relu(iq[h] . pooled[p]) for token t0 + ti
/// and every pool it sees (IndexerScores, w already scaled). A wave per
/// pool, a lane per head (strided), one wave reduction per pool.
__global__ void ScoreKernel(const float* iq, const float* iw,
                            const float* pooled, float* scores,
                            std::size_t score_stride, std::uint32_t t0,
                            std::uint32_t pos0, std::uint32_t kpool,
                            std::uint32_t ih, std::uint32_t idim) {
  extern __shared__ float lds[];
  const std::uint32_t ti = blockIdx.y;
  const std::uint32_t t = t0 + ti;
  const std::uint32_t pools = (pos0 + t + 1) / kpool;
  const std::uint32_t p_begin = blockIdx.x * kPoolsPerScoreBlock;
  if (p_begin >= pools) {
    return;
  }
  const std::uint32_t pitch = idim + 1;  // no bank conflicts across heads
  float* q_s = lds;
  float* w_s = lds + (std::size_t{ih} * pitch);
  const float* q = iq + (std::size_t{t} * ih * idim);
  for (std::uint32_t i = threadIdx.x; i < ih * idim; i += blockDim.x) {
    q_s[((i / idim) * pitch) + (i % idim)] = q[i];
  }
  for (std::uint32_t h = threadIdx.x; h < ih; h += blockDim.x) {
    w_s[h] = iw[(std::size_t{t} * ih) + h];
  }
  __syncthreads();
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t wave = threadIdx.x / kWave;
  const std::uint32_t waves = blockDim.x / kWave;
  const std::uint32_t p_end = min(pools, p_begin + kPoolsPerScoreBlock);
  for (std::uint32_t p = p_begin + wave; p < p_end; p += waves) {
    const float* pk = pooled + (std::size_t{p} * idim);
    float acc = 0.0F;
    for (std::uint32_t h = lane; h < ih; h += kWave) {
      const float* qh = q_s + (std::size_t{h} * pitch);
      float dot = 0.0F;
      for (std::uint32_t d = 0; d < idim; ++d) {
        dot += qh[d] * pk[d];
      }
      acc += fmaxf(dot, 0.0F) * w_s[h];
    }
    acc = WaveSum(acc);
    if (lane == 0) {
      scores[(std::size_t{ti} * score_stride) + p] = acc;
    }
  }
}

/// Order-preserving key of a score; +0 and -0 compare equal, as in
/// SelectPools.
__device__ __forceinline__ std::uint32_t ScoreKey(float s) {
  std::uint32_t u = __float_as_uint(s);
  if ((u & 0x7FFFFFFFU) == 0) {
    u = 0;
  }
  return (u & 0x80000000U) != 0 ? ~u : (u | 0x80000000U);
}

/// Exclusive block scan of `v` over kSelectThreads threads; `*total` gets
/// the block sum. `sums` holds kSelectThreads / kWave + 1 words.
__device__ std::uint32_t BlockScan(std::uint32_t v, std::uint32_t* sums,
                                   std::uint32_t* total) {
  const std::uint32_t lane = threadIdx.x % kWave;
  const std::uint32_t wave = threadIdx.x / kWave;
  constexpr std::uint32_t kWaves = kSelectThreads / kWave;
  std::uint32_t inc = v;
#pragma unroll
  for (std::uint32_t off = 1; off < kWave; off <<= 1) {
    const std::uint32_t up = __shfl_up(inc, off, kWave);
    if (lane >= off) {
      inc += up;
    }
  }
  if (lane == kWave - 1) {
    sums[wave] = inc;
  }
  __syncthreads();
  if (wave == 0) {
    std::uint32_t w = lane < kWaves ? sums[lane] : 0;
#pragma unroll
    for (std::uint32_t off = 1; off < kWave; off <<= 1) {
      const std::uint32_t up = __shfl_up(w, off, kWave);
      if (lane >= off) {
        w += up;
      }
    }
    if (lane < kWaves) {
      sums[lane] = w;
    }
  }
  __syncthreads();
  const std::uint32_t prefix = wave > 0 ? sums[wave - 1] : 0;
  *total = sums[kWaves - 1];
  __syncthreads();
  return prefix + inc - v;
}

/// Per token t0 + blockIdx.x: the top `budget` of its visible pools by
/// score, ties to the lower ID (SelectPools), written ascending to sel. MSB
/// radix select finds the budget-th key, then an ordered compaction keeps
/// every larger key and the lowest-ID equal ones. With top_k set, also the
/// selection ranked by descending score (llama.cpp indexer_top_k order).
__global__ void __launch_bounds__(kSelectThreads)
    SelectKernel(const float* scores, std::size_t score_stride,
                 std::uint32_t t0, std::uint32_t pos0, std::uint32_t kpool,
                 std::uint32_t budget, std::int32_t* sel, std::int32_t* top_k) {
  extern __shared__ std::int32_t chosen[];  // [budget]
  __shared__ std::uint32_t hist[256];
  __shared__ std::uint32_t sums[(kSelectThreads / kWave) + 1];
  __shared__ std::uint32_t pick[2];
  const std::uint32_t ti = blockIdx.x;
  const std::uint32_t t = t0 + ti;
  const std::uint32_t pools = (pos0 + t + 1) / kpool;
  const std::uint32_t count = min(pools, budget);
  const float* s = scores + (std::size_t{ti} * score_stride);
  if (pools <= budget) {
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
      chosen[i] = static_cast<std::int32_t>(i);
    }
  } else {
    std::uint32_t prefix = 0;
    std::uint32_t mask = 0;
    std::uint32_t remaining = budget;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (std::uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {
        hist[i] = 0;
      }
      __syncthreads();
      for (std::uint32_t p = threadIdx.x; p < pools; p += blockDim.x) {
        const std::uint32_t key = ScoreKey(s[p]);
        if ((key & mask) == prefix) {
          atomicAdd(&hist[(key >> shift) & 0xFFU], 1U);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        std::uint32_t above = 0;
        std::uint32_t digit = 0;
        for (int d = 255; d >= 0; --d) {
          if (above + hist[d] >= remaining) {
            digit = static_cast<std::uint32_t>(d);
            break;
          }
          above += hist[d];
        }
        pick[0] = digit;
        pick[1] = remaining - above;
      }
      __syncthreads();
      prefix |= pick[0] << shift;
      mask |= 0xFFU << shift;
      remaining = pick[1];
      __syncthreads();
    }
    // prefix is the budget-th largest key; take `remaining` keys equal to it.
    std::uint32_t written = 0;
    std::uint32_t equal_seen = 0;
    for (std::uint32_t base = 0; base < pools; base += blockDim.x) {
      const std::uint32_t p = base + threadIdx.x;
      const std::uint32_t key = p < pools ? ScoreKey(s[p]) : 0;
      const std::uint32_t greater = p < pools && key > prefix ? 1 : 0;
      const std::uint32_t equal = p < pools && key == prefix ? 1 : 0;
      std::uint32_t equal_total = 0;
      const std::uint32_t equal_rank =
          equal_seen + BlockScan(equal, sums, &equal_total);
      const std::uint32_t take =
          greater != 0 || (equal != 0 && equal_rank < remaining) ? 1 : 0;
      std::uint32_t take_total = 0;
      const std::uint32_t slot = written + BlockScan(take, sums, &take_total);
      if (take != 0) {
        chosen[slot] = static_cast<std::int32_t>(p);
      }
      written += take_total;
      equal_seen += equal_total;
    }
  }
  __syncthreads();
  std::int32_t* out = sel + (std::size_t{t} * budget);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    out[i] = chosen[i];
  }
  if (top_k == nullptr) {
    return;
  }
  std::int32_t* ranked = top_k + (std::size_t{t} * budget);
  for (std::uint32_t i = threadIdx.x; i < budget; i += blockDim.x) {
    if (i >= count) {
      ranked[i] = -1;
      continue;
    }
    const std::int32_t id = chosen[i];
    const std::uint32_t key = ScoreKey(s[id]);
    std::uint32_t rank = 0;
    for (std::uint32_t j = 0; j < count; ++j) {
      const std::uint32_t other = ScoreKey(s[chosen[j]]);
      rank += other > key || (other == key && chosen[j] < id) ? 1 : 0;
    }
    ranked[rank] = id;
  }
}

std::size_t ScoreLds(const Config& c) {
  return ((std::size_t{c.indexer_heads} * (c.indexer_head_dim + 1)) +
          c.indexer_heads) *
         sizeof(float);
}

std::size_t SelectLds(const Config& c) {
  return std::size_t{c.IndexerPoolBudget()} * sizeof(std::int32_t);
}

}  // namespace

bool IndexerSupported(const Config& c) noexcept {
  constexpr std::size_t kSelectStatic =
      (256 + (kSelectThreads / kWave) + 1 + 2) * sizeof(std::uint32_t);
  return c.indexer_heads > 0 && c.indexer_head_dim > 0 && c.indexer_kpool > 0 &&
         c.IndexerPoolBudget() > 0 && c.q_lora_rank > 0 &&
         ScoreLds(c) <= kMaxLds && SelectLds(c) + kSelectStatic <= kMaxLds;
}

bool IndexerForward(const Config& c, const MlaLayerDevice& w,
                    const CacheLayout& cache, const IndexerBuffers& b,
                    std::uint32_t n, std::uint32_t pos0, Stream stream) {
  const std::uint32_t ih = c.indexer_heads;
  const std::uint32_t idim = c.indexer_head_dim;
  const std::uint32_t kpool = c.indexer_kpool;
  const std::uint32_t budget = c.IndexerPoolBudget();
  const std::uint32_t pos_end = pos0 + n;
  // q from q_resid; key (LayerNorm with bias), pool gate and head weights
  // from the layer input (HF:829-832, 851-865).
  bool ok = Project(w.idx_q_b, 0, b.q_resid, b.iq, n, stream);
  ok = ok && Project(w.idx_k, 0, b.x, b.ik, n, stream);
  ok = ok && NormRows(b.ik, w.idx_k_norm, w.idx_k_norm_bias, b.ik, n, idim,
                      c.norm_eps, stream);
  ok = ok && Project(w.idx_gate, 0, b.x, b.ig, n, stream);
  ok = ok && Project(w.idx_proj, 0, b.x, b.iw, n, stream);
  if (!ok) {
    return false;
  }
  // llama.cpp's placement: the head weights carry 1/sqrt(ih * idim).
  const float scale = 1.0F / std::sqrt(static_cast<float>(ih) * idim);
  const std::size_t weights = std::size_t{n} * ih;
  ScaleKernel<<<static_cast<unsigned>((weights + 255) / 256), 256, 0, stream>>>(
      b.iw, weights, scale);

  // Pools completed by this pass, then the incomplete tail into the ring.
  const std::uint32_t p_first = pos0 / kpool;
  const std::uint32_t p_end = pos_end / kpool;
  if (p_end > p_first) {
    PoolKernel<<<p_end - p_first, kPoolThreads, 0, stream>>>(
        b.ik, b.ig, cache.ring_key, cache.ring_gate, w.idx_ape, cache.pooled,
        pos0, p_first, kpool, idim);
  }
  const std::uint32_t ring_first = std::max(pos0, p_end * kpool);
  if (pos_end > ring_first) {
    RingKernel<<<pos_end - ring_first, kPoolThreads, 0, stream>>>(
        b.ik, b.ig, cache.ring_key, cache.ring_gate, pos0, ring_first, kpool,
        idim);
  }

  // Scores and selection, kSelectTokens query tokens at a time (the score
  // rows live in the cache workspace).
  const std::size_t stride = cache.pool_capacity;
  for (std::uint32_t t0 = 0; t0 < n; t0 += kSelectTokens) {
    const std::uint32_t ns = std::min(kSelectTokens, n - t0);
    const std::uint32_t pools_last = (pos0 + t0 + ns) / kpool;
    if (pools_last > cache.pool_capacity) {
      return false;
    }
    if (pools_last > 0) {
      const dim3 grid(
          (pools_last + kPoolsPerScoreBlock - 1) / kPoolsPerScoreBlock, ns);
      ScoreKernel<<<grid, kScoreThreads, ScoreLds(c), stream>>>(
          b.iq, b.iw, cache.pooled, cache.scores, stride, t0, pos0, kpool, ih,
          idim);
    }
    SelectKernel<<<ns, kSelectThreads, SelectLds(c), stream>>>(
        cache.scores, stride, t0, pos0, kpool, budget, b.sel, b.top_k);
  }
  return hipPeekAtLastError() == hipSuccess;
}

}  // namespace gufo::models::glm53_flash::mla
