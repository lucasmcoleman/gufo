// Laguna-S-2.1 GPU attention operator. See attention.hpp.
//
// The kernels here are written with plain IEEE arithmetic (no fast-math) and
// no FP contraction, so their rounding stays close to the CPU oracle's. The
// Q6_K projections reuse the qfn_mmq kernels vendored with qwen38_flash_next
// (the routed-expert entry points with one expert and one slot, which is a
// dense matrix product); no dot product of the weights is written here.
#pragma clang fp contract(off)

#include "src/models/laguna/kernels/attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <utility>

#include "src/models/laguna/reference.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::laguna::rocm {
namespace {

using core::GgmlType;

constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kVectorMaxRows = 32;
constexpr int kMaxWaves = 4;  // 128 threads, wave32

void Assign(std::string* out, const std::string& message) {
  if (out != nullptr) {
    *out = message;
  }
}

bool CheckHip(hipError_t status, const char* what, std::string* error_msg) {
  if (status == hipSuccess) {
    return true;
  }
  Assign(error_msg, std::string(what) + ": " + hipGetErrorString(status));
  return false;
}

// ---- Kernels ---------------------------------------------------------------

/// g[t][h] = dot(Wg[h], n[t]). One block per (head, token); fixed order.
__global__ void GateLogitsKernel(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 float* __restrict__ out, std::uint32_t hidden,
                                 std::uint32_t heads) {
  const float* xr = x + std::size_t{blockIdx.y} * hidden;
  const float* wr = w + std::size_t{blockIdx.x} * hidden;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    sum += wr[i] * xr[i];
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
  }
  __shared__ float part[kMaxWaves];
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  if (lane == 0) {
    part[wave] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = part[0];
    for (int i = 1; i < static_cast<int>(blockDim.x) / warpSize; ++i) {
      total += part[i];
    }
    out[std::size_t{blockIdx.y} * heads + blockIdx.x] = total;
  }
}

struct RopeArgs {
  std::uint32_t rotary_dim;
  float theta_scale;
  float corr_low;
  float corr_high;
  float freq_scale;
  float ext_factor;
  float mscale;
};

/// grid (tokens, heads + 2*kv_heads), block 128. Block y < heads: query head
/// (norm, rope, in place). Then key heads (norm, rope, F16 into the cache),
/// then value heads (F16 into the cache). Rotation matches ggml's rope cache:
/// the angle of pair i is the position times theta_scale, i times (float),
/// blended with the interpolated angle by the YaRN ramp, and cos/sin carry
/// mscale. Norm: y = (x * rsqrt(mean(x^2) + eps)) * gamma.
__global__ void QkNormRopeAppendKernel(
    float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_gamma,
    const float* __restrict__ k_gamma, __half* __restrict__ k_cache,
    __half* __restrict__ v_cache, std::uint32_t heads, std::uint32_t kv_heads,
    float eps, std::uint32_t start_pos, std::uint32_t capacity, RopeArgs rope) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t b = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t pos = start_pos + t;
  const std::size_t kv_width = std::size_t{kv_heads} * kHeadDim;
  const std::size_t slot = pos % capacity;
  if (b >= heads + kv_heads) {
    const std::uint32_t h = b - heads - kv_heads;
    v_cache[slot * kv_width + h * kHeadDim + d] =
        __float2half(v[std::size_t{t} * kv_width + h * kHeadDim + d]);
    return;
  }
  const bool is_q = b < heads;
  const std::uint32_t h = is_q ? b : b - heads;
  float* qrow = q + (std::size_t{t} * heads + h) * kHeadDim;
  const float* krow = k + std::size_t{t} * kv_width + h * kHeadDim;
  const float x = is_q ? qrow[d] : krow[d];
  __shared__ float part[kMaxWaves];
  __shared__ float normed[kHeadDim];
  float ss = x * x;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    ss += __shfl_xor(ss, offset);
  }
  if (d % warpSize == 0) {
    part[d / warpSize] = ss;
  }
  __syncthreads();
  float total = part[0];
  for (int i = 1; i < static_cast<int>(kHeadDim) / warpSize; ++i) {
    total += part[i];
  }
  const float mean = total / static_cast<float>(kHeadDim);
  const float scale = 1.0F / sqrtf(mean + eps);
  normed[d] = (x * scale) * (is_q ? q_gamma[d] : k_gamma[d]);
  __syncthreads();
  const std::uint32_t half_rot = rope.rotary_dim / 2;
  float out = normed[d];
  if (d < rope.rotary_dim) {
    const std::uint32_t pair = d < half_rot ? d : d - half_rot;
    float theta = static_cast<float>(pos);
    for (std::uint32_t i = 0; i < pair; ++i) {
      theta *= rope.theta_scale;
    }
    const float theta_interp = rope.freq_scale * theta;
    float rotated = theta_interp;
    if (rope.ext_factor != 0.0F) {
      const float y = (static_cast<float>(pair) - rope.corr_low) /
                      fmaxf(0.001F, rope.corr_high - rope.corr_low);
      const float ramp = 1.0F - fminf(1.0F, fmaxf(0.0F, y));
      const float ramp_mix = ramp * rope.ext_factor;
      rotated = theta_interp * (1.0F - ramp_mix) + theta * ramp_mix;
    }
    const float c = cosf(rotated) * rope.mscale;
    const float s = sinf(rotated) * rope.mscale;
    const float x0 = normed[pair];
    const float x1 = normed[pair + half_rot];
    out = d < half_rot ? x0 * c - x1 * s : x0 * s + x1 * c;
  }
  if (is_q) {
    qrow[d] = out;
  } else {
    k_cache[slot * kv_width + h * kHeadDim + d] = __float2half(out);
  }
}

__device__ __forceinline__ float WaveMax(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  return v;
}
__device__ __forceinline__ float WaveSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

/// grid (kv_heads, tokens), block 128. A block serves the kGroup query heads
/// of one KV head for one row at position p. Keys stream in tiles of 128 from
/// the first visible key: thread t scores key t of the tile for every head
/// (K row from the cache, q from LDS), the tile's max and sum are reduced over
/// the block, then thread d adds the tile's V column d with online-softmax
/// rescaling. The tile boundaries depend only on p, so the sum order of a row
/// is independent of the chunking. Visible keys: j <= p and (window == 0 or
/// p - j < window), which is j >= p + 1 - window.
template<int kGroup>
__global__ void AttentionKernel(const float* __restrict__ q,
                                const __half* __restrict__ k_cache,
                                const __half* __restrict__ v_cache,
                                float* __restrict__ out, std::uint32_t heads,
                                std::uint32_t kv_heads, std::uint32_t window,
                                std::uint32_t start_pos, std::uint32_t capacity,
                                float scale) {
  const std::uint32_t kvh = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t p = start_pos + t;
  const std::uint32_t first =
      (window == 0 || p + 1 <= window) ? 0 : p + 1 - window;
  const std::size_t kv_width = std::size_t{kv_heads} * kHeadDim;
  __shared__ float qs[kGroup][kHeadDim];
  __shared__ float probs[kGroup][kHeadDim];
  __shared__ float red[kMaxWaves][kGroup];
  const int wave = d / warpSize;
  const int lane = d % warpSize;
  const int waves = kHeadDim / warpSize;
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    qs[g][d] = q[(std::size_t{t} * heads + kvh * kGroup + g) * kHeadDim + d];
  }
  __syncthreads();
  float m[kGroup];
  float l[kGroup];
  float acc[kGroup];
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.0F;
    acc[g] = 0.0F;
  }
  for (std::uint32_t ts = first; ts <= p; ts += kHeadDim) {
    const std::uint32_t j = ts + d;
    const bool valid = j <= p;
    float s[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      s[g] = 0.0F;
    }
    if (valid) {
      const uint4* krow = reinterpret_cast<const uint4*>(
          k_cache + std::size_t{j % capacity} * kv_width + kvh * kHeadDim);
      for (int c = 0; c < static_cast<int>(kHeadDim) / 8; ++c) {
        const uint4 raw = krow[c];
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
        float kf[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __half22float2(h2[e]);
          kf[2 * e] = f.x;
          kf[2 * e + 1] = f.y;
        }
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            s[g] = fmaf(qs[g][c * 8 + e], kf[e], s[g]);
          }
        }
      }
    }
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      s[g] = valid ? s[g] * scale : -INFINITY;
    }
    // Tile max per head.
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const float w = WaveMax(s[g]);
      if (lane == 0) {
        red[wave][g] = w;
      }
    }
    __syncthreads();
    float alpha[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      float tile_max = red[0][g];
      for (int i = 1; i < waves; ++i) {
        tile_max = fmaxf(tile_max, red[i][g]);
      }
      const float m_new = fmaxf(m[g], tile_max);
      alpha[g] = expf(m[g] - m_new);
      m[g] = m_new;
      s[g] = valid ? expf(s[g] - m_new) : 0.0F;
      probs[g][d] = s[g];
    }
    __syncthreads();  // red is rewritten below
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const float w = WaveSum(s[g]);
      if (lane == 0) {
        red[wave][g] = w;
      }
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      float tile_sum = red[0][g];
      for (int i = 1; i < waves; ++i) {
        tile_sum += red[i][g];
      }
      l[g] = l[g] * alpha[g] + tile_sum;
      acc[g] *= alpha[g];
    }
    const std::uint32_t count = min(kHeadDim, p - ts + 1);
    for (std::uint32_t jj = 0; jj < count; ++jj) {
      const float vv =
          __half2float(v_cache[std::size_t{(ts + jj) % capacity} * kv_width +
                               kvh * kHeadDim + d]);
#pragma unroll
      for (int g = 0; g < kGroup; ++g) {
        acc[g] = fmaf(probs[g][jj], vv, acc[g]);
      }
    }
    __syncthreads();  // probs and red are rewritten by the next tile
  }
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    out[(std::size_t{t} * heads + kvh * kGroup + g) * kHeadDim + d] =
        acc[g] / l[g];
  }
}

/// gate_act = softplus(gate) (x > 20 ? x : log(1 + exp(x)), llama.cpp's
/// form); attn[t][h][:] *= gate_act[t][h]. One block per (token, head).
__global__ void GateKernel(float* __restrict__ gate, float* __restrict__ attn) {
  const std::size_t idx = blockIdx.x;  // t * heads + h
  const float g = gate[idx];
  const float sp = g > 20.0F ? g : logf(1.0F + expf(g));
  attn[idx * kHeadDim + threadIdx.x] *= sp;
  __syncthreads();
  if (threadIdx.x == 0) {
    gate[idx] = sp;
  }
}

template<int kGroup>
bool LaunchAttention(const float* q, const __half* k, const __half* v,
                     float* out, std::uint32_t heads, std::uint32_t kv_heads,
                     std::uint32_t window, std::uint32_t start_pos,
                     std::uint32_t tokens, std::uint32_t capacity, float scale,
                     hipStream_t stream) {
  AttentionKernel<kGroup><<<dim3(kv_heads, tokens), kHeadDim, 0, stream>>>(
      q, k, v, out, heads, kv_heads, window, start_pos, capacity, scale);
  return true;
}

bool DispatchAttention(std::uint32_t group, const float* q, const __half* k,
                       const __half* v, float* out, std::uint32_t heads,
                       std::uint32_t kv_heads, std::uint32_t window,
                       std::uint32_t start_pos, std::uint32_t tokens,
                       std::uint32_t capacity, float scale,
                       hipStream_t stream) {
#define LAGUNA_ATTN_CASE(G)                                          \
  case G:                                                            \
    return LaunchAttention<G>(q, k, v, out, heads, kv_heads, window, \
                              start_pos, tokens, capacity, scale, stream);
  switch (group) {
    LAGUNA_ATTN_CASE(1)
    LAGUNA_ATTN_CASE(2)
    LAGUNA_ATTN_CASE(3)
    LAGUNA_ATTN_CASE(4)
    LAGUNA_ATTN_CASE(6)
    LAGUNA_ATTN_CASE(8)
    LAGUNA_ATTN_CASE(9)
    LAGUNA_ATTN_CASE(12)
    default:
      return false;
  }
#undef LAGUNA_ATTN_CASE
}

hipError_t DefaultAllocate(void** ptr, std::size_t bytes) {
  return hipMalloc(ptr, bytes);
}
hipError_t DefaultRelease(void* ptr) {
  return hipFree(ptr);
}

}  // namespace

// ---- LayerAttn
// ---------------------------------------------------------------

LayerAttn::LayerAttn(LayerAttn&& other) noexcept
    : w_(other.w_),
      allocations_(std::move(other.allocations_)),
      bytes_(other.bytes_) {
  other.allocations_.clear();
  other.bytes_ = 0;
  other.w_ = {};
}

LayerAttn& LayerAttn::operator=(LayerAttn&& other) noexcept {
  if (this != &other) {
    Release();
    w_ = other.w_;
    allocations_ = std::move(other.allocations_);
    bytes_ = other.bytes_;
    other.allocations_.clear();
    other.bytes_ = 0;
    other.w_ = {};
  }
  return *this;
}

LayerAttn::~LayerAttn() {
  Release();
}

void LayerAttn::Release() noexcept {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  allocations_.clear();
  bytes_ = 0;
  w_ = {};
}

namespace {

/// Copies `t` to the device (appending the allocation to `owned`). Quantized
/// matrices get 512 zeroed columns' worth of bytes after the last row, as
/// ggml's MATRIX_ROW_PADDING allocation does: the MMQ tile loaders read whole
/// 512-column tiles, so a row length that is not a multiple of 512 would read
/// past the allocation in the last row.
bool UploadTensor(const TensorRef& t, GgmlType type, std::uint64_t cols,
                  std::uint64_t rows, const char* what,
                  std::vector<void*>* owned, std::size_t* bytes,
                  const void** device, std::string* error_msg) {
  if (t.empty() || t.data == nullptr) {
    Assign(error_msg, std::string("laguna attention upload: missing ") + what);
    return false;
  }
  if (t.type != type || t.cols != cols || t.rows != rows || t.experts != 1) {
    Assign(error_msg, std::string("laguna attention upload: ") + what +
                          " has an unexpected type or shape");
    return false;
  }
  constexpr std::size_t kRowPadding = 512;  // ggml MATRIX_ROW_PADDING
  const std::size_t size = t.SizeBytes();
  const std::size_t tail =
      t.type == GgmlType::kF32 ? 0 : t.RowBytes() * kRowPadding / t.cols;
  void* ptr = nullptr;
  if (!CheckHip(hipMalloc(&ptr, size + tail), "hipMalloc", error_msg)) {
    return false;
  }
  owned->push_back(ptr);
  *bytes += size + tail;
  if (!CheckHip(hipMemcpy(ptr, t.data, size, hipMemcpyHostToDevice),
                "hipMemcpy", error_msg) ||
      (tail != 0 &&
       !CheckHip(hipMemset(static_cast<char*>(ptr) + size, 0, tail),
                 "hipMemset", error_msg))) {
    return false;
  }
  *device = ptr;
  return true;
}

}  // namespace

std::optional<LayerAttn> LayerAttn::Upload(const Config& config,
                                           const LayerWeights& layer,
                                           std::uint32_t il,
                                           std::string* error_msg) {
  const std::uint64_t hidden = config.hidden_size;
  const std::uint64_t heads = config.n_head(il);
  const std::uint64_t kvh = config.num_kv_heads;
  const std::uint64_t hd = config.head_dim;
  if (hd != kHeadDim) {
    Assign(error_msg, "laguna attention: head_dim must be 128");
    return std::nullopt;
  }
  if (kvh == 0 || heads % kvh != 0) {
    Assign(error_msg, "laguna attention: heads not a multiple of kv heads");
    return std::nullopt;
  }
  const reference::RopeParams rp = reference::RopeFor(config, il);
  if (rp.n_rot == 0 || rp.n_rot % 2 != 0 || rp.n_rot > hd) {
    Assign(error_msg, "laguna attention: bad rotary dimension");
    return std::nullopt;
  }
  LayerAttn result;
  AttnWeights& w = result.w_;
  const auto up = [&](const TensorRef& t, GgmlType type, std::uint64_t cols,
                      std::uint64_t rows, const char* what, const void** dst) {
    return UploadTensor(t, type, cols, rows, what, &result.allocations_,
                        &result.bytes_, dst, error_msg);
  };
  const auto matrix = [&](const TensorRef& t, std::uint64_t cols,
                          std::uint64_t rows, const char* what,
                          DeviceMatrix* m) {
    if (!up(t, GgmlType::kQ6_K, cols, rows, what, &m->data)) {
      return false;
    }
    m->type = GgmlType::kQ6_K;
    m->cols = static_cast<std::uint32_t>(cols);
    m->rows = static_cast<std::uint32_t>(rows);
    m->experts = 1;
    return true;
  };
  const void* gate = nullptr;
  const void* qn = nullptr;
  const void* kn = nullptr;
  const bool ok =
      matrix(layer.attn_q, hidden, heads * hd, "attn_q", &w.q) &&
      matrix(layer.attn_k, hidden, kvh * hd, "attn_k", &w.k) &&
      matrix(layer.attn_v, hidden, kvh * hd, "attn_v", &w.v) &&
      matrix(layer.attn_output, heads * hd, hidden, "attn_output", &w.o) &&
      up(layer.attn_gate, GgmlType::kF32, hidden, heads, "attn_gate", &gate) &&
      up(layer.attn_q_norm, GgmlType::kF32, hd, 1, "attn_q_norm", &qn) &&
      up(layer.attn_k_norm, GgmlType::kF32, hd, 1, "attn_k_norm", &kn);
  if (!ok) {
    return std::nullopt;
  }
  w.gate = static_cast<const float*>(gate);
  w.q_norm = static_cast<const float*>(qn);
  w.k_norm = static_cast<const float*>(kn);
  w.hidden = config.hidden_size;
  w.heads = config.n_head(il);
  w.kv_heads = config.num_kv_heads;
  w.head_dim = config.head_dim;
  w.window = config.is_full_attention(il) ? 0U : config.sliding_window;
  w.eps = config.rms_eps;
  const reference::RopeSetup setup = reference::MakeRopeSetup(rp);
  w.rope =
      RopeSpec{rp.n_rot,      setup.theta_scale, setup.corr[0], setup.corr[1],
               rp.freq_scale, rp.ext_factor,     setup.mscale};
  return result;
}

// ---- KvCache
// -----------------------------------------------------------------

KvCache::KvCache(KvCache&& other) noexcept
    : k_(other.k_),
      v_(other.v_),
      kv_heads_(other.kv_heads_),
      head_dim_(other.head_dim_),
      capacity_(other.capacity_),
      window_(other.window_),
      committed_(other.committed_),
      retained_begin_(other.retained_begin_),
      valid_(other.valid_) {
  other.k_ = nullptr;
  other.v_ = nullptr;
  other.capacity_ = 0;
}

KvCache& KvCache::operator=(KvCache&& other) noexcept {
  if (this != &other) {
    Release();
    k_ = other.k_;
    v_ = other.v_;
    kv_heads_ = other.kv_heads_;
    head_dim_ = other.head_dim_;
    capacity_ = other.capacity_;
    window_ = other.window_;
    committed_ = other.committed_;
    retained_begin_ = other.retained_begin_;
    valid_ = other.valid_;
    other.k_ = nullptr;
    other.v_ = nullptr;
    other.capacity_ = 0;
  }
  return *this;
}

KvCache::~KvCache() {
  Release();
}

void KvCache::Release() noexcept {
  if (k_ != nullptr) {
    (void)hipFree(k_);
  }
  if (v_ != nullptr) {
    (void)hipFree(v_);
  }
  k_ = nullptr;
  v_ = nullptr;
}

std::size_t KvCache::device_bytes() const noexcept {
  return std::size_t{2} * capacity_ * kv_heads_ * head_dim_ * sizeof(__half);
}

std::optional<KvCache> KvCache::Create(std::uint32_t kv_heads,
                                       std::uint32_t head_dim,
                                       std::uint32_t capacity,
                                       std::uint32_t window,
                                       std::string* error_msg) {
  if (kv_heads == 0 || head_dim != kHeadDim || capacity == 0 ||
      (window != 0 && capacity < window)) {
    Assign(error_msg, "laguna kv cache: bad geometry");
    return std::nullopt;
  }
  KvCache cache;
  cache.kv_heads_ = kv_heads;
  cache.head_dim_ = head_dim;
  cache.capacity_ = capacity;
  cache.window_ = window;
  const std::size_t bytes =
      std::size_t{capacity} * kv_heads * head_dim * sizeof(__half);
  // A failure leaves `cache` owning whatever was allocated; its destructor
  // frees it.
  if (!CheckHip(hipMalloc(&cache.k_, bytes), "hipMalloc keys", error_msg) ||
      !CheckHip(hipMalloc(&cache.v_, bytes), "hipMalloc values", error_msg)) {
    return std::nullopt;
  }
  return cache;
}

std::optional<KvCache> KvCache::Create(const Config& config, std::uint32_t il,
                                       std::uint32_t context,
                                       std::uint32_t max_chunk,
                                       std::string* error_msg) {
  const bool full = config.is_full_attention(il);
  const std::uint32_t window = full ? 0U : config.sliding_window;
  const std::uint32_t capacity = full ? context : window + max_chunk;
  return Create(config.num_kv_heads, config.head_dim, capacity, window,
                error_msg);
}

// ---- AttnOps
// -----------------------------------------------------------------

AttnOps::AttnOps(hipStream_t stream)
    : stream_(stream), allocator_{DefaultAllocate, DefaultRelease} {
  int device = 0;
  initialized_ =
      hipGetDevice(&device) == hipSuccess && qfn_mmq_init(device) == 0;
}

AttnOps::~AttnOps() {
  (void)hipStreamSynchronize(stream_);
  for (Scratch* s : {&q_, &k_, &v_, &gate_, &attn_, &ids_}) {
    Free(s);
  }
}

void AttnOps::Free(Scratch* buffer) noexcept {
  if (buffer->ptr != nullptr) {
    (void)allocator_.release(buffer->ptr);
  }
  buffer->ptr = nullptr;
  buffer->capacity = 0;
}

bool AttnOps::Reserve(Scratch* buffer, std::size_t bytes,
                      std::string* error_msg) {
  if (bytes <= buffer->capacity) {
    return true;
  }
  // Allocate the replacement first: a failure changes nothing. Only after it
  // succeeds is the stream drained (queued kernels may still read the old
  // buffer) and the old buffer freed and replaced.
  void* fresh = nullptr;
  if (!CheckHip(allocator_.allocate(&fresh, bytes), "hipMalloc scratch",
                error_msg)) {
    return false;
  }
  if (!CheckHip(hipStreamSynchronize(stream_), "hipStreamSynchronize",
                error_msg)) {
    (void)allocator_.release(fresh);
    return false;
  }
  void* old = buffer->ptr;
  buffer->ptr = fresh;
  buffer->capacity = bytes;
  if (old != nullptr) {
    (void)allocator_.release(old);
  }
  return true;
}

bool AttnOps::Emit(const char* name, const float* device, std::size_t count,
                   std::string* error_msg) {
  if (tap_) {
    if (!CheckHip(hipStreamSynchronize(stream_), "tap synchronize",
                  error_msg)) {
      return false;
    }
    tap_(name, device, count);
  }
  if (!fail_after_.empty() && fail_after_ == name) {
    Assign(error_msg,
           std::string("laguna attention: injected failure after ") + name);
    return false;
  }
  return true;
}

bool AttnOps::Project(const DeviceMatrix& w, const float* x, float* out,
                      std::uint32_t tokens, bool vector,
                      std::string* error_msg) {
  if (w.type != GgmlType::kQ6_K || w.experts != 1) {
    Assign(error_msg, "laguna attention projection needs a Q6_K matrix");
    return false;
  }
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int N = static_cast<int>(tokens);
  const auto* ids = static_cast<const std::int32_t*>(ids_.ptr);
  const int rc = vector ? qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x,
                                          ids, out, M, K, N, 1, 1, stream_)
                        : qfn_mmq_q6_K_moe_raw(w.data, x, ids, out, M, K, N, 1,
                                               1, stream_);
  if (rc != 0) {
    Assign(error_msg, "Q6_K projection failed");
    return false;
  }
  return true;
}

bool AttnOps::Forward(const AttnWeights& w, KvCache& cache, const float* n,
                      float* out, std::uint32_t tokens, std::uint32_t start_pos,
                      Path path, std::string* error_msg) {
  if (!initialized_) {
    Assign(error_msg, "qfn_mmq_init failed");
    return false;
  }
  const bool vector = path == Path::kVector ||
                      (path == Path::kAuto && tokens <= kVectorMaxTokens);
  if (tokens == 0 || (vector && tokens > kVectorMaxRows)) {
    Assign(error_msg, "laguna attention: unsupported row count for path");
    return false;
  }
  if (w.head_dim != kHeadDim || w.kv_heads == 0 || w.heads % w.kv_heads != 0 ||
      w.kv_heads != cache.kv_heads_ || cache.head_dim_ != kHeadDim ||
      w.window != cache.window_ || w.q.rows != w.heads * kHeadDim ||
      w.k.rows != w.kv_heads * kHeadDim || w.v.rows != w.k.rows ||
      w.q.cols != w.hidden || w.k.cols != w.hidden || w.v.cols != w.hidden ||
      w.o.cols != w.heads * kHeadDim || w.o.rows != w.hidden ||
      w.rope.rotary_dim == 0 || w.rope.rotary_dim % 2 != 0 ||
      w.rope.rotary_dim > kHeadDim || cache.k_ == nullptr ||
      cache.v_ == nullptr || w.hidden == 0) {
    Assign(error_msg, "laguna attention: inconsistent weights or cache");
    return false;
  }
  // Preflight everything a later stage would check: once the K/V append is
  // enqueued, a failure can no longer leave the cache untouched.
  for (const DeviceMatrix* m : {&w.q, &w.k, &w.v, &w.o}) {
    if (m->type != GgmlType::kQ6_K || m->experts != 1 || m->data == nullptr) {
      Assign(error_msg, "laguna attention projection needs a Q6_K matrix");
      return false;
    }
  }
  if (n == nullptr || out == nullptr || w.gate == nullptr ||
      w.q_norm == nullptr || w.k_norm == nullptr) {
    Assign(error_msg, "laguna attention: null input, output or tensor");
    return false;
  }
  if (!cache.valid_) {
    Assign(error_msg,
           "laguna attention: cache invalidated by a failed call; Reset() it");
    return false;
  }
  const std::uint32_t group = w.heads / w.kv_heads;
  if (group == 0 || group > 12 || group == 5 || group == 7 || group == 10 ||
      group == 11) {
    Assign(error_msg, "laguna attention: unsupported GQA group");
    return false;
  }
  if (start_pos > cache.committed_ ||
      std::uint64_t{start_pos} + tokens > UINT32_MAX) {
    Assign(error_msg, "laguna attention: start_pos beyond the cached rows");
    return false;
  }
  if (cache.window_ == 0) {
    if (std::uint64_t{start_pos} + tokens > cache.capacity_) {
      Assign(error_msg, "laguna attention: context exceeds the cache");
      return false;
    }
  } else {
    // Ring: a chunk must fit beside the window, and a rewind must leave the
    // window of keys before start_pos intact.
    if (tokens > cache.capacity_ - cache.window_ ||
        cache.committed_ - start_pos > cache.capacity_ - cache.window_) {
      Assign(error_msg, "laguna attention: chunk or rewind exceeds the ring");
      return false;
    }
  }
  // Keys before start_pos that the rows read: [first_key, start_pos). They
  // must still be in their slots; a rewind cannot bring back evicted ones.
  const std::uint32_t first_key =
      cache.window_ != 0 && start_pos + 1 > cache.window_
          ? start_pos + 1 - cache.window_
          : 0;
  if (start_pos > 0 && first_key < cache.retained_begin_) {
    Assign(
        error_msg,
        "laguna attention: keys the call needs were overwritten in the ring");
    return false;
  }
  const std::size_t q_dim = std::size_t{w.heads} * kHeadDim;
  const std::size_t kv_dim = std::size_t{w.kv_heads} * kHeadDim;
  // Reserve everything first; pointers are read only afterwards.
  if (!Reserve(&q_, tokens * q_dim * sizeof(float), error_msg) ||
      !Reserve(&k_, tokens * kv_dim * sizeof(float), error_msg) ||
      !Reserve(&v_, tokens * kv_dim * sizeof(float), error_msg) ||
      !Reserve(&gate_, std::size_t{tokens} * w.heads * sizeof(float),
               error_msg) ||
      !Reserve(&attn_, tokens * q_dim * sizeof(float), error_msg) ||
      !Reserve(&ids_, tokens * sizeof(std::int32_t), error_msg)) {
    return false;
  }
  float* q = static_cast<float*>(q_.ptr);
  float* k = static_cast<float*>(k_.ptr);
  float* v = static_cast<float*>(v_.ptr);
  float* gate = static_cast<float*>(gate_.ptr);
  float* attn = static_cast<float*>(attn_.ptr);
  if (!CheckHip(
          hipMemsetAsync(ids_.ptr, 0, tokens * sizeof(std::int32_t), stream_),
          "hipMemsetAsync", error_msg)) {
    return false;
  }
  // Each stage is emitted right after its projection, so a tap (and the
  // activation capture of tests) sees each projection's quantization under its
  // own stage name.
  if (!Project(w.q, n, q, tokens, vector, error_msg) ||
      !Emit("q_proj", q, tokens * q_dim, error_msg) ||
      !Project(w.k, n, k, tokens, vector, error_msg) ||
      !Emit("k_proj", k, tokens * kv_dim, error_msg) ||
      !Project(w.v, n, v, tokens, vector, error_msg) ||
      !Emit("v_proj", v, tokens * kv_dim, error_msg)) {
    return false;
  }
  GateLogitsKernel<<<dim3(w.heads, tokens), 128, 0, stream_>>>(
      n, w.gate, gate, w.hidden, w.heads);
  if (!CheckHip(hipGetLastError(), "gate launch", error_msg) ||
      !Emit("gate_proj", gate, std::size_t{tokens} * w.heads, error_msg)) {
    return false;
  }
  const RopeArgs rope{w.rope.rotary_dim, w.rope.theta_scale, w.rope.corr_low,
                      w.rope.corr_high,  w.rope.freq_scale,  w.rope.ext_factor,
                      w.rope.mscale};
  // From here on the call's K/V rows may be in the cache: any failure
  // invalidates it, since its slots no longer match committed().
  const auto invalidate = [&cache] {
    cache.valid_ = false;
    cache.committed_ = 0;
    cache.retained_begin_ = 0;
    return false;
  };
  QkNormRopeAppendKernel<<<dim3(tokens, w.heads + 2 * w.kv_heads), kHeadDim, 0,
                           stream_>>>(
      q, k, v, w.q_norm, w.k_norm, static_cast<__half*>(cache.k_),
      static_cast<__half*>(cache.v_), w.heads, w.kv_heads, w.eps, start_pos,
      cache.capacity_, rope);
  if (!CheckHip(hipGetLastError(), "qk norm rope launch", error_msg) ||
      !Emit("q_rope", q, tokens * q_dim, error_msg)) {
    return invalidate();
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  if (!DispatchAttention(group, q, static_cast<const __half*>(cache.k_),
                         static_cast<const __half*>(cache.v_), attn, w.heads,
                         w.kv_heads, w.window, start_pos, tokens,
                         cache.capacity_, scale, stream_) ||
      !CheckHip(hipGetLastError(), "attention launch", error_msg) ||
      !Emit("attn", attn, tokens * q_dim, error_msg)) {
    return invalidate();
  }
  GateKernel<<<std::size_t{tokens} * w.heads, kHeadDim, 0, stream_>>>(gate,
                                                                      attn);
  if (!CheckHip(hipGetLastError(), "gate activation launch", error_msg) ||
      !Emit("gate_act", gate, std::size_t{tokens} * w.heads, error_msg) ||
      !Emit("gated", attn, tokens * q_dim, error_msg)) {
    return invalidate();
  }
  // Project the gated rows. The rows were produced in float, so they are
  // quantized exactly as the oracle quantizes them.
  if (!Project(w.o, attn, out, tokens, vector, error_msg) ||
      !Emit("o", out, std::size_t{tokens} * w.hidden, error_msg)) {
    return invalidate();
  }
  // Writing [start_pos, end) evicted every position below end - capacity. A
  // call below retained_begin (only a start at 0, which needs no history)
  // keeps only its own rows.
  const std::uint32_t end = start_pos + tokens;
  std::uint32_t retained = std::min(cache.retained_begin_, start_pos);
  if (end > cache.capacity_) {
    retained = std::max(retained, end - cache.capacity_);
  }
  cache.retained_begin_ = retained;
  cache.committed_ = end;
  return true;
}

}  // namespace gufo::models::laguna::rocm
