// Laguna DFlash drafter GPU operators. See dflash_ops.hpp.
//
// The kernels here use plain IEEE arithmetic (no fast-math) and no FP
// contraction, so their rounding stays close to the CPU oracle's. Norms sum
// in double, as the oracle does. The Q4_K/Q6_K projections reuse the qfn_mmq
// kernels vendored with qwen38_flash_next (the routed-expert entry points with
// one expert and zero ids, which is a dense matrix product); no dot product of
// the weights is written here.
#pragma clang fp contract(off)

#include "src/models/laguna/kernels/dflash_ops.hpp"

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
constexpr int kNormThreads = 256;

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

hipError_t DefaultAllocate(void** ptr, std::size_t bytes) {
  return hipMalloc(ptr, bytes);
}
hipError_t DefaultRelease(void* ptr) {
  return hipFree(ptr);
}

bool ProjectionType(GgmlType type) {
  return type == GgmlType::kQ4_K || type == GgmlType::kQ6_K;
}

// ---- Kernels ---------------------------------------------------------------

/// y[r] = (x[r] * scale) * w[r % weight_rows], scale = 1 / sqrt(mean(x^2) +
/// eps) with the mean summed in double: the oracle's RmsNormRow. One block per
/// row of `n`. With weight_rows > 1, consecutive rows cycle through the weight
/// rows (the encoder's per-tap norms: row t*taps + k uses aux_norm[k]).
__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ y, std::uint32_t n,
                              std::uint32_t weight_rows, float eps) {
  const float* xr = x + std::size_t{blockIdx.x} * n;
  float* yr = y + std::size_t{blockIdx.x} * n;
  const float* wr = w + std::size_t{blockIdx.x % weight_rows} * n;
  double sum = 0.0;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    sum += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
  }
  __shared__ double part[kNormThreads];
  part[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      part[threadIdx.x] += part[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = static_cast<float>(part[0] / static_cast<double>(n));
  const float scale = 1.0F / sqrtf(mean + eps);
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    yr[i] = (xr[i] * scale) * wr[i];
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

RopeArgs ToArgs(const RopeSpec& r) {
  return RopeArgs{r.rotary_dim, r.theta_scale, r.corr_low, r.corr_high,
                  r.freq_scale, r.ext_factor,  r.mscale};
}

/// grid (rows, heads + 2*kv_heads), block 128. Block y < heads: query head
/// (norm, rope, in place in q). Then key heads (norm, rope, F16 into the
/// ring), then value heads (F16 into the ring). Row t sits at start_pos + t.
/// The rotation is ggml's rope cache in the oracle's float arithmetic (the
/// angle of pair i is the position times theta_scale, i times), as the
/// target's attention operator computes it. Norm: y = (x * rsqrt(mean(x^2) +
/// eps)) * gamma.
__global__ void QkNormRopeStoreKernel(
    float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_gamma,
    const float* __restrict__ k_gamma, __half* __restrict__ k_ring,
    __half* __restrict__ v_ring, std::uint32_t heads, std::uint32_t kv_heads,
    float eps, std::uint32_t start_pos, std::uint32_t capacity, RopeArgs rope) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t b = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t pos = start_pos + t;
  const std::size_t kv_width = std::size_t{kv_heads} * kHeadDim;
  const std::size_t slot = pos % capacity;
  if (b >= heads + kv_heads) {
    const std::uint32_t h = b - heads - kv_heads;
    v_ring[slot * kv_width + h * kHeadDim + d] =
        __float2half(v[std::size_t{t} * kv_width + h * kHeadDim + d]);
    return;
  }
  const bool is_q = b < heads;
  const std::uint32_t h = is_q ? b : b - heads;
  float* qrow = is_q ? q + (std::size_t{t} * heads + h) * kHeadDim : nullptr;
  const float* krow = k + std::size_t{t} * kv_width + h * kHeadDim;
  const float x = is_q ? qrow[d] : krow[d];
  __shared__ float part[kHeadDim / 32];
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
    k_ring[slot * kv_width + h * kHeadDim + d] = __float2half(out);
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

constexpr int kMaxWaves = kHeadDim / 32;

/// grid (kv_heads, rows), block 128: the target's attention kernel
/// (kernels/attention.hip.cpp) with the drafter's key range. A block serves
/// the kGroup query heads of one KV head for block row t at p = start + t.
/// Visible keys: first <= j <= last with first = max(p + 1 - window, floor)
/// and last = p (causal) or start + rows - 1. Keys stream in tiles of 128
/// from `first`: thread t scores key t of the tile for every head, the
/// tile's max and sum are reduced over the block, then thread d adds the
/// tile's V column d with online-softmax rescaling. The tile boundaries
/// depend only on p, so a row's sum order does not depend on the block length.
template<int kGroup>
__global__ void BlockAttentionKernel(
    const float* __restrict__ q, const __half* __restrict__ k_ring,
    const __half* __restrict__ v_ring, float* __restrict__ out,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t window,
    std::uint32_t floor, std::uint32_t start, std::uint32_t rows,
    std::uint32_t capacity, bool causal, float scale) {
  const std::uint32_t kvh = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t p = start + t;
  const std::uint32_t lo = p + 1 <= window ? 0 : p + 1 - window;
  const std::uint32_t first = lo > floor ? lo : floor;
  const std::uint32_t last = causal ? p : start + rows - 1;
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
  for (std::uint32_t ts = first; ts <= last; ts += kHeadDim) {
    const std::uint32_t j = ts + d;
    const bool valid = j <= last;
    float s[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      s[g] = 0.0F;
    }
    if (valid) {
      const uint4* krow = reinterpret_cast<const uint4*>(
          k_ring + std::size_t{j % capacity} * kv_width + kvh * kHeadDim);
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
    const std::uint32_t count = min(kHeadDim, last - ts + 1);
    for (std::uint32_t jj = 0; jj < count; ++jj) {
      const float vv =
          __half2float(v_ring[std::size_t{(ts + jj) % capacity} * kv_width +
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

struct AttentionArgs {
  const float* q;
  const __half* k;
  const __half* v;
  float* out;
  std::uint32_t heads, kv_heads, window, floor, start, rows, capacity;
  bool causal;
  float scale;
};

bool DispatchAttention(std::uint32_t group, const AttentionArgs& a,
                       hipStream_t stream) {
#define LAGUNA_DFLASH_ATTN_CASE(G)                                        \
  case G:                                                                 \
    BlockAttentionKernel<G>                                               \
        <<<dim3(a.kv_heads, a.rows), kHeadDim, 0, stream>>>(              \
            a.q, a.k, a.v, a.out, a.heads, a.kv_heads, a.window, a.floor, \
            a.start, a.rows, a.capacity, a.causal, a.scale);              \
    return true;
  switch (group) {
    LAGUNA_DFLASH_ATTN_CASE(1)
    LAGUNA_DFLASH_ATTN_CASE(2)
    LAGUNA_DFLASH_ATTN_CASE(3)
    LAGUNA_DFLASH_ATTN_CASE(4)
    LAGUNA_DFLASH_ATTN_CASE(6)
    LAGUNA_DFLASH_ATTN_CASE(8)
    LAGUNA_DFLASH_ATTN_CASE(9)
    LAGUNA_DFLASH_ATTN_CASE(12)
    default:
      return false;
  }
#undef LAGUNA_DFLASH_ATTN_CASE
}

bool SupportedGroup(std::uint32_t group) {
  return group == 1 || group == 2 || group == 3 || group == 4 || group == 6 ||
         group == 8 || group == 9 || group == 12;
}

/// gate = softplus(gate) (x > 20 ? x : log(1 + exp(x)), llama.cpp's form);
/// attn[t][h][:] *= gate[t][h]. One block of 128 per (row, head).
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

/// act = silu(gate) * up, silu(g) = g / (1 + exp(-g)) (the oracle's).
__global__ void SwiGluKernel(const float* __restrict__ gate,
                             const float* __restrict__ up,
                             float* __restrict__ act, std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count) {
    const float g = gate[i];
    act[i] = (g / (1.0F + expf(-g))) * up[i];
  }
}

__global__ void AddKernel(const float* __restrict__ a,
                          const float* __restrict__ b, float* __restrict__ out,
                          std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = a[i] + b[i];
  }
}

constexpr int kArgmaxThreads = 256;

/// drafts[r] = the first index of the largest value of logits row r (NaN
/// never wins; -1 when no element is larger than -inf). One block per row.
__global__ void ArgmaxKernel(const float* __restrict__ logits,
                             std::int32_t* __restrict__ drafts,
                             std::uint32_t vocab) {
  const float* row = logits + std::size_t{blockIdx.x} * vocab;
  float best = -INFINITY;
  std::int32_t arg = -1;
  for (std::uint32_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    if (row[i] > best) {  // strict: the first maximum of this thread wins
      best = row[i];
      arg = static_cast<std::int32_t>(i);
    }
  }
  __shared__ float value[kArgmaxThreads];
  __shared__ std::int32_t index[kArgmaxThreads];
  value[threadIdx.x] = best;
  index[threadIdx.x] = arg;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float v = value[threadIdx.x + stride];
      const std::int32_t j = index[threadIdx.x + stride];
      const std::int32_t i = index[threadIdx.x];
      if (j >= 0 && (i < 0 || v > value[threadIdx.x] ||
                     (v == value[threadIdx.x] && j < i))) {
        value[threadIdx.x] = v;
        index[threadIdx.x] = j;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    drafts[blockIdx.x] = index[0];
  }
}

// ---- Upload ----------------------------------------------------------------

/// Copies `t` to the device (appending the allocation to `owned`). Quantized
/// matrices get 512 zeroed columns' worth of bytes after the last row, as
/// ggml's MATRIX_ROW_PADDING allocation does: the MMQ tile loaders read whole
/// 512-column tiles.
bool UploadTensor(const TensorRef& t, std::uint64_t cols, std::uint64_t rows,
                  bool quantized, const char* what, std::vector<void*>* owned,
                  std::size_t* bytes, const void** device,
                  std::string* error_msg) {
  if (t.empty() || t.data == nullptr) {
    Assign(error_msg, std::string("laguna dflash upload: missing ") + what);
    return false;
  }
  const bool type_ok =
      quantized ? ProjectionType(t.type) : t.type == GgmlType::kF32;
  if (!type_ok || t.cols != cols || t.rows != rows || t.experts != 1) {
    Assign(error_msg, std::string("laguna dflash upload: ") + what +
                          " has an unexpected type or shape");
    return false;
  }
  constexpr std::size_t kRowPadding = 512;  // ggml MATRIX_ROW_PADDING
  const std::size_t size = t.SizeBytes();
  const std::size_t tail = quantized ? t.RowBytes() * kRowPadding / t.cols : 0;
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

// ---- DeviceDrafter ----------------------------------------------------------

DeviceDrafter::DeviceDrafter(DeviceDrafter&& other) noexcept
    : w_(std::move(other.w_)),
      allocations_(std::move(other.allocations_)),
      bytes_(other.bytes_) {
  other.allocations_.clear();
  other.bytes_ = 0;
  other.w_ = {};
}

DeviceDrafter& DeviceDrafter::operator=(DeviceDrafter&& other) noexcept {
  if (this != &other) {
    Release();
    w_ = std::move(other.w_);
    allocations_ = std::move(other.allocations_);
    bytes_ = other.bytes_;
    other.allocations_.clear();
    other.bytes_ = 0;
    other.w_ = {};
  }
  return *this;
}

DeviceDrafter::~DeviceDrafter() {
  Release();
}

void DeviceDrafter::Release() noexcept {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  allocations_.clear();
  bytes_ = 0;
  w_ = {};
}

std::optional<DeviceDrafter> DeviceDrafter::Upload(const DFlashWeights& dw,
                                                   std::string* error_msg) {
  const DFlashConfig& c = dw.config;
  if (c.head_dim != kHeadDim || c.rotary_dim == 0 || c.rotary_dim % 2 != 0 ||
      c.rotary_dim > kHeadDim) {
    Assign(error_msg, "laguna dflash: head_dim must be 128 with an even rope");
    return std::nullopt;
  }
  if (c.num_kv_heads == 0 || c.num_heads % c.num_kv_heads != 0 ||
      c.tap_count() == 0 || c.hidden_size == 0 || c.ffn_size == 0 ||
      c.sliding_window == 0 || c.block_size < 2 ||
      dw.layers.size() != c.num_layers) {
    Assign(error_msg, "laguna dflash: unsupported drafter geometry");
    return std::nullopt;
  }
  DeviceDrafter result;
  DrafterWeights& w = result.w_;
  const std::uint64_t H = c.hidden_size;
  const auto up = [&](const TensorRef& t, std::uint64_t cols,
                      std::uint64_t rows, bool quantized, const char* what,
                      const void** dst) {
    return UploadTensor(t, cols, rows, quantized, what, &result.allocations_,
                        &result.bytes_, dst, error_msg);
  };
  const auto f32 = [&](const TensorRef& t, std::uint64_t cols,
                       std::uint64_t rows, const char* what,
                       const float** dst) {
    const void* p = nullptr;
    if (!up(t, cols, rows, false, what, &p)) {
      return false;
    }
    *dst = static_cast<const float*>(p);
    return true;
  };
  const auto matrix = [&](const TensorRef& t, std::uint64_t cols,
                          std::uint64_t rows, const char* what,
                          DeviceMatrix* m) {
    if (!up(t, cols, rows, true, what, &m->data)) {
      return false;
    }
    m->type = t.type;
    m->cols = static_cast<std::uint32_t>(cols);
    m->rows = static_cast<std::uint32_t>(rows);
    m->experts = 1;
    return true;
  };
  const std::uint64_t taps = c.tap_count();
  if (!f32(dw.aux_norm, H, taps, "enc.aux_norm", &w.aux_norm) ||
      !matrix(dw.fc, taps * H, H, "fc", &w.fc) ||
      !f32(dw.enc_output_norm, H, 1, "enc.output_norm", &w.enc_norm) ||
      !f32(dw.output_norm, H, 1, "output_norm", &w.output_norm)) {
    return std::nullopt;
  }
  const std::uint64_t qd = std::uint64_t{c.num_heads} * kHeadDim;
  const std::uint64_t kvd = std::uint64_t{c.num_kv_heads} * kHeadDim;
  w.layers.resize(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DFlashLayerWeights& s = dw.layers[il];
    DrafterLayer& l = w.layers[il];
    if (!f32(s.attn_norm, H, 1, "attn_norm", &l.attn_norm) ||
        !matrix(s.attn_q, H, qd, "attn_q", &l.q) ||
        !matrix(s.attn_k, H, kvd, "attn_k", &l.k) ||
        !matrix(s.attn_v, H, kvd, "attn_v", &l.v) ||
        !matrix(s.attn_output, qd, H, "attn_output", &l.o) ||
        !matrix(s.attn_gate, H, c.num_heads, "attn_gate", &l.gate) ||
        !f32(s.attn_q_norm, kHeadDim, 1, "attn_q_norm", &l.q_norm) ||
        !f32(s.attn_k_norm, kHeadDim, 1, "attn_k_norm", &l.k_norm) ||
        !f32(s.ffn_norm, H, 1, "ffn_norm", &l.ffn_norm) ||
        !matrix(s.ffn_gate, H, c.ffn_size, "ffn_gate", &l.ffn_gate) ||
        !matrix(s.ffn_up, H, c.ffn_size, "ffn_up", &l.ffn_up) ||
        !matrix(s.ffn_down, c.ffn_size, H, "ffn_down", &l.ffn_down)) {
      return std::nullopt;
    }
  }
  w.hidden = c.hidden_size;
  w.ffn = c.ffn_size;
  w.heads = c.num_heads;
  w.kv_heads = c.num_kv_heads;
  w.head_dim = c.head_dim;
  w.taps = c.tap_count();
  w.window = c.sliding_window;
  w.block_size = c.block_size;
  w.eps = c.rms_eps;
  const reference::RopeParams rp{.n_rot = c.rotary_dim,
                                 .freq_base = c.rope_theta};
  const reference::RopeSetup setup = reference::MakeRopeSetup(rp);
  w.rope =
      RopeSpec{rp.n_rot,      setup.theta_scale, setup.corr[0], setup.corr[1],
               rp.freq_scale, rp.ext_factor,     setup.mscale};
  w.causal_block = true;
  return result;
}

// ---- DrafterKv --------------------------------------------------------------

DrafterKv::DrafterKv(DrafterKv&& other) noexcept
    : k_(std::move(other.k_)),
      v_(std::move(other.v_)),
      kv_heads_(other.kv_heads_),
      capacity_(other.capacity_),
      window_(other.window_),
      committed_(other.committed_),
      retained_begin_(other.retained_begin_),
      floor_(other.floor_),
      valid_(other.valid_) {
  other.k_.clear();
  other.v_.clear();
  other.capacity_ = 0;
}

DrafterKv& DrafterKv::operator=(DrafterKv&& other) noexcept {
  if (this != &other) {
    Release();
    k_ = std::move(other.k_);
    v_ = std::move(other.v_);
    kv_heads_ = other.kv_heads_;
    capacity_ = other.capacity_;
    window_ = other.window_;
    committed_ = other.committed_;
    retained_begin_ = other.retained_begin_;
    floor_ = other.floor_;
    valid_ = other.valid_;
    other.k_.clear();
    other.v_.clear();
    other.capacity_ = 0;
  }
  return *this;
}

DrafterKv::~DrafterKv() {
  Release();
}

void DrafterKv::Release() noexcept {
  for (std::vector<void*>* list : {&k_, &v_}) {
    for (void* p : *list) {
      if (p != nullptr) {
        (void)hipFree(p);
      }
    }
    list->clear();
  }
}

std::string DrafterKv::RejectContinue(std::uint32_t position) const {
  if (!valid_) {
    return "laguna dflash: kv invalidated by a failed call; Reset() it";
  }
  if (position < floor_ || position > committed_) {
    return "laguna dflash: position outside [floor, committed]";
  }
  const std::uint32_t lo = position + 1 <= window_ ? 0 : position + 1 - window_;
  if (std::max(lo, floor_) < retained_begin_) {
    return "laguna dflash: context keys the block needs were overwritten";
  }
  return {};
}

bool DrafterKv::RewindTo(std::uint32_t position, std::string* error_msg) {
  if (std::string why = RejectContinue(position); !why.empty()) {
    Assign(error_msg, why);
    return false;
  }
  retained_begin_ = std::min(retained_begin_, position);
  committed_ = position;
  return true;
}

std::optional<DrafterKv> DrafterKv::Create(const DrafterWeights& w,
                                           std::uint32_t max_rows,
                                           std::string* error_msg) {
  if (w.kv_heads == 0 || w.head_dim != kHeadDim || w.window == 0 ||
      w.layers.empty() || max_rows == 0) {
    Assign(error_msg, "laguna dflash kv: bad geometry");
    return std::nullopt;
  }
  DrafterKv kv;
  kv.kv_heads_ = w.kv_heads;
  kv.window_ = w.window;
  kv.capacity_ = w.window + std::max(max_rows, w.block_size);
  const std::size_t bytes = std::size_t{kv.capacity_} * kv.row_bytes();
  // A failure leaves `kv` owning whatever was allocated; its destructor frees
  // it.
  for (std::size_t il = 0; il < w.layers.size(); ++il) {
    for (std::vector<void*>* list : {&kv.k_, &kv.v_}) {
      void* p = nullptr;
      if (!CheckHip(hipMalloc(&p, bytes), "hipMalloc drafter kv", error_msg)) {
        return std::nullopt;
      }
      list->push_back(p);
    }
  }
  return kv;
}

// ---- DFlashOps --------------------------------------------------------------

DFlashOps::DFlashOps(hipStream_t stream)
    : stream_(stream), allocator_{DefaultAllocate, DefaultRelease} {
  int device = 0;
  initialized_ =
      hipGetDevice(&device) == hipSuccess && qfn_mmq_init(device) == 0;
}

DFlashOps::~DFlashOps() {
  (void)hipStreamSynchronize(stream_);
  for (Scratch* s : {&ids_, &aux_, &u_, &fused_, &x_, &k_, &v_, &q_, &gate_,
                     &attn_, &fg_, &fu_, &fa_, &h_, &n_, &o_, &r_}) {
    Free(s);
  }
}

std::size_t DFlashOps::scratch_bytes() const noexcept {
  std::size_t total = 0;
  for (const Scratch* s :
       {&ids_, &aux_, &u_, &fused_, &x_, &k_, &v_, &q_, &gate_, &attn_, &fg_,
        &fu_, &fa_, &h_, &n_, &o_, &r_}) {
    total += s->capacity;
  }
  return total;
}

void DFlashOps::Free(Scratch* buffer) noexcept {
  if (buffer->ptr != nullptr) {
    (void)allocator_.release(buffer->ptr);
  }
  buffer->ptr = nullptr;
  buffer->capacity = 0;
}

bool DFlashOps::Reserve(Scratch* buffer, std::size_t bytes,
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

bool DFlashOps::Emit(const char* name, int layer, const float* device,
                     std::size_t count, std::string* error_msg) {
  if (!tap_ && fail_after_.empty()) {
    return true;
  }
  const std::string full = layer >= 0
                               ? std::string(name) + "-" + std::to_string(layer)
                               : std::string(name);
  if (tap_) {
    if (!CheckHip(hipStreamSynchronize(stream_), "tap synchronize",
                  error_msg)) {
      return false;
    }
    tap_(full, device, count);
  }
  if (!fail_after_.empty() && fail_after_ == full) {
    Assign(error_msg, "laguna dflash: injected failure after " + full);
    return false;
  }
  return true;
}

bool DFlashOps::Project(const DeviceMatrix& w, const float* x, float* out,
                        std::uint32_t rows, bool vector,
                        std::string* error_msg) {
  if (!ProjectionType(w.type) || w.experts != 1 || w.data == nullptr) {
    Assign(error_msg, "laguna dflash projection needs a Q4_K or Q6_K matrix");
    return false;
  }
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int N = static_cast<int>(rows);
  const auto* ids = static_cast<const std::int32_t*>(ids_.ptr);
  int rc = -1;
  if (vector) {
    rc = qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x, ids, out, M, K, N,
                         1, 1, stream_);
  } else if (w.type == GgmlType::kQ4_K) {
    rc = qfn_mmq_q4_K_moe_raw(w.data, x, ids, out, M, K, N, 1, 1, stream_);
  } else {
    rc = qfn_mmq_q6_K_moe_raw(w.data, x, ids, out, M, K, N, 1, 1, stream_);
  }
  if (rc != 0) {
    Assign(error_msg, std::string("laguna dflash ") +
                          std::string(core::ToString(w.type)) +
                          " projection failed");
    return false;
  }
  return true;
}

bool DFlashOps::Ready(Path path, std::uint32_t rows, bool* vector,
                      std::string* error_msg) const {
  if (!initialized_) {
    Assign(error_msg, "qfn_mmq_init failed");
    return false;
  }
  *vector = path == Path::kVector ||
            (path == Path::kAuto && rows <= kVectorMaxTokens);
  if (rows == 0 || (*vector && rows > kVectorMaxRows)) {
    Assign(error_msg, "laguna dflash: unsupported row count for path");
    return false;
  }
  return true;
}

bool DFlashOps::PrepareIds(std::uint32_t rows, std::string* error_msg) {
  const std::size_t bytes = std::size_t{rows} * sizeof(std::int32_t);
  return Reserve(&ids_, bytes, error_msg) &&
         CheckHip(hipMemsetAsync(ids_.ptr, 0, bytes, stream_), "hipMemsetAsync",
                  error_msg);
}

bool DFlashOps::Encode(const DrafterWeights& w, const float* features,
                       float* fused, std::uint32_t rows, Path path,
                       std::string* error_msg) {
  bool vector = false;
  return Ready(path, rows, &vector, error_msg) &&
         EncodeImpl(w, features, fused, rows, vector, error_msg);
}

bool DFlashOps::EncodeImpl(const DrafterWeights& w, const float* features,
                           float* fused, std::uint32_t rows, bool vector,
                           std::string* error_msg) {
  const std::size_t width = std::size_t{w.taps} * w.hidden;
  if (w.taps == 0 || w.hidden == 0 || w.fc.cols != width ||
      w.fc.rows != w.hidden || !ProjectionType(w.fc.type) ||
      w.fc.data == nullptr || w.aux_norm == nullptr || w.enc_norm == nullptr) {
    Assign(error_msg, "laguna dflash encoder: inconsistent weights");
    return false;
  }
  if (features == nullptr || fused == nullptr) {
    Assign(error_msg, "laguna dflash encoder: null input or output");
    return false;
  }
  if (!PrepareIds(rows, error_msg) ||
      !Reserve(&aux_, rows * width * sizeof(float), error_msg) ||
      !Reserve(&u_, std::size_t{rows} * w.hidden * sizeof(float), error_msg)) {
    return false;
  }
  float* aux = static_cast<float*>(aux_.ptr);
  float* u = static_cast<float*>(u_.ptr);
  RmsNormKernel<<<rows * w.taps, kNormThreads, 0, stream_>>>(
      features, w.aux_norm, aux, w.hidden, w.taps, w.eps);
  if (!CheckHip(hipGetLastError(), "aux norm launch", error_msg) ||
      !Emit("enc_aux_norm", -1, aux, rows * width, error_msg) ||
      !Project(w.fc, aux, u, rows, vector, error_msg) ||
      !Emit("fc_out", -1, u, std::size_t{rows} * w.hidden, error_msg)) {
    return false;
  }
  RmsNormKernel<<<rows, kNormThreads, 0, stream_>>>(u, w.enc_norm, fused,
                                                    w.hidden, 1, w.eps);
  return CheckHip(hipGetLastError(), "encoder norm launch", error_msg) &&
         Emit("enc_norm_out", -1, fused, std::size_t{rows} * w.hidden,
              error_msg);
}

namespace {

/// Shapes and pointers of one layer's attention tensors.
bool LayerConsistent(const DrafterWeights& w, const DrafterLayer& l) {
  const std::uint32_t qd = w.heads * kHeadDim;
  const std::uint32_t kvd = w.kv_heads * kHeadDim;
  const auto ok = [](const DeviceMatrix& m, std::uint32_t cols,
                     std::uint32_t rows) {
    return ProjectionType(m.type) && m.experts == 1 && m.data != nullptr &&
           m.cols == cols && m.rows == rows;
  };
  return ok(l.q, w.hidden, qd) && ok(l.k, w.hidden, kvd) &&
         ok(l.v, w.hidden, kvd) && ok(l.o, qd, w.hidden) &&
         ok(l.gate, w.hidden, w.heads) && ok(l.ffn_gate, w.hidden, w.ffn) &&
         ok(l.ffn_up, w.hidden, w.ffn) && ok(l.ffn_down, w.ffn, w.hidden) &&
         l.attn_norm != nullptr && l.q_norm != nullptr && l.k_norm != nullptr &&
         l.ffn_norm != nullptr;
}

bool GeometryConsistent(const DrafterWeights& w) {
  if (w.head_dim != kHeadDim || w.kv_heads == 0 || w.heads % w.kv_heads != 0 ||
      w.hidden == 0 || w.ffn == 0 || w.window == 0 || w.layers.empty() ||
      w.rope.rotary_dim == 0 || w.rope.rotary_dim % 2 != 0 ||
      w.rope.rotary_dim > kHeadDim) {
    return false;
  }
  return std::all_of(
      w.layers.begin(), w.layers.end(),
      [&](const DrafterLayer& l) { return LayerConsistent(w, l); });
}

}  // namespace

bool DFlashOps::CheckInject(const DrafterWeights& w, const DrafterKv& kv,
                            std::uint32_t rows, std::uint32_t start,
                            std::string* error_msg) const {
  if (!GeometryConsistent(w)) {
    Assign(error_msg, "laguna dflash: inconsistent drafter weights");
    return false;
  }
  if (kv.k_.size() != w.layers.size() || kv.kv_heads_ != w.kv_heads ||
      kv.window_ != w.window) {
    Assign(error_msg, "laguna dflash: kv rings do not match the drafter");
    return false;
  }
  if (!kv.valid_) {
    Assign(error_msg,
           "laguna dflash: kv invalidated by a failed call; Reset() it");
    return false;
  }
  if (start < kv.floor_ || start > kv.committed_) {
    Assign(error_msg,
           "laguna dflash: injection must start in [floor, committed]");
    return false;
  }
  if (rows > kv.max_rows() || std::uint64_t{start} + rows > UINT32_MAX) {
    Assign(error_msg, "laguna dflash: injection exceeds the ring");
    return false;
  }
  return true;
}

bool DFlashOps::InjectKv(const DrafterWeights& w, DrafterKv& kv,
                         const float* fused, std::uint32_t rows,
                         std::uint32_t start, Path path,
                         std::string* error_msg) {
  bool vector = false;
  if (!Ready(path, rows, &vector, error_msg) ||
      !CheckInject(w, kv, rows, start, error_msg)) {
    return false;
  }
  if (fused == nullptr) {
    Assign(error_msg, "laguna dflash: null injection input");
    return false;
  }
  return InjectImpl(w, kv, fused, rows, start, vector, error_msg);
}

bool DFlashOps::Inject(const DrafterWeights& w, DrafterKv& kv,
                       const float* features, std::uint32_t rows,
                       std::uint32_t start, Path path, std::string* error_msg) {
  bool vector = false;
  if (!Ready(path, rows, &vector, error_msg) ||
      !CheckInject(w, kv, rows, start, error_msg) ||
      !Reserve(&fused_, std::size_t{rows} * w.hidden * sizeof(float),
               error_msg)) {
    return false;
  }
  float* fused = static_cast<float*>(fused_.ptr);
  return EncodeImpl(w, features, fused, rows, vector, error_msg) &&
         InjectImpl(w, kv, fused, rows, start, vector, error_msg);
}

bool DFlashOps::InjectImpl(const DrafterWeights& w, DrafterKv& kv,
                           const float* fused, std::uint32_t rows,
                           std::uint32_t start, bool vector,
                           std::string* error_msg) {
  const std::size_t kvd = std::size_t{w.kv_heads} * kHeadDim;
  if (!PrepareIds(rows, error_msg) ||
      !Reserve(&x_, std::size_t{rows} * w.hidden * sizeof(float), error_msg) ||
      !Reserve(&k_, rows * kvd * sizeof(float), error_msg) ||
      !Reserve(&v_, rows * kvd * sizeof(float), error_msg)) {
    return false;
  }
  float* x = static_cast<float*>(x_.ptr);
  float* k = static_cast<float*>(k_.ptr);
  float* v = static_cast<float*>(v_.ptr);
  const RopeArgs rope = ToArgs(w.rope);
  bool stored = false;  // a ring may hold part of the call's rows
  const auto fail = [&] {
    if (stored) {
      kv.valid_ = false;
      kv.committed_ = kv.floor_;
      kv.retained_begin_ = kv.floor_;
    }
    return false;
  };
  for (std::uint32_t il = 0; il < w.layers.size(); ++il) {
    const DrafterLayer& l = w.layers[il];
    const int layer = static_cast<int>(il);
    RmsNormKernel<<<rows, kNormThreads, 0, stream_>>>(fused, l.attn_norm, x,
                                                      w.hidden, 1, w.eps);
    if (!CheckHip(hipGetLastError(), "kv norm launch", error_msg) ||
        !Emit("kv_inp_normed", layer, x, std::size_t{rows} * w.hidden,
              error_msg) ||
        !Project(l.k, x, k, rows, vector, error_msg) ||
        !Emit("Kproj_injected", layer, k, rows * kvd, error_msg) ||
        !Project(l.v, x, v, rows, vector, error_msg) ||
        !Emit("Vcur_injected", layer, v, rows * kvd, error_msg)) {
      return fail();
    }
    stored = true;
    QkNormRopeStoreKernel<<<dim3(rows, 2 * w.kv_heads), kHeadDim, 0, stream_>>>(
        nullptr, k, v, nullptr, l.k_norm, static_cast<__half*>(kv.k_[il]),
        static_cast<__half*>(kv.v_[il]), 0, w.kv_heads, w.eps, start,
        kv.capacity_, rope);
    if (!CheckHip(hipGetLastError(), "kv store launch", error_msg) ||
        !Emit("kv_stored", layer, nullptr, 0, error_msg)) {
      return fail();
    }
  }
  // Writing [start, end) evicted every position below end - capacity; a start
  // below retained_begin keeps only the rows just written.
  const std::uint32_t end = start + rows;
  std::uint32_t retained = std::min(kv.retained_begin_, start);
  if (end > kv.capacity_) {
    retained = std::max(retained, end - kv.capacity_);
  }
  kv.retained_begin_ = retained;
  kv.committed_ = end;
  return true;
}

bool DFlashOps::CheckAttention(const DrafterWeights& w, std::uint32_t layer,
                               const DrafterKv& kv, std::uint32_t rows,
                               std::string* error_msg) const {
  if (!GeometryConsistent(w) || layer >= w.layers.size()) {
    Assign(error_msg, "laguna dflash: inconsistent drafter weights or layer");
    return false;
  }
  if (kv.k_.size() != w.layers.size() || kv.kv_heads_ != w.kv_heads ||
      kv.window_ != w.window) {
    Assign(error_msg, "laguna dflash: kv rings do not match the drafter");
    return false;
  }
  if (!kv.valid_) {
    Assign(error_msg,
           "laguna dflash: kv invalidated by a failed call; Reset() it");
    return false;
  }
  if (rows > kv.max_rows() ||
      std::uint64_t{kv.committed_} + rows > UINT32_MAX) {
    Assign(error_msg, "laguna dflash: block exceeds the ring");
    return false;
  }
  // The first context key row 0 reads must still be in its slot.
  if (std::string why = kv.RejectContinue(kv.committed_); !why.empty()) {
    Assign(error_msg, why);
    return false;
  }
  return true;
}

bool DFlashOps::Attention(const DrafterWeights& w, std::uint32_t layer,
                          DrafterKv& kv, const float* normed,
                          std::uint32_t rows, float* out, Path path,
                          std::string* error_msg) {
  bool vector = false;
  if (!Ready(path, rows, &vector, error_msg) ||
      !CheckAttention(w, layer, kv, rows, error_msg)) {
    return false;
  }
  if (normed == nullptr || out == nullptr) {
    Assign(error_msg, "laguna dflash: null attention input or output");
    return false;
  }
  return AttentionImpl(w, layer, kv, normed, rows, out, vector, error_msg);
}

bool DFlashOps::AttentionImpl(const DrafterWeights& w, std::uint32_t layer,
                              DrafterKv& kv, const float* normed,
                              std::uint32_t rows, float* out, bool vector,
                              std::string* error_msg) {
  const DrafterLayer& l = w.layers[layer];
  const int il = static_cast<int>(layer);
  const std::size_t qd = std::size_t{w.heads} * kHeadDim;
  const std::size_t kvd = std::size_t{w.kv_heads} * kHeadDim;
  if (!PrepareIds(rows, error_msg) ||
      !Reserve(&q_, rows * qd * sizeof(float), error_msg) ||
      !Reserve(&k_, rows * kvd * sizeof(float), error_msg) ||
      !Reserve(&v_, rows * kvd * sizeof(float), error_msg) ||
      !Reserve(&gate_, std::size_t{rows} * w.heads * sizeof(float),
               error_msg) ||
      !Reserve(&attn_, rows * qd * sizeof(float), error_msg)) {
    return false;
  }
  float* q = static_cast<float*>(q_.ptr);
  float* k = static_cast<float*>(k_.ptr);
  float* v = static_cast<float*>(v_.ptr);
  float* gate = static_cast<float*>(gate_.ptr);
  float* attn = static_cast<float*>(attn_.ptr);
  // Each stage is emitted right after its projection, so a tap (and the
  // activation capture of tests) sees each projection under its own name.
  if (!Project(l.q, normed, q, rows, vector, error_msg) ||
      !Emit("Qproj", il, q, rows * qd, error_msg) ||
      !Project(l.k, normed, k, rows, vector, error_msg) ||
      !Emit("Kproj", il, k, rows * kvd, error_msg) ||
      !Project(l.v, normed, v, rows, vector, error_msg) ||
      !Emit("Vcur", il, v, rows * kvd, error_msg) ||
      !Project(l.gate, normed, gate, rows, vector, error_msg) ||
      !Emit("attn_gate_proj", il, gate, std::size_t{rows} * w.heads,
            error_msg)) {
    return false;
  }
  // From here on the block's K/V may be in the ring.
  const auto invalidate = [&kv] {
    kv.valid_ = false;
    kv.committed_ = kv.floor_;
    kv.retained_begin_ = kv.floor_;
    return false;
  };
  const std::uint32_t n = kv.committed_;
  QkNormRopeStoreKernel<<<dim3(rows, w.heads + 2 * w.kv_heads), kHeadDim, 0,
                          stream_>>>(
      q, k, v, l.q_norm, l.k_norm, static_cast<__half*>(kv.k_[layer]),
      static_cast<__half*>(kv.v_[layer]), w.heads, w.kv_heads, w.eps, n,
      kv.capacity_, ToArgs(w.rope));
  if (!CheckHip(hipGetLastError(), "qk norm rope launch", error_msg) ||
      !Emit("Qcur", il, q, rows * qd, error_msg)) {
    return invalidate();
  }
  const AttentionArgs args{q,
                           static_cast<const __half*>(kv.k_[layer]),
                           static_cast<const __half*>(kv.v_[layer]),
                           attn,
                           w.heads,
                           w.kv_heads,
                           w.window,
                           kv.floor_,
                           n,
                           rows,
                           kv.capacity_,
                           w.causal_block,
                           1.0F / std::sqrt(static_cast<float>(kHeadDim))};
  if (!DispatchAttention(w.heads / w.kv_heads, args, stream_) ||
      !CheckHip(hipGetLastError(), "attention launch", error_msg) ||
      !Emit("attn_out", il, attn, rows * qd, error_msg)) {
    return invalidate();
  }
  GateKernel<<<std::size_t{rows} * w.heads, kHeadDim, 0, stream_>>>(gate, attn);
  if (!CheckHip(hipGetLastError(), "gate launch", error_msg) ||
      !Emit("attn_gate_softplus", il, gate, std::size_t{rows} * w.heads,
            error_msg) ||
      !Emit("attn_gated", il, attn, rows * qd, error_msg) ||
      !Project(l.o, attn, out, rows, vector, error_msg) ||
      !Emit("attn_o_proj", il, out, std::size_t{rows} * w.hidden, error_msg)) {
    return invalidate();
  }
  // The block rows evicted every position below n + rows - capacity.
  if (n + rows > kv.capacity_) {
    kv.retained_begin_ = std::max(kv.retained_begin_, n + rows - kv.capacity_);
  }
  return true;
}

bool DFlashOps::Ffn(const DrafterWeights& w, std::uint32_t layer,
                    const float* normed, float* out, std::uint32_t rows,
                    Path path, std::string* error_msg) {
  bool vector = false;
  if (!Ready(path, rows, &vector, error_msg)) {
    return false;
  }
  if (!GeometryConsistent(w) || layer >= w.layers.size()) {
    Assign(error_msg, "laguna dflash: inconsistent drafter weights or layer");
    return false;
  }
  if (normed == nullptr || out == nullptr) {
    Assign(error_msg, "laguna dflash: null ffn input or output");
    return false;
  }
  return FfnImpl(w, layer, normed, out, rows, vector, error_msg);
}

bool DFlashOps::FfnImpl(const DrafterWeights& w, std::uint32_t layer,
                        const float* normed, float* out, std::uint32_t rows,
                        bool vector, std::string* error_msg) {
  const DrafterLayer& l = w.layers[layer];
  const int il = static_cast<int>(layer);
  const std::size_t count = std::size_t{rows} * w.ffn;
  if (!PrepareIds(rows, error_msg) ||
      !Reserve(&fg_, count * sizeof(float), error_msg) ||
      !Reserve(&fu_, count * sizeof(float), error_msg) ||
      !Reserve(&fa_, count * sizeof(float), error_msg)) {
    return false;
  }
  float* gate = static_cast<float*>(fg_.ptr);
  float* up = static_cast<float*>(fu_.ptr);
  float* act = static_cast<float*>(fa_.ptr);
  if (!Project(l.ffn_gate, normed, gate, rows, vector, error_msg) ||
      !Emit("ffn_gate", il, gate, count, error_msg) ||
      !Project(l.ffn_up, normed, up, rows, vector, error_msg) ||
      !Emit("ffn_up", il, up, count, error_msg)) {
    return false;
  }
  SwiGluKernel<<<(count + 255) / 256, 256, 0, stream_>>>(gate, up, act, count);
  return CheckHip(hipGetLastError(), "swiglu launch", error_msg) &&
         Emit("ffn_swiglu", il, act, count, error_msg) &&
         Project(l.ffn_down, act, out, rows, vector, error_msg) &&
         Emit("ffn_out", il, out, std::size_t{rows} * w.hidden, error_msg);
}

bool DFlashOps::Head(const DrafterWeights& w, const DeviceMatrix& head,
                     const float* h, std::uint32_t rows, float* logits,
                     std::int32_t* drafts, Path path, std::string* error_msg) {
  bool vector = false;
  if (!Ready(path, rows, &vector, error_msg)) {
    return false;
  }
  if (rows < 2 || w.output_norm == nullptr || w.hidden == 0) {
    Assign(error_msg, "laguna dflash head: bad rows or weights");
    return false;
  }
  if (!ProjectionType(head.type) || head.experts != 1 || head.data == nullptr ||
      head.cols != w.hidden || head.rows == 0) {
    Assign(error_msg, "laguna dflash head: the target head does not match");
    return false;
  }
  if (h == nullptr || logits == nullptr || drafts == nullptr) {
    Assign(error_msg, "laguna dflash head: null input or output");
    return false;
  }
  return HeadImpl(w, head, h, rows, logits, drafts, vector, error_msg);
}

bool DFlashOps::HeadImpl(const DrafterWeights& w, const DeviceMatrix& head,
                         const float* h, std::uint32_t rows, float* logits,
                         std::int32_t* drafts, bool vector,
                         std::string* error_msg) {
  const std::size_t H = w.hidden;
  if (!PrepareIds(rows, error_msg) ||
      !Reserve(&n_, rows * H * sizeof(float), error_msg)) {
    return false;
  }
  float* normed = static_cast<float*>(n_.ptr);
  RmsNormKernel<<<rows, kNormThreads, 0, stream_>>>(h, w.output_norm, normed,
                                                    w.hidden, 1, w.eps);
  // Row 0 (the anchor) is normalized for the tap but not projected.
  if (!CheckHip(hipGetLastError(), "output norm launch", error_msg) ||
      !Emit("result_norm", -1, normed, rows * H, error_msg) ||
      !Project(head, normed + H, logits, rows - 1, vector, error_msg) ||
      !Emit("result_output", -1, logits, std::size_t{rows - 1} * head.rows,
            error_msg)) {
    return false;
  }
  ArgmaxKernel<<<rows - 1, kArgmaxThreads, 0, stream_>>>(logits, drafts,
                                                         head.rows);
  return CheckHip(hipGetLastError(), "argmax launch", error_msg);
}

bool DFlashOps::Block(const DrafterWeights& w, const DeviceMatrix& head,
                      DrafterKv& kv, const float* embeddings,
                      std::uint32_t rows, float* logits, std::int32_t* drafts,
                      Path path, std::string* error_msg) {
  bool vector = false;
  if (!Ready(path, rows, &vector, error_msg) ||
      !CheckAttention(w, 0, kv, rows, error_msg)) {
    return false;
  }
  if (rows < 2 || rows > w.block_size || w.output_norm == nullptr) {
    Assign(error_msg, "laguna dflash: a block has 2 to block_size rows");
    return false;
  }
  if (!ProjectionType(head.type) || head.experts != 1 || head.data == nullptr ||
      head.cols != w.hidden || head.rows == 0) {
    Assign(error_msg, "laguna dflash head: the target head does not match");
    return false;
  }
  if (embeddings == nullptr || logits == nullptr || drafts == nullptr) {
    Assign(error_msg, "laguna dflash: null block input or output");
    return false;
  }
  const std::size_t count = std::size_t{rows} * w.hidden;
  // Reserve the residual buffers before anything is written; the stage
  // helpers reserve their own scratch the same way.
  if (!Reserve(&h_, count * sizeof(float), error_msg) ||
      !Reserve(&r_, count * sizeof(float), error_msg) ||
      !Reserve(&o_, count * sizeof(float), error_msg) ||
      !Reserve(&n_, count * sizeof(float), error_msg)) {
    return false;
  }
  float* h = static_cast<float*>(h_.ptr);
  float* x = static_cast<float*>(r_.ptr);  // ffn_inp
  float* o = static_cast<float*>(o_.ptr);
  float* n = static_cast<float*>(n_.ptr);
  if (!CheckHip(hipMemcpyAsync(h, embeddings, count * sizeof(float),
                               hipMemcpyDeviceToDevice, stream_),
                "embedding copy", error_msg) ||
      !Emit("inp_noise_embd", -1, h, count, error_msg)) {
    return false;
  }
  const unsigned add_blocks = static_cast<unsigned>((count + 255) / 256);
  // After layer 0's attention the block's K/V are in the ring: a later
  // failure invalidates it.
  const auto invalidate = [&kv] {
    kv.valid_ = false;
    kv.committed_ = kv.floor_;
    kv.retained_begin_ = kv.floor_;
    return false;
  };
  for (std::uint32_t il = 0; il < w.layers.size(); ++il) {
    const DrafterLayer& l = w.layers[il];
    const int layer = static_cast<int>(il);
    RmsNormKernel<<<rows, kNormThreads, 0, stream_>>>(h, l.attn_norm, n,
                                                      w.hidden, 1, w.eps);
    if (!CheckHip(hipGetLastError(), "noise norm launch", error_msg) ||
        !Emit("noise_norm", layer, n, count, error_msg) ||
        !AttentionImpl(w, il, kv, n, rows, o, vector, error_msg)) {
      // AttentionImpl invalidates the ring itself once it has written; a
      // failure before layer 0's write leaves it untouched.
      return il == 0 ? false : invalidate();
    }
    AddKernel<<<add_blocks, 256, 0, stream_>>>(o, h, x, count);
    if (!CheckHip(hipGetLastError(), "residual launch", error_msg) ||
        !Emit("ffn_inp", layer, x, count, error_msg)) {
      return invalidate();
    }
    RmsNormKernel<<<rows, kNormThreads, 0, stream_>>>(x, l.ffn_norm, n,
                                                      w.hidden, 1, w.eps);
    if (!CheckHip(hipGetLastError(), "ffn norm launch", error_msg) ||
        !Emit("ffn_norm", layer, n, count, error_msg) ||
        !FfnImpl(w, il, n, o, rows, vector, error_msg)) {
      return invalidate();
    }
    AddKernel<<<add_blocks, 256, 0, stream_>>>(o, x, h, count);
    if (!CheckHip(hipGetLastError(), "residual launch", error_msg) ||
        !Emit("l_out", layer, h, count, error_msg)) {
      return invalidate();
    }
  }
  return HeadImpl(w, head, h, rows, logits, drafts, vector, error_msg) ||
         invalidate();
}

// ---- DrafterSession --------------------------------------------------------

DrafterSession::DrafterSession(const DrafterWeights& w,
                               const DeviceMatrix& head,
                               const TensorRef& token_embd,
                               std::int32_t mask_token, hipStream_t stream)
    : w_(w),
      head_(head),
      token_embd_(token_embd),
      mask_token_(mask_token),
      stream_(stream),
      ops_(stream) {}

DrafterSession::~DrafterSession() {
  (void)hipStreamSynchronize(stream_);
  for (void* p :
       {static_cast<void*>(features_), static_cast<void*>(embeddings_),
        static_cast<void*>(logits_), static_cast<void*>(drafts_)}) {
    if (p != nullptr) {
      (void)hipFree(p);
    }
  }
}

std::unique_ptr<DrafterSession> DrafterSession::Create(
    const DrafterWeights& w, const DeviceMatrix& head,
    const TensorRef& token_embd, std::int32_t mask_token,
    std::uint32_t max_inject_rows, hipStream_t stream, std::string* error_msg) {
  if (!GeometryConsistent(w) || w.block_size < 2 || max_inject_rows == 0) {
    Assign(error_msg, "laguna dflash session: inconsistent drafter");
    return nullptr;
  }
  if (!ProjectionType(head.type) || head.data == nullptr ||
      head.cols != w.hidden || head.experts != 1 ||
      token_embd.cols != w.hidden || token_embd.rows != head.rows ||
      token_embd.data == nullptr || mask_token < 0 ||
      static_cast<std::uint64_t>(mask_token) >= token_embd.rows) {
    Assign(error_msg,
           "laguna dflash session: target head or embedding does not match");
    return nullptr;
  }
  std::unique_ptr<DrafterSession> s(
      new DrafterSession(w, head, token_embd, mask_token, stream));
  auto kv = DrafterKv::Create(w, max_inject_rows, error_msg);
  if (!kv.has_value()) {
    return nullptr;
  }
  s->kv_ = std::move(*kv);
  s->max_inject_rows_ = max_inject_rows;
  const std::size_t H = w.hidden;
  const std::size_t B = w.block_size;
  // A failure leaves `s` owning what was allocated; its destructor frees it.
  if (!CheckHip(hipMalloc(&s->features_, std::size_t{max_inject_rows} * w.taps *
                                             H * sizeof(float)),
                "hipMalloc features", error_msg) ||
      !CheckHip(hipMalloc(&s->embeddings_, B * H * sizeof(float)),
                "hipMalloc embeddings", error_msg) ||
      !CheckHip(hipMalloc(&s->logits_, (B - 1) * head.rows * sizeof(float)),
                "hipMalloc logits", error_msg) ||
      !CheckHip(hipMalloc(&s->drafts_, (B - 1) * sizeof(std::int32_t)),
                "hipMalloc drafts", error_msg)) {
    return nullptr;
  }
  return s;
}

bool DrafterSession::Inject(std::span<const float> features,
                            std::uint32_t start, Path path,
                            std::string* error_msg) {
  const std::size_t width = std::size_t{w_.taps} * w_.hidden;
  if (features.empty() || features.size() % width != 0 ||
      features.size() / width > max_inject_rows_) {
    Assign(error_msg, "laguna dflash session: bad feature rows");
    return false;
  }
  const auto rows = static_cast<std::uint32_t>(features.size() / width);
  // The copy is ordered after earlier work on the stream (which may still
  // read the buffer); waiting for it releases the caller's memory.
  if (!CheckHip(hipMemcpyAsync(features_, features.data(),
                               features.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream_),
                "feature upload", error_msg) ||
      !CheckHip(hipStreamSynchronize(stream_), "feature upload sync",
                error_msg)) {
    return false;
  }
  return ops_.Inject(w_, kv_, features_, rows, start, path, error_msg);
}

bool DrafterSession::InjectDevice(const float* features, std::uint32_t rows,
                                  std::uint32_t start, Path path,
                                  std::string* error_msg) {
  if (features == nullptr || rows == 0 || rows > max_inject_rows_) {
    Assign(error_msg, "laguna dflash session: bad feature rows");
    return false;
  }
  return ops_.Inject(w_, kv_, features, rows, start, path, error_msg);
}

bool DrafterSession::Propose(std::int32_t anchor, std::uint32_t block_rows,
                             std::vector<std::int32_t>* drafts,
                             std::vector<float>* logits, Path path,
                             std::string* error_msg) {
  if (drafts == nullptr || block_rows < 2 || block_rows > w_.block_size ||
      anchor < 0 || static_cast<std::uint64_t>(anchor) >= token_embd_.rows) {
    Assign(error_msg, "laguna dflash session: bad anchor or block length");
    return false;
  }
  const std::size_t H = w_.hidden;
  host_embed_.resize(block_rows * H);
  for (std::uint32_t r = 0; r < block_rows; ++r) {
    const std::int32_t token = r == 0 ? anchor : mask_token_;
    reference::DequantizeRow(
        token_embd_.type,
        static_cast<const std::uint8_t*>(token_embd_.data) +
            static_cast<std::size_t>(token) * token_embd_.RowBytes(),
        host_embed_.data() + r * H, H);
  }
  const std::size_t rows = block_rows - 1;
  drafts->resize(rows);
  if (logits != nullptr) {
    logits->resize(rows * head_.rows);
  }
  const bool ok = CheckHip(hipMemcpyAsync(embeddings_, host_embed_.data(),
                                          host_embed_.size() * sizeof(float),
                                          hipMemcpyHostToDevice, stream_),
                           "embedding upload", error_msg) &&
                  ops_.Block(w_, head_, kv_, embeddings_, block_rows, logits_,
                             drafts_, path, error_msg) &&
                  CheckHip(hipMemcpyAsync(drafts->data(), drafts_,
                                          rows * sizeof(std::int32_t),
                                          hipMemcpyDeviceToHost, stream_),
                           "draft download", error_msg) &&
                  (logits == nullptr ||
                   CheckHip(hipMemcpyAsync(logits->data(), logits_,
                                           logits->size() * sizeof(float),
                                           hipMemcpyDeviceToHost, stream_),
                            "logit download", error_msg));
  // Wait in every case: the host buffers must outlive the copies.
  const bool synced =
      CheckHip(hipStreamSynchronize(stream_), "propose sync", error_msg);
  return ok && synced;
}

}  // namespace gufo::models::laguna::rocm
