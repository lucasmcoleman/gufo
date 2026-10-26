// Laguna-S-2.1 GPU FFN operators. See ffn.hpp.
//
// Router and combine kernels are written here, with plain IEEE arithmetic
// (no fast-math) and no FP contraction, so their rounding matches the CPU
// oracle's. The projections reuse the qfn_mmq kernels vendored with
// qwen38_flash_next and add no dot products of their own.
#pragma clang fp contract(off)

#include "src/models/laguna/kernels/ffn.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <utility>

#include "src/core/hip/weight_upload.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::laguna::rocm {
namespace {

using core::GgmlType;

constexpr int kMaxExperts = 1024;
constexpr int kMaxUsed = 32;
constexpr int kRouterThreads = 256;
constexpr int kLogitThreads = 128;
constexpr std::uint32_t kVectorMaxRows = 32;  // qfn_mmq_q8_0_dense_vec_preq
constexpr float kRouterSumClamp = 6.103515625e-5F;  // 2^-14

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

/// logits[t][e] = dot(w[e], x[t]). One block per (expert, token); the sum
/// order is fixed (strided lanes, then a fixed shuffle/shared reduction).
__global__ void RouterLogitsKernel(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   float* __restrict__ logits,
                                   std::uint32_t hidden,
                                   std::uint32_t experts) {
  const float* xr = x + std::size_t{blockIdx.y} * hidden;
  const float* wr = w + std::size_t{blockIdx.x} * hidden;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    sum += wr[i] * xr[i];
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
  }
  __shared__ float part[kLogitThreads / 32];
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
    logits[std::size_t{blockIdx.y} * experts + blockIdx.x] = total;
  }
}

__device__ bool Better(float v, int i, float bv, int bi) {
  return v > bv || (v == bv && i < bi);
}

/// Sigmoid routing for one token per block. s = sigmoid(logit); selection
/// uses s + bias (bias selection-only); a tie picks the lower expert id; the
/// weights use the unbiased s, normalized by max(sum, 2^-14) when `normalize`,
/// then scaled. A non-finite biased score ranks last.
__global__ void RouterTopKKernel(const float* __restrict__ logits,
                                 const float* __restrict__ bias,
                                 std::uint32_t experts, std::uint32_t used,
                                 bool normalize, float scale,
                                 std::int32_t* __restrict__ ids,
                                 float* __restrict__ weights) {
  __shared__ float choice[kMaxExperts];
  __shared__ float prob[kMaxExperts];
  __shared__ bool taken[kMaxExperts];
  __shared__ float red_v[kRouterThreads];
  __shared__ int red_i[kRouterThreads];
  __shared__ int sel[kMaxUsed];
  const std::size_t token = blockIdx.x;
  const int tid = threadIdx.x;
  for (std::uint32_t e = tid; e < experts; e += blockDim.x) {
    const float s = 1.0F / (1.0F + expf(-logits[token * experts + e]));
    const float c = s + (bias != nullptr ? bias[e] : 0.0F);
    prob[e] = s;
    choice[e] = isfinite(c) ? c : -INFINITY;
    taken[e] = false;
  }
  __syncthreads();
  for (std::uint32_t k = 0; k < used; ++k) {
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (std::uint32_t e = tid; e < experts; e += blockDim.x) {
      if (!taken[e] && Better(choice[e], static_cast<int>(e), bv, bi)) {
        bv = choice[e];
        bi = static_cast<int>(e);
      }
    }
    red_v[tid] = bv;
    red_i[tid] = bi;
    __syncthreads();
    for (int stride = kRouterThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride && Better(red_v[tid + stride], red_i[tid + stride],
                                 red_v[tid], red_i[tid])) {
        red_v[tid] = red_v[tid + stride];
        red_i[tid] = red_i[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      sel[k] = red_i[0];
      taken[red_i[0]] = true;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float sum = 0.0F;
    for (std::uint32_t k = 0; k < used; ++k) {
      sum += prob[sel[k]];
    }
    const float denominator = normalize ? fmaxf(sum, kRouterSumClamp) : 1.0F;
    for (std::uint32_t k = 0; k < used; ++k) {
      float w = prob[sel[k]];
      if (normalize) {
        w /= denominator;
      }
      if (scale != 0.0F && scale != 1.0F) {
        w *= scale;
      }
      ids[token * used + k] = sel[k];
      weights[token * used + k] = w;
    }
  }
}

/// act = silu(gate) * up, silu(g) = g / (1 + exp(-g)).
__global__ void SwiGluKernel(const float* __restrict__ gate,
                             const float* __restrict__ up,
                             float* __restrict__ act, std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count) {
    const float g = gate[i];
    act[i] = (g / (1.0F + expf(-g))) * up[i];
  }
}

/// out[t][d] = sum_s (w[t][s] * down[t*used + s][d]) + shared[t][d], summed
/// over s left to right in float, then the shared expert added.
__global__ void CombineKernel(const float* __restrict__ down,
                              const float* __restrict__ weights,
                              const float* __restrict__ shared,
                              float* __restrict__ out, std::uint32_t used,
                              std::uint32_t hidden) {
  const std::uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t t = blockIdx.y;
  if (d >= hidden) {
    return;
  }
  const float* w = weights + t * used;
  const float* rows = down + t * used * hidden;
  float acc = w[0] * rows[d];
  for (std::uint32_t s = 1; s < used; ++s) {
    const float weighted = w[s] * rows[std::size_t{s} * hidden + d];
    acc += weighted;
  }
  out[t * hidden + d] = acc + shared[t * hidden + d];
}

// ---- Routed GEMM dispatch ---------------------------------------------------

using RawMoe = int (*)(const void*, const float*, const std::int32_t*, float*,
                       int, int, int, int, int, hipStream_t);

RawMoe BatchedRoutedKernel(GgmlType type) {
  switch (type) {
    case GgmlType::kQ6_K:
      return qfn_mmq_q6_K_moe_raw;
    case GgmlType::kQ5_K:
      return qfn_mmq_q5_K_moe_raw;
    case GgmlType::kQ4_K:
      return qfn_mmq_q4_K_moe_raw;
    case GgmlType::kIQ4_XS:
      return qfn_mmq_iq4_xs_moe_raw;
    case GgmlType::kQ8_0:
      return qfn_mmq_q8_0_moe_raw;
    default:
      return nullptr;
  }
}

bool SameShape(const DeviceMatrix& a, const DeviceMatrix& b) {
  return a.type == b.type && a.cols == b.cols && a.rows == b.rows &&
         a.experts == b.experts;
}

}  // namespace

// ---- LayerFfn
// ----------------------------------------------------------------

LayerFfn::LayerFfn(LayerFfn&& other) noexcept
    : moe_(other.moe_),
      dense_(other.dense_),
      routed_(other.routed_),
      allocations_(std::move(other.allocations_)),
      bytes_(other.bytes_) {
  other.allocations_.clear();
  other.bytes_ = 0;
}

LayerFfn& LayerFfn::operator=(LayerFfn&& other) noexcept {
  if (this != &other) {
    Release();
    moe_ = other.moe_;
    dense_ = other.dense_;
    routed_ = other.routed_;
    allocations_ = std::move(other.allocations_);
    bytes_ = other.bytes_;
    other.allocations_.clear();
    other.bytes_ = 0;
  }
  return *this;
}

LayerFfn::~LayerFfn() {
  Release();
}

void LayerFfn::Release() noexcept {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  allocations_.clear();
  bytes_ = 0;
  dense_ = {};
  routed_ = {};
}

namespace {

/// Copies `t` to the device (appending the allocation to `owned`) and
/// returns its descriptor. Quantized matrices get 512 zeroed columns' worth
/// of bytes after the last row, as ggml's MATRIX_ROW_PADDING allocation
/// does: the MMQ tile loaders read whole 512-column tiles, so a row length
/// that is not a multiple of 512 (the shared expert's 320) would otherwise
/// read past the allocation in the last row, and a non-finite value there
/// poisons that output (then sanitized to 0).
bool UploadTensor(const TensorRef& t, std::vector<void*>* owned,
                  std::size_t* bytes, DeviceMatrix* out,
                  hip::WeightUpload* stager, std::string* error_msg) {
  if (t.empty() || t.data == nullptr) {
    Assign(error_msg, "laguna ffn upload: missing tensor");
    return false;
  }
  constexpr std::size_t kRowPadding = 512;  // ggml MATRIX_ROW_PADDING
  const std::size_t size = t.SizeBytes();
  const std::size_t tail =
      t.type == GgmlType::kF32 ? 0 : t.RowBytes() * kRowPadding / t.cols;
  void* device = nullptr;
  if (!CheckHip(hipMalloc(&device, size + tail), "hipMalloc", error_msg)) {
    return false;
  }
  owned->push_back(device);
  *bytes += size + tail;
  const bool copied =
      stager != nullptr
          ? stager->Copy(t.shard, t.file_offset, size, device, error_msg)
          : CheckHip(hipMemcpy(device, t.data, size, hipMemcpyHostToDevice),
                     "hipMemcpy", error_msg);
  if (!copied ||
      (tail != 0 &&
       !CheckHip(hipMemset(static_cast<char*>(device) + size, 0, tail),
                 "hipMemset", error_msg))) {
    return false;
  }
  out->data = device;
  out->type = t.type;
  out->cols = static_cast<std::uint32_t>(t.cols);
  out->rows = static_cast<std::uint32_t>(t.rows);
  out->experts = static_cast<std::uint32_t>(t.experts);
  return true;
}

bool Expect(const TensorRef& t, GgmlType type, std::uint64_t cols,
            std::uint64_t rows, std::uint64_t experts, const char* what,
            std::string* error_msg) {
  if (t.type != type || t.cols != cols || t.rows != rows ||
      t.experts != experts) {
    Assign(error_msg, std::string("laguna ffn upload: ") + what +
                          " has an unexpected type or shape");
    return false;
  }
  return true;
}

}  // namespace

std::optional<LayerFfn> LayerFfn::Upload(const Config& config,
                                         const LayerWeights& layer,
                                         std::string* error_msg,
                                         hip::WeightUpload* stager) {
  LayerFfn result;
  result.moe_ = layer.moe;
  const std::uint64_t hidden = config.hidden_size;
  bool ok = true;
  const auto up = [&](const TensorRef& t, DeviceMatrix* m) {
    ok = ok && UploadTensor(t, &result.allocations_, &result.bytes_, m, stager,
                            error_msg);
  };
  DenseFfnWeights* dense = &result.dense_;
  if (!layer.moe) {
    const std::uint64_t ff = config.ffn_size;
    ok = Expect(layer.ffn_gate, GgmlType::kQ8_0, hidden, ff, 1, "ffn_gate",
                error_msg) &&
         Expect(layer.ffn_up, GgmlType::kQ8_0, hidden, ff, 1, "ffn_up",
                error_msg) &&
         Expect(layer.ffn_down, GgmlType::kQ8_0, ff, hidden, 1, "ffn_down",
                error_msg);
    if (ok) {
      up(layer.ffn_gate, &dense->gate);
      up(layer.ffn_up, &dense->up);
      up(layer.ffn_down, &dense->down);
    }
  } else {
    const std::uint64_t e = config.num_experts;
    const std::uint64_t ff = config.expert_ff;
    const std::uint64_t sff = config.shared_expert_ff;
    ok = Expect(layer.ffn_gate_inp, GgmlType::kF32, hidden, e, 1,
                "ffn_gate_inp", error_msg) &&
         Expect(layer.ffn_exp_probs_b, GgmlType::kF32, e, 1, 1, "exp_probs_b",
                error_msg) &&
         Expect(layer.ffn_gate_exps, layer.ffn_gate_exps.type, hidden, ff, e,
                "ffn_gate_exps", error_msg) &&
         Expect(layer.ffn_up_exps, layer.ffn_gate_exps.type, hidden, ff, e,
                "ffn_up_exps", error_msg) &&
         Expect(layer.ffn_down_exps, layer.ffn_gate_exps.type, ff, hidden, e,
                "ffn_down_exps", error_msg) &&
         Expect(layer.ffn_gate_shexp, GgmlType::kQ8_0, hidden, sff, 1,
                "ffn_gate_shexp", error_msg) &&
         Expect(layer.ffn_up_shexp, GgmlType::kQ8_0, hidden, sff, 1,
                "ffn_up_shexp", error_msg) &&
         Expect(layer.ffn_down_shexp, GgmlType::kQ8_0, sff, hidden, 1,
                "ffn_down_shexp", error_msg);
    if (ok) {
      MoeFfnWeights& r = result.routed_;
      DeviceMatrix router;
      DeviceMatrix bias;
      up(layer.ffn_gate_inp, &router);
      up(layer.ffn_exp_probs_b, &bias);
      up(layer.ffn_gate_exps, &r.gate_exps);
      up(layer.ffn_up_exps, &r.up_exps);
      up(layer.ffn_down_exps, &r.down_exps);
      up(layer.ffn_gate_shexp, &r.shared.gate);
      up(layer.ffn_up_shexp, &r.shared.up);
      up(layer.ffn_down_shexp, &r.shared.down);
      r.router = static_cast<const float*>(router.data);
      r.bias = static_cast<const float*>(bias.data);
      r.experts = config.num_experts;
      r.used = config.num_experts_used;
      r.normalize = config.expert_weights_norm;
      r.scale = config.expert_weights_scale;
    }
  }
  // Queued copies write into `result`'s allocations: wait for them before the
  // result is returned or (on failure) freed.
  if (stager != nullptr) {
    std::string finish_error;
    if (!stager->Finish(&finish_error) && ok) {
      ok = false;
      Assign(error_msg, finish_error);
    }
  }
  if (!ok) {
    return std::nullopt;
  }
  return result;
}

bool LayerFfn::ReplaceRouter(const std::vector<float>& values,
                             std::string* error_msg) {
  if (!moe_ || routed_.router == nullptr ||
      values.size() != std::size_t{routed_.experts} * routed_.gate_exps.cols) {
    Assign(error_msg, "ReplaceRouter: not a MoE layer or wrong size");
    return false;
  }
  return CheckHip(
      hipMemcpy(const_cast<float*>(routed_.router), values.data(),
                values.size() * sizeof(float), hipMemcpyHostToDevice),
      "hipMemcpy", error_msg);
}

// ---- FfnOps
// ------------------------------------------------------------------

namespace {
hipError_t DefaultAlloc(void** ptr, std::size_t bytes) {
  return hipMalloc(ptr, bytes);
}
hipError_t DefaultFree(void* ptr) {
  return hipFree(ptr);
}
}  // namespace

FfnOps::FfnOps(hipStream_t stream)
    : stream_(stream), alloc_(DefaultAlloc), free_(DefaultFree) {
  int device = 0;
  initialized_ =
      hipGetDevice(&device) == hipSuccess && qfn_mmq_init(device) == 0;
}

FfnOps::~FfnOps() {
  (void)hipStreamSynchronize(stream_);
  for (void* p :
       {static_cast<void*>(logits_), static_cast<void*>(ids_),
        static_cast<void*>(weights_), static_cast<void*>(gate_),
        static_cast<void*>(up_), static_cast<void*>(act_),
        static_cast<void*>(down_), static_cast<void*>(shared_), q8_}) {
    if (p != nullptr) {
      (void)free_(p);
    }
  }
}

std::size_t FfnOps::scratch_bytes() const noexcept {
  return logits_cap_ + ids_cap_ + weights_cap_ + gate_cap_ + up_cap_ +
         act_cap_ + down_cap_ + shared_cap_ + q8_cap_;
}

void FfnOps::SetAllocatorForTesting(DeviceAlloc alloc, DeviceFree release) {
  alloc_ = alloc != nullptr ? alloc : DefaultAlloc;
  free_ = release != nullptr ? release : DefaultFree;
}

template<class T>
bool FfnOps::Reserve(T** ptr, std::size_t* capacity, std::size_t bytes,
                     std::string* error_msg) {
  if (bytes <= *capacity) {
    return true;
  }
  // Allocate the replacement first: on failure the owner keeps its old,
  // still valid buffer and capacity. Growth waits for the stream before
  // freeing the old buffer (nothing else uses this scratch). Preallocate to
  // the largest call to keep pointers stable.
  void* fresh = nullptr;
  if (!CheckHip(alloc_(&fresh, bytes), "hipMalloc scratch", error_msg)) {
    return false;
  }
  if (!CheckHip(hipStreamSynchronize(stream_), "hipStreamSynchronize",
                error_msg)) {
    (void)free_(fresh);
    return false;
  }
  if (*ptr != nullptr) {
    (void)free_(static_cast<void*>(*ptr));
  }
  *ptr = static_cast<T*>(fresh);
  *capacity = bytes;
  return true;
}

bool FfnOps::Emit(const char* prefix, const char* name, const void* device,
                  std::size_t count, bool is_int32, std::string* error_msg) {
  if (!tap_) {
    return true;
  }
  if (!CheckHip(hipStreamSynchronize(stream_), "tap synchronize", error_msg)) {
    return false;
  }
  const std::string full =
      prefix[0] != '\0' ? std::string(prefix) + "_" + name : std::string(name);
  tap_(full, device, count, is_int32);
  return true;
}

bool FfnOps::Route(const MoeFfnWeights& w, const float* x, std::uint32_t hidden,
                   std::uint32_t tokens, std::string* error_msg) {
  if (!initialized_) {
    Assign(error_msg, "qfn_mmq_init failed");
    return false;
  }
  if (w.router == nullptr || tokens == 0 || hidden == 0 || w.experts == 0 ||
      w.experts > kMaxExperts || w.used == 0 || w.used > w.experts ||
      w.used > kMaxUsed) {
    Assign(error_msg, "laguna router: bad shape");
    return false;
  }
  const std::size_t e = w.experts;
  if (!Reserve(&logits_, &logits_cap_, tokens * e * sizeof(float), error_msg) ||
      !Reserve(&ids_, &ids_cap_, std::size_t{tokens} * w.used * sizeof(int),
               error_msg) ||
      !Reserve(&weights_, &weights_cap_,
               std::size_t{tokens} * w.used * sizeof(float), error_msg)) {
    return false;
  }
  RouterLogitsKernel<<<dim3(w.experts, tokens), kLogitThreads, 0, stream_>>>(
      x, w.router, logits_, hidden, w.experts);
  RouterTopKKernel<<<tokens, kRouterThreads, 0, stream_>>>(
      logits_, w.bias, w.experts, w.used, w.normalize, w.scale, ids_, weights_);
  return CheckHip(hipGetLastError(), "router launch", error_msg) &&
         Emit("", "router_logits", logits_, tokens * e, false, error_msg) &&
         Emit("", "topk", ids_, std::size_t{tokens} * w.used, true,
              error_msg) &&
         Emit("", "weights", weights_, std::size_t{tokens} * w.used, false,
              error_msg);
}

bool FfnOps::ProjectDense(const DeviceMatrix& w, const float* x, float* out,
                          std::uint32_t tokens, bool vector,
                          std::string* error_msg) {
  if (w.type != GgmlType::kQ8_0 || w.experts != 1) {
    Assign(error_msg, "laguna dense projection needs a Q8_0 matrix");
    return false;
  }
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int N = static_cast<int>(tokens);
  int rc = 0;
  if (vector) {
    if (!Reserve(&q8_, &q8_cap_, qfn_mmq_q8_1_bytes(N, K), error_msg)) {
      return false;
    }
    rc = qfn_mmq_quantize_q8_1(x, q8_, N, K, stream_);
    if (rc == 0) {
      rc = qfn_mmq_q8_0_dense_vec_preq(w.data, nullptr, q8_, out, M, N, K,
                                       stream_);
    }
  } else {
    rc = qfn_mmq_q8_0_dense(w.data, x, out, M, N, K, stream_);
  }
  if (rc != 0) {
    Assign(error_msg, "Q8_0 projection failed");
    return false;
  }
  return true;
}

bool FfnOps::DenseImpl(const DenseFfnWeights& w, const float* x, float* out,
                       std::uint32_t tokens, bool vector, const char* prefix,
                       std::string* error_msg) {
  const std::size_t ff = w.gate.rows;
  if (w.up.rows != ff || w.up.cols != w.gate.cols || w.down.cols != ff ||
      w.down.rows != w.gate.cols) {
    Assign(error_msg, "laguna dense ffn: inconsistent shapes");
    return false;
  }
  const std::size_t bytes = tokens * ff * sizeof(float);
  if (!Reserve(&gate_, &gate_cap_, bytes, error_msg) ||
      !Reserve(&up_, &up_cap_, bytes, error_msg) ||
      !Reserve(&act_, &act_cap_, bytes, error_msg)) {
    return false;
  }
  const std::size_t count = tokens * ff;
  if (!ProjectDense(w.gate, x, gate_, tokens, vector, error_msg) ||
      !ProjectDense(w.up, x, up_, tokens, vector, error_msg) ||
      !Emit(prefix, "gate", gate_, count, false, error_msg) ||
      !Emit(prefix, "up", up_, count, false, error_msg)) {
    return false;
  }
  SwiGluKernel<<<(count + 255) / 256, 256, 0, stream_>>>(gate_, up_, act_,
                                                         count);
  return CheckHip(hipGetLastError(), "swiglu launch", error_msg) &&
         Emit(prefix, "act", act_, count, false, error_msg) &&
         ProjectDense(w.down, act_, out, tokens, vector, error_msg) &&
         Emit(prefix, "out", out, std::size_t{tokens} * w.down.rows, false,
              error_msg);
}

bool FfnOps::Dense(const DenseFfnWeights& w, const float* x, float* out,
                   std::uint32_t tokens, Path path, std::string* error_msg) {
  if (!initialized_) {
    Assign(error_msg, "qfn_mmq_init failed");
    return false;
  }
  const bool vector = path == Path::kVector ||
                      (path == Path::kAuto && tokens <= kVectorMaxTokens);
  if (tokens == 0 || (vector && tokens > kVectorMaxRows)) {
    Assign(error_msg, "laguna dense ffn: unsupported row count for path");
    return false;
  }
  return DenseImpl(w, x, out, tokens, vector, "dense", error_msg);
}

bool FfnOps::Experts(const DeviceMatrix& w, const float* x,
                     const std::int32_t* ids, float* out, std::uint32_t rows,
                     std::uint32_t used, bool vector, const DeviceMatrix* pair,
                     float* pair_out, std::string* error_msg) {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(rows);
  const int U = static_cast<int>(used);
  int rc = -1;
  if (vector) {
    rc = qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x, ids, out, M, K, T,
                         E, U, stream_, pair != nullptr ? pair->data : nullptr,
                         pair_out);
  } else if (const RawMoe raw = BatchedRoutedKernel(w.type); raw != nullptr) {
    rc = raw(w.data, x, ids, out, M, K, T, E, U, stream_);
    if (rc == 0 && pair != nullptr) {
      rc = raw(pair->data, x, ids, pair_out, M, K, T, E, U, stream_);
    }
  }
  if (rc != 0) {
    Assign(error_msg, std::string("routed expert projection failed for ") +
                          std::string(core::ToString(w.type)));
    return false;
  }
  return true;
}

bool FfnOps::Moe(const MoeFfnWeights& w, const float* x, float* out,
                 std::uint32_t hidden, std::uint32_t tokens, Path path,
                 std::string* error_msg) {
  if (!initialized_) {
    Assign(error_msg, "qfn_mmq_init failed");
    return false;
  }
  const bool vector = path == Path::kVector ||
                      (path == Path::kAuto && tokens <= kVectorMaxTokens);
  if (tokens == 0 || (vector && tokens > kVectorMaxRows)) {
    Assign(error_msg, "laguna moe ffn: unsupported row count for path");
    return false;
  }
  const std::uint32_t ff = w.gate_exps.rows;
  if (!SameShape(w.gate_exps, w.up_exps) || w.gate_exps.cols != hidden ||
      w.down_exps.type != w.gate_exps.type || w.down_exps.cols != ff ||
      w.down_exps.rows != hidden || w.gate_exps.experts != w.experts ||
      w.down_exps.experts != w.experts || w.shared.gate.cols != hidden ||
      w.shared.down.rows != hidden) {
    Assign(error_msg, "laguna moe ffn: inconsistent shapes");
    return false;
  }
  if (!Route(w, x, hidden, tokens, error_msg)) {
    return false;
  }
  const std::size_t slots = std::size_t{tokens} * w.used;
  const auto reserve = [&](float** buffer, std::size_t* capacity,
                           std::size_t floats) {
    return Reserve(buffer, capacity, floats * sizeof(float), error_msg);
  };
  if (!reserve(&gate_, &gate_cap_, slots * ff) ||
      !reserve(&up_, &up_cap_, slots * ff) ||
      !reserve(&act_, &act_cap_, slots * ff) ||
      !reserve(&down_, &down_cap_, slots * hidden) ||
      !reserve(&shared_, &shared_cap_, std::size_t{tokens} * hidden)) {
    return false;
  }
  // Gate and up share one set of activation rows; down sees one row per
  // (token, slot) and one expert each (ids_ is already that flat list).
  if (!Experts(w.gate_exps, x, ids_, gate_, tokens, w.used, vector, &w.up_exps,
               up_, error_msg) ||
      !Emit("", "gate", gate_, slots * ff, false, error_msg) ||
      !Emit("", "up", up_, slots * ff, false, error_msg)) {
    return false;
  }
  SwiGluKernel<<<(slots * ff + 255) / 256, 256, 0, stream_>>>(gate_, up_, act_,
                                                              slots * ff);
  if (!CheckHip(hipGetLastError(), "swiglu launch", error_msg) ||
      !Emit("", "act", act_, slots * ff, false, error_msg) ||
      !Experts(w.down_exps, act_, ids_, down_,
               static_cast<std::uint32_t>(slots), 1, vector, nullptr, nullptr,
               error_msg) ||
      !Emit("", "down", down_, slots * hidden, false, error_msg)) {
    return false;
  }
  // The shared expert reuses gate_/up_/act_ after the routed work above, in
  // stream order.
  if (!DenseImpl(w.shared, x, shared_, tokens, vector, "shared", error_msg)) {
    return false;
  }
  CombineKernel<<<dim3((hidden + 255) / 256, tokens), 256, 0, stream_>>>(
      down_, weights_, shared_, out, w.used, hidden);
  return CheckHip(hipGetLastError(), "combine launch", error_msg);
}

}  // namespace gufo::models::laguna::rocm
