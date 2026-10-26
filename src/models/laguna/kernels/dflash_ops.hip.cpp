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

// ---- DFlashOps --------------------------------------------------------------

DFlashOps::DFlashOps(hipStream_t stream)
    : stream_(stream), allocator_{DefaultAllocate, DefaultRelease} {
  int device = 0;
  initialized_ =
      hipGetDevice(&device) == hipSuccess && qfn_mmq_init(device) == 0;
}

DFlashOps::~DFlashOps() {
  (void)hipStreamSynchronize(stream_);
  for (Scratch* s : {&ids_, &aux_, &u_}) {
    Free(s);
  }
}

std::size_t DFlashOps::scratch_bytes() const noexcept {
  std::size_t total = 0;
  for (const Scratch* s : {&ids_, &aux_, &u_}) {
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

bool DFlashOps::Encode(const DrafterWeights& w, const float* features,
                       float* fused, std::uint32_t rows, Path path,
                       std::string* error_msg) {
  bool vector = false;
  if (!Ready(path, rows, &vector, error_msg)) {
    return false;
  }
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
  const std::size_t ids_bytes = std::size_t{rows} * sizeof(std::int32_t);
  if (!Reserve(&ids_, ids_bytes, error_msg) ||
      !Reserve(&aux_, rows * width * sizeof(float), error_msg) ||
      !Reserve(&u_, std::size_t{rows} * w.hidden * sizeof(float), error_msg)) {
    return false;
  }
  if (!CheckHip(hipMemsetAsync(ids_.ptr, 0, ids_bytes, stream_),
                "hipMemsetAsync", error_msg)) {
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

}  // namespace gufo::models::laguna::rocm
