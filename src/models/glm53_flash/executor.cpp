#include "src/models/glm53_flash/executor.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <utility>

#include "src/models/glm53_flash/kernels/kda.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::models::glm53_flash {
namespace {

using core::GgmlType;

// q/k l2norm eps inside the KDA kernel call (HF:458), as in reference.cpp.
constexpr float kKdaL2Eps = 1e-6F;
constexpr std::size_t kAlign = 256;

constexpr std::size_t AlignUp(std::size_t n) noexcept {
  return (n + kAlign - 1) / kAlign * kAlign;
}

bool Fail(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return false;
}

std::string ShapeOf(const TensorRef& t) {
  return std::string(t.name) + " [" + std::to_string(t.cols) + " x " +
         std::to_string(t.rows) + " x " + std::to_string(t.experts) + "]";
}

bool CheckMatrix(const TensorRef& t, const char* what, std::uint64_t cols,
                 std::uint64_t rows, std::string* error) {
  if (t.empty()) {
    return Fail(error, std::string("missing tensor ") + what);
  }
  if (t.cols != cols || t.rows != rows || t.experts != 1) {
    return Fail(error, "tensor " + ShapeOf(t) + " expected [" +
                           std::to_string(cols) + " x " + std::to_string(rows) +
                           "]");
  }
  return true;
}

/// Copies the F32 tensors `parts` (each `count` elements) back to back into
/// one new device allocation.
const float* UploadF32(DeviceArena& arena,
                       std::initializer_list<const TensorRef*> parts,
                       std::size_t count, const char* what,
                       std::string* error) {
  for (const TensorRef* t : parts) {
    if (t->empty() || t->type != GgmlType::kF32 || t->ElementCount() != count) {
      Fail(error, std::string(what) + ": expected an F32 tensor of " +
                      std::to_string(count) + " elements, got " +
                      (t->empty() ? std::string("none") : ShapeOf(*t)));
      return nullptr;
    }
  }
  const std::size_t bytes = count * sizeof(float);
  auto* dst = static_cast<std::uint8_t*>(arena.Allocate(bytes * parts.size()));
  if (dst == nullptr) {
    Fail(error, std::string(what) + ": device allocation failed");
    return nullptr;
  }
  std::size_t offset = 0;
  for (const TensorRef* t : parts) {
    if (hipMemcpy(dst + offset, t->data, bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
      Fail(error, std::string(what) + ": host-to-device copy failed");
      return nullptr;
    }
    offset += bytes;
  }
  return reinterpret_cast<const float*>(dst);
}

/// Stacks single-expert matrices of one encoding and width along rows.
bool UploadStacked(DeviceArena& arena,
                   std::initializer_list<const TensorRef*> parts,
                   DeviceMatrix* out, std::string* error) {
  const TensorRef& first = **parts.begin();
  std::uint64_t rows = 0;
  for (const TensorRef* t : parts) {
    if (t->type != first.type || t->cols != first.cols || t->experts != 1) {
      return Fail(error, "cannot stack " + ShapeOf(*t) + " (type " +
                             std::to_string(static_cast<int>(t->type)) +
                             ") onto " + ShapeOf(first) + " (type " +
                             std::to_string(static_cast<int>(first.type)) +
                             ")");
    }
    rows += t->rows;
  }
  if (!SupportsProjection(first.type, first.cols)) {
    return Fail(error, "no projection route for " + ShapeOf(first) + " type " +
                           std::to_string(static_cast<int>(first.type)));
  }
  const std::size_t row_bytes = first.RowBytes();
  auto* dst = static_cast<std::uint8_t*>(arena.Allocate(row_bytes * rows));
  if (dst == nullptr) {
    return Fail(error, "device allocation failed for " + ShapeOf(first));
  }
  std::size_t offset = 0;
  for (const TensorRef* t : parts) {
    if (hipMemcpy(dst + offset, t->data, t->SizeBytes(),
                  hipMemcpyHostToDevice) != hipSuccess) {
      return Fail(error, "host-to-device copy failed for " + ShapeOf(*t));
    }
    offset += t->SizeBytes();
  }
  *out = DeviceMatrix{.data = dst,
                      .type = first.type,
                      .cols = first.cols,
                      .rows = rows,
                      .experts = 1,
                      .row_bytes = row_bytes};
  return true;
}

/// Scratch regions of one KdaMixerForward pass of `tokens` tokens, in floats.
/// The q|k|v projection region is reused after the conv for qn | kn | o.
struct KdaScratch {
  float* qkv{nullptr};      ///< [T][3W]; then qn [T][W] | kn [T][W] | o [T][W]
  float* conv{nullptr};     ///< [T][3W]
  float* low{nullptr};      ///< [T][D] f_a, then g_a output
  float* f{nullptr};        ///< [T][W]
  float* gate{nullptr};     ///< [T][W]
  float* decay{nullptr};    ///< [T][W]
  float* beta_in{nullptr};  ///< [T][H]
  float* beta{nullptr};     ///< [T][H]
};

/// Lays out (when `base` is non-null) and sizes the scratch for `tokens`.
std::size_t LayoutKdaScratch(const Config& config, std::uint32_t tokens,
                             void* base, KdaScratch* s) {
  const std::size_t t = tokens;
  const std::size_t w = config.KdaDim();
  const std::size_t d = config.kda_head_dim;
  const std::size_t h = config.num_heads;
  std::size_t offset = 0;
  const auto take = [&](float*& region, std::size_t floats) {
    if (base != nullptr) {
      region =
          reinterpret_cast<float*>(static_cast<std::uint8_t*>(base) + offset);
    }
    offset += AlignUp(floats * sizeof(float));
  };
  KdaScratch unused;
  KdaScratch& r = s != nullptr ? *s : unused;
  take(r.qkv, t * 3 * w);
  take(r.conv, t * 3 * w);
  take(r.low, t * d);
  take(r.f, t * w);
  take(r.gate, t * w);
  take(r.decay, t * w);
  take(r.beta_in, t * h);
  take(r.beta, t * h);
  return offset;
}

/// One pass over `n` tokens whose scratch fits.
bool KdaPass(const OpContext& ctx, const Config& config,
             const KdaLayerDevice& w, const KdaState& state, const float* x,
             std::uint32_t n, float* y) {
  const std::uint32_t heads = config.num_heads;
  const std::uint32_t width = config.KdaDim();
  const std::uint32_t channels = 3 * width;
  const std::size_t tw = std::size_t{n} * width;
  KdaScratch s;
  LayoutKdaScratch(config, n, ctx.scratch, &s);
  float* qn = s.qkv;
  float* kn = s.qkv + tw;
  float* o = s.qkv + (2 * tw);
  const Stream stream = ctx.stream;

  // Projections (HF:676-686, 723-733), then the conv over q|k|v (HF:688-716).
  bool ok = LaunchProjection(w.qkv, x, s.qkv, n, stream);
  ok = ok && rocm::KdaConv(s.qkv, channels, w.conv_w, state.conv, s.conv, n,
                           channels, stream);
  ok = ok && LaunchProjection(w.f_a, x, s.low, n, stream);
  ok = ok && LaunchProjection(w.f_b, s.low, s.f, n, stream);
  ok = ok && LaunchProjection(w.g_a, x, s.low, n, stream);
  ok = ok && LaunchProjection(w.g_b, s.low, s.gate, n, stream);
  ok = ok && LaunchProjection(w.beta, x, s.beta_in, n, stream);
  // Forget gate (HF:353-369), l2norm + beta (HF:451-459, 733), recurrence
  // (HF:497-511), gated RMSNorm (HF:381-393), o_proj.
  ok = ok &&
       rocm::KdaGate(s.f, width, w.dt_bias, w.a, config.kda_gate_lower_bound,
                     nullptr, s.decay, n, heads, stream);
  ok = ok && rocm::KdaPrep(s.conv, channels, s.beta_in, heads, qn, kn, s.beta,
                           n, heads, kKdaL2Eps, stream);
  ok = ok && rocm::KdaRecurrent(qn, kn, s.conv + (2 * width), channels, s.decay,
                                s.beta, state.recurrent, o, n, heads, stream);
  ok = ok && rocm::KdaOutput(o, s.gate, width, w.norm, config.rms_eps, o, n,
                             heads, stream);
  ok = ok && LaunchProjection(w.out, o, y, n, stream);
  return ok && hipPeekAtLastError() == hipSuccess;
}

}  // namespace

// ---- DeviceArena ----

DeviceArena::~DeviceArena() {
  for (void* p : blocks_) {
    (void)hipFree(p);
  }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : blocks_(std::exchange(other.blocks_, {})),
      bytes_(std::exchange(other.bytes_, 0)) {}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
  if (this != &other) {
    for (void* p : blocks_) {
      (void)hipFree(p);
    }
    blocks_ = std::exchange(other.blocks_, {});
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

void* DeviceArena::Allocate(std::size_t bytes) {
  const std::size_t size = AlignUp(std::max<std::size_t>(bytes, 1));
  void* p = nullptr;
  if (hipMalloc(&p, size) != hipSuccess) {
    return nullptr;
  }
  blocks_.push_back(p);
  bytes_ += size;
  return p;
}

// ---- Projections ----

bool SupportsProjection(GgmlType type, std::uint64_t cols) noexcept {
  if (cols == 0) {
    return false;
  }
  switch (type) {
    case GgmlType::kQ8_0:
      return cols % 32 == 0;
    case GgmlType::kQ8_K:
      return cols % 256 == 0;
    case GgmlType::kF32:
    case GgmlType::kBF16:
      // The dense GEMV reads rows as float4 / 8 x bf16 vectors.
      return cols % 8 == 0;
    default:
      return ::gufo::hip::detail::IsNativeWmmaQuant(type) && cols % 256 == 0;
  }
}

bool UploadMatrix(DeviceArena& arena, const TensorRef& t, DeviceMatrix* out,
                  std::string* error) {
  if (t.empty()) {
    return Fail(error, "missing tensor " + std::string(t.name));
  }
  if (!SupportsProjection(t.type, t.cols)) {
    return Fail(error, "no projection route for " + ShapeOf(t) + " type " +
                           std::to_string(static_cast<int>(t.type)));
  }
  void* dst = arena.Allocate(t.SizeBytes());
  if (dst == nullptr) {
    return Fail(error, "device allocation failed for " + ShapeOf(t));
  }
  if (hipMemcpy(dst, t.data, t.SizeBytes(), hipMemcpyHostToDevice) !=
      hipSuccess) {
    return Fail(error, "host-to-device copy failed for " + ShapeOf(t));
  }
  *out = DeviceMatrix{.data = dst,
                      .type = t.type,
                      .cols = t.cols,
                      .rows = t.rows,
                      .experts = t.experts,
                      .row_bytes = t.RowBytes()};
  return true;
}

bool LaunchProjection(const DeviceMatrix& w, const float* x, float* y,
                      std::uint32_t tokens, Stream stream) {
  if (w.empty() || w.experts != 1 || !SupportsProjection(w.type, w.cols)) {
    return false;
  }
  if (tokens == 0) {
    return true;
  }
  if (w.type == GgmlType::kF32 || w.type == GgmlType::kBF16) {
    for (std::uint32_t t = 0; t < tokens; ++t) {
      ::gufo::hip::LaunchGEMV(w.data, w.type, x + (std::size_t{t} * w.cols),
                              y + (std::size_t{t} * w.rows), w.rows, w.cols,
                              stream);
    }
    return true;
  }
  ::gufo::hip::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, y, tokens, w.rows,
                                          w.cols, stream);
  return true;
}

// ---- KDA layer ----

bool UploadKdaLayer(DeviceArena& arena, const Config& config,
                    const LayerWeights& l, std::uint32_t il,
                    KdaLayerDevice* out, std::string* error) {
  if (!config.IsKdaLayer(il) || l.mixer != Config::Mixer::kKda) {
    return Fail(error, "layer " + std::to_string(il) + " is not a KDA layer");
  }
  if (config.kda_head_dim != rocm::kKdaHeadDim ||
      config.ssm_conv_kernel != rocm::kKdaConvTaps) {
    return Fail(error, "KDA kernels need head dim 128 and conv kernel 4, got " +
                           std::to_string(config.kda_head_dim) + " and " +
                           std::to_string(config.ssm_conv_kernel));
  }
  const std::uint64_t hidden = config.hidden_size;
  const std::uint64_t dim = config.KdaDim();
  const std::uint64_t d = config.kda_head_dim;
  const std::uint64_t heads = config.num_heads;
  if (!CheckMatrix(l.kda_q, "attn_q", hidden, dim, error) ||
      !CheckMatrix(l.kda_k, "attn_k", hidden, dim, error) ||
      !CheckMatrix(l.kda_v, "attn_v", hidden, dim, error) ||
      !CheckMatrix(l.kda_f_a, "ssm_f_a", hidden, d, error) ||
      !CheckMatrix(l.kda_f_b, "ssm_f_b", d, dim, error) ||
      !CheckMatrix(l.kda_g_a, "ssm_g_a", hidden, d, error) ||
      !CheckMatrix(l.kda_g_b, "ssm_g_b", d, dim, error) ||
      !CheckMatrix(l.kda_beta, "ssm_beta", hidden, heads, error) ||
      !CheckMatrix(l.attn_out, "attn_output", dim, hidden, error)) {
    return false;
  }
  KdaLayerDevice w;
  if (!UploadStacked(arena, {&l.kda_q, &l.kda_k, &l.kda_v}, &w.qkv, error) ||
      !UploadMatrix(arena, l.kda_f_a, &w.f_a, error) ||
      !UploadMatrix(arena, l.kda_f_b, &w.f_b, error) ||
      !UploadMatrix(arena, l.kda_g_a, &w.g_a, error) ||
      !UploadMatrix(arena, l.kda_g_b, &w.g_b, error) ||
      !UploadMatrix(arena, l.kda_beta, &w.beta, error) ||
      !UploadMatrix(arena, l.attn_out, &w.out, error)) {
    return false;
  }
  // The three GGUF convs [4, 1, dim] concatenate to the kernel's
  // [3*dim][4] (reference.cpp Kda).
  w.conv_w = UploadF32(arena, {&l.kda_conv_q, &l.kda_conv_k, &l.kda_conv_v},
                       dim * config.ssm_conv_kernel, "ssm_conv1d", error);
  if (w.conv_w == nullptr) {
    return false;
  }
  w.dt_bias = UploadF32(arena, {&l.kda_dt_bias}, dim, "ssm_dt.bias", error);
  if (w.dt_bias == nullptr) {
    return false;
  }
  w.a = UploadF32(arena, {&l.kda_a}, heads, "ssm_a", error);
  if (w.a == nullptr) {
    return false;
  }
  w.norm = UploadF32(arena, {&l.kda_norm}, d, "ssm_norm", error);
  if (w.norm == nullptr) {
    return false;
  }
  *out = w;
  return true;
}

std::size_t KdaStateBytes(const Config& config) noexcept {
  const std::size_t conv = std::size_t{config.ssm_conv_kernel - 1} * 3 *
                           config.KdaDim() * sizeof(float);
  const std::size_t recurrent = std::size_t{config.num_heads} *
                                config.kda_head_dim * config.kda_head_dim *
                                sizeof(float);
  return AlignUp(conv) + recurrent;
}

bool AllocateKdaState(DeviceArena& arena, const Config& config, KdaState* out) {
  auto* base =
      static_cast<std::uint8_t*>(arena.Allocate(KdaStateBytes(config)));
  if (base == nullptr) {
    return false;
  }
  const std::size_t conv = std::size_t{config.ssm_conv_kernel - 1} * 3 *
                           config.KdaDim() * sizeof(float);
  KdaState s{.conv = reinterpret_cast<float*>(base),
             .recurrent = reinterpret_cast<float*>(base + AlignUp(conv))};
  if (hipMemset(base, 0, KdaStateBytes(config)) != hipSuccess) {
    return false;
  }
  *out = s;
  return true;
}

bool ResetKdaState(const Config& config, const KdaState& state, Stream stream) {
  if (state.conv == nullptr || state.recurrent == nullptr) {
    return false;
  }
  const std::size_t conv = std::size_t{config.ssm_conv_kernel - 1} * 3 *
                           config.KdaDim() * sizeof(float);
  const std::size_t recurrent = std::size_t{config.num_heads} *
                                config.kda_head_dim * config.kda_head_dim *
                                sizeof(float);
  return hipMemsetAsync(state.conv, 0, conv, stream) == hipSuccess &&
         hipMemsetAsync(state.recurrent, 0, recurrent, stream) == hipSuccess;
}

std::size_t KdaScratchBytes(const Config& config,
                            std::uint32_t max_tokens) noexcept {
  return LayoutKdaScratch(config, max_tokens, nullptr, nullptr);
}

bool KdaMixerForward(const OpContext& ctx, const Config& config,
                     std::uint32_t layer, const KdaLayerDevice& w,
                     KdaState& state, const float* x, std::uint32_t tokens,
                     float* y) {
  if (tokens == 0) {
    return true;
  }
  if (!config.IsKdaLayer(layer) || x == nullptr || y == nullptr ||
      state.conv == nullptr || state.recurrent == nullptr ||
      ctx.scratch == nullptr || w.conv_w == nullptr || w.dt_bias == nullptr ||
      w.a == nullptr || w.norm == nullptr) {
    return false;
  }
  // Largest chunk whose scratch fits: KdaScratchBytes(c) <= c *
  // KdaScratchBytes(1) gives a lower bound, then bisect up to `tokens`.
  const std::size_t one = KdaScratchBytes(config, 1);
  if (ctx.scratch_bytes < one) {
    return false;
  }
  std::uint32_t lo = static_cast<std::uint32_t>(
      std::min<std::size_t>(tokens, ctx.scratch_bytes / one));
  std::uint32_t hi = tokens;
  while (lo < hi) {
    const std::uint32_t mid = lo + ((hi - lo + 1) / 2);
    if (KdaScratchBytes(config, mid) <= ctx.scratch_bytes) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const std::uint32_t chunk = lo;
  const std::size_t hidden = config.hidden_size;
  for (std::uint32_t t0 = 0; t0 < tokens; t0 += chunk) {
    const std::uint32_t n = std::min(chunk, tokens - t0);
    if (!KdaPass(ctx, config, w, state, x + (t0 * hidden), n,
                 y + (t0 * hidden))) {
      return false;
    }
  }
  if (ctx.tap != nullptr && *ctx.tap) {
    (*ctx.tap)("kda_out", layer,
               TapView{.device = y,
                       .dtype = TapView::Dtype::kF32,
                       .ne0 = hidden,
                       .ne1 = tokens});
  }
  return true;
}

}  // namespace gufo::models::glm53_flash
