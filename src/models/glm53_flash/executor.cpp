#include "src/models/glm53_flash/executor.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <utility>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/glm53_flash/kernels/hc.hpp"
#include "src/models/glm53_flash/kernels/kda.hpp"
#include "src/models/glm53_flash/kernels/prefill_gemm.hpp"
#include "src/models/glm53_flash/profile.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

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
                 std::uint64_t rows, std::string* error,
                 std::uint64_t experts = 1) {
  if (t.empty()) {
    return Fail(error, std::string("missing tensor ") + what);
  }
  if (t.cols != cols || t.rows != rows || t.experts != experts) {
    return Fail(error, "tensor " + ShapeOf(t) + " expected [" +
                           std::to_string(cols) + " x " + std::to_string(rows) +
                           " x " + std::to_string(experts) + "]");
  }
  return true;
}

/// Copies `t` (all experts) to the device in its GGUF encoding, with no
/// projection-route check: operator-owned weights (MLA, FFN, embedding).
bool UploadEncoded(DeviceArena& arena, const TensorRef& t, DeviceMatrix* out,
                   std::string* error) {
  if (t.empty()) {
    return Fail(error, "missing tensor " + std::string(t.name));
  }
  if (t.SizeBytes() == 0) {
    return Fail(error, "unsized encoding for " + ShapeOf(t));
  }
  void* dst = arena.Allocate(t.SizeBytes());
  if (dst == nullptr) {
    return Fail(error, "device allocation failed for " + ShapeOf(t));
  }
  if (!arena.Copy(dst, t, t.SizeBytes())) {
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

/// True if gufo's batched embedding lookup decodes `type` rows of `cols`.
bool SupportsEmbedding(GgmlType type, std::uint64_t cols) noexcept {
  switch (type) {
    case GgmlType::kF32:
    case GgmlType::kBF16:
      return cols > 0;
    case GgmlType::kQ8_0:
    case GgmlType::kIQ4_NL:
      return cols > 0 && cols % 32 == 0;
    case GgmlType::kQ3_K:
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
    case GgmlType::kIQ4_XS:
    case GgmlType::kIQ3_S:
      return cols > 0 && cols % 256 == 0;
    default:
      return false;
  }
}

float ClampAt(const std::vector<float>& clamps, std::uint32_t il) {
  return il < clamps.size() ? clamps[il] : 0.0F;
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
    if (!arena.Copy(dst + offset, *t, bytes)) {
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
    if (!arena.Copy(dst + offset, *t, t->SizeBytes())) {
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
  // Sub-scopes ("kda/..." rows) nest inside the executor's "kda" scope and
  // cost a pointer test when GUFO_GLM53_PROFILE is off.
  std::optional<OpScope> scope;
  scope.emplace(ctx.profiler, "kda/proj_qkv", stream);
  bool ok = LaunchProjection(w.qkv, x, s.qkv, n, stream);
  scope.emplace(ctx.profiler, "kda/conv", stream);
  ok = ok && rocm::KdaConv(s.qkv, channels, w.conv_w, state.conv, s.conv, n,
                           channels, stream,
                           state.record ? state.conv_ckpt : nullptr);
  scope.emplace(ctx.profiler, "kda/proj_f", stream);
  ok = ok && LaunchProjection(w.f_a, x, s.low, n, stream);
  ok = ok && LaunchProjection(w.f_b, s.low, s.f, n, stream);
  scope.emplace(ctx.profiler, "kda/proj_g", stream);
  ok = ok && LaunchProjection(w.g_a, x, s.low, n, stream);
  ok = ok && LaunchProjection(w.g_b, s.low, s.gate, n, stream);
  scope.emplace(ctx.profiler, "kda/proj_beta", stream);
  ok = ok && LaunchProjection(w.beta, x, s.beta_in, n, stream);
  // Forget gate (HF:353-369), l2norm + beta (HF:451-459, 733), recurrence
  // (HF:497-511), gated RMSNorm (HF:381-393), o_proj.
  scope.emplace(ctx.profiler, "kda/gate_prep", stream);
  ok = ok &&
       rocm::KdaGate(s.f, width, w.dt_bias, w.a, config.kda_gate_lower_bound,
                     nullptr, s.decay, n, heads, stream);
  ok = ok && rocm::KdaPrep(s.conv, channels, s.beta_in, heads, qn, kn, s.beta,
                           n, heads, kKdaL2Eps, stream);
  scope.emplace(ctx.profiler, "kda/recurrent", stream);
  ok = ok && rocm::KdaRecurrent(qn, kn, s.conv + (2 * width), channels, s.decay,
                                s.beta, state.recurrent, o, n, heads, stream,
                                state.record ? state.recurrent_ckpt : nullptr);
  scope.emplace(ctx.profiler, "kda/out_norm", stream);
  ok = ok && rocm::KdaOutput(o, s.gate, width, w.norm, config.rms_eps, o, n,
                             heads, stream);
  scope.emplace(ctx.profiler, "kda/out_proj", stream);
  ok = ok && LaunchProjection(w.out, o, y, n, stream);
  scope.reset();
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
      bytes_(std::exchange(other.bytes_, 0)),
      upload_(std::exchange(other.upload_, nullptr)) {}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
  if (this != &other) {
    for (void* p : blocks_) {
      (void)hipFree(p);
    }
    blocks_ = std::exchange(other.blocks_, {});
    bytes_ = std::exchange(other.bytes_, 0);
    upload_ = std::exchange(other.upload_, nullptr);
  }
  return *this;
}

void* DeviceArena::Allocate(std::size_t bytes) {
  const std::size_t size = AlignUp(std::max<std::size_t>(bytes, 1));
  void* p = nullptr;
  // Managed memory, always: on the gfx1151 APU hipMalloc stalls (it never
  // returns OOM) near 56 GiB while hipMemGetInfo still reports the 112 GiB
  // TTM heap free, so the AllocateDevice fallback never triggers and the
  // 89.5 GiB Maya-S upload hangs. Managed pages come from the same DRAM.
  if (gufo::hip::AllocateManaged(&p, size) != hipSuccess) {
    return nullptr;
  }
  blocks_.push_back(p);
  bytes_ += size;
  return p;
}

bool DeviceArena::Copy(void* dst, const TensorRef& t, std::size_t bytes) const {
  if (upload_ != nullptr) {
    return upload_->Copy(t.shard, t.file_offset, bytes, dst, nullptr);
  }
  return hipMemcpy(dst, t.data, bytes, hipMemcpyHostToDevice) == hipSuccess;
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
  return UploadEncoded(arena, t, out, error);
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
  if (PrefillQuantGemm(w.type, w.data, x, y, tokens, w.rows, w.cols, stream))
    return true;
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
  // Checkpoints index tokens within one pass.
  if (state.record && chunk < tokens) {
    return false;
  }
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

// ---- MLA / FFN layers ----

bool UploadMlaLayer(DeviceArena& arena, const Config& config,
                    const LayerWeights& l, std::uint32_t il,
                    MlaLayerDevice* out, std::string* error) {
  if (!config.IsMlaLayer(il) || l.mixer != Config::Mixer::kMla) {
    return Fail(error, "layer " + std::to_string(il) + " is not an MLA layer");
  }
  const std::uint64_t hidden = config.hidden_size;
  const std::uint64_t heads = config.num_heads;
  const std::uint64_t q_lora = config.q_lora_rank;
  const std::uint64_t kv_lora = config.kv_lora_rank;
  const std::uint64_t nope = config.mla_head_dim;
  const std::uint64_t v_dim = config.mla_value_dim;
  const std::uint64_t ih = config.indexer_heads;
  const std::uint64_t idim = config.indexer_head_dim;
  if (!CheckMatrix(l.mla_q_a, "attn_q_a", hidden, q_lora, error) ||
      !CheckMatrix(l.mla_q_b, "attn_q_b", q_lora, heads * nope, error) ||
      !CheckMatrix(l.mla_kv_a, "attn_kv_a_mqa", hidden, kv_lora, error) ||
      !CheckMatrix(l.mla_k_b, "attn_k_b", nope, kv_lora, error, heads) ||
      !CheckMatrix(l.mla_v_b, "attn_v_b", kv_lora, v_dim, error, heads) ||
      !CheckMatrix(l.attn_out, "attn_output", heads * v_dim, hidden, error) ||
      !CheckMatrix(l.idx_q_b, "indexer.attn_q_b", q_lora, ih * idim, error) ||
      !CheckMatrix(l.idx_k, "indexer.attn_k", hidden, idim, error) ||
      !CheckMatrix(l.idx_proj, "indexer.proj", hidden, ih, error) ||
      !CheckMatrix(l.idx_gate, "indexer_compressor_gate", hidden, idim,
                   error)) {
    return false;
  }
  MlaLayerDevice w;
  if (!UploadEncoded(arena, l.mla_q_a, &w.q_a, error) ||
      !UploadEncoded(arena, l.mla_q_b, &w.q_b, error) ||
      !UploadEncoded(arena, l.mla_kv_a, &w.kv_a, error) ||
      !UploadEncoded(arena, l.mla_k_b, &w.k_b, error) ||
      !UploadEncoded(arena, l.mla_v_b, &w.v_b, error) ||
      !UploadEncoded(arena, l.attn_out, &w.out, error) ||
      !UploadEncoded(arena, l.idx_q_b, &w.idx_q_b, error) ||
      !UploadEncoded(arena, l.idx_k, &w.idx_k, error) ||
      !UploadEncoded(arena, l.idx_proj, &w.idx_proj, error) ||
      !UploadEncoded(arena, l.idx_gate, &w.idx_gate, error)) {
    return false;
  }
  const auto vec = [&](const float*& dst, const TensorRef& t, std::size_t n,
                       const char* what) {
    dst = UploadF32(arena, {&t}, n, what, error);
    return dst != nullptr;
  };
  if (!vec(w.q_a_norm, l.mla_q_a_norm, q_lora, "attn_q_a_norm") ||
      !vec(w.kv_a_norm, l.mla_kv_a_norm, kv_lora, "attn_kv_a_norm") ||
      !vec(w.idx_k_norm, l.idx_k_norm, idim, "indexer.k_norm") ||
      !vec(w.idx_k_norm_bias, l.idx_k_norm_bias, idim, "indexer.k_norm.bias") ||
      !vec(w.idx_ape, l.idx_ape, config.indexer_kpool * idim,
           "indexer_compressor_ape")) {
    return false;
  }
  *out = w;
  return true;
}

bool UploadFfnLayer(DeviceArena& arena, const Config& config,
                    const LayerWeights& l, std::uint32_t il,
                    FfnLayerDevice* out, std::string* error) {
  FfnLayerDevice w;
  w.dense = config.IsDenseFfnLayer(il);
  if (w.dense != l.dense_ffn) {
    return Fail(error, "layer " + std::to_string(il) +
                           " dense-FFN binding disagrees with the config");
  }
  w.clamp_exp = ClampAt(config.swiglu_clamp_exp, il);
  w.clamp_shexp = ClampAt(config.swiglu_clamp_shexp, il);
  const std::uint64_t hidden = config.hidden_size;
  if (w.dense) {
    const std::uint64_t ff = config.dense_ff;
    if (!CheckMatrix(l.ffn_gate, "ffn_gate", hidden, ff, error) ||
        !CheckMatrix(l.ffn_up, "ffn_up", hidden, ff, error) ||
        !CheckMatrix(l.ffn_down, "ffn_down", ff, hidden, error) ||
        !UploadEncoded(arena, l.ffn_gate, &w.gate, error) ||
        !UploadEncoded(arena, l.ffn_up, &w.up, error) ||
        !UploadEncoded(arena, l.ffn_down, &w.down, error)) {
      return false;
    }
    *out = w;
    return true;
  }
  const std::uint64_t ne = config.num_experts;
  const std::uint64_t eff = config.expert_ff;
  const std::uint64_t sff = config.shared_expert_ff;
  if (!CheckMatrix(l.ffn_gate_exps, "ffn_gate_exps", hidden, eff, error, ne) ||
      !CheckMatrix(l.ffn_up_exps, "ffn_up_exps", hidden, eff, error, ne) ||
      !CheckMatrix(l.ffn_down_exps, "ffn_down_exps", eff, hidden, error, ne) ||
      !CheckMatrix(l.shexp_gate, "ffn_gate_shexp", hidden, sff, error) ||
      !CheckMatrix(l.shexp_up, "ffn_up_shexp", hidden, sff, error) ||
      !CheckMatrix(l.shexp_down, "ffn_down_shexp", sff, hidden, error)) {
    return false;
  }
  w.router = UploadF32(arena, {&l.router}, ne * hidden, "ffn_gate_inp", error);
  if (w.router == nullptr) {
    return false;
  }
  w.router_bias = UploadF32(arena, {&l.router_bias}, ne, "exp_probs_b", error);
  if (w.router_bias == nullptr) {
    return false;
  }
  if (!UploadEncoded(arena, l.ffn_gate_exps, &w.gate_exps, error) ||
      !UploadEncoded(arena, l.ffn_up_exps, &w.up_exps, error) ||
      !UploadEncoded(arena, l.ffn_down_exps, &w.down_exps, error) ||
      !UploadEncoded(arena, l.shexp_gate, &w.shexp_gate, error) ||
      !UploadEncoded(arena, l.shexp_up, &w.shexp_up, error) ||
      !UploadEncoded(arena, l.shexp_down, &w.shexp_down, error)) {
    return false;
  }
  *out = w;
  return true;
}

// ---- PRODUCTION BINDING (Lane B) ----

#if defined(GLM53_LANE_B_OPS)
LayerOps LaneBOps() {
  return LayerOps{.mla_cache_bytes = &MlaCacheBytes,
                  .op_scratch_bytes = &OpScratchBytes,
                  .mla = &MlaForward,
                  .ffn = &FfnForward};
}
#endif

// ---- Executor ----

Executor::Executor(const ModelWeights& weights, const ExecutorOptions& options,
                   LayerOps ops)
    : weights_(weights),
      config_(weights.config),
      options_(options),
      ops_(std::move(ops)) {}

Executor::~Executor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
    if (profiler_ != nullptr) {
      PrintProfile(std::cerr);
    }
    (void)hipStreamDestroy(stream_);
  }
}

std::unique_ptr<Executor> Executor::Create(const ModelWeights& weights,
                                           const ExecutorOptions& options,
                                           LayerOps ops, std::string* error) {
  std::unique_ptr<Executor> e(new Executor(weights, options, std::move(ops)));
  if (!e->Init(error)) {
    return nullptr;
  }
  return e;
}

void Executor::PrintProfile(std::ostream& os) const {
  if (profiler_ != nullptr) {
    profiler_->Print(os);
  }
}

bool Executor::Init(std::string* error) {
  const Config& c = config_;
  profiler_ = OpProfiler::FromEnv();
  if (options_.last_layer == 0) {
    options_.last_layer = c.num_layers;
  }
  const std::uint32_t first = options_.first_layer;
  const std::uint32_t last = options_.last_layer;
  if (first >= last || last > c.num_layers || last > weights_.layers.size() ||
      c.mixers.size() < last) {
    return Fail(error, "bad layer range [" + std::to_string(first) + ", " +
                           std::to_string(last) + ") of " +
                           std::to_string(c.num_layers) + " trunk layers");
  }
  if (c.hc_count != rocm::kHcStreams || c.HcMixDim() != rocm::kHcMixDim) {
    return Fail(error, "hyper-connection kernels need 4 streams, got " +
                           std::to_string(c.hc_count));
  }
  if (options_.max_tokens == 0) {
    return Fail(error, "max_tokens must be positive");
  }
  if (options_.embed && first != 0) {
    return Fail(error,
                "the embedding feeds layer 0; use ForwardStreams for a "
                "range starting at layer " +
                    std::to_string(first));
  }
  bool any_kda = false;
  bool any_mla = false;
  for (std::uint32_t il = first; il < last; ++il) {
    any_kda = any_kda || c.IsKdaLayer(il);
    any_mla = any_mla || c.IsMlaLayer(il);
  }
  if (any_mla && (!ops_.mla || !ops_.mla_cache_bytes)) {
    return Fail(error,
                "the layer range has MLA layers but no MLA operator "
                "is bound (LayerOps::mla / mla_cache_bytes)");
  }
  if (!ops_.ffn) {
    return Fail(error, "no FFN operator is bound (LayerOps::ffn)");
  }
  // KdaConv checkpoints need the verify pass in one conv tile (8), and the
  // indexer ring (2 * kpool rows) keeps a rolled-back pass's kept rows only
  // while it is at most kpool + 1 tokens.
  const std::uint32_t verify_max =
      std::min<std::uint32_t>(8, config_.indexer_kpool + 1);
  if (options_.max_verify_tokens == 0 ||
      options_.max_verify_tokens > verify_max) {
    return Fail(error, "max_verify_tokens must be 1.." +
                           std::to_string(verify_max));
  }
  if (hipStreamCreate(&stream_) != hipSuccess) {
    return Fail(error, "hipStreamCreate failed");
  }
  // Weight copies queue on `upload` until Finish below. On an early return
  // the guard detaches it before its destructor drains the queue.
  std::unique_ptr<hip::WeightUpload> upload;
  if (!options_.shards.empty()) {
    upload = hip::WeightUpload::Create(options_.shards, error);
    if (upload == nullptr) {
      return false;
    }
  }
  struct StageGuard {
    DeviceArena& arena;
    ~StageGuard() { arena.StageFrom(nullptr); }
  } stage_guard{arena_};
  arena_.StageFrom(upload.get());

  const std::size_t hidden = c.hidden_size;
  const std::size_t mix_dim = c.HcMixDim();
  // With the NextN block, its LayerDevice and MlaCache follow the trunk's.
  const std::uint32_t slots = c.num_layers + (options_.nextn ? 1 : 0);
  layers_.assign(slots, {});
  kda_.assign(c.num_layers, {});
  mla_.assign(slots, {});
  for (std::uint32_t il = first; il < last; ++il) {
    const LayerWeights& l = weights_.layers[il];
    LayerDevice& d = layers_[il];
    std::string why;
    const auto layer_fail = [&](const std::string& what) {
      return Fail(error, "layer " + std::to_string(il) + ": " + what);
    };
    if (l.nextn) {
      return layer_fail("NextN block in the trunk range");
    }
    for (int k = 0; k < 2; ++k) {
      const HcMixer& m = k == 0 ? l.hc_attn : l.hc_ffn;
      LayerDevice::Hc& h = k == 0 ? d.hc_attn : d.hc_ffn;
      h.fn = UploadF32(arena_, {&m.fn}, mix_dim * c.HcDim(), "hc fn", &why);
      h.base = h.fn != nullptr
                   ? UploadF32(arena_, {&m.base}, mix_dim, "hc base", &why)
                   : nullptr;
      h.scale = h.base != nullptr
                    ? UploadF32(arena_, {&m.scale}, 3, "hc scale", &why)
                    : nullptr;
      if (h.scale == nullptr) {
        return layer_fail(why);
      }
    }
    d.attn_norm = UploadF32(arena_, {&l.attn_norm}, hidden, "attn_norm", &why);
    d.ffn_norm = d.attn_norm != nullptr ? UploadF32(arena_, {&l.ffn_norm},
                                                    hidden, "ffn_norm", &why)
                                        : nullptr;
    if (d.ffn_norm == nullptr) {
      return layer_fail(why);
    }
    if (c.IsKdaLayer(il)) {
      if (!UploadKdaLayer(arena_, c, l, il, &d.kda, &why)) {
        return layer_fail(why);
      }
      if (!AllocateKdaState(arena_, c, &kda_[il])) {
        return layer_fail("KDA state allocation failed");
      }
      if (options_.max_verify_tokens > 1) {
        const std::size_t keep = options_.max_verify_tokens - 1;
        const std::size_t conv = std::size_t{c.ssm_conv_kernel - 1} * 3 *
                                 c.KdaDim() * sizeof(float);
        const std::size_t recurrent = std::size_t{c.num_heads} *
                                      c.kda_head_dim * c.kda_head_dim *
                                      sizeof(float);
        kda_[il].conv_ckpt =
            static_cast<float*>(arena_.Allocate(keep * conv));
        kda_[il].recurrent_ckpt =
            static_cast<float*>(arena_.Allocate(keep * recurrent));
        if (kda_[il].conv_ckpt == nullptr ||
            kda_[il].recurrent_ckpt == nullptr) {
          return layer_fail("KDA verify checkpoint allocation failed");
        }
      }
    } else {
      if (!UploadMlaLayer(arena_, c, l, il, &d.mla, &why)) {
        return layer_fail(why);
      }
      MlaCache& cache = mla_[il];
      cache.bytes = ops_.mla_cache_bytes(c, options_.capacity);
      cache.capacity = options_.capacity;
      if (cache.bytes != 0) {
        cache.data = arena_.Allocate(cache.bytes);
        if (cache.data == nullptr ||
            hipMemset(cache.data, 0, cache.bytes) != hipSuccess) {
          return layer_fail("MLA cache allocation failed (" +
                            std::to_string(cache.bytes) + " bytes)");
        }
      }
    }
    if (!UploadFfnLayer(arena_, c, l, il, &d.ffn, &why)) {
      return layer_fail(why);
    }
  }

  if (options_.nextn) {
    const std::uint32_t il = c.num_layers;
    std::string why;
    const auto nextn_fail = [&](const std::string& what) {
      return Fail(error, "NextN block: " + what);
    };
    if (!options_.embed || !options_.head || weights_.layers.size() <= il ||
        !weights_.layers[il].nextn || !c.IsMlaLayer(il)) {
      return nextn_fail(
          "needs the weights bound with NextN, the embedding and the head");
    }
    const LayerWeights& l = weights_.layers[il];
    LayerDevice& d = layers_[il];
    d.attn_norm = UploadF32(arena_, {&l.attn_norm}, hidden, "attn_norm", &why);
    d.ffn_norm = d.attn_norm != nullptr ? UploadF32(arena_, {&l.ffn_norm},
                                                    hidden, "ffn_norm", &why)
                                        : nullptr;
    enorm_ = d.ffn_norm != nullptr
                 ? UploadF32(arena_, {&l.nextn_enorm}, hidden, "enorm", &why)
                 : nullptr;
    hnorm_ = enorm_ != nullptr
                 ? UploadF32(arena_, {&l.nextn_hnorm}, hidden, "hnorm", &why)
                 : nullptr;
    // shared_head_norm, else the trunk's output_norm (llama.cpp graph_mtp).
    const TensorRef& head_norm = l.nextn_head_norm.empty()
                                     ? weights_.output_norm
                                     : l.nextn_head_norm;
    nextn_head_norm_ = hnorm_ != nullptr ? UploadF32(arena_, {&head_norm},
                                                     hidden, "head_norm", &why)
                                         : nullptr;
    if (nextn_head_norm_ == nullptr) {
      return nextn_fail(why);
    }
    if (!CheckMatrix(l.nextn_eh_proj, "eh_proj", 2 * hidden, hidden, error) ||
        !UploadMatrix(arena_, l.nextn_eh_proj, &eh_proj_, error)) {
      return false;
    }
    if (!UploadMlaLayer(arena_, c, l, il, &d.mla, &why) ||
        !UploadFfnLayer(arena_, c, l, il, &d.ffn, &why)) {
      return nextn_fail(why);
    }
    MlaCache& cache = mla_[il];
    cache.bytes = ops_.mla_cache_bytes(c, options_.capacity);
    cache.capacity = options_.capacity;
    if (cache.bytes != 0) {
      cache.data = arena_.Allocate(cache.bytes);
      if (cache.data == nullptr ||
          hipMemset(cache.data, 0, cache.bytes) != hipSuccess) {
        return nextn_fail("MLA cache allocation failed");
      }
    }
  }

  if (options_.embed) {
    const TensorRef& t = weights_.token_embd;
    if (!CheckMatrix(t, "token_embd", hidden, c.vocab_size, error)) {
      return false;
    }
    if (!SupportsEmbedding(t.type, t.cols)) {
      return Fail(error, "no embedding lookup for " + ShapeOf(t) + " type " +
                             std::to_string(static_cast<int>(t.type)));
    }
    if (!UploadEncoded(arena_, t, &embd_, error)) {
      return false;
    }
  }
  if (options_.head) {
    output_norm_ = UploadF32(arena_, {&weights_.output_norm}, hidden,
                             "output_norm", error);
    if (output_norm_ == nullptr) {
      return false;
    }
    if (!weights_.output.empty() &&
        (!CheckMatrix(weights_.output, "output", hidden, c.vocab_size, error) ||
         !UploadMatrix(arena_, weights_.output, &output_, error))) {
      return false;
    }
  }

  arena_.StageFrom(nullptr);
  if (upload != nullptr && !upload->Finish(error)) {
    return false;
  }

  // Activations for one pass of max_tokens.
  const std::size_t t = options_.max_tokens;
  const auto floats = [&](float*& dst, std::size_t n) {
    dst = static_cast<float*>(arena_.Allocate(n * sizeof(float)));
    return dst != nullptr;
  };
  bool ok = floats(streams_, t * c.HcDim()) && floats(x_, t * hidden) &&
            floats(xn_, t * hidden) && floats(a_, t * hidden) &&
            floats(split_, t * mix_dim) && floats(tap_post_, t * c.hc_count) &&
            floats(tap_comb_, t * c.hc_count * c.hc_count);
  if (ok && !output_.empty()) {
    ok = floats(logits_, t * c.vocab_size);
  }
  if (ok && !embd_.empty()) {
    ids_ =
        static_cast<std::uint32_t*>(arena_.Allocate(t * sizeof(std::uint32_t)));
    ok = ids_ != nullptr;
  }
  scratch_bytes_ = std::max(
      any_kda ? KdaScratchBytes(c, options_.max_tokens) : std::size_t{0},
      ops_.op_scratch_bytes ? ops_.op_scratch_bytes(c, options_.max_tokens)
                            : std::size_t{0});
  if (ok && scratch_bytes_ != 0) {
    scratch_ = arena_.Allocate(scratch_bytes_);
    ok = scratch_ != nullptr;
  }
  if (!ok) {
    return Fail(error, "activation / scratch allocation failed");
  }
  return true;
}

bool Executor::Reset() {
  bool ok = true;
  for (std::uint32_t il = options_.first_layer; il < options_.last_layer;
       ++il) {
    if (config_.IsKdaLayer(il)) {
      ok = ok && ResetKdaState(config_, kda_[il], stream_);
    } else if (mla_[il].data != nullptr) {
      ok = ok && hipMemsetAsync(mla_[il].data, 0, mla_[il].bytes, stream_) ==
                     hipSuccess;
    }
  }
  if (options_.nextn && mla_[config_.num_layers].data != nullptr) {
    const MlaCache& cache = mla_[config_.num_layers];
    ok = ok && hipMemsetAsync(cache.data, 0, cache.bytes, stream_) ==
                   hipSuccess;
  }
  ok = ok && hipStreamSynchronize(stream_) == hipSuccess;
  position_ = 0;
  nextn_position_ = 0;
  verify_open_ = false;
  failed_ = !ok;
  return ok;
}

bool Executor::Forward(std::span<const std::int32_t> tokens, bool want_logits,
                       ForwardResult* out, const TapFn* tap,
                       std::string* error) {
  if (embd_.empty()) {
    return Fail(error, "Forward needs the embedding (ExecutorOptions::embed)");
  }
  for (const std::int32_t id : tokens) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= config_.vocab_size) {
      return Fail(error, "token id " + std::to_string(id) + " out of range");
    }
  }
  return Run(tokens.data(), nullptr, static_cast<std::uint32_t>(tokens.size()),
             want_logits, Head::kPerToken, out, tap, error);
}

bool Executor::ForwardVerify(std::span<const std::int32_t> tokens,
                             bool want_logits, ForwardResult* out,
                             std::string* error) {
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > options_.max_verify_tokens) {
    return Fail(error, "ForwardVerify of " + std::to_string(n) +
                           " tokens; max_verify_tokens is " +
                           std::to_string(options_.max_verify_tokens));
  }
  if (n > options_.max_tokens) {
    return Fail(error, "ForwardVerify needs one pass (max_tokens)");
  }
  const std::uint32_t base = position_;
  for (auto& k : kda_) {
    k.record = k.recurrent_ckpt != nullptr;
  }
  const bool ok = Forward(tokens, want_logits, out, nullptr, error);
  for (auto& k : kda_) {
    k.record = false;
  }
  if (ok) {
    verify_open_ = true;
    verify_base_ = base;
    verify_n_ = n;
  }
  return ok;
}

bool Executor::Rollback(std::uint32_t keep) {
  if (!verify_open_ || failed_ || keep == 0 || keep > verify_n_) {
    return false;
  }
  verify_open_ = false;
  if (keep == verify_n_) {
    return true;
  }
  const Config& c = config_;
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * 3 * c.KdaDim() * sizeof(float);
  const std::size_t recurrent = std::size_t{c.num_heads} * c.kda_head_dim *
                                c.kda_head_dim * sizeof(float);
  bool ok = true;
  for (std::uint32_t il = options_.first_layer; ok && il < options_.last_layer;
       ++il) {
    if (!c.IsKdaLayer(il)) {
      continue;
    }
    const KdaState& k = kda_[il];
    const std::size_t at = keep - 1;
    ok = hipMemcpyAsync(k.recurrent,
                        reinterpret_cast<const std::uint8_t*>(k.recurrent_ckpt) +
                            (at * recurrent),
                        recurrent, hipMemcpyDeviceToDevice,
                        stream_) == hipSuccess &&
         hipMemcpyAsync(k.conv,
                        reinterpret_cast<const std::uint8_t*>(k.conv_ckpt) +
                            (at * conv),
                        conv, hipMemcpyDeviceToDevice, stream_) == hipSuccess;
  }
  ok = ok && hipStreamSynchronize(stream_) == hipSuccess;
  if (!ok) {
    failed_ = true;
    return false;
  }
  position_ = verify_base_ + keep;
  return true;
}

bool Executor::ForwardNextn(std::span<const std::int32_t> tokens,
                            std::span<const float> h, bool want_logits,
                            ForwardResult* out, std::string* error,
                            bool all_rows) {
  const Config& c = config_;
  const std::uint32_t il = c.num_layers;
  const std::size_t hidden = c.hidden_size;
  if (!options_.nextn) {
    return Fail(error, "ForwardNextn needs ExecutorOptions::nextn");
  }
  if (failed_) {
    return Fail(error, "a previous call failed partway; call Reset()");
  }
  if (out == nullptr || h.size() != tokens.size() * hidden) {
    return Fail(error, "ForwardNextn: null result or h size mismatch");
  }
  *out = ForwardResult{};
  for (const std::int32_t id : tokens) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= c.vocab_size) {
      return Fail(error, "token id " + std::to_string(id) + " out of range");
    }
  }
  const auto n_tokens = static_cast<std::uint32_t>(tokens.size());
  if (std::uint64_t{nextn_position_} + n_tokens > mla_[il].capacity) {
    return Fail(error, "NextN MLA cache capacity exceeded");
  }
  const LayerDevice& d = layers_[il];
  const OpContext ctx{.stream = stream_,
                      .scratch = scratch_,
                      .scratch_bytes = scratch_bytes_,
                      .tap = nullptr,
                      .profiler = profiler_.get()};
  const auto download = [&](std::vector<float>& dst, const float* src,
                            std::uint32_t n, std::size_t width) {
    const std::size_t at = dst.size();
    dst.resize(at + (n * width));
    return hipMemcpyAsync(dst.data() + at, src, n * width * sizeof(float),
                          hipMemcpyDeviceToHost, stream_) == hipSuccess;
  };
  failed_ = true;
  for (std::uint32_t t0 = 0; t0 < n_tokens; t0 += options_.max_tokens) {
    const std::uint32_t n = std::min(options_.max_tokens, n_tokens - t0);
    const std::vector<std::uint32_t> ids(tokens.begin() + t0,
                                         tokens.begin() + t0 + n);
    // [enorm(embed) | hnorm(h)] rows in streams_ (hc_dim >= 2 * hidden).
    bool ok = hipMemcpyAsync(ids_, ids.data(), n * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, stream_) == hipSuccess &&
              hipMemcpyAsync(a_, h.data() + (std::size_t{t0} * hidden),
                             n * hidden * sizeof(float), hipMemcpyHostToDevice,
                             stream_) == hipSuccess;
    ::gufo::hip::LaunchBatchedEmbeddingLookup(embd_.data, embd_.type, ids_, x_,
                                              n, hidden, stream_);
    ::gufo::hip::LaunchBatchedRMSNorm(x_, enorm_, xn_, nullptr, n, hidden,
                                      c.rms_eps, stream_);
    ::gufo::hip::LaunchBatchedRMSNorm(a_, hnorm_, x_, nullptr, n, hidden,
                                      c.rms_eps, stream_);
    const std::size_t row = hidden * sizeof(float);
    ok = ok &&
         hipMemcpy2DAsync(streams_, 2 * row, xn_, row, row, n,
                          hipMemcpyDeviceToDevice, stream_) == hipSuccess &&
         hipMemcpy2DAsync(streams_ + hidden, 2 * row, x_, row, row, n,
                          hipMemcpyDeviceToDevice, stream_) == hipSuccess;
    // x_ = eh_proj(...) is the attention residual.
    ok = ok && LaunchProjection(eh_proj_, streams_, x_, n, stream_);
    ::gufo::hip::LaunchBatchedRMSNorm(x_, d.attn_norm, xn_, nullptr, n, hidden,
                                      c.rms_eps, stream_);
    ok = ok && ops_.mla(ctx, c, il, d.mla, mla_[il], xn_, n,
                        nextn_position_ + t0, a_);
    ::gufo::hip::LaunchResidualAdd(x_, a_, x_, n * hidden, stream_);
    ::gufo::hip::LaunchBatchedRMSNorm(x_, d.ffn_norm, xn_, nullptr, n, hidden,
                                      c.rms_eps, stream_);
    ok = ok && ops_.ffn(ctx, c, il, d.ffn, xn_, n, a_);
    ::gufo::hip::LaunchResidualAdd(x_, a_, x_, n * hidden, stream_);
    ::gufo::hip::LaunchBatchedRMSNorm(x_, nextn_head_norm_, xn_, nullptr, n,
                                      hidden, c.rms_eps, stream_);
    ok = ok && download(out->hidden, xn_, n, hidden);
    if (want_logits && !output_.empty() && all_rows) {
      ok = ok && LaunchProjection(output_, xn_, logits_, n, stream_) &&
           download(out->logits, logits_, n, c.vocab_size);
    } else if (want_logits && !output_.empty() && t0 + n == n_tokens) {
      ok = ok &&
           LaunchProjection(output_, xn_ + ((n - 1) * hidden), logits_, 1,
                            stream_) &&
           download(out->logits, logits_, 1, c.vocab_size);
    }
    ok = ok && hipStreamSynchronize(stream_) == hipSuccess &&
         hipPeekAtLastError() == hipSuccess;
    if (!ok) {
      return Fail(error, "NextN pass failed");
    }
  }
  nextn_position_ += n_tokens;
  failed_ = false;
  return true;
}

bool Executor::RewindNextn(std::uint32_t position) {
  if (position > nextn_position_ ||
      nextn_position_ - position > config_.indexer_kpool + 1) {
    return false;
  }
  nextn_position_ = position;
  return true;
}

bool Executor::ForwardLast(std::span<const std::int32_t> tokens,
                           bool want_logits, ForwardResult* out,
                           const TapFn* tap, std::string* error) {
  if (embd_.empty()) {
    return Fail(error,
                "ForwardLast needs the embedding (ExecutorOptions::embed)");
  }
  for (const std::int32_t id : tokens) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= config_.vocab_size) {
      return Fail(error, "token id " + std::to_string(id) + " out of range");
    }
  }
  return Run(tokens.data(), nullptr, static_cast<std::uint32_t>(tokens.size()),
             want_logits, Head::kLastRow, out, tap, error);
}

bool Executor::ForwardStreams(const float* streams, std::uint32_t n_tokens,
                              bool want_logits, ForwardResult* out,
                              const TapFn* tap, std::string* error) {
  if (streams == nullptr && n_tokens != 0) {
    return Fail(error, "null streams");
  }
  return Run(nullptr, streams, n_tokens, want_logits, Head::kPerToken, out, tap,
             error);
}

bool Executor::Run(const std::int32_t* tokens, const float* streams,
                   std::uint32_t n_tokens, bool want_logits, Head head,
                   ForwardResult* out, const TapFn* tap, std::string* error) {
  if (failed_) {
    return Fail(error, "a previous call failed partway; call Reset()");
  }
  if (out == nullptr) {
    return Fail(error, "null result");
  }
  *out = ForwardResult{};
  if (n_tokens == 0) {
    return true;
  }
  for (std::uint32_t il = options_.first_layer; il < options_.last_layer;
       ++il) {
    if (config_.IsMlaLayer(il) &&
        std::uint64_t{position_} + n_tokens > mla_[il].capacity) {
      return Fail(error,
                  "MLA cache capacity " + std::to_string(mla_[il].capacity) +
                      " exceeded at position " + std::to_string(position_));
    }
  }
  // Any other call ends the window in which Rollback can undo a verify.
  verify_open_ = false;
  // State is advanced pass by pass; a failure partway leaves it ahead of
  // position_, so the executor refuses further calls until Reset().
  failed_ = true;
  const std::size_t hc_dim = config_.HcDim();
  for (std::uint32_t t0 = 0; t0 < n_tokens; t0 += options_.max_tokens) {
    const std::uint32_t n = std::min(options_.max_tokens, n_tokens - t0);
    if (!Pass(tokens != nullptr ? tokens + t0 : nullptr,
              streams != nullptr ? streams + (t0 * hc_dim) : nullptr, n,
              want_logits, head, t0 + n == n_tokens, out, tap, error)) {
      return false;
    }
    position_ += n;
  }
  failed_ = false;
  if (profiler_ != nullptr && profiler_->per_run()) {
    PrintProfile(std::cerr);
  }
  return true;
}

bool Executor::Pass(const std::int32_t* tokens, const float* streams,
                    std::uint32_t n, bool want_logits, Head head,
                    bool last_pass, ForwardResult* out, const TapFn* tap,
                    std::string* error) {
  const Config& c = config_;
  const std::size_t hidden = c.hidden_size;
  const std::size_t hc_dim = c.HcDim();
  const std::size_t row = hidden * sizeof(float);
  // Spans open when a failure returns are dropped; declared first so it runs
  // after the scopes below have closed.
  struct PassGuard {
    OpProfiler* p;
    bool done{false};
    ~PassGuard() {
      if (p != nullptr && !done) {
        p->AbortPass();
      }
    }
  } guard{profiler_.get()};
  if (profiler_ != nullptr) {
    profiler_->BeginPass(n);
  }
  std::optional<OpScope> pass_scope;
  pass_scope.emplace(profiler_.get(), "pass", stream_);
  std::optional<OpScope> embed_scope;
  if (tokens != nullptr) {
    embed_scope.emplace(profiler_.get(), "embedding", stream_);
    // Four copies of the token embedding (HF:1493).
    const std::vector<std::uint32_t> ids(tokens, tokens + n);
    bool ok = hipMemcpyAsync(ids_, ids.data(), n * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, stream_) == hipSuccess;
    ::gufo::hip::LaunchBatchedEmbeddingLookup(embd_.data, embd_.type, ids_, x_,
                                              n, hidden, stream_);
    for (std::uint32_t s = 0; ok && s < c.hc_count; ++s) {
      ok = hipMemcpy2DAsync(streams_ + (s * hidden), hc_dim * sizeof(float), x_,
                            row, row, n, hipMemcpyDeviceToDevice,
                            stream_) == hipSuccess;
    }
    embed_scope.reset();
    if (!ok || hipStreamSynchronize(stream_) != hipSuccess) {
      return Fail(error, "embedding lookup failed");
    }
  } else if (hipMemcpyAsync(streams_, streams, n * hc_dim * sizeof(float),
                            hipMemcpyHostToDevice, stream_) != hipSuccess) {
    return Fail(error, "stream upload failed");
  }
  embed_scope.reset();

  for (std::uint32_t il = options_.first_layer; il < options_.last_layer;
       ++il) {
    if (!HalfLayer(il, /*ffn=*/false, n, position_, tap)) {
      return Fail(error, "layer " + std::to_string(il) + " " +
                             (c.IsKdaLayer(il) ? "KDA" : "MLA") +
                             " half-layer failed at position " +
                             std::to_string(position_));
    }
    if (!HalfLayer(il, /*ffn=*/true, n, position_, tap)) {
      return Fail(error, "layer " + std::to_string(il) +
                             " FFN half-layer failed at position " +
                             std::to_string(position_));
    }
  }

  const auto download = [&](std::vector<float>& dst, const float* src,
                            std::size_t width) {
    const std::size_t at = dst.size();
    dst.resize(at + (n * width));
    return hipMemcpyAsync(dst.data() + at, src, n * width * sizeof(float),
                          hipMemcpyDeviceToHost, stream_) == hipSuccess;
  };
  bool ok = true;
  // kLastRow needs the head only for the last pass's logits row (and for taps,
  // which see the whole head); it never downloads `hidden`.
  const bool last_row = head == Head::kLastRow;
  const bool need_logits = want_logits && !output_.empty();
  if (output_norm_ != nullptr &&
      (!last_row || (last_pass && need_logits) || (tap != nullptr && *tap))) {
    // Mean head, then the final RMSNorm (HF:1509) and the output projection.
    {
      const OpScope scope(profiler_.get(), "head_norm", stream_);
      ok = rocm::HcHeadMean(streams_, a_, n, c.hidden_size, stream_);
      Emit(tap, "hc_head.out", kNoLayer, a_, hidden, n);
      ::gufo::hip::LaunchBatchedRMSNorm(a_, output_norm_, xn_, nullptr, n,
                                        hidden, c.rms_eps, stream_);
      Emit(tap, "final_norm.out", kNoLayer, xn_, hidden, n);
    }
    if (!last_row) {
      const OpScope scope(profiler_.get(), "head_hidden_download", stream_);
      ok = ok && download(out->hidden, xn_, hidden);
    }
    if (need_logits && !last_row) {
      const OpScope scope(profiler_.get(), "head_logits", stream_);
      ok = ok && LaunchProjection(output_, xn_, logits_, n, stream_) &&
           download(out->logits, logits_, c.vocab_size);
    } else if (need_logits && last_pass) {
      // One row: the final token of the call, projected inside this pass.
      const OpScope scope(profiler_.get(), "head_logits", stream_);
      out->logits.resize(c.vocab_size);
      ok = ok &&
           LaunchProjection(output_, xn_ + ((n - 1) * hidden), logits_, 1,
                            stream_) &&
           hipMemcpyAsync(out->logits.data(), logits_,
                          c.vocab_size * sizeof(float), hipMemcpyDeviceToHost,
                          stream_) == hipSuccess;
    }
  } else if (output_norm_ == nullptr) {
    const OpScope scope(profiler_.get(), "streams_download", stream_);
    ok = download(out->streams, streams_, hc_dim);
  }
  pass_scope.reset();
  ok = ok && hipStreamSynchronize(stream_) == hipSuccess &&
       hipPeekAtLastError() == hipSuccess;
  if (profiler_ != nullptr && ok) {
    profiler_->EndPass();
    guard.done = true;
  }
  return ok || Fail(error, "head failed");
}

bool Executor::HalfLayer(std::uint32_t il, bool ffn, std::uint32_t n,
                         std::uint32_t pos0, const TapFn* tap) {
  // HcPre -> RMSNorm -> sub-layer -> HcPost (HF:1324-1346).
  static constexpr const char* kHcTaps[2][3] = {
      {"attn_hc.out0", "attn_hc.out1", "attn_hc.out2"},
      {"ffn_hc.out0", "ffn_hc.out1", "ffn_hc.out2"}};
  const Config& c = config_;
  const LayerDevice& d = layers_[il];
  const LayerDevice::Hc& w = ffn ? d.hc_ffn : d.hc_attn;
  const std::size_t hidden = c.hidden_size;
  const std::size_t hc = c.hc_count;
  const std::size_t mix_bytes = c.HcMixDim() * sizeof(float);
  const int k = ffn ? 1 : 0;
  Emit(tap, ffn ? "ffn_hc.in" : "in", il, streams_, hidden, hc, n);
  {
    const OpScope scope(profiler_.get(), "hc_pre", stream_);
    if (!rocm::HcPre(streams_, w.fn, w.base, w.scale, x_, split_, nullptr, n,
                     c.hidden_size, c.hc_sinkhorn_iters, c.hc_eps, c.rms_eps,
                     stream_)) {
      return false;
    }
  }
  if (tap != nullptr && *tap) {
    // post and comb are strided inside split [T][pre | post | comb].
    if (hipMemcpy2DAsync(tap_post_, hc * sizeof(float), split_ + hc, mix_bytes,
                         hc * sizeof(float), n, hipMemcpyDeviceToDevice,
                         stream_) != hipSuccess ||
        hipMemcpy2DAsync(tap_comb_, hc * hc * sizeof(float), split_ + (2 * hc),
                         mix_bytes, hc * hc * sizeof(float), n,
                         hipMemcpyDeviceToDevice, stream_) != hipSuccess) {
      return false;
    }
    Emit(tap, kHcTaps[k][0], il, tap_post_, hc, n);
    Emit(tap, kHcTaps[k][1], il, tap_comb_, hc * hc, n);
    Emit(tap, kHcTaps[k][2], il, x_, hidden, n);
  }
  {
    const OpScope scope(profiler_.get(), "norm", stream_);
    ::gufo::hip::LaunchBatchedRMSNorm(x_, ffn ? d.ffn_norm : d.attn_norm, xn_,
                                      nullptr, n, hidden, c.rms_eps, stream_);
  }
  Emit(tap, ffn ? "post_attention_layernorm.out" : "input_layernorm.out", il,
       xn_, hidden, n);
  const OpContext ctx{.stream = stream_,
                      .scratch = scratch_,
                      .scratch_bytes = scratch_bytes_,
                      .tap = tap,
                      .profiler = profiler_.get()};
  bool ok = false;
  if (ffn) {
    const OpScope scope(profiler_.get(), d.ffn.dense ? "ffn_dense" : "ffn_moe",
                        stream_);
    ok = ops_.ffn(ctx, c, il, d.ffn, xn_, n, a_);
  } else if (c.IsKdaLayer(il)) {
    const OpScope scope(profiler_.get(), "kda", stream_);
    ok = KdaMixerForward(ctx, c, il, d.kda, kda_[il], xn_, n, a_);
  } else {
    const OpScope scope(profiler_.get(), "mla", stream_);
    ok = ops_.mla(ctx, c, il, d.mla, mla_[il], xn_, n, pos0, a_);
  }
  if (!ok) {
    return false;
  }
  Emit(tap, ffn ? "mlp.out" : "self_attn.out", il, a_, hidden, n);
  {
    const OpScope scope(profiler_.get(), "hc_post", stream_);
    if (!rocm::HcPost(a_, streams_, split_, streams_, n, c.hidden_size,
                      stream_)) {
      return false;
    }
  }
  if (ffn) {
    Emit(tap, "out0", il, streams_, hidden, hc, n);
  }
  return hipPeekAtLastError() == hipSuccess;
}

void Executor::Emit(const TapFn* tap, std::string_view name,
                    std::uint32_t layer, const float* data, std::uint64_t ne0,
                    std::uint64_t ne1, std::uint64_t ne2) {
  if (tap == nullptr || !*tap) {
    return;
  }
  (void)hipStreamSynchronize(stream_);
  (*tap)(name, layer,
         TapView{.device = data,
                 .dtype = TapView::Dtype::kF32,
                 .ne0 = ne0,
                 .ne1 = ne1,
                 .ne2 = ne2});
}

}  // namespace gufo::models::glm53_flash
