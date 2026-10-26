#ifndef GUFO_TESTS_MODELS_LAGUNA_FFN_GPU_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_FFN_GPU_HPP_

// Shared pieces of the Laguna GPU FFN operator tests: random quantized layer
// fixtures (every weight random, sizes unequal), device helpers, and the
// comparison of an operator run against the phase 2 oracle.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/models/laguna/kernels/ffn.hpp"
#include "src/models/laguna/reference.hpp"
#include "src/models/laguna/weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "tests/models/laguna/support.hpp"

namespace gufo::testing::laguna {

namespace ml = gufo::models::laguna;
namespace ref = gufo::models::laguna::reference;
namespace gpu = gufo::models::laguna::rocm;
using core::GgmlType;

inline void CheckHip(hipError_t status, const char* what) {
  Require(status == hipSuccess,
          std::string(what) + ": " + hipGetErrorString(status));
}

template<class T>
T* ToDevice(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<class T>
std::vector<T> ToHost(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

inline std::uint16_t HalfBitsOf(float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

/// One stored tensor.
struct HostMat {
  GgmlType type{GgmlType::kF32};
  std::size_t cols{0};
  std::size_t rows{1};
  std::size_t experts{1};
  std::vector<std::uint8_t> bytes;
};

/// Random blocks of `type`: random quantized payloads with finite fp16 scales
/// sized so a unit-variance input gives outputs of order one. Any bit pattern
/// is a valid payload, so the oracle's decoder and the GPU decode the same
/// weights.
inline HostMat RandomQuant(GgmlType type, std::size_t cols, std::size_t rows,
                           std::size_t experts, std::mt19937_64& rng) {
  std::size_t block_elems = 256;
  std::size_t block_bytes = 0;
  float d_lo = 0.0F, d_hi = 0.0F;
  switch (type) {
    case GgmlType::kQ8_0:
      block_elems = 32;
      block_bytes = 34;
      d_lo = 4e-4F;
      d_hi = 8e-4F;
      break;
    case GgmlType::kQ6_K:
      block_bytes = 210;
      d_lo = 1.5e-4F;
      d_hi = 2.5e-4F;
      break;
    case GgmlType::kQ5_K:
      block_bytes = 176;
      d_lo = 6e-5F;
      d_hi = 1e-4F;
      break;
    case GgmlType::kIQ4_XS:
      block_bytes = 136;
      d_lo = 3e-5F;
      d_hi = 5e-5F;
      break;
    default:
      Require(false, "RandomQuant: unsupported type");
  }
  Require(cols % block_elems == 0, "RandomQuant: cols not block aligned");
  HostMat m;
  m.type = type;
  m.cols = cols;
  m.rows = rows;
  m.experts = experts;
  const std::size_t blocks = cols / block_elems * rows * experts;
  m.bytes.resize(blocks * block_bytes);
  std::uniform_real_distribution<float> d_dist(d_lo, d_hi);
  for (auto& byte : m.bytes) {
    byte = static_cast<std::uint8_t>(rng() & 0xFF);
  }
  const auto put_half = [&](std::uint8_t* p, float v) {
    const std::uint16_t bits = HalfBitsOf(v);
    std::memcpy(p, &bits, 2);
  };
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = m.bytes.data() + b * block_bytes;
    const float d = d_dist(rng);
    switch (type) {
      case GgmlType::kQ8_0:
        put_half(blk, d);
        break;
      case GgmlType::kQ6_K:
        put_half(blk + 208, d);
        for (int i = 0; i < 16; ++i) {  // int8 sub-block scales, |s| <= 24
          blk[192 + i] = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(static_cast<int>(rng() % 49) - 24));
        }
        break;
      case GgmlType::kQ5_K:
        put_half(blk, d);
        put_half(blk + 2, d * 15.5F);
        break;
      case GgmlType::kIQ4_XS:
        put_half(blk, d);
        break;
      default:
        break;
    }
  }
  return m;
}

inline HostMat RandomF32(std::size_t cols, std::size_t rows, float std_dev,
                         std::mt19937_64& rng) {
  HostMat m;
  m.cols = cols;
  m.rows = rows;
  std::normal_distribution<float> dist(0.0F, std_dev);
  std::vector<float> values(cols * rows);
  for (float& v : values) {
    v = dist(rng);
  }
  m.bytes.resize(values.size() * sizeof(float));
  std::memcpy(m.bytes.data(), values.data(), m.bytes.size());
  return m;
}

inline std::vector<float> F32Values(const HostMat& m) {
  std::vector<float> values(m.bytes.size() / sizeof(float));
  std::memcpy(values.data(), m.bytes.data(), m.bytes.size());
  return values;
}
inline void SetF32Values(HostMat* m, const std::vector<float>& values) {
  m->bytes.resize(values.size() * sizeof(float));
  std::memcpy(m->bytes.data(), values.data(), m->bytes.size());
}

/// A synthetic layer: config + tensors, owning its bytes.
class TinyLayer {
public:
  /// `routed` is the routed expert type (Q6_K, Q5_K, IQ4_XS); dense and
  /// shared matrices are Q8_0, as in the model.
  TinyLayer(GgmlType routed, bool moe, std::uint64_t seed) : rng_(seed) {
    config_.hidden_size = 512;
    config_.ffn_size = 768;
    config_.num_experts = 32;
    config_.num_experts_used = 10;
    config_.expert_ff = 256;
    config_.shared_expert_ff = 320;  // unequal to expert_ff on purpose
    config_.expert_weights_norm = true;
    config_.expert_weights_scale = 2.5F;
    layer_.moe = moe;
    const std::size_t H = config_.hidden_size;
    if (!moe) {
      gate_ = RandomQuant(GgmlType::kQ8_0, H, config_.ffn_size, 1, rng_);
      up_ = RandomQuant(GgmlType::kQ8_0, H, config_.ffn_size, 1, rng_);
      down_ = RandomQuant(GgmlType::kQ8_0, config_.ffn_size, H, 1, rng_);
      layer_.ffn_gate = Ref(gate_, "blk.0.ffn_gate.weight");
      layer_.ffn_up = Ref(up_, "blk.0.ffn_up.weight");
      layer_.ffn_down = Ref(down_, "blk.0.ffn_down.weight");
      return;
    }
    const std::size_t E = config_.num_experts;
    router_ = RandomF32(H, E, 3.0F / std::sqrt(static_cast<float>(H)), rng_);
    bias_ = RandomF32(E, 1, 0.25F, rng_);
    gate_ = RandomQuant(routed, H, config_.expert_ff, E, rng_);
    up_ = RandomQuant(routed, H, config_.expert_ff, E, rng_);
    down_ = RandomQuant(routed, config_.expert_ff, H, E, rng_);
    sgate_ = RandomQuant(GgmlType::kQ8_0, H, config_.shared_expert_ff, 1, rng_);
    sup_ = RandomQuant(GgmlType::kQ8_0, H, config_.shared_expert_ff, 1, rng_);
    sdown_ = RandomQuant(GgmlType::kQ8_0, config_.shared_expert_ff, H, 1, rng_);
    BindMoe();
  }
  TinyLayer(const TinyLayer&) = delete;
  TinyLayer& operator=(const TinyLayer&) = delete;

  [[nodiscard]] const ml::Config& config() const { return config_; }
  [[nodiscard]] const ml::LayerWeights& layer() const { return layer_; }
  [[nodiscard]] std::mt19937_64& rng() { return rng_; }
  [[nodiscard]] HostMat& router() { return router_; }
  [[nodiscard]] HostMat& bias() { return bias_; }
  /// Re-points the oracle's view after router/bias bytes changed.
  void BindMoe() {
    layer_.ffn_gate_inp = Ref(router_, "blk.1.ffn_gate_inp.weight");
    layer_.ffn_exp_probs_b = Ref(bias_, "blk.1.exp_probs_b.bias");
    layer_.ffn_gate_exps = Ref(gate_, "blk.1.ffn_gate_exps.weight");
    layer_.ffn_up_exps = Ref(up_, "blk.1.ffn_up_exps.weight");
    layer_.ffn_down_exps = Ref(down_, "blk.1.ffn_down_exps.weight");
    layer_.ffn_gate_shexp = Ref(sgate_, "blk.1.ffn_gate_shexp.weight");
    layer_.ffn_up_shexp = Ref(sup_, "blk.1.ffn_up_shexp.weight");
    layer_.ffn_down_shexp = Ref(sdown_, "blk.1.ffn_down_shexp.weight");
  }

private:
  ml::TensorRef Ref(const HostMat& m, const std::string& name) {
    names_.push_back(name);
    ml::TensorRef t;
    t.data = m.bytes.data();
    t.type = m.type;
    t.cols = m.cols;
    t.rows = m.rows;
    t.experts = m.experts;
    t.name = names_.back();
    Require(t.SizeBytes() == m.bytes.size(), "fixture byte size mismatch");
    return t;
  }
  std::mt19937_64 rng_;
  ml::Config config_;
  ml::LayerWeights layer_;
  HostMat router_, bias_, gate_, up_, down_, sgate_, sup_, sdown_;
  std::deque<std::string> names_;
};

/// Activation rows like an RMSNorm output: N(0,1) with a few large channels
/// (outliers stress the Q8 block scale).
inline std::vector<float> RandomRows(std::size_t rows, std::size_t hidden,
                                     std::mt19937_64& rng) {
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> x(rows * hidden);
  for (std::size_t t = 0; t < rows; ++t) {
    for (std::size_t i = 0; i < hidden; ++i) {
      float v = dist(rng);
      if (i % 61 == 7) {
        v *= 6.0F;
      }
      x[t * hidden + i] = v;
    }
  }
  return x;
}

/// Oracle result with the routing taps it emitted.
struct OracleRun {
  std::vector<float> out;
  std::vector<std::int32_t> topk;   // [T][used]
  std::vector<float> weights;       // [T][used], scale included
  std::vector<float> biased_probs;  // [T][E]
  std::vector<float> logits;        // [T][E]
};

inline OracleRun RunOracle(const ml::Config& config,
                           const ml::LayerWeights& layer,
                           const std::vector<float>& x, std::size_t tokens,
                           ref::ActivationQuant mode) {
  ref::Options options;
  options.activation_quant = mode;
  OracleRun run;
  const auto tap = [&](const ref::Tap& t) {
    const auto copy_f32 = [&](std::vector<float>* dst) {
      dst->assign(static_cast<const float*>(t.data),
                  static_cast<const float*>(t.data) + t.Count());
    };
    if (t.name.rfind("ffn_moe_topk", 0) == 0) {
      run.topk.assign(static_cast<const std::int32_t*>(t.data),
                      static_cast<const std::int32_t*>(t.data) + t.Count());
    } else if (t.name.rfind("ffn_moe_weights_scaled", 0) == 0) {
      copy_f32(&run.weights);
    } else if (t.name.rfind("ffn_moe_probs_biased", 0) == 0) {
      copy_f32(&run.biased_probs);
    } else if (t.name.rfind("ffn_moe_logits", 0) == 0) {
      copy_f32(&run.logits);
    }
  };
  run.out = ref::FfnForward(config, layer, x.data(), tokens, options, tap);
  return run;
}

/// Smallest gap between the last selected and the first rejected biased score
/// over the tokens (the router's decision margin). 0 means an exact tie at the
/// selection boundary.
inline double SelectionMargin(const OracleRun& run, std::size_t tokens,
                              std::size_t experts, std::size_t used) {
  double margin = 1e30;
  for (std::size_t t = 0; t < tokens; ++t) {
    std::vector<float> s(run.biased_probs.begin() + t * experts,
                         run.biased_probs.begin() + (t + 1) * experts);
    std::sort(s.begin(), s.end(), std::greater<float>());
    margin = std::min(margin, static_cast<double>(s[used - 1]) - s[used]);
  }
  return margin;
}

struct ErrorStats {
  double max_abs{0};
  double ref_rms{0};
  double max_over_rms{0};  ///< max_abs / ref_rms
  std::size_t index{0};
};

inline ErrorStats Compare(const std::vector<float>& got,
                          const std::vector<float>& want) {
  Require(got.size() == want.size(), "Compare: size mismatch");
  ErrorStats s;
  double sq = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    Require(std::isfinite(got[i]), "GPU output is not finite");
    const double e = std::fabs(static_cast<double>(got[i]) - want[i]);
    if (e > s.max_abs) {
      s.max_abs = e;
      s.index = i;
    }
    sq += static_cast<double>(want[i]) * want[i];
  }
  s.ref_rms = std::sqrt(sq / static_cast<double>(got.size()));
  s.max_over_rms = s.ref_rms > 0 ? s.max_abs / s.ref_rms : s.max_abs;
  return s;
}

inline const char* PathName(gpu::Path p) {
  return p == gpu::Path::kVector    ? "vector"
         : p == gpu::Path::kBatched ? "batched"
                                    : "auto";
}

/// Owns a stream and the operator bound to it.
class GpuContext {
public:
  GpuContext() {
    CheckHip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
             "stream");
    ops_ = std::make_unique<gpu::FfnOps>(stream_);
  }
  ~GpuContext() {
    ops_.reset();
    (void)hipStreamSynchronize(stream_);
    qfn_mmq_release_stream(stream_);
    (void)hipStreamDestroy(stream_);
  }
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;
  [[nodiscard]] hipStream_t stream() const { return stream_; }
  [[nodiscard]] gpu::FfnOps& ops() { return *ops_; }

private:
  hipStream_t stream_{};
  std::unique_ptr<gpu::FfnOps> ops_;
};

// ---------------------------------------------------------------------------
// Stage-by-stage and end-to-end checks of one operator call.
// ---------------------------------------------------------------------------

/// Which activation rounding the GPU path applies: the vector (MMVQ) kernels
/// quantize rows to Q8_1 blocks; the batched (MMQ tile) kernels use the MMQ
/// layout. See ref::ActivationQuant.
inline bool UsesVector(gpu::Path path, std::uint32_t tokens) {
  return path == gpu::Path::kVector ||
         (path == gpu::Path::kAuto && tokens <= gpu::kVectorMaxTokens);
}
inline ref::ActivationQuant ModeFor(bool vector) {
  return vector ? ref::ActivationQuant::kGpuQ8_1
                : ref::ActivationQuant::kGpuMmq;
}

/// One buffer as qfn_mmq handed it to the capture hook.
struct Capture {
  int layout{QFN_MMQ_ACT_Q8_1};
  std::size_t rows{0};
  std::size_t k{0};
  std::size_t k_padded{0};
  std::vector<std::uint8_t> bytes;
  std::vector<std::int32_t> src;  ///< input row per buffer row; empty: same
  std::vector<std::int32_t> dst;  ///< output row per buffer row; empty: none
};

struct Taps {
  std::map<std::string, std::vector<float>> f32;
  std::map<std::string, std::vector<std::int32_t>> i32;
  /// Activation buffers captured since the previous tap, then filed under
  /// the next stage the operator emits (the stage whose projection read
  /// them: "gate" for gate/up, "down", "<prefix>_gate", "<prefix>_out").
  std::vector<Capture> pending;
  std::map<std::string, std::vector<Capture>> captures;
  std::string capture_error;
  const std::vector<float>& F(const std::string& name) const {
    const auto it = f32.find(name);
    Require(it != f32.end(), "missing tap " + name);
    return it->second;
  }
  const std::vector<std::int32_t>& I(const std::string& name) const {
    const auto it = i32.find(name);
    Require(it != i32.end(), "missing tap " + name);
    return it->second;
  }
  const std::vector<Capture>& C(const std::string& name) const {
    static const std::vector<Capture> kNone;
    const auto it = captures.find(name);
    return it != captures.end() ? it->second : kNone;
  }
};

/// qfn_mmq capture hook: copies the packed buffer (and its row maps) while
/// qfn_mmq still holds it. Errors are recorded, not thrown through qfn_mmq.
inline void OnActivationCapture(void* user, int layout, const void* data,
                                std::size_t bytes, int rows, int k,
                                int k_padded, const std::int32_t* row_src,
                                const std::int32_t* row_dst,
                                hipStream_t stream) {
  auto* taps = static_cast<Taps*>(user);
  const auto ok = [&](hipError_t e) {
    if (e != hipSuccess && taps->capture_error.empty()) {
      taps->capture_error = hipGetErrorString(e);
    }
    return e == hipSuccess;
  };
  Capture c;
  c.layout = layout;
  c.rows = static_cast<std::size_t>(rows);
  c.k = static_cast<std::size_t>(k);
  c.k_padded = static_cast<std::size_t>(k_padded);
  c.bytes.resize(bytes);
  if (!ok(hipStreamSynchronize(stream)) ||
      !ok(hipMemcpy(c.bytes.data(), data, bytes, hipMemcpyDeviceToHost))) {
    return;
  }
  for (const auto& [device, host] :
       {std::pair{row_src, &c.src}, std::pair{row_dst, &c.dst}}) {
    if (device != nullptr) {
      host->resize(c.rows);
      if (!ok(hipMemcpy(host->data(), device, c.rows * sizeof(std::int32_t),
                        hipMemcpyDeviceToHost))) {
        return;
      }
    }
  }
  taps->pending.push_back(std::move(c));
}

/// Taps every stage of `ops` into `taps` and captures the activation buffers
/// qfn_mmq quantizes meanwhile. Remove with RemoveTap.
inline void InstallTap(gpu::FfnOps& ops, Taps* taps) {
  qfn_mmq_set_activation_capture(OnActivationCapture, taps);
  ops.SetTap([taps](std::string_view name, const void* device,
                    std::size_t count, bool is_int32) {
    const std::string key(name);
    if (is_int32) {
      taps->i32[key] = ToHost(static_cast<const std::int32_t*>(device), count);
    } else {
      taps->f32[key] = ToHost(static_cast<const float*>(device), count);
    }
    auto& filed = taps->captures[key];
    for (auto& c : taps->pending) {
      filed.push_back(std::move(c));
    }
    taps->pending.clear();
  });
}

inline void RemoveTap(gpu::FfnOps& ops) {
  ops.SetTap({});
  qfn_mmq_set_activation_capture(nullptr, nullptr);
}

/// Distance (in Q8 codes) that AvoidTies keeps the elements of a test input
/// from a rounding boundary of their first quantization, so the end-to-end
/// comparison against ref::FfnForward is not dominated by a flip in the first
/// projection. The stage checks do not rely on it: they consume the GPU's own
/// quantized activations (CheckCapturedProjection).
inline constexpr double kTieMargin = 4e-5;

/// Elements of `row` (length `cols`) whose x * 127 / amax lies within `margin`
/// of a rounding boundary of their 32-element block.
inline std::vector<std::size_t> NearTies(const float* row, std::size_t cols,
                                         double margin) {
  std::vector<std::size_t> out;
  for (std::size_t b = 0; b < cols; b += 32) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(row[b + j]));
    }
    if (amax == 0.0F) {
      continue;
    }
    const double scale = 127.0 / amax;
    for (std::size_t j = 0; j < 32; ++j) {
      const double v = static_cast<double>(row[b + j]) * scale;
      const double frac = std::fabs(v) - std::floor(std::fabs(v));
      if (std::fabs(frac - 0.5) < margin) {
        out.push_back(b + j);
      }
    }
  }
  return out;
}

/// Moves every element of the rows `x` ([rows][cols]) that lies within
/// 4 * kTieMargin of a Q8 rounding boundary a few percent of a code away from
/// it, so the first quantization of a test input is unambiguous (intermediate
/// rows are produced by the operator and cannot be treated this way).
inline void AvoidTies(std::vector<float>* x, std::size_t rows,
                      std::size_t cols) {
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = x->data() + r * cols;
    for (const std::size_t j : NearTies(row, cols, 4.0 * kTieMargin)) {
      const std::size_t b = j / 32 * 32;
      float amax = 0.0F;
      for (std::size_t k = 0; k < 32; ++k) {
        amax = std::max(amax, std::fabs(row[b + k]));
      }
      const double step = amax / 127.0;
      const double v = static_cast<double>(row[j]) / step;
      const double lo = std::floor(std::fabs(v));
      row[j] = static_cast<float>((v < 0 ? -1 : 1) * (lo + 0.5 + 0.05) * step);
    }
  }
}

// ---------------------------------------------------------------------------
// Captured activations: the packed buffer each GPU projection actually read
// (qfn_mmq_set_activation_capture), decoded, validated against the float
// input it was quantized from, then used as the exact input of the oracle's
// matrix product.
// ---------------------------------------------------------------------------

/// A decoded capture: codes [rows][k], and per 32-element group the stored
/// scale d and (Q8_1, MMQ DS4) the stored sum d * sum(q), as float values.
struct QuantRows {
  int layout{QFN_MMQ_ACT_Q8_1};
  std::size_t rows{0};
  std::size_t k{0};
  bool has_sum{false};
  std::vector<std::int8_t> q;
  std::vector<float> d;
  std::vector<float> s;
  std::vector<std::int32_t> src;
  std::vector<std::int32_t> dst;
};

inline const char* LayoutName(int layout) {
  switch (layout) {
    case QFN_MMQ_ACT_Q8_1:
      return "Q8_1";
    case QFN_MMQ_ACT_MMQ_D4:
      return "MMQ D4";
    case QFN_MMQ_ACT_MMQ_DS4:
      return "MMQ DS4";
    default:
      return "MMQ D2S6";
  }
}

/// The layout each path's projection of `type` weights quantizes into
/// (mmq_get_q8_1_ds_layout for the MMQ tile kernels).
inline int ExpectedLayout(GgmlType type, bool vector) {
  if (vector) {
    return QFN_MMQ_ACT_Q8_1;
  }
  return type == GgmlType::kQ4_K || type == GgmlType::kQ5_K
             ? QFN_MMQ_ACT_MMQ_DS4
             : QFN_MMQ_ACT_MMQ_D4;
}

inline float HalfAt(const std::uint8_t* p) {
  _Float16 h;
  std::memcpy(&h, p, sizeof(h));
  return static_cast<float>(h);
}

/// Round to nearest even fp16, from a double (no double rounding via float).
inline float RoundHalf(double v) {
  return static_cast<float>(static_cast<_Float16>(v));
}

inline QuantRows DecodeCapture(const Capture& c) {
  Require(c.k % 32 == 0 && c.k <= c.k_padded && c.rows > 0,
          "capture: bad shape");
  QuantRows out;
  out.layout = c.layout;
  out.rows = c.rows;
  out.k = c.k;
  out.src = c.src;
  out.dst = c.dst;
  const std::size_t groups = c.k / 32;
  out.q.resize(c.rows * c.k);
  out.d.resize(c.rows * groups);
  out.s.assign(c.rows * groups, std::nanf(""));
  if (c.layout == QFN_MMQ_ACT_Q8_1) {
    // block_q8_1: half2 {d, d * sum(q)}, then 32 int8 codes.
    out.has_sum = true;
    const std::size_t row_blocks = c.k_padded / 32;
    Require(c.bytes.size() >= c.rows * row_blocks * 36,
            "capture: Q8_1 buffer too small");
    for (std::size_t r = 0; r < c.rows; ++r) {
      for (std::size_t g = 0; g < groups; ++g) {
        const std::uint8_t* p = c.bytes.data() + (r * row_blocks + g) * 36;
        out.d[r * groups + g] = HalfAt(p);
        out.s[r * groups + g] = HalfAt(p + 2);
        std::memcpy(out.q.data() + r * c.k + g * 32, p + 4, 32);
      }
    }
    return out;
  }
  Require(c.layout == QFN_MMQ_ACT_MMQ_D4 || c.layout == QFN_MMQ_ACT_MMQ_DS4,
          std::string("capture: unsupported layout ") + LayoutName(c.layout));
  // block_q8_1_mmq: 16 scale bytes (D4: float d[4]; DS4: half2 {d, sum}[4]),
  // then 128 codes; block (column / 128) * rows + row.
  out.has_sum = c.layout == QFN_MMQ_ACT_MMQ_DS4;
  Require(c.k_padded % 128 == 0 &&
              c.bytes.size() >= c.k_padded / 128 * c.rows * 144,
          "capture: MMQ buffer too small");
  for (std::size_t r = 0; r < c.rows; ++r) {
    for (std::size_t g = 0; g < groups; ++g) {
      const std::size_t col = g * 32;
      const std::uint8_t* p = c.bytes.data() + (col / 128 * c.rows + r) * 144;
      const std::size_t gi = col % 128 / 32;
      if (out.has_sum) {
        out.d[r * groups + g] = HalfAt(p + 4 * gi);
        out.s[r * groups + g] = HalfAt(p + 4 * gi + 2);
      } else {
        std::memcpy(&out.d[r * groups + g], p + 4 * gi, 4);
      }
      std::memcpy(out.q.data() + r * c.k + col, p + 16 + col % 128, 32);
    }
  }
  return out;
}

/// Relative error bounds of the GPU quantizers, in units of u = 2^-24 (float
/// unit roundoff), per layout. Both kernels are built with -ffast-math, so a
/// division may be a v_rcp_f32 reciprocal (at most 1 ulp = 2u, RDNA3 ISA)
/// times a correctly rounded multiply (u). amax is exact (fabs/max).
///
///  Q8_1 (quantize_q8_1, MMVQ decode):
///    d = amax / 127            -> rcp + mul                         3u
///    q = roundf(x / d) = x * rcp(d): multiplier 1/d off by 3u + 2u  5u
///      plus the product's rounding                              -> 6u
///    stored d  = half(d);  stored sum = half(d * sum(q)): 3u + u    4u
///  MMQ D4/DS4 (quantize_mmq_q8_1, prefill tiles):
///    d_inv = 127 / amax        -> rcp + mul                         3u
///    q = roundf(x * d_inv)     -> plus the product's rounding       4u
///    d = 1 / d_inv             -> 3u + rcp 2u                       5u
///    D4 stores that float d; DS4 stores half(d) and half(d * sum(q)) 6u
///
/// Each bound below adds 1u for second-order terms and the double evaluation
/// of the exact quantities. `multiplier` is the code bound without the
/// product's own rounding (which differs per element).
struct QuantBounds {
  double code;        ///< |x * m - x * 127 / amax| / |x * 127 / amax|
  double multiplier;  ///< |m - 127 / amax| / (127 / amax), m shared per block
  double scale;       ///< |d_unrounded - amax / 127| / (amax / 127)
  double sum;         ///< |sum_unrounded - amax / 127 * sum(q)| / same
  bool half_scale;    ///< d is stored as fp16 (else float)
};

inline QuantBounds BoundsFor(int layout) {
  constexpr double u = 0x1p-24;
  if (layout == QFN_MMQ_ACT_Q8_1) {
    return {7 * u, 6 * u, 4 * u, 5 * u, true};
  }
  return {5 * u, 4 * u, 6 * u, 7 * u, layout == QFN_MMQ_ACT_MMQ_DS4};
}

/// What validating captured quantizations found. Raw statistics are kept for
/// every block, not only failing ones.
struct QuantStats {
  std::size_t blocks{0};
  std::size_t boundary_codes{0};  ///< codes != round(x * 127 / amax)
  double worst_boundary_gap{0};   ///< max |frac(|v|) - 0.5| of those codes
  double max_scale_rel{0};        ///< max |d_stored - amax/127| / (amax/127)
  std::size_t violations{0};
  std::string first;  ///< first violation, for the failure message
};

inline void Violation(QuantStats* st, const std::string& why) {
  if (st->violations++ == 0) {
    st->first = why;
  }
}

/// Checks one 32-element block of captured codes `q`, stored scale `d` and,
/// when `has_sum`, stored sum `s` against the float input `x` it was
/// quantized from. A code is accepted only if (1) it is within the derived
/// error of round(x * 127 / amax): a code other than the nearest one needs
/// x * 127 / amax within `code` (relative) of a .5 boundary; (2) equal
/// inputs have equal codes and larger inputs no smaller codes (one shared
/// multiplier and a monotone rounding); and (3) one multiplier within the
/// derived interval reproduces every code of the block jointly. The scale
/// and sum must be the fp16 (or float) rounding of a value within the derived
/// error of the unrounded amax / 127 and amax / 127 * sum(q).
inline void ValidateBlock(const float* x, const std::int8_t* q, float d,
                          float s, bool has_sum, int layout, QuantStats* st) {
  const QuantBounds b = BoundsFor(layout);
  constexpr double u = 0x1p-24;
  ++st->blocks;
  float amax = 0.0F;
  for (int j = 0; j < 32; ++j) {
    amax = std::max(amax, std::fabs(x[j]));
  }
  std::ostringstream why;
  why << LayoutName(layout) << " block (amax " << amax << "): ";
  if (amax == 0.0F) {
    bool ok = d == 0.0F && (!has_sum || s == 0.0F);
    for (int j = 0; j < 32; ++j) {
      ok = ok && q[j] == 0;
    }
    if (!ok) {
      why << "an all-zero block must have zero codes, scale and sum";
      Violation(st, why.str());
    }
    return;
  }
  const double t = static_cast<double>(amax) / 127.0;
  const double scale = 127.0 / static_cast<double>(amax);
  int sumq = 0;
  double rho_lo = 1.0 - b.multiplier;
  double rho_hi = 1.0 + b.multiplier;
  for (int j = 0; j < 32; ++j) {
    const double v = static_cast<double>(x[j]) * scale;
    const double lo = std::round(v - std::fabs(v) * b.code);
    const double hi = std::round(v + std::fabs(v) * b.code);
    sumq += q[j];
    if (q[j] < lo || q[j] > hi) {
      why << "element " << j << " x=" << x[j] << " (x*127/amax=" << v
          << ") has code " << int{q[j]} << ", outside [" << lo << ", " << hi
          << "]";
      Violation(st, why.str());
      return;
    }
    if (q[j] != std::round(v)) {
      ++st->boundary_codes;
      const double av = std::fabs(v);
      st->worst_boundary_gap = std::max(st->worst_boundary_gap,
                                        std::fabs(av - std::floor(av) - 0.5));
    }
    if (v != 0.0) {
      // |x| * m * (1 + delta), |delta| <= u, must round to |q|.
      const double av = std::fabs(v);
      const double aq = std::abs(int{q[j]});
      rho_lo = std::max(rho_lo, (aq - 0.5) / (av * (1.0 + u)));
      rho_hi = std::min(rho_hi, (aq + 0.5) / (av * (1.0 - u)));
    }
  }
  std::array<int, 32> order{};
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int c) { return x[a] < x[c]; });
  for (int i = 1; i < 32; ++i) {
    const int a = order[i - 1];
    const int c = order[i];
    if ((x[a] == x[c] && q[a] != q[c]) || (x[a] < x[c] && q[a] > q[c])) {
      why << "elements " << a << " and " << c << " (x=" << x[a] << ", " << x[c]
          << ") have codes " << int{q[a]} << ", " << int{q[c]}
          << ": no shared monotone rounding gives these";
      Violation(st, why.str());
      return;
    }
  }
  // roundf of x * m with one positive m is odd-symmetric: q(-x) = -q(x). So a
  // code never has the opposite sign of its input, and equal magnitudes get
  // equal code magnitudes (larger magnitudes no smaller ones).
  for (int j = 0; j < 32; ++j) {
    if ((x[j] > 0.0F && q[j] < 0) || (x[j] < 0.0F && q[j] > 0) ||
        (x[j] == 0.0F && q[j] != 0)) {
      why << "element " << j << " x=" << x[j] << " has code " << int{q[j]}
          << " of the opposite sign";
      Violation(st, why.str());
      return;
    }
  }
  std::sort(order.begin(), order.end(),
            [&](int a, int c) { return std::fabs(x[a]) < std::fabs(x[c]); });
  for (int i = 1; i < 32; ++i) {
    const int a = order[i - 1];
    const int c = order[i];
    const int qa = std::abs(int{q[a]});
    const int qc = std::abs(int{q[c]});
    const float xa = std::fabs(x[a]);
    const float xc = std::fabs(x[c]);
    if ((xa == xc && qa != qc) || (xa < xc && qa > qc)) {
      why << "elements " << a << " and " << c << " (x=" << x[a] << ", " << x[c]
          << ") have codes " << int{q[a]} << ", " << int{q[c]}
          << ": rounding with one positive multiplier is odd-symmetric";
      Violation(st, why.str());
      return;
    }
  }
  if (rho_lo > rho_hi) {
    why << "no single multiplier within " << b.multiplier / u
        << "u of 127/amax reproduces every code of the block";
    Violation(st, why.str());
    return;
  }
  st->max_scale_rel = std::max(st->max_scale_rel, std::fabs(d - t) / t);
  const double d_lo = t * (1.0 - b.scale);
  const double d_hi = t * (1.0 + b.scale);
  const bool d_ok = b.half_scale ? d >= RoundHalf(d_lo) && d <= RoundHalf(d_hi)
                                 : d >= d_lo && d <= d_hi;
  if (!d_ok) {
    why << "stored scale " << std::setprecision(10) << d
        << " is not the rounding of amax/127 = " << t << " within "
        << b.scale / u << "u";
    Violation(st, why.str());
    return;
  }
  if (has_sum) {
    const double ts = t * sumq;
    const double e = std::fabs(ts) * b.sum;
    const bool s_ok = sumq == 0
                          ? s == 0.0F
                          : s >= RoundHalf(ts - e) && s <= RoundHalf(ts + e);
    if (!s_ok) {
      why << "stored sum " << std::setprecision(10) << s
          << " is not the fp16 rounding of amax/127 * sum(q) = " << ts
          << " within " << b.sum / u << "u";
      Violation(st, why.str());
    }
  }
}

/// Validates every row of `qr` against the float rows `in` ([in_rows][k]):
/// buffer row i was quantized from input row src[i] (or i).
inline QuantStats ValidateQuantRows(const QuantRows& qr,
                                    const std::vector<float>& in,
                                    std::size_t in_rows) {
  Require(in.size() == in_rows * qr.k, "ValidateQuantRows: shape mismatch");
  QuantStats st;
  const std::size_t groups = qr.k / 32;
  for (std::size_t i = 0; i < qr.rows; ++i) {
    const std::size_t src =
        qr.src.empty() ? i : static_cast<std::size_t>(qr.src[i]);
    if (src >= in_rows) {
      Violation(&st, "capture row maps to a missing input row");
      continue;
    }
    for (std::size_t g = 0; g < groups; ++g) {
      ValidateBlock(in.data() + src * qr.k + g * 32,
                    qr.q.data() + i * qr.k + g * 32, qr.d[i * groups + g],
                    qr.s[i * groups + g], qr.has_sum, qr.layout, &st);
    }
  }
  return st;
}

/// Projections that share one input must have consumed one quantization of
/// it: every buffer row (of every listed capture) quantized from the same
/// input row must hold the same codes, scales and sums, bit for bit. Returns
/// the empty string when they agree.
inline std::string SharedInputMismatch(
    const std::vector<const QuantRows*>& captures) {
  std::map<std::int32_t, std::pair<const QuantRows*, std::size_t>> first;
  for (const QuantRows* qr : captures) {
    const std::size_t groups = qr->k / 32;
    for (std::size_t i = 0; i < qr->rows; ++i) {
      const std::int32_t src =
          qr->src.empty() ? static_cast<std::int32_t>(i) : qr->src[i];
      const auto [it, inserted] = first.try_emplace(src, qr, i);
      if (inserted) {
        continue;
      }
      const QuantRows* a = it->second.first;
      const std::size_t ai = it->second.second;
      if (a->layout != qr->layout || a->k != qr->k ||
          std::memcmp(a->q.data() + ai * a->k, qr->q.data() + i * qr->k,
                      qr->k) != 0 ||
          std::memcmp(a->d.data() + ai * groups, qr->d.data() + i * groups,
                      groups * sizeof(float)) != 0 ||
          (qr->has_sum &&
           std::memcmp(a->s.data() + ai * groups, qr->s.data() + i * groups,
                       groups * sizeof(float)) != 0)) {
        return "input row " + std::to_string(src) +
               " was quantized differently for projections that share it";
      }
    }
  }
  return {};
}

template<class Fn>
void ParallelFor(std::size_t count, const Fn& fn) {
  const std::size_t workers = std::min<std::size_t>(
      std::max(1U, std::thread::hardware_concurrency()), count);
  std::atomic<std::size_t> next{0};
  const auto work = [&] {
    for (std::size_t i = next++; i < count; i = next++) {
      fn(i);
    }
  };
  std::vector<std::thread> pool;
  for (std::size_t i = 1; i < workers; ++i) {
    pool.emplace_back(work);
  }
  work();
  for (auto& t : pool) {
    t.join();
  }
}

/// Oracle product of captured activations: y[n][m] = W[expert][m] . x'_n in
/// double, where x'_n is buffer row rows[n] with x' = q * d per group (exact
/// in double), and sabs[n][m] the sum of the magnitudes of the terms. MMQ
/// tiles for Q4_K/Q5_K weights (DS4 layout) are modelled as the tile kernel
/// computes them: per 32-element sub-block, x_dm.x * d8 * sum(qw * qx) +
/// x_dm.y * s8 with x_dm = half2(d * sc, -dmin * m) and the captured fp16 d8
/// and s8 (see ref::MatMulRows, kGpuMmq).
inline void CapturedProduct(const ml::TensorRef& w, std::uint64_t expert,
                            const QuantRows& qr,
                            const std::vector<std::size_t>& rows, double* y,
                            double* sabs) {
  const std::size_t K = w.cols;
  const std::size_t M = w.rows;
  const std::size_t groups = K / 32;
  const std::uint8_t* base = w.Expert(expert);
  const std::size_t row_bytes = w.RowBytes();
  const bool kquant_tiles =
      qr.layout == QFN_MMQ_ACT_MMQ_DS4 &&
      (w.type == GgmlType::kQ4_K || w.type == GgmlType::kQ5_K);
  if (qr.layout == QFN_MMQ_ACT_MMQ_DS4) {
    Require(kquant_tiles, "CapturedProduct: DS4 activations need Q4_K/Q5_K");
  }
  if (kquant_tiles) {
    const bool q5 = w.type == GgmlType::kQ5_K;
    const std::size_t block_bytes = q5 ? 176 : 144;
    ParallelFor(M, [&](std::size_t m) {
      for (std::size_t n = 0; n < rows.size(); ++n) {
        const std::size_t r = rows[n];
        double acc = 0.0;
        double mag = 0.0;
        for (std::size_t b = 0; b < K / 256; ++b) {
          const std::uint8_t* blk = base + m * row_bytes + b * block_bytes;
          const float dw = HalfAt(blk);
          const float dmin = HalfAt(blk + 2);
          const std::uint8_t* scales = blk + 4;
          const std::uint8_t* qh = q5 ? blk + 16 : nullptr;
          const std::uint8_t* qs = blk + (q5 ? 48 : 16);
          for (int j = 0; j < 8; ++j) {
            int sc = 0;
            int mn = 0;
            if (j < 4) {
              sc = scales[j] & 63;
              mn = scales[j + 4] & 63;
            } else {
              sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
              mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }
            const double dm_x = RoundHalf(dw * static_cast<float>(sc));
            const double dm_y = -RoundHalf(dmin * static_cast<float>(mn));
            long dot = 0;
            const std::int8_t* qx = qr.q.data() + r * K + b * 256 + j * 32;
            for (int l = 0; l < 32; ++l) {
              const std::uint8_t byte = qs[32 * (j / 2) + l];
              int qw = (j & 1) != 0 ? (byte >> 4) : (byte & 0xF);
              if (q5 && ((qh[l] >> j) & 1) != 0) {
                qw += 16;
              }
              dot += static_cast<long>(qw) * qx[l];
            }
            const std::size_t g = b * 8 + j;
            const double t1 =
                dm_x * qr.d[r * groups + g] * static_cast<double>(dot);
            const double t2 = dm_y * qr.s[r * groups + g];
            acc += t1 + t2;
            mag += std::fabs(t1) + std::fabs(t2);
          }
        }
        y[n * M + m] = acc;
        sabs[n * M + m] = mag;
      }
    });
    return;
  }
  std::vector<double> xq(rows.size() * K);
  for (std::size_t n = 0; n < rows.size(); ++n) {
    for (std::size_t c = 0; c < K; ++c) {
      xq[n * K + c] = static_cast<double>(qr.q[rows[n] * K + c]) *
                      qr.d[rows[n] * groups + c / 32];
    }
  }
  ParallelFor(M, [&](std::size_t m) {
    std::vector<float> wrow(K);
    ref::DequantizeRow(w.type, base + m * row_bytes, wrow.data(), K);
    for (std::size_t n = 0; n < rows.size(); ++n) {
      double acc = 0.0;
      double mag = 0.0;
      for (std::size_t c = 0; c < K; ++c) {
        const double term = static_cast<double>(wrow[c]) * xq[n * K + c];
        acc += term;
        mag += std::fabs(term);
      }
      y[n * M + m] = acc;
      sabs[n * M + m] = mag;
    }
  });
}

/// Float-accumulation bound of a GPU projection output against the exact
/// product of the same quantized operands, as a multiple of u times the sum
/// of the terms' magnitudes (Higham's gamma_n for recursive summation): at
/// most one float term per 16 activations (Q6_K's 16-element scale groups
/// are the finest among Q8_0, Q4_K, Q5_K, Q6_K and IQ4_XS), each formed with
/// up to 3 roundings, summed with at most 8 tree-reduction levels (16), plus
/// 2 for the weights' own float dequantization in the oracle (d * sc * q).
inline double AccumulationUnits(std::size_t k) {
  return static_cast<double>(k) / 16.0 + 18.0;
}

struct StageStats {
  /// Worst |gpu - oracle| / oracle rms over every output (accepted ones
  /// included): the raw discrepancy.
  double max_over_rms{0};
  /// Worst |gpu - oracle| / accumulation bound (<= 1 passes).
  double max_over_bound{0};
  double ref_rms{0};
  std::size_t rows{0};
  std::size_t unexplained{0};  ///< outputs beyond the bound
  /// Legacy fields of the tolerance-search checker; always 0 now.
  std::size_t flip_rows{0};
  std::size_t scale_rows{0};
  QuantStats quant;
  std::string first;  ///< first failing output, for the failure message
};

/// Checks one projection stage against its captured activations. `in`
/// ([in_rows][K]) is the float input the GPU quantized, `gpu_out` ([R][M])
/// its output, row r computed with matrix expert_of_row[r]. Buffer row i
/// feeds output row dst[i] (or, without dst, output rows i * fan_out ..
/// i * fan_out + fan_out - 1: the vector path's gate/up read one row per token
/// for every slot). The capture is validated first (ValidateQuantRows); only
/// a valid capture is used as the oracle's input, and every output must then
/// match the exact product of those quantized values within the float
/// accumulation bound. Nothing is searched or nudged.
inline StageStats CheckCapturedProjection(
    const ml::TensorRef& w, const std::vector<std::int32_t>& expert_of_row,
    const std::vector<float>& in, std::size_t in_rows,
    const std::vector<float>& gpu_out, const QuantRows& qr,
    std::size_t fan_out = 1) {
  const std::size_t K = w.cols;
  const std::size_t M = w.rows;
  const std::size_t R = expert_of_row.size();
  Require(qr.k == K && in.size() == in_rows * K && gpu_out.size() == R * M,
          "CheckCapturedProjection: shape mismatch");
  StageStats stats;
  stats.rows = R;
  stats.quant = ValidateQuantRows(qr, in, in_rows);
  if (stats.quant.violations != 0) {
    return stats;
  }
  // Output row -> buffer row.
  std::vector<std::size_t> quant_of(R, SIZE_MAX);
  if (!qr.dst.empty()) {
    for (std::size_t i = 0; i < qr.rows; ++i) {
      const auto d = static_cast<std::size_t>(qr.dst[i]);
      const auto s = static_cast<std::size_t>(qr.src[i]);
      Require(d < R && quant_of[d] == SIZE_MAX && s == d / fan_out,
              "capture: output row map is not a permutation of the slots");
      quant_of[d] = i;
    }
  } else {
    Require(qr.rows * fan_out == R && qr.rows == in_rows,
            "capture: row count does not match the projection");
    for (std::size_t r = 0; r < R; ++r) {
      quant_of[r] = r / fan_out;
    }
  }
  std::vector<double> expected(R * M);
  std::vector<double> magnitude(R * M);
  std::map<std::int32_t, std::vector<std::size_t>> by_expert;
  for (std::size_t r = 0; r < R; ++r) {
    by_expert[expert_of_row[r]].push_back(r);
  }
  for (const auto& [e, list] : by_expert) {
    std::vector<std::size_t> rows(list.size());
    for (std::size_t i = 0; i < list.size(); ++i) {
      rows[i] = quant_of[list[i]];
    }
    std::vector<double> y(list.size() * M);
    std::vector<double> mag(list.size() * M);
    CapturedProduct(w, static_cast<std::uint64_t>(e), qr, rows, y.data(),
                    mag.data());
    for (std::size_t i = 0; i < list.size(); ++i) {
      std::copy_n(y.data() + i * M, M, expected.data() + list[i] * M);
      std::copy_n(mag.data() + i * M, M, magnitude.data() + list[i] * M);
    }
  }
  double sq = 0;
  for (const double v : expected) {
    sq += v * v;
  }
  stats.ref_rms = std::sqrt(sq / static_cast<double>(expected.size()));
  const double units = AccumulationUnits(K) * 0x1p-24;
  for (std::size_t i = 0; i < R * M; ++i) {
    Require(std::isfinite(gpu_out[i]), "GPU stage output is not finite");
    const double err = std::fabs(gpu_out[i] - expected[i]);
    // The floor covers outputs whose terms are all zero (denormal results).
    const double bound = units * magnitude[i] + 1e-30;
    stats.max_over_rms = std::max(
        stats.max_over_rms, stats.ref_rms > 0 ? err / stats.ref_rms : err);
    stats.max_over_bound = std::max(stats.max_over_bound, err / bound);
    if (err > bound) {
      if (stats.unexplained++ == 0) {
        std::ostringstream m;
        m << "row " << i / M << " col " << i % M << " expert "
          << expert_of_row[i / M] << ": gpu " << gpu_out[i] << " oracle "
          << expected[i] << ", error " << err << " > bound " << bound;
        stats.first = m.str();
      }
    }
  }
  return stats;
}

/// Legacy strict check without captured activations, kept for callers that
/// have not moved to CheckCapturedProjection: compares with the oracle's own
/// quantization in `mode` and counts every row whose error exceeds `tol`
/// (relative to the oracle rms) as unexplained. It explains nothing away, so
/// a genuine rounding-boundary difference also fails here.
inline StageStats CheckProjection(
    const ml::TensorRef& w, const std::vector<std::int32_t>& expert_of_row,
    const std::vector<float>& in, const std::vector<float>& gpu_out,
    ref::ActivationQuant mode, double tol) {
  const std::size_t K = w.cols;
  const std::size_t M = w.rows;
  const std::size_t R = expert_of_row.size();
  Require(in.size() == R * K && gpu_out.size() == R * M,
          "CheckProjection: shape mismatch");
  ref::Options options;
  options.activation_quant = mode;
  std::vector<float> expected(R * M);
  std::map<std::int32_t, std::vector<std::size_t>> by_expert;
  for (std::size_t r = 0; r < R; ++r) {
    by_expert[expert_of_row[r]].push_back(r);
  }
  for (const auto& [e, list] : by_expert) {
    std::vector<float> rows(list.size() * K);
    for (std::size_t i = 0; i < list.size(); ++i) {
      std::copy_n(in.data() + list[i] * K, K, rows.data() + i * K);
    }
    std::vector<float> y(list.size() * M);
    ref::MatMulRows(w, static_cast<std::uint64_t>(e), rows.data(), list.size(),
                    y.data(), options);
    for (std::size_t i = 0; i < list.size(); ++i) {
      std::copy_n(y.data() + i * M, M, expected.data() + list[i] * M);
    }
  }
  StageStats stats;
  stats.rows = R;
  double sq = 0;
  for (const float v : expected) {
    sq += static_cast<double>(v) * v;
  }
  stats.ref_rms = std::sqrt(sq / static_cast<double>(expected.size()));
  for (std::size_t r = 0; r < R; ++r) {
    double worst = 0;
    for (std::size_t c = 0; c < M; ++c) {
      Require(std::isfinite(gpu_out[r * M + c]),
              "GPU stage output is not finite");
      worst =
          std::max(worst, std::fabs(static_cast<double>(gpu_out[r * M + c]) -
                                    expected[r * M + c]));
    }
    stats.max_over_rms = std::max(stats.max_over_rms, worst / stats.ref_rms);
    if (worst > tol * stats.ref_rms) {
      ++stats.unexplained;
    }
  }
  return stats;
}

/// One projection stage's measurements, kept for every case.
struct StageReport {
  double over_rms{0};    ///< worst raw |gpu - oracle| / oracle rms
  double over_bound{0};  ///< worst |gpu - oracle| / accumulation bound
  std::size_t blocks{0};
  std::size_t boundary_codes{0};
  double max_scale_rel{0};
};

/// What one operator case measured.
struct CaseReport {
  std::string label;
  double e2e_over_rms{0};    ///< final output vs FfnForward, / oracle rms
  double f32_over_rms{0};    ///< final output vs the F32 oracle (Q8 noise)
  double stage_over_rms{0};  ///< worst projection-stage error / rms
  double stage_over_bound{0};
  double e2e_abs{0};
  std::size_t boundary_codes{0};  ///< captured codes not the nearest one
  std::map<std::string, StageReport> stages;
  double margin{-1};  ///< router selection margin (MoE)
};

struct CaseOptions {
  bool expect_tie{false};    ///< the selection boundary is an exact tie
  bool expect_clamp{false};  ///< the 2^-14 clamp must engage
  double e2e_tol{5e-3};      ///< see the ffn_ops_test header comment
  double min_margin{1e-6};   ///< router decision margin needed to compare ids
  bool f32_reference{true};  ///< also run the F32 oracle (Q8 noise level)
};

inline void ExpectNear(const std::vector<float>& got,
                       const std::vector<float>& want, double rel,
                       double abs_floor, const std::string& what) {
  Require(got.size() == want.size(), what + ": size mismatch");
  for (std::size_t i = 0; i < got.size(); ++i) {
    Require(std::isfinite(got[i]), what + ": GPU value not finite");
    const double bound = rel * std::fabs(want[i]) + abs_floor;
    if (std::fabs(static_cast<double>(got[i]) - want[i]) > bound) {
      std::ostringstream m;
      m << what << ": element " << i << " gpu " << got[i] << " oracle "
        << want[i];
      Require(false, m.str());
    }
  }
}

/// Runs the operator once on `x`, tapped and with its quantized activations
/// captured, and checks it stage by stage: router stages against the oracle,
/// each projection against the oracle's product of the captured (and
/// independently validated) quantized input, SwiGLU and combine from the
/// GPU's own stage outputs. Then end to end against ref::FfnForward in the
/// matching activation mode; then reruns untapped (the result must be
/// bit-identical) and returns the measurements. Throws on any failed check.
inline CaseReport RunOperatorCase(
    const std::string& label, const ml::Config& config,
    const ml::LayerWeights& lw, const gpu::LayerFfn& device,
    const std::vector<float>& x_in, std::uint32_t tokens, gpu::Path path,
    GpuContext& ctx, const CaseOptions& opt, Taps* keep = nullptr) {
  CaseReport report;
  report.label = label;
  const std::size_t H = config.hidden_size;
  const bool vector = UsesVector(path, tokens);
  const ref::ActivationQuant mode = ModeFor(vector);
  Require(x_in.size() == std::size_t{tokens} * H, "RunOperatorCase: bad input");
  std::vector<float> x = x_in;
  AvoidTies(&x, tokens, H);
  float* dx = ToDevice(x);
  float* dout = nullptr;
  CheckHip(hipMalloc(&dout, x.size() * sizeof(float)), "alloc out");
  const auto launch = [&] {
    std::string error;
    const bool ok =
        device.moe()
            ? ctx.ops().Moe(device.routed(), dx, dout,
                            static_cast<std::uint32_t>(H), tokens, path, &error)
            : ctx.ops().Dense(device.dense(), dx, dout, tokens, path, &error);
    Require(ok, label + ": operator failed: " + error);
  };

  // Tapped run, with every quantized activation buffer captured.
  Taps taps;
  {
    struct Guard {
      gpu::FfnOps& ops;
      ~Guard() { RemoveTap(ops); }
    } guard{ctx.ops()};
    InstallTap(ctx.ops(), &taps);
    launch();
    CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  }
  Require(taps.capture_error.empty(),
          label + ": activation capture failed: " + taps.capture_error);
  Require(taps.pending.empty(),
          label + ": activation buffers captured after the last stage");
  const auto tapped = ToHost(dout, x.size());

  // Oracle end to end (matching mode) and F32 oracle (Q8 noise reference).
  const OracleRun matched = RunOracle(config, lw, x, tokens, mode);
  OracleRun exact;
  if (opt.f32_reference) {
    exact = RunOracle(config, lw, x, tokens, ref::ActivationQuant::kNone);
  }
  const auto e2e = Compare(tapped, matched.out);
  report.e2e_over_rms = e2e.max_over_rms;
  report.e2e_abs = e2e.max_abs;
  if (opt.f32_reference) {
    report.f32_over_rms = Compare(tapped, exact.out).max_over_rms;
  }

  // The captures a stage's projections read: `count` buffers in the layout
  // the path uses for `type` weights.
  const auto captured = [&](const std::string& stage, std::size_t count,
                            GgmlType type) {
    const auto& list = taps.C(stage);
    std::ostringstream m;
    m << label << ": stage " << stage << " captured " << list.size()
      << " activation buffers, expected " << count;
    Require(list.size() == count, m.str());
    std::vector<QuantRows> out;
    for (const Capture& c : list) {
      Require(c.layout == ExpectedLayout(type, vector),
              label + ": stage " + stage + " read " + LayoutName(c.layout) +
                  " activations, expected " +
                  LayoutName(ExpectedLayout(type, vector)));
      out.push_back(DecodeCapture(c));
    }
    return out;
  };
  const auto stage = [&](const std::string& name, const StageStats& s) {
    if (s.quant.violations != 0) {
      std::ostringstream m;
      m << label << ": stage " << name << " captured activations fail "
        << "validation (" << s.quant.violations
        << " blocks): " << s.quant.first;
      Require(false, m.str());
    }
    StageReport& r = report.stages[name];
    r.over_rms = s.max_over_rms;
    r.over_bound = s.max_over_bound;
    r.blocks = s.quant.blocks;
    r.boundary_codes = s.quant.boundary_codes;
    r.max_scale_rel = s.quant.max_scale_rel;
    report.stage_over_rms = std::max(report.stage_over_rms, s.max_over_rms);
    report.stage_over_bound =
        std::max(report.stage_over_bound, s.max_over_bound);
    report.boundary_codes += s.quant.boundary_codes;
    if (s.unexplained != 0) {
      std::ostringstream m;
      m << label << ": stage " << name << " has " << s.unexplained
        << " outputs beyond the accumulation bound (worst " << s.max_over_bound
        << " of the bound, " << s.max_over_rms
        << " of rms); first: " << s.first;
      Require(false, m.str());
    }
  };
  const auto shared = [&](const std::string& what,
                          const std::vector<const QuantRows*>& list) {
    const std::string why = SharedInputMismatch(list);
    Require(why.empty(), label + ": " + what + ": " + why);
  };
  // Dense expert-less stages (layer 0 or the shared expert): gate and up each
  // quantize x (two buffers that must agree), down quantizes the act tap.
  std::vector<QuantRows> x_captures;  // every quantization of x
  const auto dense_stages = [&](const char* prefix, const ml::TensorRef& gate,
                                const ml::TensorRef& up,
                                const ml::TensorRef& down) {
    const std::string p = prefix;
    const std::vector<std::int32_t> zeros(tokens, 0);
    const auto gu = captured(p + "_gate", 2, gate.type);
    shared(p + " gate/up input", {&gu[0], &gu[1]});
    stage(p + "_gate", CheckCapturedProjection(gate, zeros, x, tokens,
                                               taps.F(p + "_gate"), gu[0]));
    stage(p + "_up", CheckCapturedProjection(up, zeros, x, tokens,
                                             taps.F(p + "_up"), gu[1]));
    x_captures.insert(x_captures.end(), gu.begin(), gu.end());
    const auto& g = taps.F(p + "_gate");
    const auto& u = taps.F(p + "_up");
    std::vector<float> act(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      act[i] = (g[i] / (1.0F + std::exp(-g[i]))) * u[i];
    }
    ExpectNear(taps.F(p + "_act"), act, 1e-5, 1e-7, p + " swiglu");
    const auto dn = captured(p + "_out", 1, down.type);
    stage(p + "_down",
          CheckCapturedProjection(down, zeros, taps.F(p + "_act"), tokens,
                                  taps.F(p + "_out"), dn[0]));
  };

  if (!device.moe()) {
    dense_stages("dense", lw.ffn_gate, lw.ffn_up, lw.ffn_down);
    // The dense output is the layer output.
    ExpectNear(tapped, taps.F("dense_out"), 0.0, 0.0, "dense output");
  } else {
    const std::size_t E = config.num_experts;
    const std::size_t U = config.num_experts_used;
    // Router: logits, then the selection and weights.
    ExpectNear(taps.F("router_logits"), matched.logits, 2e-5, 5e-5,
               "router logits");
    const auto& ids = taps.I("topk");
    Require(ids.size() == matched.topk.size() && ids == matched.topk,
            label + ": router top-k differs from the oracle");
    report.margin = SelectionMargin(matched, tokens, E, U);
    if (opt.expect_tie) {
      Require(report.margin == 0.0, label + ": fixture lost its exact tie");
    } else {
      Require(report.margin >= opt.min_margin,
              label + ": router decision margin too small to compare ids");
    }
    ExpectNear(taps.F("weights"), matched.weights, 5e-6, 1e-12,
               "router weights");
    if (opt.expect_clamp) {
      for (std::size_t t = 0; t < tokens; ++t) {
        double sum = 0;
        for (std::size_t s = 0; s < U; ++s) {
          const double l = taps.F("router_logits")[t * E + ids[t * U + s]];
          sum += 1.0 / (1.0 + std::exp(-l));
        }
        Require(sum < 6.103515625e-5, label + ": clamp not engaged");
      }
    }
    // Routed experts. Each (token, slot) row is one projection row. The
    // vector path quantizes x once for gate and up (one row per token, read
    // by every slot); the batched path quantizes one gathered row per slot
    // for each of the two projections.
    const std::vector<std::int32_t> expert_of_slot(ids.begin(), ids.end());
    const GgmlType rt = lw.ffn_gate_exps.type;
    const auto gu = captured("gate", vector ? 1 : 2, rt);
    const QuantRows& gq = gu[0];
    const QuantRows& uq = vector ? gu[0] : gu[1];
    if (!vector) {
      shared("routed gate/up input", {&gu[0], &gu[1]});
    }
    stage("gate", CheckCapturedProjection(lw.ffn_gate_exps, expert_of_slot, x,
                                          tokens, taps.F("gate"), gq, U));
    stage("up", CheckCapturedProjection(lw.ffn_up_exps, expert_of_slot, x,
                                        tokens, taps.F("up"), uq, U));
    x_captures.insert(x_captures.end(), gu.begin(), gu.end());
    const auto& g = taps.F("gate");
    const auto& u = taps.F("up");
    std::vector<float> act(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      act[i] = (g[i] / (1.0F + std::exp(-g[i]))) * u[i];
    }
    ExpectNear(taps.F("act"), act, 1e-5, 1e-7, "routed swiglu");
    const auto dq = captured("down", 1, rt);
    stage("down", CheckCapturedProjection(lw.ffn_down_exps, expert_of_slot,
                                          taps.F("act"), tokens * U,
                                          taps.F("down"), dq[0]));
    dense_stages("shared", lw.ffn_gate_shexp, lw.ffn_up_shexp,
                 lw.ffn_down_shexp);
    // Combine: sum_s w_s * down_s, left to right, plus the shared expert.
    std::vector<float> combined(tokens * H);
    for (std::size_t t = 0; t < tokens; ++t) {
      for (std::size_t d = 0; d < H; ++d) {
        float acc = taps.F("weights")[t * U] * taps.F("down")[(t * U) * H + d];
        for (std::size_t s = 1; s < U; ++s) {
          acc += taps.F("weights")[t * U + s] *
                 taps.F("down")[(t * U + s) * H + d];
        }
        combined[t * H + d] = acc + taps.F("shared_out")[t * H + d];
      }
    }
    ExpectNear(tapped, combined, 1e-5, 1e-6, "combine");
  }
  // Every quantization of x in one layout (routed and shared expert alike)
  // must be the same: one deterministic quantizer, one input.
  std::map<int, std::vector<const QuantRows*>> by_layout;
  for (const QuantRows& q : x_captures) {
    by_layout[q.layout].push_back(&q);
  }
  for (const auto& [layout, list] : by_layout) {
    shared(std::string("every ") + LayoutName(layout) + " quantization of x",
           list);
  }

  if (e2e.max_over_rms > opt.e2e_tol) {
    std::ostringstream m;
    m << label << ": end-to-end error vs the oracle " << e2e.max_over_rms
      << " of rms exceeds the bound " << opt.e2e_tol;
    Require(false, m.str());
  }

  // Untapped reruns: same kernels without taps or synchronization. Two calls
  // are queued back to back before the first read; both must reproduce the
  // tapped result bit for bit.
  CheckHip(hipMemset(dout, 0xFF, x.size() * sizeof(float)), "clear");
  launch();
  launch();
  CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  const auto untapped = ToHost(dout, x.size());
  Require(std::memcmp(untapped.data(), tapped.data(),
                      tapped.size() * sizeof(float)) == 0,
          label + ": untapped result differs from the tapped result");
  (void)hipFree(dx);
  (void)hipFree(dout);
  if (keep != nullptr) {
    *keep = std::move(taps);
  }
  return report;
}

inline std::string CaseLabel(const std::string& what, std::uint32_t tokens,
                             gpu::Path path) {
  std::ostringstream l;
  l << what << " T=" << tokens << " " << PathName(path);
  return l.str();
}

inline void PrintReport(const CaseReport& r) {
  std::cout << std::setprecision(3) << std::scientific << "  " << r.label
            << ": e2e " << r.e2e_over_rms << " of rms (abs " << r.e2e_abs
            << "), stage " << r.stage_over_rms << " of rms ("
            << r.stage_over_bound << " of bound), Q8 noise " << r.f32_over_rms
            << ", boundary codes " << r.boundary_codes;
  if (r.margin >= 0) {
    std::cout << ", margin " << std::defaultfloat << r.margin;
  }
  std::cout << std::defaultfloat << "\n";
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_FFN_GPU_HPP_
