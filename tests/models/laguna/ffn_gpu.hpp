#ifndef GUFO_TESTS_MODELS_LAGUNA_FFN_GPU_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_FFN_GPU_HPP_

// Shared pieces of the Laguna GPU FFN operator tests: random quantized layer
// fixtures (every weight random, sizes unequal), device helpers, and the
// comparison of an operator run against the phase 2 oracle.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
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

struct Taps {
  std::map<std::string, std::vector<float>> f32;
  std::map<std::string, std::vector<std::int32_t>> i32;
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
};

inline void InstallTap(gpu::FfnOps& ops, Taps* taps) {
  ops.SetTap([taps](std::string_view name, const void* device,
                    std::size_t count, bool is_int32) {
    if (is_int32) {
      taps->i32[std::string(name)] =
          ToHost(static_cast<const std::int32_t*>(device), count);
    } else {
      taps->f32[std::string(name)] =
          ToHost(static_cast<const float*>(device), count);
    }
  });
}

/// The GPU's quantizer computes its scale and 1/scale with approximate
/// reciprocals, a few ulps (about 2e-5 in units of one Q8 code at the largest
/// codes) from the correctly rounded value; a margin of twice that is searched.
inline constexpr double kTieMargin = 4e-5;

/// Both sides quantize each 32-element block with scale amax/127. A value
/// whose x * 127 / amax sits within `margin` of a rounding boundary can round
/// differently on the GPU, whose division is not correctly rounded
/// (-ffast-math reciprocal), without any arithmetic being wrong. Returns the
/// elements of `row` (length `cols`) that close to a boundary.
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
/// kTieMargin of a Q8 rounding boundary a few percent of a code away from it,
/// so the first quantization of a test input is unambiguous (intermediate rows
/// are produced by the operator and cannot be treated this way).
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

struct StageStats {
  double max_over_rms{0};  ///< worst unexplained |gpu - oracle| / oracle rms
  double ref_rms{0};
  std::size_t rows{0};
  std::size_t flip_rows{0};   ///< rows whose mismatch is one-LSB tie flips
  std::size_t scale_rows{0};  ///< rows explained by one fp16 scale boundary
  std::size_t unexplained{0};
};

/// Compares a projection's GPU output (`gpu_out[rows][M]`) with the oracle's
/// product of the same input rows (`in[rows][K]`) and weights, matrix
/// `expert_of_row[row]`, in activation mode `mode`. Rows whose error exceeds
/// `tol` (relative to the oracle rms) are accepted only if flipping the
/// rounding of some of their near-tie input elements reproduces the GPU row
/// to `tol`; anything else counts as unexplained.
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
  const double tol_abs = tol * stats.ref_rms;
  const auto row_error = [&](const float* a, const float* b) {
    double worst = 0;
    for (std::size_t c = 0; c < M; ++c) {
      Require(std::isfinite(a[c]), "GPU stage output is not finite");
      worst = std::max(worst, std::fabs(static_cast<double>(a[c]) - b[c]));
    }
    return worst;
  };
  for (std::size_t r = 0; r < R; ++r) {
    const double err =
        row_error(gpu_out.data() + r * M, expected.data() + r * M);
    if (err <= tol_abs) {
      stats.max_over_rms = std::max(stats.max_over_rms, err / stats.ref_rms);
      continue;
    }
    // Try to explain the row by rounding flips of near-tie inputs.
    const auto ties = NearTies(in.data() + r * K, K, kTieMargin);
    bool explained = false;
    if (!ties.empty() && ties.size() <= 6) {
      for (std::uint32_t subset = 1; subset < (1U << ties.size()) && !explained;
           ++subset) {
        std::vector<float> variant(in.begin() + r * K,
                                   in.begin() + (r + 1) * K);
        for (std::size_t i = 0; i < ties.size(); ++i) {
          if (((subset >> i) & 1U) == 0) {
            continue;
          }
          const std::size_t j = ties[i];
          const std::size_t b = j / 32 * 32;
          float amax = 0.0F;
          for (std::size_t k = 0; k < 32; ++k) {
            amax = std::max(amax, std::fabs(variant[b + k]));
          }
          // The value sits on a rounding boundary: its two candidate codes are
          // floor(|v|) and floor(|v|) + 1. The oracle chose one in float
          // arithmetic of the mode; the variant takes the other.
          const double v =
              std::fabs(static_cast<double>(variant[j])) * (127.0 / amax);
          const double lo = std::floor(v);
          const float scale = mode == ref::ActivationQuant::kGpuMmq
                                  ? 127.0F / amax
                                  : 1.0F / (amax / 127.0F);
          const double oracle_code = std::fabs(std::round(variant[j] * scale));
          const double alt = oracle_code == lo ? lo + 1 : lo;
          variant[j] = static_cast<float>((variant[j] < 0 ? -alt : alt) *
                                          (amax / 127.0));
        }
        std::vector<float> y(M);
        ref::MatMulRows(w, static_cast<std::uint64_t>(expert_of_row[r]),
                        variant.data(), 1, y.data(), options);
        const double verr = row_error(gpu_out.data() + r * M, y.data());
        if (verr <= tol_abs) {
          explained = true;
        }
      }
    }
    bool scale_boundary = false;
    if (!explained) {
      // Or one stored fp16 scale (or block sum) that rounded the other way.
      const bool kquant_mmq =
          mode == ref::ActivationQuant::kGpuMmq &&
          (w.type == GgmlType::kQ4_K || w.type == GgmlType::kQ5_K);
      if (mode == ref::ActivationQuant::kGpuQ8_1 || kquant_mmq) {
        for (int g = 0; g < static_cast<int>(K / 32) && !scale_boundary; ++g) {
          for (int kind = 0; kind < (kquant_mmq ? 2 : 1) && !scale_boundary;
               ++kind) {
            for (const int steps : {-1, 1}) {
              ref::Options nudged = options;
              nudged.nudge = ref::ScaleNudge{g, kind, steps};
              std::vector<float> y(M);
              ref::MatMulRows(w, static_cast<std::uint64_t>(expert_of_row[r]),
                              in.data() + r * K, 1, y.data(), nudged);
              if (row_error(gpu_out.data() + r * M, y.data()) <= tol_abs) {
                scale_boundary = true;
                break;
              }
            }
          }
        }
      }
    }
    if (scale_boundary) {
      ++stats.scale_rows;
    } else if (explained) {
      ++stats.flip_rows;
    } else {
      std::cerr << "  unexplained row " << r << " expert " << expert_of_row[r]
                << " error/rms " << err / stats.ref_rms << " near-ties "
                << ties.size();
      for (const std::size_t j : ties) {
        std::cerr << " [" << j << "]";
      }
      std::cerr << "\n";
      ++stats.unexplained;
      stats.max_over_rms = std::max(stats.max_over_rms, err / stats.ref_rms);
    }
  }
  return stats;
}

/// What one operator case measured.
struct CaseReport {
  std::string label;
  double e2e_over_rms{0};    ///< final output vs FfnForward, / oracle rms
  double f32_over_rms{0};    ///< final output vs the F32 oracle (Q8 noise)
  double stage_over_rms{0};  ///< worst unexplained stage error / rms
  double e2e_abs{0};
  std::size_t flip_rows{0};
  std::size_t scale_rows{0};
  double margin{-1};  ///< router selection margin (MoE)
};

struct CaseOptions {
  bool expect_tie{false};    ///< the selection boundary is an exact tie
  bool expect_clamp{false};  ///< the 2^-14 clamp must engage
  double stage_tol{2e-5};    ///< see CheckProjection
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

/// Runs the operator once on `x`, tapped, and checks it stage by stage
/// against the oracle's building blocks, then end to end against
/// ref::FfnForward in the matching activation mode; then reruns untapped (the
/// result must be bit-identical) and returns the measurements. Throws on any
/// failed check.
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

  // Tapped run.
  Taps taps;
  InstallTap(ctx.ops(), &taps);
  launch();
  CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  ctx.ops().SetTap({});
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

  const auto stage = [&](const char* name, const StageStats& s) {
    report.stage_over_rms = std::max(report.stage_over_rms, s.max_over_rms);
    report.flip_rows += s.flip_rows;
    report.scale_rows += s.scale_rows;
    if (s.unexplained != 0) {
      std::ostringstream m;
      m << label << ": stage " << name << " has " << s.unexplained
        << " unexplained mismatching rows, worst " << s.max_over_rms
        << " of rms";
      Require(false, m.str());
    }
  };
  // Dense expert-less stages (layer 0 or the shared expert).
  const auto dense_stages = [&](const char* prefix, const ml::TensorRef& gate,
                                const ml::TensorRef& up,
                                const ml::TensorRef& down) {
    const std::string p = prefix;
    const std::vector<std::int32_t> zeros(tokens, 0);
    stage((p + "_gate").c_str(),
          CheckProjection(gate, zeros, x, taps.F(p + "_gate"), mode,
                          opt.stage_tol));
    stage((p + "_up").c_str(), CheckProjection(up, zeros, x, taps.F(p + "_up"),
                                               mode, opt.stage_tol));
    const auto& g = taps.F(p + "_gate");
    const auto& u = taps.F(p + "_up");
    std::vector<float> act(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      act[i] = (g[i] / (1.0F + std::exp(-g[i]))) * u[i];
    }
    ExpectNear(taps.F(p + "_act"), act, 1e-5, 1e-7, p + " swiglu");
    stage((p + "_out").c_str(),
          CheckProjection(down, zeros, taps.F(p + "_act"), taps.F(p + "_out"),
                          mode, opt.stage_tol));
  };

  if (!device.moe()) {
    dense_stages("dense", lw.ffn_gate, lw.ffn_up, lw.ffn_down);
    // The dense output is the layer output.
    ExpectNear(tapped, taps.F("dense_out"), 0.0, 0.0, "dense output");
  } else {
    const std::size_t E = config.num_experts;
    const std::size_t U = config.num_experts_used;
    const std::size_t F = config.expert_ff;
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
    // Routed experts. Each (token, slot) row is one projection row.
    std::vector<std::int32_t> expert_of_slot(ids.begin(), ids.end());
    std::vector<float> slot_x(tokens * U * H);
    for (std::size_t slot = 0; slot < tokens * U; ++slot) {
      std::copy_n(x.data() + (slot / U) * H, H, slot_x.data() + slot * H);
    }
    stage("gate", CheckProjection(lw.ffn_gate_exps, expert_of_slot, slot_x,
                                  taps.F("gate"), mode, opt.stage_tol));
    stage("up", CheckProjection(lw.ffn_up_exps, expert_of_slot, slot_x,
                                taps.F("up"), mode, opt.stage_tol));
    const auto& g = taps.F("gate");
    const auto& u = taps.F("up");
    std::vector<float> act(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      act[i] = (g[i] / (1.0F + std::exp(-g[i]))) * u[i];
    }
    ExpectNear(taps.F("act"), act, 1e-5, 1e-7, "routed swiglu");
    stage("down",
          CheckProjection(lw.ffn_down_exps, expert_of_slot, taps.F("act"),
                          taps.F("down"), mode, opt.stage_tol));
    (void)F;
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
            << "), stage " << r.stage_over_rms << ", Q8 noise "
            << r.f32_over_rms << ", tie-flip rows " << r.flip_rows
            << ", scale-boundary rows " << r.scale_rows;
  if (r.margin >= 0) {
    std::cout << ", margin " << std::defaultfloat << r.margin;
  }
  std::cout << std::defaultfloat << "\n";
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_FFN_GPU_HPP_
