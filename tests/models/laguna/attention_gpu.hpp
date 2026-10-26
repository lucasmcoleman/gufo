#ifndef GUFO_TESTS_MODELS_LAGUNA_ATTENTION_GPU_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_ATTENTION_GPU_HPP_

// Shared pieces of the Laguna GPU attention operator tests: random tiny layer
// fixtures (every weight random, dimensions unequal), a driver that runs a
// chunk plan through the operator, and the stage-by-stage and end-to-end
// comparison with the phase 2 oracle (reference::AttnForward and its building
// blocks). See attention_ops_test.cpp for the tolerances and why they hold.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
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
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/laguna/kernels/attention.hpp"
#include "tests/models/laguna/ffn_gpu.hpp"

namespace gufo::testing::laguna {

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct AttnShape {
  const char* name;
  std::uint32_t hidden;
  std::uint32_t kv_heads;
  std::uint32_t heads_full;  ///< layer 0 (full attention)
  std::uint32_t heads_swa;   ///< layer 1 (sliding window)
  std::uint32_t rotary_full;
  std::uint32_t rotary_swa;
  std::uint32_t window;
};

/// The model's own head geometry (GQA 6 on full layers, 9 on sliding ones,
/// rotary 64 / 128, window 512) at a small hidden size.
inline constexpr AttnShape kModelShape{
    "model-geometry", 512, 2, 12, 18, 64, 128, 512};
/// Unequal everything else: hidden 768, 3 KV heads, GQA 2 and 4, rotary 32 and
/// 96, a short window.
inline constexpr AttnShape kOddShape{"odd-A", 768, 3, 6, 12, 32, 96, 37};
/// GQA 8 and 12, one KV head, hidden 1024, rotary 128 / 64, window 64.
inline constexpr AttnShape kWideShape{"odd-B", 1024, 1, 8, 12, 128, 64, 64};

inline ml::Config MakeAttnConfig(const AttnShape& s) {
  ml::Config c;
  c.num_layers = 2;
  c.hidden_size = s.hidden;
  c.context_length = 262144;
  c.heads_per_layer = {s.heads_full, s.heads_swa};
  c.num_kv_heads = s.kv_heads;
  c.head_dim = 128;
  c.sliding_window = s.window;
  c.rope_theta_full = 500000.0F;
  c.rope_theta_swa = 10000.0F;
  c.rotary_dim_full = s.rotary_full;
  c.rotary_dim_swa = s.rotary_swa;
  c.yarn.factor = 32.0F;
  c.yarn.original_context = 8192;
  c.yarn.beta_fast = 32.0F;
  c.yarn.beta_slow = 1.0F;
  c.rope_attn_factor = 1.0F;
  c.rms_eps = 1e-6F;
  return c;
}

inline HostMat RandomUniformF32(std::size_t count, float lo, float hi,
                                std::mt19937_64& rng) {
  HostMat m;
  m.cols = count;
  m.rows = 1;
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> values(count);
  for (float& v : values) {
    v = dist(rng);
  }
  SetF32Values(&m, values);
  return m;
}

/// Two layers (0: full attention, 1: sliding window) with every tensor random.
class TinyAttn {
public:
  TinyAttn(const AttnShape& shape, std::uint64_t seed)
      : rng_(seed), shape_(shape), config_(MakeAttnConfig(shape)) {
    const std::size_t H = shape.hidden;
    const std::size_t kv = std::size_t{shape.kv_heads} * 128;
    for (std::uint32_t il = 0; il < 2; ++il) {
      const std::size_t heads = config_.n_head(il);
      Mats& m = mats_[il];
      m.q = RandomQuant(GgmlType::kQ6_K, H, heads * 128, 1, rng_);
      m.k = RandomQuant(GgmlType::kQ6_K, H, kv, 1, rng_);
      m.v = RandomQuant(GgmlType::kQ6_K, H, kv, 1, rng_);
      m.o = RandomQuant(GgmlType::kQ6_K, heads * 128, H, 1, rng_);
      m.gate =
          RandomF32(H, heads, 1.5F / std::sqrt(static_cast<float>(H)), rng_);
      m.qn = RandomUniformF32(128, 0.5F, 1.5F, rng_);
      m.kn = RandomUniformF32(128, 0.5F, 1.5F, rng_);
      ml::LayerWeights& lw = layers_[il];
      lw.full_attention = config_.is_full_attention(il);
      const std::string p = "blk." + std::to_string(il) + ".";
      lw.attn_q = Ref(m.q, p + "attn_q.weight");
      lw.attn_k = Ref(m.k, p + "attn_k.weight");
      lw.attn_v = Ref(m.v, p + "attn_v.weight");
      lw.attn_output = Ref(m.o, p + "attn_output.weight");
      lw.attn_gate = Ref(m.gate, p + "attn_gate.weight");
      lw.attn_q_norm = Ref(m.qn, p + "attn_q_norm.weight");
      lw.attn_k_norm = Ref(m.kn, p + "attn_k_norm.weight");
    }
  }
  TinyAttn(const TinyAttn&) = delete;
  TinyAttn& operator=(const TinyAttn&) = delete;

  [[nodiscard]] const ml::Config& config() const { return config_; }
  [[nodiscard]] const ml::LayerWeights& layer(std::uint32_t il) const {
    return layers_.at(il);
  }
  [[nodiscard]] const AttnShape& shape() const { return shape_; }
  [[nodiscard]] std::mt19937_64& rng() { return rng_; }

private:
  struct Mats {
    HostMat q, k, v, o, gate, qn, kn;
  };
  ml::TensorRef Ref(const HostMat& m, const std::string& name) {
    names_.push_back(name);
    ml::TensorRef t;
    t.data = m.bytes.data();
    t.type = m.type;
    t.cols = m.cols;
    t.rows = m.rows;
    t.experts = 1;
    t.name = names_.back();
    Require(t.SizeBytes() == m.bytes.size(), "fixture byte size mismatch");
    return t;
  }
  std::mt19937_64 rng_;
  AttnShape shape_;
  ml::Config config_;
  std::array<Mats, 2> mats_;
  std::array<ml::LayerWeights, 2> layers_;
  std::deque<std::string> names_;
};

// ---------------------------------------------------------------------------
// Device driver
// ---------------------------------------------------------------------------

class AttnContext {
public:
  AttnContext() {
    CheckHip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
             "stream");
    ops_ = std::make_unique<gpu::AttnOps>(stream_);
  }
  ~AttnContext() {
    ops_.reset();
    (void)hipStreamSynchronize(stream_);
    qfn_mmq_release_stream(stream_);
    (void)hipStreamDestroy(stream_);
  }
  AttnContext(const AttnContext&) = delete;
  AttnContext& operator=(const AttnContext&) = delete;
  [[nodiscard]] hipStream_t stream() const { return stream_; }
  [[nodiscard]] gpu::AttnOps& ops() { return *ops_; }

private:
  hipStream_t stream_{};
  std::unique_ptr<gpu::AttnOps> ops_;
};

/// A sequence split into chunks, each with the projection path it forces.
struct Plan {
  std::vector<std::uint32_t> chunks;
  std::vector<gpu::Path> paths;
  [[nodiscard]] std::uint32_t total() const {
    return std::accumulate(chunks.begin(), chunks.end(), 0U);
  }
  [[nodiscard]] std::uint32_t max_chunk() const {
    return *std::max_element(chunks.begin(), chunks.end());
  }
};

inline Plan UniformPlan(std::vector<std::uint32_t> chunks, gpu::Path path) {
  Plan p;
  p.paths.assign(chunks.size(), path);
  p.chunks = std::move(chunks);
  return p;
}

/// `count` rows of one token each.
inline std::vector<std::uint32_t> Singles(std::uint32_t count) {
  return std::vector<std::uint32_t>(count, 1U);
}
inline std::vector<std::uint32_t> Cat(std::vector<std::uint32_t> a,
                                      const std::vector<std::uint32_t>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

struct GpuRun {
  std::vector<float> out;                           ///< [T][hidden]
  std::map<std::string, std::vector<float>> stage;  ///< tapped runs only
  std::vector<std::uint16_t> khist, vhist;          ///< [T][kv_dim] F16 bits
  /// Activation buffers qfn_mmq quantized for each projection stage
  /// ("q_proj", "k_proj", "v_proj", "o"), one per chunk, in chunk order.
  std::map<std::string, std::vector<Capture>> captures;
};

/// Runs `plan` over the rows `n` ([T][hidden]) on a fresh cache. Tapped runs
/// synchronize after every stage and keep every stage's rows and the cache
/// rows of each chunk; untapped runs queue every chunk back to back and
/// synchronize once.
inline GpuRun RunPlan(AttnContext& ctx, const gpu::AttnWeights& w,
                      const ml::Config& config, std::uint32_t il,
                      const std::vector<float>& n, const Plan& plan,
                      bool tapped) {
  const std::uint32_t T = plan.total();
  const std::size_t H = config.hidden_size;
  const std::size_t kvd = config.kv_dim();
  Require(n.size() == T * H, "RunPlan: input size");
  Require(plan.paths.size() == plan.chunks.size(), "RunPlan: paths");
  std::string error;
  auto cache = gpu::KvCache::Create(config, il, T, plan.max_chunk(), &error);
  Require(cache.has_value(), "kv cache: " + error);
  GpuRun run;
  float* dn = ToDevice(n);
  float* dout = nullptr;
  CheckHip(hipMalloc(&dout, T * H * sizeof(float)), "alloc out");
  Taps cap;  // the capture hook's pending buffers
  struct Guard {
    bool active;
    AttnContext& ctx;
    ~Guard() {
      if (active) {
        ctx.ops().SetTap({});
        qfn_mmq_set_activation_capture(nullptr, nullptr);
      }
    }
  } guard{tapped, ctx};
  if (tapped) {
    run.khist.assign(T * kvd, 0);
    run.vhist.assign(T * kvd, 0);
    qfn_mmq_set_activation_capture(OnActivationCapture, &cap);
    ctx.ops().SetTap([&run, &cap](std::string_view name, const float* device,
                                  std::size_t count) {
      const std::string key(name);
      auto& dst = run.stage[key];
      const auto rows = ToHost(device, count);
      dst.insert(dst.end(), rows.begin(), rows.end());
      // The buffers quantized since the previous stage belong to this stage's
      // projection.
      auto& filed = run.captures[key];
      for (auto& c : cap.pending) {
        filed.push_back(std::move(c));
      }
      cap.pending.clear();
    });
  }
  std::uint32_t pos = 0;
  for (std::size_t c = 0; c < plan.chunks.size(); ++c) {
    const std::uint32_t rows = plan.chunks[c];
    const bool ok = ctx.ops().Forward(w, *cache, dn + pos * H, dout + pos * H,
                                      rows, pos, plan.paths[c], &error);
    Require(ok, "Forward failed: " + error);
    if (tapped) {
      CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
      for (std::uint32_t p = pos; p < pos + rows; ++p) {
        const std::size_t off = std::size_t{cache->Slot(p)} * kvd * 2;
        CheckHip(hipMemcpy(run.khist.data() + p * kvd,
                           static_cast<const char*>(cache->keys()) + off,
                           kvd * 2, hipMemcpyDeviceToHost),
                 "copy keys");
        CheckHip(hipMemcpy(run.vhist.data() + p * kvd,
                           static_cast<const char*>(cache->values()) + off,
                           kvd * 2, hipMemcpyDeviceToHost),
                 "copy values");
      }
    }
    pos += rows;
  }
  CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  if (tapped) {
    Require(cap.capture_error.empty(),
            "activation capture failed: " + cap.capture_error);
    Require(cap.pending.empty(),
            "activation buffers captured after the last stage");
  }
  run.out = ToHost(dout, T * H);
  (void)hipFree(dn);
  (void)hipFree(dout);
  return run;
}

inline bool BitIdentical(const std::vector<float>& a,
                         const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// ---------------------------------------------------------------------------
// Oracle side
// ---------------------------------------------------------------------------

/// The oracle's output rows and the gated attention rows it feeds to Wo.
struct OracleOut {
  std::vector<float> out;
  std::vector<float> gated;
};

inline OracleOut OracleAttn(const ml::Config& config,
                            const ml::LayerWeights& lw, std::uint32_t il,
                            const std::vector<float>& n, std::size_t T,
                            ref::ActivationQuant mode) {
  ref::Options options;
  options.activation_quant = mode;
  options.round_kv_f16 = true;  // llama.cpp's default F16 cache
  OracleOut result;
  const auto tap = [&](const ref::Tap& t) {
    if (t.name.rfind("attn_gated", 0) == 0) {
      result.gated.assign(static_cast<const float*>(t.data),
                          static_cast<const float*>(t.data) + t.Count());
    }
  };
  result.out = ref::AttnForward(config, lw, il, n.data(), T, options, tap);
  return result;
}

/// ggml_compute_forward_rms_norm_f32 then the weight, as the oracle does it.
inline void RefRmsNorm(const float* x, const float* weight, std::size_t n,
                       float eps, float* y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * x[i];
  }
  const float mean = static_cast<float>(sum / static_cast<double>(n));
  const float scale = 1.0F / std::sqrt(mean + eps);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = (x[i] * scale) * weight[i];
  }
}

inline float HalfToFloat(std::uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, sizeof(h));
  return static_cast<float>(h);
}

/// Half a unit in the last place of the binary16 value nearest `x`.
inline double HalfUlp(double x) {
  const double a = std::fabs(x);
  if (a < 6.103515625e-5) {  // subnormal range: fixed spacing 2^-24
    return std::ldexp(1.0, -24);
  }
  return std::ldexp(1.0, static_cast<int>(std::floor(std::log2(a))) - 10);
}

inline double Rms(const std::vector<float>& v) {
  double sq = 0;
  for (const float x : v) {
    sq += static_cast<double>(x) * x;
  }
  return std::sqrt(sq / static_cast<double>(v.size()));
}

/// Attention in plain single precision, one head at a time: sequential dot,
/// float softmax, sequential weighted sum. Same visibility rule as the oracle.
inline std::vector<float> FloatAttention(const float* q, const float* k,
                                         const float* v, std::size_t T,
                                         std::uint32_t heads,
                                         std::uint32_t kv_heads,
                                         std::uint32_t window, float scale) {
  const std::uint32_t group = heads / kv_heads;
  const std::size_t qrow = std::size_t{heads} * 128;
  const std::size_t kvrow = std::size_t{kv_heads} * 128;
  std::vector<float> out(T * qrow);
  std::vector<float> score;
  for (std::size_t p = 0; p < T; ++p) {
    const std::size_t first =
        window == 0 || p + 1 <= window ? 0 : p + 1 - window;
    score.assign(p - first + 1, 0.0F);
    for (std::uint32_t h = 0; h < heads; ++h) {
      const float* qh = q + p * qrow + std::size_t{h} * 128;
      const std::size_t kvh = h / group;
      float top = -INFINITY;
      for (std::size_t j = first; j <= p; ++j) {
        const float* kj = k + j * kvrow + kvh * 128;
        float dot = 0.0F;
        for (int d = 0; d < 128; ++d) {
          dot += qh[d] * kj[d];
        }
        score[j - first] = dot * scale;
        top = std::max(top, score[j - first]);
      }
      float denom = 0.0F;
      for (float& s : score) {
        s = std::exp(s - top);
        denom += s;
      }
      float* oh = out.data() + p * qrow + std::size_t{h} * 128;
      for (int d = 0; d < 128; ++d) {
        float acc = 0.0F;
        for (std::size_t j = first; j <= p; ++j) {
          acc += score[j - first] * v[j * kvrow + kvh * 128 + d];
        }
        oh[d] = acc / denom;
      }
    }
  }
  return out;
}

/// The oracle's attention block recomposed from its public building blocks
/// (MatMulRows, RmsNorm, ApplyRope, GqaAttention): the same arithmetic as
/// ref::AttnForward, with one addition. `khist`/`vhist` (F16 bits, [T][kv_dim])
/// replace the oracle's own F16 rounding of K and V when given: the GPU rounds
/// float values that differ from the oracle's by float noise, so an element
/// within that noise of an F16 rounding boundary is stored differently, and
/// feeding the oracle the GPU's stored rows removes that discretization from
/// the end-to-end comparison (CheckRun proves the composition without the
/// override is bit-identical to AttnForward).
inline OracleOut ComposeOracle(const ml::Config& c, const ml::LayerWeights& lw,
                               std::uint32_t il, const std::vector<float>& n,
                               std::size_t T, ref::ActivationQuant mode,
                               const std::vector<std::uint16_t>* khist,
                               const std::vector<std::uint16_t>* vhist) {
  ref::Options options;
  options.activation_quant = mode;
  const std::size_t H = c.hidden_size;
  const std::uint32_t heads = c.n_head(il);
  const std::uint32_t kvh = c.num_kv_heads;
  const std::size_t qd = std::size_t{heads} * 128;
  const std::size_t kvd = std::size_t{kvh} * 128;
  std::vector<float> q(T * qd), k(T * kvd), v(T * kvd), gate(T * heads);
  ref::MatMulRows(lw.attn_q, 0, n.data(), T, q.data(), options);
  ref::MatMulRows(lw.attn_k, 0, n.data(), T, k.data(), options);
  ref::MatMulRows(lw.attn_v, 0, n.data(), T, v.data(), options);
  ref::MatMulRows(lw.attn_gate, 0, n.data(), T, gate.data(), options);
  const auto gamma = [&](const ml::TensorRef& t) {
    std::vector<float> g(128);
    ref::DequantizeRow(t.type, t.data, g.data(), 128);
    return g;
  };
  const auto qg = gamma(lw.attn_q_norm);
  const auto kg = gamma(lw.attn_k_norm);
  for (std::size_t i = 0; i < T * heads; ++i) {
    RefRmsNorm(q.data() + i * 128, qg.data(), 128, c.rms_eps,
               q.data() + i * 128);
  }
  for (std::size_t i = 0; i < T * kvh; ++i) {
    RefRmsNorm(k.data() + i * 128, kg.data(), 128, c.rms_eps,
               k.data() + i * 128);
  }
  std::vector<std::int32_t> positions(T);
  std::iota(positions.begin(), positions.end(), 0);
  const ref::RopeParams rope = ref::RopeFor(c, il);
  ref::ApplyRope(q.data(), positions, heads, 128, rope);
  ref::ApplyRope(k.data(), positions, kvh, 128, rope);
  for (std::size_t i = 0; i < k.size(); ++i) {
    k[i] = khist != nullptr ? HalfToFloat((*khist)[i]) : ref::RoundToF16(k[i]);
    v[i] = vhist != nullptr ? HalfToFloat((*vhist)[i]) : ref::RoundToF16(v[i]);
  }
  OracleOut result;
  std::vector<float> attn(T * qd);
  ref::GqaAttention(q.data(), k.data(), v.data(), T, heads, kvh, 128,
                    c.is_full_attention(il) ? 0U : c.sliding_window,
                    1.0F / std::sqrt(128.0F), attn.data());
  for (std::size_t i = 0; i < T * heads; ++i) {
    const float g = gate[i];
    const float sp = g > 20.0F ? g : std::log(1.0F + std::exp(g));
    for (int d = 0; d < 128; ++d) {
      attn[i * 128 + d] *= sp;
    }
  }
  result.gated = attn;
  result.out.resize(T * H);
  ref::MatMulRows(lw.attn_output, 0, attn.data(), T, result.out.data(),
                  options);
  return result;
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

struct AttnTolerances {
  /// q/k after norm + rope, / the head's largest |x|. About 10 float
  /// roundings separate the two sides of one element (squared-sum order and
  /// the reciprocal root, two products by scale and gamma, one library ulp
  /// each in cos and sin, two products and a sum in the rotation): 10 u with
  /// u = 2^-24 is 6e-7; measured at most 3.6e-7.
  double rope_head{1e-6};
  double attn_float_multiple{3.0};   ///< x the plain-float CPU error
  double attn_floor_rms{2e-6};       ///< floor of that bound, / rms
  double chain_float_multiple{4.0};  ///< x plain-float attention noise
  double chain_floor_rms{5e-6};      ///< floor, / rms of the gated rows
  double raw_rms{3e-2};  ///< output vs AttnForward (Q8 flips), / rms
};

struct AttnReport {
  std::string label;
  double proj_over_rms{0};    ///< worst raw q/k/v/o stage error / oracle rms
  double proj_over_bound{0};  ///< worst stage error / accumulation bound
  std::size_t boundary_codes{0};  ///< captured codes not the nearest one
  double max_scale_rel{0};        ///< worst |stored scale - amax/127| relative
  double rope_over_rms{0};
  double kv_excess{0};  ///< worst K/V cache error beyond half an F16 ulp, /|x|
  double attn_over_rms{0};
  double attn_float_over_rms{0};  ///< plain float CPU attention, same inputs
  double e2e_over_rms{-1};        ///< -1: not compared (mixed paths)
  double e2e_abs{0};
  std::size_t e2e_row{0};      ///< row holding the largest end-to-end error
  double e2e_raw_over_rms{0};  ///< output vs the oracle with no correction
  double q8_noise{-1};         ///< output vs the F32 oracle, / rms
  double mscale_dev{0};        ///< worst |rotary-norm ratio / mscale - 1|
  double mscale{0};
  double gate_rel{0};
  double window_ratio{-1};  ///< boundary probe: wrong-window error / ours
};

struct CheckOptions {
  AttnTolerances tol;
  bool f32_reference{false};
  bool e2e{true};  ///< false for plans that mix projection modes
  /// Query rows at which the window boundary is probed (empty: none).
  std::vector<std::uint32_t> boundary_rows;
  /// Precomputed oracle outputs by activation mode (shared across plans).
  std::map<int, OracleOut>* oracle_cache{nullptr};
};

inline void ExpectRelNear(const std::vector<float>& got,
                          const std::vector<float>& want, double rel,
                          double abs_floor, double* worst,
                          const std::string& what) {
  Require(got.size() == want.size(), what + ": size mismatch");
  for (std::size_t i = 0; i < got.size(); ++i) {
    Require(std::isfinite(got[i]), what + ": GPU value not finite");
    const double err = std::fabs(static_cast<double>(got[i]) - want[i]);
    const double bound = rel * std::fabs(want[i]) + abs_floor;
    if (worst != nullptr && bound > 0) {
      *worst = std::max(*worst, err / (std::fabs(want[i]) + abs_floor / rel));
    }
    if (err > bound) {
      std::ostringstream m;
      m << what << ": element " << i << " gpu " << got[i] << " want "
        << want[i];
      Require(false, m.str());
    }
  }
}

inline double MaxAbsDiff(const std::vector<float>& a,
                         const std::vector<float>& b) {
  Require(a.size() == b.size(), "MaxAbsDiff: size");
  double worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return worst;
}

/// Runs the tapped run's stage checks and (for single-mode plans) the end to
/// end comparison; throws on any failure.
inline AttnReport CheckRun(const std::string& label, const ml::Config& config,
                           const ml::LayerWeights& lw, std::uint32_t il,
                           const std::vector<float>& n, const Plan& plan,
                           const GpuRun& run, const CheckOptions& opt) {
  AttnReport rep;
  rep.label = label;
  const std::uint32_t T = plan.total();
  const std::size_t H = config.hidden_size;
  const std::uint32_t heads = config.n_head(il);
  const std::uint32_t kvh = config.num_kv_heads;
  const std::size_t qd = std::size_t{heads} * 128;
  const std::size_t kvd = std::size_t{kvh} * 128;
  const std::uint32_t window =
      config.is_full_attention(il) ? 0U : config.sliding_window;
  const auto& st = run.stage;
  const auto need = [&](const char* name, std::size_t count) -> const auto& {
    const auto it = st.find(name);
    Require(it != st.end(), std::string("missing stage ") + name);
    Require(it->second.size() == count,
            std::string("stage ") + name + " has the wrong size");
    return it->second;
  };
  const auto& q_proj = need("q_proj", T * qd);
  const auto& k_proj = need("k_proj", T * kvd);
  const auto& v_proj = need("v_proj", T * kvd);
  const auto& gate_proj = need("gate_proj", std::size_t{T} * heads);
  const auto& q_rope = need("q_rope", T * qd);
  const auto& attn = need("attn", T * qd);
  const auto& gate_act = need("gate_act", std::size_t{T} * heads);
  const auto& gated = need("gated", T * qd);
  const auto& o = need("o", T * H);
  Require(BitIdentical(o, run.out), label + ": output differs from the o tap");

  // ---- projections: captured activations, then an exact oracle ----
  // For every projection the GPU's packed Q8 activation buffer is captured
  // where the kernel reads it and validated against the float input it was
  // quantized from (ValidateQuantRows); only a valid capture feeds the
  // oracle's exact product, which the GPU output must match within the float
  // accumulation bound (CheckCapturedProjection). Nothing is searched or
  // nudged. q, k and v read the same input and must have been given the same
  // quantization.
  {
    const std::size_t chunks = plan.chunks.size();
    for (const char* name : {"q_proj", "k_proj", "v_proj", "o"}) {
      const auto it = run.captures.find(name);
      Require(it != run.captures.end() && it->second.size() == chunks,
              label + ": stage " + name +
                  " captured the wrong number of "
                  "activation buffers");
    }
    struct Stage {
      const char* name;
      const ml::TensorRef* w;
      const std::vector<float>* in;
      std::size_t in_width;
      const std::vector<float>* out;
      std::size_t out_width;
    };
    const Stage stages[] = {{"q_proj", &lw.attn_q, &n, H, &q_proj, qd},
                            {"k_proj", &lw.attn_k, &n, H, &k_proj, kvd},
                            {"v_proj", &lw.attn_v, &n, H, &v_proj, kvd},
                            {"o", &lw.attn_output, &gated, qd, &o, H}};
    // Decoded captures per stage and chunk.
    std::map<std::string, std::vector<QuantRows>> decoded;
    for (const Stage& st : stages) {
      for (std::size_t c = 0; c < chunks; ++c) {
        const bool vector = UsesVector(plan.paths[c], plan.chunks[c]);
        const Capture& cap = run.captures.at(st.name)[c];
        Require(cap.layout == ExpectedLayout(st.w->type, vector),
                label + ": stage " + st.name + " read " +
                    LayoutName(cap.layout) + " activations, expected " +
                    LayoutName(ExpectedLayout(st.w->type, vector)));
        Require(cap.rows == plan.chunks[c],
                label + ": stage " + st.name + " captured the wrong rows");
        decoded[st.name].push_back(DecodeCapture(cap));
      }
    }
    for (std::size_t c = 0; c < chunks; ++c) {
      const std::string why =
          SharedInputMismatch({&decoded["q_proj"][c], &decoded["k_proj"][c],
                               &decoded["v_proj"][c]});
      Require(why.empty(),
              label + ": q/k/v input, chunk " + std::to_string(c) + ": " + why);
    }
    for (const Stage& st : stages) {
      for (const bool vector : {false, true}) {
        // Merge the chunks of this activation mode, in chunk order.
        QuantRows merged;
        std::vector<float> in;
        std::vector<float> out;
        std::size_t pos = 0;
        std::size_t rows = 0;
        for (std::size_t c = 0; c < chunks; ++c) {
          const std::size_t r = plan.chunks[c];
          if (UsesVector(plan.paths[c], plan.chunks[c]) == vector) {
            const QuantRows& q = decoded[st.name][c];
            if (rows == 0) {
              merged.layout = q.layout;
              merged.k = q.k;
              merged.has_sum = q.has_sum;
            }
            merged.q.insert(merged.q.end(), q.q.begin(), q.q.end());
            merged.d.insert(merged.d.end(), q.d.begin(), q.d.end());
            merged.s.insert(merged.s.end(), q.s.begin(), q.s.end());
            for (const std::int32_t v : q.src) {
              merged.src.push_back(v + static_cast<std::int32_t>(rows));
            }
            for (const std::int32_t v : q.dst) {
              merged.dst.push_back(v + static_cast<std::int32_t>(rows));
            }
            in.insert(in.end(), st.in->begin() + pos * st.in_width,
                      st.in->begin() + (pos + r) * st.in_width);
            out.insert(out.end(), st.out->begin() + pos * st.out_width,
                       st.out->begin() + (pos + r) * st.out_width);
            rows += r;
          }
          pos += r;
        }
        if (rows == 0) {
          continue;
        }
        merged.rows = rows;
        const std::vector<std::int32_t> zeros(rows, 0);
        const StageStats s =
            CheckCapturedProjection(*st.w, zeros, in, rows, out, merged);
        Require(s.quant.violations == 0,
                label + ": stage " + st.name +
                    " captured activations fail "
                    "validation (" +
                    std::to_string(s.quant.violations) +
                    " blocks): " + s.quant.first);
        Require(s.unexplained == 0,
                label + ": stage " + st.name + " has " +
                    std::to_string(s.unexplained) +
                    " outputs beyond the accumulation bound (worst " +
                    std::to_string(s.max_over_bound) + " of the bound, " +
                    std::to_string(s.max_over_rms) +
                    " of rms); first: " + s.first);
        rep.proj_over_rms = std::max(rep.proj_over_rms, s.max_over_rms);
        rep.proj_over_bound = std::max(rep.proj_over_bound, s.max_over_bound);
        rep.boundary_codes += s.quant.boundary_codes;
        rep.max_scale_rel = std::max(rep.max_scale_rel, s.quant.max_scale_rel);
      }
    }
  }

  // ---- output gate projection: F32 weights, exact inputs ----
  {
    ref::Options exact;
    std::vector<float> want(std::size_t{T} * heads);
    ref::MatMulRows(lw.attn_gate, 0, n.data(), T, want.data(), exact);
    ExpectRelNear(gate_proj, want, 2e-5, 2e-5 * Rms(want), nullptr,
                  label + " gate projection");
    // softplus (x > 20 ? x : log(1 + exp(x))) of the GPU's own gate rows.
    std::vector<float> sp(gate_proj.size());
    for (std::size_t i = 0; i < sp.size(); ++i) {
      const double g = gate_proj[i];
      sp[i] = static_cast<float>(g > 20.0 ? g : std::log(1.0 + std::exp(g)));
    }
    ExpectRelNear(gate_act, sp, 2e-6, 1e-7, &rep.gate_rel, label + " softplus");
  }

  // ---- QK norm and RoPE, from the GPU's own projections ----
  const ref::RopeParams rope = ref::RopeFor(config, il);
  const ref::RopeSetup setup = ref::MakeRopeSetup(rope);
  rep.mscale = setup.mscale;
  std::vector<std::int32_t> positions(T);
  std::iota(positions.begin(), positions.end(), 0);
  const auto norm_heads = [&](const std::vector<float>& proj,
                              const ml::TensorRef& gamma_ref,
                              std::uint32_t head_count) {
    std::vector<float> gamma(128);
    ref::DequantizeRow(gamma_ref.type, gamma_ref.data, gamma.data(), 128);
    std::vector<float> out(proj.size());
    for (std::size_t i = 0; i < std::size_t{T} * head_count; ++i) {
      RefRmsNorm(proj.data() + i * 128, gamma.data(), 128, config.rms_eps,
                 out.data() + i * 128);
    }
    return out;
  };
  const std::vector<float> qn = norm_heads(q_proj, lw.attn_q_norm, heads);
  std::vector<float> q_want = qn;
  ref::ApplyRope(q_want.data(), positions, heads, 128, rope);
  // A rotated element is a sum of two products and a normalized one a product
  // of three roundings, so its float disagreement scales with the largest
  // element of its head, not with the whole array's rms (a head with a large
  // norm gain has elements many times the rms). Bound: opt.tol.rope_head of
  // the head's largest |element|.
  const auto head_amax = [&](const std::vector<float>& v, std::size_t head) {
    float amax = 0.0F;
    for (int d = 0; d < 128; ++d) {
      amax = std::max(amax, std::fabs(v[head * 128 + d]));
    }
    return static_cast<double>(amax);
  };
  {
    for (std::size_t h = 0; h < std::size_t{T} * heads; ++h) {
      const double amax = head_amax(q_want, h);
      for (int d = 0; d < 128; ++d) {
        const double err = std::fabs(static_cast<double>(q_rope[h * 128 + d]) -
                                     q_want[h * 128 + d]);
        rep.rope_over_rms = std::max(rep.rope_over_rms, err / amax);
        Require(err <= opt.tol.rope_head * amax,
                label + ": q after norm+rope differs from the oracle, " +
                    std::to_string(err / amax) +
                    " of the head's largest element");
      }
    }
    // The rotated block keeps its length times mscale (YaRN's attention
    // factor); compare lengths of the rotary channels before and after.
    const std::uint32_t rot = rope.n_rot;
    for (std::size_t i = 0; i < std::size_t{T} * heads; ++i) {
      double before = 0, after = 0;
      for (std::uint32_t d = 0; d < rot; ++d) {
        before += static_cast<double>(qn[i * 128 + d]) * qn[i * 128 + d];
        after += static_cast<double>(q_rope[i * 128 + d]) * q_rope[i * 128 + d];
      }
      if (before > 0) {
        const double ratio = std::sqrt(after / before);
        rep.mscale_dev =
            std::max(rep.mscale_dev, std::fabs(ratio / setup.mscale - 1.0));
      }
    }
    Require(rep.mscale_dev < 2e-5,
            label + ": rotary length is not mscale times the input's");
  }
  // K and V as stored (F16) against the oracle's values rounded to F16: the
  // stored value must be a binary16 within half an ulp of the oracle's float
  // (plus the float-level disagreement before rounding).
  {
    std::vector<float> k_want = norm_heads(k_proj, lw.attn_k_norm, kvh);
    ref::ApplyRope(k_want.data(), positions, kvh, 128, rope);
    // `slack` is the float disagreement before the rounding, as a fraction of
    // the head's largest element (K, as for q above; 0 for V, which is stored
    // from the v_proj values themselves) plus 3e-6 relative to the element.
    const auto check_f16 = [&](const std::vector<std::uint16_t>& hist,
                               const std::vector<float>& want, double slack,
                               const char* what) {
      for (std::size_t i = 0; i < want.size(); ++i) {
        const double got = HalfToFloat(hist[i]);
        const double err = std::fabs(got - want[i]);
        const double half_ulp = 0.5 * HalfUlp(want[i]);
        const double amax = slack > 0 ? head_amax(want, i / 128) : 0.0;
        const double allowed = half_ulp * (1.0 + 1e-6) +
                               (slack > 0 ? 3e-6 * std::fabs(want[i]) : 0.0) +
                               slack * amax;
        rep.kv_excess = std::max(
            rep.kv_excess,
            std::max(0.0, err - half_ulp) /
                std::max(static_cast<double>(std::fabs(want[i])), 1e-3));
        if (err > allowed) {
          std::ostringstream m;
          m << label << ": cached " << what << " element " << i << " is " << got
            << ", oracle " << want[i];
          Require(false, m.str());
        }
      }
    };
    check_f16(run.khist, k_want, opt.tol.rope_head, "key");
    // V is stored from the very values of the v_proj tap: pure rounding.
    check_f16(run.vhist, v_proj, 0.0, "value");
  }

  // ---- attention over the cache as stored, in double ----
  double plain_gated_abs = 0;
  std::vector<float> kf(run.khist.size());
  std::vector<float> vf(run.vhist.size());
  for (std::size_t i = 0; i < kf.size(); ++i) {
    kf[i] = HalfToFloat(run.khist[i]);
    vf[i] = HalfToFloat(run.vhist[i]);
  }
  const float scale = 1.0F / std::sqrt(128.0F);
  std::vector<float> attn_want(T * qd);
  ref::GqaAttention(q_rope.data(), kf.data(), vf.data(), T, heads, kvh, 128,
                    window, scale, attn_want.data());
  {
    // The GPU sums in float, so it cannot match the double oracle bit for bit.
    // How far a plain float implementation of the same arithmetic (sequential
    // dot, softmax and weighted sum on the CPU) lies from the double result is
    // the float noise of this input; the GPU must be within a small multiple of
    // it (and of a floor for inputs where that is accidentally tiny).
    const double rms = Rms(attn_want);
    const double err = MaxAbsDiff(attn, attn_want);
    const std::vector<float> plain = FloatAttention(
        q_rope.data(), kf.data(), vf.data(), T, heads, kvh, window, scale);
    rep.attn_over_rms = err / rms;
    rep.attn_float_over_rms = MaxAbsDiff(plain, attn_want) / rms;
    // The same float noise as it reaches the gated rows (for the chain check).
    {
      double worst = 0;
      for (std::size_t i = 0; i < plain.size(); ++i) {
        worst = std::max(
            worst, std::fabs(static_cast<double>(plain[i]) - attn_want[i]) *
                       gate_act[i / 128]);
      }
      plain_gated_abs = worst;
    }
    const double bound =
        std::max(opt.tol.attn_float_multiple * rep.attn_float_over_rms,
                 opt.tol.attn_floor_rms);
    Require(rep.attn_over_rms <= bound,
            label + ": attention differs from the double-precision oracle, " +
                std::to_string(rep.attn_over_rms) +
                " of rms; a plain float implementation is " +
                std::to_string(rep.attn_float_over_rms) + " away");
  }
  // The window boundary: the same rows computed with the window one key
  // narrower and one key wider must be far from the GPU's rows, so the check
  // above would have caught an off-by-one.
  if (!opt.boundary_rows.empty() && window != 0) {
    double ours = 0;
    double wrong = 1e30;
    for (const std::uint32_t variant : {window - 1, window + 1}) {
      std::vector<float> alt(T * qd);
      ref::GqaAttention(q_rope.data(), kf.data(), vf.data(), T, heads, kvh, 128,
                        variant, scale, alt.data());
      double worst_alt = 0;
      for (const std::uint32_t r : opt.boundary_rows) {
        for (std::size_t i = 0; i < qd; ++i) {
          ours =
              std::max(ours, std::fabs(static_cast<double>(attn[r * qd + i]) -
                                       attn_want[r * qd + i]));
          worst_alt = std::max(worst_alt,
                               std::fabs(static_cast<double>(attn[r * qd + i]) -
                                         alt[r * qd + i]));
        }
      }
      wrong = std::min(wrong, worst_alt);
    }
    rep.window_ratio = wrong / std::max(ours, 1e-12);
    Require(rep.window_ratio > 100.0,
            label + ": window probe too weak to separate window +-1 (ratio " +
                std::to_string(rep.window_ratio) + ")");
  }
  // gated = attn * softplus(gate), the same single float multiply.
  {
    std::vector<float> want(attn.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
      want[i] = attn[i] * gate_act[i / 128];
    }
    Require(BitIdentical(gated, want), label + ": gating is not attn * gate");
  }

  // ---- end to end, for plans that use one projection mode ----
  if (opt.e2e) {
    const bool vector = UsesVector(plan.paths[0], plan.chunks[0]);
    for (std::size_t i = 0; i < plan.chunks.size(); ++i) {
      Require(UsesVector(plan.paths[i], plan.chunks[i]) == vector,
              "CheckRun: e2e needs one projection mode");
    }
    const ref::ActivationQuant mode = ModeFor(vector);
    OracleOut base;
    const int key = static_cast<int>(mode);
    if (opt.oracle_cache != nullptr && opt.oracle_cache->count(key) != 0) {
      base = opt.oracle_cache->at(key);
    } else {
      base = OracleAttn(config, lw, il, n, T, mode);
      const OracleOut again =
          ComposeOracle(config, lw, il, n, T, mode, nullptr, nullptr);
      Require(BitIdentical(again.out, base.out) &&
                  BitIdentical(again.gated, base.gated),
              label + ": the recomposed oracle differs from AttnForward");
      if (opt.oracle_cache != nullptr) {
        (*opt.oracle_cache)[key] = base;
      }
    }
    const OracleOut oracle =
        ComposeOracle(config, lw, il, n, T, mode, &run.khist, &run.vhist);
    // Chain check: everything before Wo (projections, norm, rope, attention,
    // gate) end to end in float terms, against the recomposed oracle fed the
    // GPU's stored K/V. The only discretization left upstream of Wo is the Q8
    // rounding of n, which AvoidTies keeps off any boundary, so the rows agree
    // to float noise. (The Wo input's own Q8 rounding is covered by the stage
    // check of "o": that stage takes the GPU's gated rows as its input.)
    {
      const double rms = Rms(oracle.gated);
      const double err = MaxAbsDiff(gated, oracle.gated);
      rep.e2e_over_rms = err / rms;
      rep.e2e_abs = err;
      const double bound = opt.tol.chain_float_multiple * plain_gated_abs +
                           opt.tol.chain_floor_rms * rms;
      Require(err <= bound,
              label +
                  ": gated attention rows differ from the recomposed "
                  "oracle, " +
                  std::to_string(err / rms) +
                  " of rms; a plain float attention is " +
                  std::to_string(plain_gated_abs / rms) + " away");
    }
    // The final output against ref::AttnForward itself. Here the Wo input of
    // each side is rounded to Q8 separately, and a near-tie element of that
    // row (an element within float noise of a rounding boundary) takes a
    // different code on each side, moving one output by about one Q8 step
    // times a weight per flip, up to a few 1e-3 of rms, with copies of the
    // same head output (row 0, attention over a single key) flipping together.
    // That is expected Q8 behaviour, not an error, so the bound is a
    // sanity bound far below the Q8 noise printed below and far above any
    // flip, which a wiring defect would exceed by orders of magnitude.
    {
      const double rms = Rms(base.out);
      rep.e2e_raw_over_rms = MaxAbsDiff(run.out, base.out) / rms;
      Require(rep.e2e_raw_over_rms <= opt.tol.raw_rms,
              label + ": output differs from AttnForward by " +
                  std::to_string(rep.e2e_raw_over_rms) + " of rms, bound " +
                  std::to_string(opt.tol.raw_rms));
    }
    if (opt.f32_reference) {
      const auto exact =
          OracleAttn(config, lw, il, n, T, ref::ActivationQuant::kNone);
      rep.q8_noise = Compare(run.out, exact.out).max_over_rms;
    }
  }
  return rep;
}

inline void PrintAttnReport(const AttnReport& r) {
  std::cout << std::setprecision(3) << std::scientific << "  " << r.label
            << ": proj " << r.proj_over_rms << " (" << r.proj_over_bound
            << " of bound), rope " << r.rope_over_rms << ", attn "
            << r.attn_over_rms << " (plain float " << r.attn_float_over_rms
            << "), gate " << r.gate_rel;
  if (r.e2e_over_rms >= 0) {
    std::cout << ", chain " << r.e2e_over_rms << ", output vs AttnForward "
              << r.e2e_raw_over_rms;
  } else {
    std::cout << ", chain and output n/a (mixed projection paths)";
  }
  if (r.q8_noise >= 0) {
    std::cout << ", Q8 noise " << r.q8_noise;
  }
  std::cout << ", mscale " << std::defaultfloat << std::setprecision(5)
            << r.mscale << std::setprecision(3) << " dev " << std::scientific
            << r.mscale_dev << ", " << r.boundary_codes << " boundary codes";
  if (r.window_ratio >= 0) {
    std::cout << ", window probe x" << std::defaultfloat << r.window_ratio;
  }
  std::cout << std::defaultfloat << "\n";
}

/// Aggregates worst-case numbers per case group for the summary.
struct GroupWorst {
  std::size_t cases{0};
  AttnReport worst;
  void Add(const AttnReport& r) {
    ++cases;
    worst.proj_over_rms = std::max(worst.proj_over_rms, r.proj_over_rms);
    worst.rope_over_rms = std::max(worst.rope_over_rms, r.rope_over_rms);
    worst.attn_over_rms = std::max(worst.attn_over_rms, r.attn_over_rms);
    worst.attn_float_over_rms =
        std::max(worst.attn_float_over_rms, r.attn_float_over_rms);
    worst.e2e_over_rms = std::max(worst.e2e_over_rms, r.e2e_over_rms);
    worst.e2e_raw_over_rms =
        std::max(worst.e2e_raw_over_rms, r.e2e_raw_over_rms);
    worst.e2e_abs = std::max(worst.e2e_abs, r.e2e_abs);
    worst.mscale_dev = std::max(worst.mscale_dev, r.mscale_dev);
    worst.proj_over_bound = std::max(worst.proj_over_bound, r.proj_over_bound);
    worst.boundary_codes += r.boundary_codes;
    if (r.window_ratio >= 0) {
      worst.window_ratio = worst.window_ratio < 0
                               ? r.window_ratio
                               : std::min(worst.window_ratio, r.window_ratio);
    }
  }
};

inline void PrintGroup(const std::string& name, const GroupWorst& g) {
  std::cout << "  " << name << ": " << g.cases << " cases, worst (of rms) proj "
            << std::scientific << std::setprecision(2) << g.worst.proj_over_rms
            << ", rope " << g.worst.rope_over_rms << ", attn "
            << g.worst.attn_over_rms << " (plain float "
            << g.worst.attn_float_over_rms << "), chain "
            << g.worst.e2e_over_rms << ", output vs AttnForward "
            << g.worst.e2e_raw_over_rms << ", mscale dev " << g.worst.mscale_dev
            << ", worst projection error " << g.worst.proj_over_bound
            << " of the accumulation bound, " << g.worst.boundary_codes
            << " boundary codes";
  if (g.worst.window_ratio >= 0) {
    std::cout << ", min window probe x" << std::defaultfloat
              << g.worst.window_ratio;
  }
  std::cout << std::defaultfloat << "\n";
}

/// Model-like input rows: RMS-normalized random rows times a norm gain; or
/// RandomRows (outliers) with quantizer ties moved away.
inline std::vector<float> TinyRows(std::size_t T, std::size_t H,
                                   std::mt19937_64& rng) {
  auto x = RandomRows(T, H, rng);
  AvoidTies(&x, T, H);
  return x;
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_ATTENTION_GPU_HPP_
