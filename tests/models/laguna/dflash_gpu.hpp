#ifndef GUFO_TESTS_MODELS_LAGUNA_DFLASH_GPU_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_DFLASH_GPU_HPP_

// Shared pieces of the Laguna DFlash drafter GPU operator tests (phase 6c):
// tiny random drafters with Q4_K/Q6_K projections and unequal dimensions on a
// tiny random target (embedding and head), the device context, tap and
// activation capture, the CPU oracle (reference::DFlashModel) with all of its
// taps, and the per-operator checks. See dflash_ops_test.cpp.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/laguna/dflash.hpp"
#include "src/models/laguna/dflash_reference.hpp"
#include "src/models/laguna/kernels/dflash_ops.hpp"
#include "tests/models/laguna/attention_gpu.hpp"

namespace gufo::testing::laguna {

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct DrafterShape {
  const char* name;
  std::uint32_t hidden;
  std::uint32_t ffn;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t taps;
  std::uint32_t window;
  std::uint32_t layers;
  std::uint32_t vocab;
  float theta;
};

/// The real drafter's head geometry (GQA 9, head 128, window 512, theta
/// 500000, block 16) at a small hidden size; three taps, three layers.
inline constexpr DrafterShape kDraftGeom{"geom", 512, 768, 18,   2,
                                         3,      512, 3,   1000, 500000.0F};
/// Unequal everything else: hidden 768, GQA 4 over 3 KV heads, FFN smaller
/// than hidden, two taps, a short window, another theta.
inline constexpr DrafterShape kDraftOdd{"odd", 768, 512, 12,  3,
                                        2,     37,  2,   777, 10000.0F};

/// A random drafter (every tensor random; the last layer carries Q6_K attn_v
/// and ffn_down as layers 2 and 5 of the real file do) plus a tiny target that
/// supplies token_embd (Q8_0) and the head (Q6_K).
class TinyDrafter {
public:
  TinyDrafter(const DrafterShape& shape, std::uint64_t seed)
      : rng_(seed), shape_(shape) {
    ml::DFlashConfig& c = drafter.config;
    c.num_layers = shape.layers;
    c.hidden_size = shape.hidden;
    c.ffn_size = shape.ffn;
    c.vocab_size = shape.vocab;
    c.context_length = 1U << 20;
    c.rms_eps = 1e-6F;
    c.num_heads = shape.heads;
    c.num_kv_heads = shape.kv_heads;
    c.head_dim = 128;
    c.sliding_window = shape.window;
    c.rope_theta = shape.theta;
    c.rotary_dim = 128;
    c.block_size = 16;
    c.mask_token_id = 12;
    for (std::uint32_t k = 0; k < shape.taps; ++k) {
      c.target_layers.push_back(2 + 7 * k);
    }
    const std::size_t H = shape.hidden;
    const std::size_t taps = shape.taps;
    const std::size_t qd = std::size_t{shape.heads} * 128;
    const std::size_t kvd = std::size_t{shape.kv_heads} * 128;
    aux_ = RandomUniformF32(H * taps, 0.4F, 1.6F, rng_);
    aux_.cols = H;
    aux_.rows = taps;
    fc_ = RandomQuant(GgmlType::kQ4_K, taps * H, H, 1, rng_);
    enc_norm_ = RandomUniformF32(H, 0.5F, 1.5F, rng_);
    out_norm_ = RandomUniformF32(H, 0.5F, 1.5F, rng_);
    drafter.aux_norm = Ref(aux_, "enc.aux_norm.weight");
    drafter.fc = Ref(fc_, "fc.weight");
    drafter.enc_output_norm = Ref(enc_norm_, "enc.output_norm.weight");
    drafter.output_norm = Ref(out_norm_, "output_norm.weight");
    mats_.resize(shape.layers);
    for (std::uint32_t il = 0; il < shape.layers; ++il) {
      const bool q6 = il + 1 == shape.layers;
      const GgmlType vt = q6 ? GgmlType::kQ6_K : GgmlType::kQ4_K;
      Mats& m = mats_[il];
      m.attn_norm = RandomUniformF32(H, 0.5F, 1.5F, rng_);
      m.q = RandomQuant(GgmlType::kQ4_K, H, qd, 1, rng_);
      m.k = RandomQuant(GgmlType::kQ4_K, H, kvd, 1, rng_);
      m.v = RandomQuant(vt, H, kvd, 1, rng_);
      m.o = RandomQuant(GgmlType::kQ4_K, qd, H, 1, rng_);
      m.gate = RandomQuant(GgmlType::kQ4_K, H, shape.heads, 1, rng_);
      m.qn = RandomUniformF32(128, 0.5F, 1.5F, rng_);
      m.kn = RandomUniformF32(128, 0.5F, 1.5F, rng_);
      m.ffn_norm = RandomUniformF32(H, 0.5F, 1.5F, rng_);
      m.fg = RandomQuant(GgmlType::kQ4_K, H, shape.ffn, 1, rng_);
      m.fu = RandomQuant(GgmlType::kQ4_K, H, shape.ffn, 1, rng_);
      m.fd = RandomQuant(vt, shape.ffn, H, 1, rng_);
      for (HostMat* f : {&m.attn_norm, &m.qn, &m.kn, &m.ffn_norm}) {
        f->rows = 1;
      }
      const std::string p = "blk." + std::to_string(il) + ".";
      ml::DFlashLayerWeights l;
      l.attn_norm = Ref(m.attn_norm, p + "attn_norm.weight");
      l.attn_q = Ref(m.q, p + "attn_q.weight");
      l.attn_k = Ref(m.k, p + "attn_k.weight");
      l.attn_v = Ref(m.v, p + "attn_v.weight");
      l.attn_output = Ref(m.o, p + "attn_output.weight");
      l.attn_q_norm = Ref(m.qn, p + "attn_q_norm.weight");
      l.attn_k_norm = Ref(m.kn, p + "attn_k_norm.weight");
      l.attn_gate = Ref(m.gate, p + "attn_gate.weight");
      l.ffn_norm = Ref(m.ffn_norm, p + "ffn_norm.weight");
      l.ffn_gate = Ref(m.fg, p + "ffn_gate.weight");
      l.ffn_up = Ref(m.fu, p + "ffn_up.weight");
      l.ffn_down = Ref(m.fd, p + "ffn_down.weight");
      drafter.layers.push_back(l);
    }
    // Target: embeddings of unit-ish scale, a Q6_K head.
    embd_ = RandomQuant(GgmlType::kQ8_0, H, shape.vocab, 1, rng_);
    {
      // Q8_0 scales sized for weights; embeddings want O(1) values.
      std::uniform_real_distribution<float> d(0.006F, 0.012F);
      for (std::size_t b = 0; b < embd_.bytes.size(); b += 34) {
        const std::uint16_t bits = HalfBitsOf(d(rng_));
        std::memcpy(embd_.bytes.data() + b, &bits, 2);
      }
    }
    head_ = RandomQuant(GgmlType::kQ6_K, H, shape.vocab, 1, rng_);
    target.config.hidden_size = shape.hidden;
    target.config.vocab_size = shape.vocab;
    target.token_embd = Ref(embd_, "token_embd.weight");
    target.output = Ref(head_, "output.weight");
  }
  TinyDrafter(const TinyDrafter&) = delete;
  TinyDrafter& operator=(const TinyDrafter&) = delete;

  ml::DFlashWeights drafter;
  ml::ModelWeights target;
  [[nodiscard]] const DrafterShape& shape() const { return shape_; }
  [[nodiscard]] std::mt19937_64& rng() { return rng_; }
  [[nodiscard]] const ml::DFlashConfig& config() const {
    return drafter.config;
  }

private:
  struct Mats {
    HostMat attn_norm, q, k, v, o, gate, qn, kn, ffn_norm, fg, fu, fd;
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
  DrafterShape shape_;
  HostMat aux_, fc_, enc_norm_, out_norm_, embd_, head_;
  std::vector<Mats> mats_;
  std::deque<std::string> names_;
};

/// Context features like the target's taps: per tap a different scale (and
/// row-dependent scales up to 1e4, as attention-sink rows reach), plus
/// outlier channels.
inline std::vector<float> RandomFeatures(std::size_t rows, std::size_t taps,
                                         std::size_t hidden,
                                         std::mt19937_64& rng) {
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> f(rows * taps * hidden);
  for (std::size_t t = 0; t < rows; ++t) {
    const float row_scale = t % 7 == 3 ? 1e4F : 1.0F + 0.1F * (t % 5);
    for (std::size_t k = 0; k < taps; ++k) {
      const float scale = row_scale * (0.5F + 1.7F * k);
      for (std::size_t i = 0; i < hidden; ++i) {
        float v = dist(rng) * scale;
        if (i % 67 == 11 + k) {
          v *= 8.0F;
        }
        f[(t * taps + k) * hidden + i] = v;
      }
    }
  }
  return f;
}

// ---------------------------------------------------------------------------
// Device context and taps
// ---------------------------------------------------------------------------

class DrafterCtx {
public:
  explicit DrafterCtx(const ml::DFlashWeights& w) {
    CheckHip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
             "stream");
    ops_ = std::make_unique<gpu::DFlashOps>(stream_);
    std::string error;
    auto up = gpu::DeviceDrafter::Upload(w, &error);
    Require(up.has_value(), "drafter upload: " + error);
    device_ = std::move(*up);
  }
  ~DrafterCtx() {
    ops_.reset();
    (void)hipStreamSynchronize(stream_);
    qfn_mmq_release_stream(stream_);
    (void)hipStreamDestroy(stream_);
  }
  DrafterCtx(const DrafterCtx&) = delete;
  DrafterCtx& operator=(const DrafterCtx&) = delete;
  [[nodiscard]] hipStream_t stream() const { return stream_; }
  [[nodiscard]] gpu::DFlashOps& ops() { return *ops_; }
  [[nodiscard]] const gpu::DrafterWeights& weights() const {
    return device_.weights();
  }

private:
  hipStream_t stream_{};
  std::unique_ptr<gpu::DFlashOps> ops_;
  gpu::DeviceDrafter device_;
};

/// Installs a tap on `ops` that copies every stage to `taps` and files the
/// activation buffers qfn_mmq quantized since the previous stage under the
/// stage (so a projection's capture is filed under the stage it emits right
/// after itself). Removed on destruction.
class TapScope {
public:
  TapScope(gpu::DFlashOps& ops, Taps* taps) : ops_(ops) {
    qfn_mmq_set_activation_capture(OnActivationCapture, taps);
    ops.SetTap([taps](std::string_view name, const float* device,
                      std::size_t count) {
      const std::string key(name);
      taps->f32[key] = device != nullptr && count != 0 ? ToHost(device, count)
                                                       : std::vector<float>{};
      auto& filed = taps->captures[key];
      for (auto& c : taps->pending) {
        filed.push_back(std::move(c));
      }
      taps->pending.clear();
    });
  }
  ~TapScope() {
    ops_.SetTap({});
    qfn_mmq_set_activation_capture(nullptr, nullptr);
  }
  TapScope(const TapScope&) = delete;
  TapScope& operator=(const TapScope&) = delete;

private:
  gpu::DFlashOps& ops_;
};

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

struct OracleTaps {
  std::map<std::string, std::vector<float>> f32;
  std::vector<float> logits;
  const std::vector<float>& F(const std::string& name) const {
    const auto it = f32.find(name);
    Require(it != f32.end(), "missing oracle tap " + name);
    return it->second;
  }
};

/// The oracle in activation mode `mode`, with F16 K/V (the GPU cache's) unless
/// `mode` is kNone (the F32 reference, whose distance from the GPU is the
/// expected quantization noise).
inline ref::Options OracleOptions(ref::ActivationQuant mode) {
  ref::Options o;
  o.activation_quant = mode;
  o.round_kv_f16 = mode != ref::ActivationQuant::kNone;
  return o;
}

inline OracleTaps RunDrafterOracle(const TinyDrafter& fx,
                                   const std::vector<float>& features,
                                   const std::vector<std::int32_t>& ctx_pos,
                                   const std::vector<std::int32_t>& block,
                                   std::int32_t block_pos,
                                   ref::ActivationQuant mode) {
  const ref::DFlashModel model(fx.drafter, fx.target, OracleOptions(mode));
  ref::DFlashInputs in;
  in.features = features;
  in.context_positions = ctx_pos;
  in.block_tokens = block;
  in.block_position = block_pos;
  OracleTaps out;
  out.logits = model.Forward(in, [&](const ref::Tap& t) {
    const auto* p = static_cast<const float*>(t.data);
    out.f32[std::string(t.name)] = std::vector<float>(p, p + t.Count());
  });
  return out;
}

inline std::vector<std::int32_t> Iota(std::int32_t begin, std::size_t n) {
  std::vector<std::int32_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = begin + static_cast<std::int32_t>(i);
  }
  return v;
}

/// [anchor, MASK x (B - 1)].
inline std::vector<std::int32_t> BlockTokens(std::int32_t anchor,
                                             std::uint32_t B,
                                             std::int32_t mask) {
  std::vector<std::int32_t> b(B, mask);
  b[0] = anchor;
  return b;
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

/// Decoded captures of one stage, each validated for the expected layout.
inline std::vector<QuantRows> StageCaptures(const Taps& taps,
                                            const std::string& stage,
                                            std::size_t count, GgmlType type,
                                            bool vector,
                                            const std::string& label) {
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
}

/// Worst projection-stage measurements of a run.
struct ProjStats {
  double over_rms{0};
  double over_bound{0};
  std::size_t boundary_codes{0};
  std::size_t stages{0};
  void Add(const StageStats& s) {
    over_rms = std::max(over_rms, s.max_over_rms);
    over_bound = std::max(over_bound, s.max_over_bound);
    boundary_codes += s.quant.boundary_codes;
    ++stages;
  }
};

/// One projection stage against its captured activations: the capture is
/// validated against the float input `in` ([rows][w.cols]) and its exact
/// product must match the GPU output within the float accumulation bound
/// (CheckCapturedProjection). Throws otherwise.
inline void CheckStage(const std::string& label, const std::string& stage,
                       const ml::TensorRef& w, const std::vector<float>& in,
                       std::size_t rows, const std::vector<float>& gpu_out,
                       const QuantRows& capture, ProjStats* stats) {
  const std::vector<std::int32_t> zeros(rows, 0);
  const StageStats s =
      CheckCapturedProjection(w, zeros, in, rows, gpu_out, capture);
  if (s.quant.violations != 0) {
    Require(false, label + ": stage " + stage +
                       " captured activations fail validation (" +
                       std::to_string(s.quant.violations) +
                       " blocks): " + s.quant.first);
  }
  if (s.unexplained != 0) {
    std::ostringstream m;
    m << label << ": stage " << stage << " has " << s.unexplained
      << " outputs beyond the accumulation bound (worst " << s.max_over_bound
      << " of the bound, " << s.max_over_rms << " of rms); first: " << s.first;
    Require(false, m.str());
  }
  if (stats != nullptr) {
    stats->Add(s);
  }
}

/// Every row of `x` ([rows][n]) RMS-normalized with weight row (r % w_rows) of
/// `weight`, the oracle's RmsNormVector.
inline std::vector<float> NormRows(const std::vector<float>& x, std::size_t n,
                                   const float* weight, std::size_t w_rows,
                                   float eps) {
  std::vector<float> y(x.size());
  for (std::size_t r = 0; r < x.size() / n; ++r) {
    ref::RmsNormVector(x.data() + r * n, weight + (r % w_rows) * n, n, eps,
                       y.data() + r * n);
  }
  return y;
}

inline std::vector<float> Values(const ml::TensorRef& t) {
  return ref::DequantizeTensor(t);
}

/// rms(got - want) / rms(want). An isolated Q8 code flip at a rounding tie
/// moves single elements by up to a few percent of rms, so end-to-end
/// distances are judged by this (and the max only loosely).
inline double RelRms(const std::vector<float>& got,
                     const std::vector<float>& want) {
  Require(got.size() == want.size(), "RelRms: size mismatch");
  double e = 0, r = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    Require(std::isfinite(got[i]), "GPU value not finite");
    const double d = static_cast<double>(got[i]) - want[i];
    e += d * d;
    r += static_cast<double>(want[i]) * want[i];
  }
  return r > 0 ? std::sqrt(e / r) : std::sqrt(e);
}

/// End-to-end distance of a GPU result from the oracle in the same activation
/// mode, next to the quantization noise (its distance from the F32 oracle).
struct E2e {
  double rms{0};        ///< RelRms to the matching oracle
  double max{0};        ///< max |diff| / rms to the matching oracle
  double noise_rms{0};  ///< RelRms to the F32 oracle
  double noise_max{0};
  void Merge(const E2e& o) {
    rms = std::max(rms, o.rms);
    max = std::max(max, o.max);
    noise_rms = std::max(noise_rms, o.noise_rms);
    noise_max = std::max(noise_max, o.noise_max);
  }
};

inline E2e MeasureE2e(const std::vector<float>& got,
                      const std::vector<float>& matched,
                      const std::vector<float>& exact) {
  E2e e;
  e.rms = RelRms(got, matched);
  e.max = Compare(got, matched).max_over_rms;
  e.noise_rms = RelRms(got, exact);
  e.noise_max = Compare(got, exact).max_over_rms;
  return e;
}

/// The arithmetic is checked exactly stage by stage (captured activations);
/// end to end the two sides quantize their own (float-noise different)
/// intermediates, so Q8 codes at rounding ties flip and the flips compound
/// through later projections. The end-to-end bound is therefore a sanity
/// bound that any wiring error (wrong weight, mask, position, tap) exceeds:
/// it is at or above the quantization noise itself. The GPU must sit at most
/// a quarter of the noise's rms distance from the oracle in its own mode, and
/// its worst element at most half the noise's worst.
inline void RequireE2e(const std::string& what, const E2e& e) {
  if (!(e.rms <= 0.25 * e.noise_rms && e.max <= 0.5 * e.noise_max)) {
    std::ostringstream m;
    m << what << ": differs from the oracle by " << e.rms << " rms / " << e.max
      << " max of rms; Q8 noise " << e.noise_rms << " / " << e.noise_max;
    Require(false, m.str());
  }
}

inline std::string E2eText(const E2e& e) {
  std::ostringstream m;
  m << std::setprecision(3) << std::scientific << e.rms << " rms / " << e.max
    << " max vs Q8 noise " << e.noise_rms << " / " << e.noise_max;
  return m.str();
}

/// Relative closeness of a norm output computed by two implementations of
/// the same formula (double sum, one float rounding per product): 2e-6 of the
/// element plus 1e-6 of the rms covers one ulp of the shared scale.
inline void CheckNorm(const std::string& what, const std::vector<float>& got,
                      const std::vector<float>& want, double* worst) {
  ExpectRelNear(got, want, 2e-6, 1e-6 * Rms(want), worst, what);
}

struct EncoderReport {
  std::string label;
  double aux_rel{0};
  double norm_rel{0};
  ProjStats proj;
  E2e e2e;
};

/// The encoder's stages of one tapped run against the oracle's building
/// blocks (the GPU's own upstream stage as input). Fills the stage fields of
/// `rep`; throws on any failure.
inline void CheckEncoderStages(const std::string& label, const TinyDrafter& fx,
                               const std::vector<float>& features,
                               std::uint32_t T, bool vector, const Taps& taps,
                               EncoderReport* rep) {
  const ml::DFlashConfig& c = fx.config();
  const std::size_t H = c.hidden_size;
  const std::size_t width = c.fused_input_width();
  const auto aux_w = Values(fx.drafter.aux_norm);
  const auto aux_want =
      NormRows(features, H, aux_w.data(), c.tap_count(), c.rms_eps);
  const auto& aux = taps.F("enc_aux_norm");
  Require(aux.size() == T * width, label + ": enc_aux_norm size");
  CheckNorm(label + " enc_aux_norm", aux, aux_want, &rep->aux_rel);
  const auto& fc_out = taps.F("fc_out");
  const auto caps =
      StageCaptures(taps, "fc_out", 1, fx.drafter.fc.type, vector, label);
  CheckStage(label, "fc_out", fx.drafter.fc, aux, T, fc_out, caps[0],
             &rep->proj);
  const auto enc_w = Values(fx.drafter.enc_output_norm);
  CheckNorm(label + " enc_norm_out", taps.F("enc_norm_out"),
            NormRows(fc_out, H, enc_w.data(), 1, c.rms_eps), &rep->norm_rel);
}

/// Checks one encoder run (taps of a tapped run, `fused` its output) stage by
/// stage and end to end against the oracle. `features` are what the oracle
/// is given (a mutated GPU input must therefore fail). Throws on any failure.
inline EncoderReport CheckEncoder(const std::string& label,
                                  const TinyDrafter& fx,
                                  const std::vector<float>& features,
                                  std::uint32_t T, bool vector,
                                  const Taps& taps,
                                  const std::vector<float>& fused) {
  EncoderReport rep;
  rep.label = label;
  const ml::DFlashConfig& c = fx.config();
  CheckEncoderStages(label, fx, features, T, vector, taps, &rep);
  Require(BitIdentical(taps.F("enc_norm_out"), fused),
          label + ": output differs from the enc_norm_out tap");
  // End to end against DFlashModel itself (its encoder taps), in the GPU's
  // activation mode, and the F32 oracle for the quantization noise.
  const auto pos = Iota(0, T);
  const auto block =
      BlockTokens(1, 2, static_cast<std::int32_t>(c.mask_token_id));
  const ref::ActivationQuant mode = ModeFor(vector);
  const OracleTaps want = RunDrafterOracle(fx, features, pos, block,
                                           static_cast<std::int32_t>(T), mode);
  const OracleTaps exact =
      RunDrafterOracle(fx, features, pos, block, static_cast<std::int32_t>(T),
                       ref::ActivationQuant::kNone);
  // The two sides quantize the same aux rows (equal to float noise), so they
  // differ only by an occasional code flip at a rounding tie and by the
  // summation order.
  rep.e2e = MeasureE2e(fused, want.F("enc_norm_out"), exact.F("enc_norm_out"));
  RequireE2e(label + " encoder output", rep.e2e);
  return rep;
}

inline void PrintEncoder(const EncoderReport& r) {
  std::cout << std::setprecision(3) << std::scientific << "  " << r.label
            << ": aux " << r.aux_rel << ", fc " << r.proj.over_rms
            << " of rms (" << r.proj.over_bound << " of bound, boundary codes "
            << r.proj.boundary_codes << "), norm " << r.norm_rel << ", e2e "
            << E2eText(r.e2e) << std::defaultfloat << "\n";
}

// ---------------------------------------------------------------------------
// K/V injection
// ---------------------------------------------------------------------------

/// The F16 K and V rows of positions [begin, end) of every layer, read from
/// the rings: [position][layer][kv_dim] raw bits.
struct KvRows {
  std::uint32_t begin{0};
  std::uint32_t end{0};
  std::size_t layers{0};
  std::size_t kvd{0};
  std::vector<std::uint16_t> k, v;
  [[nodiscard]] const std::uint16_t* K(std::uint32_t p, std::size_t l) const {
    return k.data() + ((p - begin) * layers + l) * kvd;
  }
  [[nodiscard]] const std::uint16_t* V(std::uint32_t p, std::size_t l) const {
    return v.data() + ((p - begin) * layers + l) * kvd;
  }
};

inline KvRows ReadRing(const gpu::DrafterKv& kv, std::uint32_t begin,
                       std::uint32_t end) {
  KvRows r;
  r.begin = begin;
  r.end = end;
  r.layers = kv.layers();
  r.kvd = kv.row_bytes() / 2;
  r.k.resize((end - begin) * r.layers * r.kvd);
  r.v.resize(r.k.size());
  for (std::size_t l = 0; l < r.layers; ++l) {
    const auto keys = ToHost(static_cast<const std::uint16_t*>(
                                 kv.keys(static_cast<std::uint32_t>(l))),
                             std::size_t{kv.capacity()} * r.kvd);
    const auto vals = ToHost(static_cast<const std::uint16_t*>(
                                 kv.values(static_cast<std::uint32_t>(l))),
                             std::size_t{kv.capacity()} * r.kvd);
    for (std::uint32_t p = begin; p < end; ++p) {
      const std::size_t slot = kv.Slot(p);
      std::copy_n(keys.data() + slot * r.kvd, r.kvd,
                  r.k.data() + ((p - begin) * r.layers + l) * r.kvd);
      std::copy_n(vals.data() + slot * r.kvd, r.kvd,
                  r.v.data() + ((p - begin) * r.layers + l) * r.kvd);
    }
  }
  return r;
}

struct InjectReport {
  std::string label;
  EncoderReport enc;
  double kv_norm_rel{0};
  ProjStats proj;
  double rope_over_amax{0};  ///< K float disagreement before F16 / head amax
  double k_excess{0};        ///< K beyond half an F16 ulp, / |x|
  double v_excess{0};
  bool has_e2e{false};
  E2e e2e_k;  ///< stored K vs the oracle
  E2e e2e_v;
};

/// F16 storage check of K/V: each stored binary16 must be within half an ulp
/// of `want` plus `slack` of the head's largest element (K: the float
/// disagreement of norm and rope; V is stored from the v projection itself, so
/// its slack is 0: pure rounding). Returns the worst excess over half an ulp
/// relative to the element; throws on a violation.
inline double CheckF16Rows(const std::string& what, const std::uint16_t* got,
                           const float* want, std::size_t count, double slack) {
  double excess = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double g = HalfToFloat(got[i]);
    const double w = want[i];
    const double err = std::fabs(g - w);
    const double half_ulp = 0.5 * HalfUlp(w);
    double amax = 0;
    if (slack > 0) {
      const std::size_t h = i / 128;
      for (std::size_t d = 0; d < 128; ++d) {
        amax =
            std::max(amax, static_cast<double>(std::fabs(want[h * 128 + d])));
      }
    }
    const double allowed = half_ulp * (1.0 + 1e-6) +
                           (slack > 0 ? 3e-6 * std::fabs(w) : 0.0) +
                           slack * amax;
    excess = std::max(
        excess, std::max(0.0, err - half_ulp) / std::max(std::fabs(w), 1e-3));
    if (err > allowed) {
      std::ostringstream m;
      m << what << ": element " << i << " stored " << g << ", want " << w;
      Require(false, m.str());
    }
  }
  return excess;
}

/// NeoX RoPE of the drafter at `positions` (the oracle's ApplyRope).
inline void DrafterRope(const ml::DFlashConfig& c, float* x,
                        const std::vector<std::int32_t>& positions,
                        std::uint32_t heads) {
  ref::RopeParams rope;
  rope.n_rot = c.rotary_dim;
  rope.freq_base = c.rope_theta;
  ref::ApplyRope(x, positions, heads, c.head_dim, rope);
}

/// Per-head RMSNorm with a [128] weight, the oracle's.
inline std::vector<float> NormHeads(const std::vector<float>& x,
                                    const ml::TensorRef& gamma, float eps) {
  const auto g = Values(gamma);
  return NormRows(x, 128, g.data(), 1, eps);
}

/// Stages of one injection chunk (rows at positions pos .. pos + T - 1,
/// features `features`) from its taps, and the ring rows it stored (`ring`,
/// read right after the call). Throws on any failure.
inline void CheckInjectChunk(const std::string& label, const TinyDrafter& fx,
                             const std::vector<float>& features,
                             std::uint32_t T, std::uint32_t pos, bool vector,
                             const Taps& taps, const KvRows& ring,
                             InjectReport* rep) {
  const ml::DFlashConfig& c = fx.config();
  const std::size_t H = c.hidden_size;
  const std::size_t kvd = c.kv_dim();
  CheckEncoderStages(label, fx, features, T, vector, taps, &rep->enc);
  const auto& fused = taps.F("enc_norm_out");
  const auto positions = Iota(static_cast<std::int32_t>(pos), T);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const ml::DFlashLayerWeights& lw = fx.drafter.layers[il];
    const std::string sfx = "-" + std::to_string(il);
    const auto norm_w = Values(lw.attn_norm);
    const auto& x = taps.F("kv_inp_normed" + sfx);
    CheckNorm(label + " kv_inp_normed" + sfx, x,
              NormRows(fused, H, norm_w.data(), 1, c.rms_eps),
              &rep->kv_norm_rel);
    const auto kc = StageCaptures(taps, "Kproj_injected" + sfx, 1,
                                  lw.attn_k.type, vector, label);
    const auto vc = StageCaptures(taps, "Vcur_injected" + sfx, 1,
                                  lw.attn_v.type, vector, label);
    if (kc[0].layout == vc[0].layout) {
      const std::string why = SharedInputMismatch({&kc[0], &vc[0]});
      Require(why.empty(), label + ": K/V input" + sfx + ": " + why);
    }
    const auto& kp = taps.F("Kproj_injected" + sfx);
    const auto& vp = taps.F("Vcur_injected" + sfx);
    CheckStage(label, "Kproj_injected" + sfx, lw.attn_k, x, T, kp, kc[0],
               &rep->proj);
    CheckStage(label, "Vcur_injected" + sfx, lw.attn_v, x, T, vp, vc[0],
               &rep->proj);
    // Stored rows: K = rope(norm(GPU projection)), V = GPU projection.
    std::vector<float> k_want = NormHeads(kp, lw.attn_k_norm, c.rms_eps);
    DrafterRope(c, k_want.data(), positions, c.num_kv_heads);
    for (std::uint32_t t = 0; t < T; ++t) {
      const std::uint32_t p = pos + t;
      Require(p >= ring.begin && p < ring.end, label + ": ring rows missing");
      rep->k_excess = std::max(
          rep->k_excess,
          CheckF16Rows(label + " stored K" + sfx + " pos " + std::to_string(p),
                       ring.K(p, il), k_want.data() + t * kvd, kvd, 1e-6));
      rep->v_excess = std::max(
          rep->v_excess,
          CheckF16Rows(label + " stored V" + sfx + " pos " + std::to_string(p),
                       ring.V(p, il), vp.data() + t * kvd, kvd, 0.0));
    }
  }
}

/// Stored K/V of positions [0, N) (rows `hist`) against the oracle's
/// Kcur_injected / Vcur_injected for the same features (same activation mode)
/// and against the F32 oracle (the quantization noise).
inline void CheckInjectEndToEnd(const std::string& label, const TinyDrafter& fx,
                                const std::vector<float>& features,
                                std::uint32_t N, bool vector,
                                const KvRows& hist, InjectReport* rep) {
  const ml::DFlashConfig& c = fx.config();
  const std::size_t kvd = c.kv_dim();
  const auto pos = Iota(0, N);
  const auto block =
      BlockTokens(1, 2, static_cast<std::int32_t>(c.mask_token_id));
  const OracleTaps want = RunDrafterOracle(
      fx, features, pos, block, static_cast<std::int32_t>(N), ModeFor(vector));
  const OracleTaps exact =
      RunDrafterOracle(fx, features, pos, block, static_cast<std::int32_t>(N),
                       ref::ActivationQuant::kNone);
  rep->has_e2e = true;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::string sfx = "-" + std::to_string(il);
    std::vector<float> k(N * kvd), v(N * kvd);
    for (std::uint32_t p = 0; p < N; ++p) {
      for (std::size_t i = 0; i < kvd; ++i) {
        k[p * kvd + i] = HalfToFloat(hist.K(p, il)[i]);
        v[p * kvd + i] = HalfToFloat(hist.V(p, il)[i]);
      }
    }
    rep->e2e_k.Merge(MeasureE2e(k, want.F("Kcur_injected" + sfx),
                                exact.F("Kcur_injected" + sfx)));
    rep->e2e_v.Merge(MeasureE2e(v, want.F("Vcur_injected" + sfx),
                                exact.F("Vcur_injected" + sfx)));
  }
  RequireE2e(label + " stored K", rep->e2e_k);
  RequireE2e(label + " stored V", rep->e2e_v);
}

inline void PrintInject(const InjectReport& r) {
  std::cout << std::setprecision(3) << std::scientific << "  " << r.label
            << ": fc " << r.enc.proj.over_bound << " of bound, k/v proj "
            << r.proj.over_rms << " of rms (" << r.proj.over_bound
            << " of bound, boundary codes " << r.proj.boundary_codes
            << "), K/V F16 excess " << r.k_excess << " / " << r.v_excess;
  if (r.has_e2e) {
    std::cout << "\n    e2e K " << E2eText(r.e2e_k) << "\n    e2e V "
              << E2eText(r.e2e_v);
  }
  std::cout << std::defaultfloat << "\n";
}

// ---------------------------------------------------------------------------
// Block attention
// ---------------------------------------------------------------------------

/// Attention of query rows at `qpos` over keys at `kpos` (F16-rounded rows as
/// stored), key j visible from p iff lo(p) <= kpos[j] <= hi(p) with lo(p) =
/// max(p + 1 - window, floor) and hi(p) = p (causal) or `last`. Double
/// precision (`plain` false: the oracle) or plain float with sequential sums
/// (`plain` true: the float noise of this input).
inline std::vector<float> AttentionAt(
    const std::vector<float>& q, const std::vector<std::int32_t>& qpos,
    const std::vector<float>& k, const std::vector<float>& v,
    const std::vector<std::int32_t>& kpos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t window, std::int32_t floor,
    bool causal, std::int32_t last, bool plain) {
  const std::uint32_t group = heads / kv_heads;
  const std::size_t qrow = std::size_t{heads} * 128;
  const std::size_t kvrow = std::size_t{kv_heads} * 128;
  const float scale = 1.0F / std::sqrt(128.0F);
  std::vector<float> out(qpos.size() * qrow, 0.0F);
  for (std::size_t r = 0; r < qpos.size(); ++r) {
    const std::int64_t p = qpos[r];
    const std::int64_t lo = std::max<std::int64_t>(p + 1 - window, floor);
    const std::int64_t hi = causal ? p : last;
    std::vector<std::size_t> keys;
    for (std::size_t j = 0; j < kpos.size(); ++j) {
      if (kpos[j] >= lo && kpos[j] <= hi) {
        keys.push_back(j);
      }
    }
    for (std::uint32_t h = 0; h < heads; ++h) {
      const float* qh = q.data() + r * qrow + std::size_t{h} * 128;
      const std::size_t kvh = h / group;
      float* oh = out.data() + r * qrow + std::size_t{h} * 128;
      if (keys.empty()) {
        continue;
      }
      if (plain) {
        std::vector<float> sc(keys.size());
        float top = -INFINITY;
        for (std::size_t i = 0; i < keys.size(); ++i) {
          const float* kj = k.data() + keys[i] * kvrow + kvh * 128;
          float dot = 0.0F;
          for (int d = 0; d < 128; ++d) {
            dot += qh[d] * kj[d];
          }
          sc[i] = dot * scale;
          top = std::max(top, sc[i]);
        }
        float den = 0.0F;
        for (float& x : sc) {
          x = std::exp(x - top);
          den += x;
        }
        for (int d = 0; d < 128; ++d) {
          float acc = 0.0F;
          for (std::size_t i = 0; i < keys.size(); ++i) {
            acc += sc[i] * v[keys[i] * kvrow + kvh * 128 + d];
          }
          oh[d] = acc / den;
        }
      } else {
        std::vector<double> sc(keys.size());
        double top = -INFINITY;
        for (std::size_t i = 0; i < keys.size(); ++i) {
          const float* kj = k.data() + keys[i] * kvrow + kvh * 128;
          double dot = 0.0;
          for (int d = 0; d < 128; ++d) {
            dot += static_cast<double>(qh[d]) * kj[d];
          }
          sc[i] = dot * static_cast<double>(scale);
          top = std::max(top, sc[i]);
        }
        double den = 0.0;
        for (double& x : sc) {
          x = std::exp(x - top);
          den += x;
        }
        for (int d = 0; d < 128; ++d) {
          double acc = 0.0;
          for (std::size_t i = 0; i < keys.size(); ++i) {
            acc += sc[i] * v[keys[i] * kvrow + kvh * 128 + d];
          }
          oh[d] = static_cast<float>(acc / den);
        }
      }
    }
  }
  return out;
}

/// What a block attention case runs: the context [floor, n) already in the
/// ring, B block rows at n through drafter layer `layer`.
struct BlockSpec {
  std::uint32_t layer{0};
  std::uint32_t floor{0};
  std::uint32_t n{0};
  std::uint32_t B{16};
  bool vector{true};
};

struct AttnCheckReport {
  std::string label;
  ProjStats proj;
  double rope_over_amax{0};
  double k_excess{0};
  double v_excess{0};
  double attn_over_rms{0};
  double attn_float_over_rms{0};
  double window_ratio{-1};  ///< wrong-window distance / ours (boundary rows)
  double causal_ratio{-1};  ///< non-causal distance / ours (rows 0..B-2)
  double gate_rel{0};
  bool has_e2e{false};
  E2e gated;  ///< attn_gated vs the oracle
  E2e o;      ///< attn_o_proj vs the oracle (ffn_inp - h)
};

/// Stage checks of one tapped Attention call: projections against captured
/// activations, q norm and rope, the block's stored K/V, attention against
/// the double oracle over the ring as stored (with window and causality
/// probes), the softplus gate, and the output projection. `ring` holds the
/// rows [ring.begin, n + B) read after the call. Throws on any failure.
inline void CheckAttentionStages(const std::string& label,
                                 const TinyDrafter& fx, const BlockSpec& b,
                                 const std::vector<float>& normed,
                                 const Taps& taps, const KvRows& ring,
                                 AttnCheckReport* rep) {
  const ml::DFlashConfig& c = fx.config();
  const ml::DFlashLayerWeights& lw = fx.drafter.layers[b.layer];
  const std::string sfx = "-" + std::to_string(b.layer);
  const std::uint32_t B = b.B;
  const std::uint32_t heads = c.num_heads;
  const std::uint32_t kvh = c.num_kv_heads;
  const std::size_t qd = c.q_dim();
  const std::size_t kvd = c.kv_dim();
  const auto& qp = taps.F("Qproj" + sfx);
  const auto& kp = taps.F("Kproj" + sfx);
  const auto& vp = taps.F("Vcur" + sfx);
  const auto& gp = taps.F("attn_gate_proj" + sfx);
  // Projections of the shared input; the same-layout ones must have used one
  // quantization of it.
  struct P {
    const char* name;
    const ml::TensorRef* w;
    const std::vector<float>* out;
  };
  const P projections[] = {{"Qproj", &lw.attn_q, &qp},
                           {"Kproj", &lw.attn_k, &kp},
                           {"Vcur", &lw.attn_v, &vp},
                           {"attn_gate_proj", &lw.attn_gate, &gp}};
  std::vector<QuantRows> caps;
  for (const P& pr : projections) {
    caps.push_back(
        StageCaptures(taps, pr.name + sfx, 1, pr.w->type, b.vector, label)[0]);
    CheckStage(label, pr.name + sfx, *pr.w, normed, B, *pr.out, caps.back(),
               &rep->proj);
  }
  {
    std::map<int, std::vector<const QuantRows*>> by_layout;
    for (const QuantRows& q : caps) {
      by_layout[q.layout].push_back(&q);
    }
    for (const auto& [layout, list] : by_layout) {
      const std::string why = SharedInputMismatch(list);
      Require(why.empty(), label + ": block input" + sfx + ": " + why);
    }
  }
  // q after norm and rope; stored block K and V.
  const auto qpos = Iota(static_cast<std::int32_t>(b.n), B);
  std::vector<float> q_want = NormHeads(qp, lw.attn_q_norm, c.rms_eps);
  DrafterRope(c, q_want.data(), qpos, heads);
  const auto& q_rope = taps.F("Qcur" + sfx);
  for (std::size_t h = 0; h < std::size_t{B} * heads; ++h) {
    double amax = 0;
    for (int d = 0; d < 128; ++d) {
      amax =
          std::max(amax, static_cast<double>(std::fabs(q_want[h * 128 + d])));
    }
    for (int d = 0; d < 128; ++d) {
      const double err = std::fabs(static_cast<double>(q_rope[h * 128 + d]) -
                                   q_want[h * 128 + d]);
      rep->rope_over_amax = std::max(rep->rope_over_amax, err / amax);
      Require(err <= 1e-6 * amax,
              label + ": Qcur" + sfx + " differs from norm+rope, " +
                  std::to_string(err / amax) + " of the head's largest");
    }
  }
  std::vector<float> k_want = NormHeads(kp, lw.attn_k_norm, c.rms_eps);
  DrafterRope(c, k_want.data(), qpos, kvh);
  for (std::uint32_t r = 0; r < B; ++r) {
    rep->k_excess = std::max(
        rep->k_excess,
        CheckF16Rows(label + " block K" + sfx, ring.K(b.n + r, b.layer),
                     k_want.data() + r * kvd, kvd, 1e-6));
    rep->v_excess =
        std::max(rep->v_excess, CheckF16Rows(label + " block V" + sfx,
                                             ring.V(b.n + r, b.layer),
                                             vp.data() + r * kvd, kvd, 0.0));
  }
  // Attention over the ring as stored.
  std::vector<std::int32_t> kpos;
  std::vector<float> kf, vf;
  for (std::uint32_t p = ring.begin; p < b.n + B; ++p) {
    kpos.push_back(static_cast<std::int32_t>(p));
    for (std::size_t i = 0; i < kvd; ++i) {
      kf.push_back(HalfToFloat(ring.K(p, b.layer)[i]));
      vf.push_back(HalfToFloat(ring.V(p, b.layer)[i]));
    }
  }
  const auto fl = static_cast<std::int32_t>(b.floor);
  const auto last = static_cast<std::int32_t>(b.n + B - 1);
  const auto& attn = taps.F("attn_out" + sfx);
  const auto want = AttentionAt(q_rope, qpos, kf, vf, kpos, heads, kvh,
                                c.sliding_window, fl, true, last, false);
  const auto plain = AttentionAt(q_rope, qpos, kf, vf, kpos, heads, kvh,
                                 c.sliding_window, fl, true, last, true);
  const double rms = Rms(want);
  const double ours = MaxAbsDiff(attn, want);
  rep->attn_over_rms = ours / rms;
  rep->attn_float_over_rms = MaxAbsDiff(plain, want) / rms;
  Require(rep->attn_over_rms <= std::max(3.0 * rep->attn_float_over_rms, 2e-6),
          label + ": attention" + sfx + " differs from the double oracle, " +
              std::to_string(rep->attn_over_rms) + " of rms; plain float is " +
              std::to_string(rep->attn_float_over_rms));
  // Probes: the same rows with the window one key narrower / wider, and with a
  // non-causal block, must be far from the GPU (where they differ at all).
  {
    double worst = 1e30;
    bool probed = false;
    for (const std::uint32_t wnd :
         {c.sliding_window - 1, c.sliding_window + 1}) {
      const auto alt = AttentionAt(q_rope, qpos, kf, vf, kpos, heads, kvh, wnd,
                                   fl, true, last, false);
      const double d = MaxAbsDiff(alt, want);
      if (d > 0) {
        probed = true;
        worst = std::min(worst, MaxAbsDiff(attn, alt));
      }
    }
    if (probed) {
      rep->window_ratio = worst / std::max(ours, 1e-12);
      Require(rep->window_ratio > 100.0,
              label + ": window probe too weak (ratio " +
                  std::to_string(rep->window_ratio) + ")");
    }
    if (B > 1) {
      const auto alt = AttentionAt(q_rope, qpos, kf, vf, kpos, heads, kvh,
                                   c.sliding_window, fl, false, last, false);
      rep->causal_ratio = MaxAbsDiff(attn, alt) / std::max(ours, 1e-12);
      Require(rep->causal_ratio > 100.0,
              label + ": causality probe too weak (ratio " +
                  std::to_string(rep->causal_ratio) + ")");
    }
  }
  // Gate: softplus of the GPU's gate rows; gated = attn * gate, bitwise.
  const auto& sp = taps.F("attn_gate_softplus" + sfx);
  {
    std::vector<float> want_sp(gp.size());
    for (std::size_t i = 0; i < gp.size(); ++i) {
      const double g = gp[i];
      want_sp[i] =
          static_cast<float>(g > 20.0 ? g : std::log(1.0 + std::exp(g)));
    }
    ExpectRelNear(sp, want_sp, 2e-6, 1e-7, &rep->gate_rel,
                  label + " softplus" + sfx);
    std::vector<float> gated(attn.size());
    for (std::size_t i = 0; i < gated.size(); ++i) {
      gated[i] = attn[i] * sp[i / 128];
    }
    Require(BitIdentical(taps.F("attn_gated" + sfx), gated),
            label + ": gating is not attn * softplus(gate)");
  }
  const auto oc = StageCaptures(taps, "attn_o_proj" + sfx, 1,
                                lw.attn_output.type, b.vector, label);
  CheckStage(label, "attn_o_proj" + sfx, lw.attn_output,
             taps.F("attn_gated" + sfx), B, taps.F("attn_o_proj" + sfx), oc[0],
             &rep->proj);
}

/// The oracle's attention block of `layer` (gated rows and output projection)
/// against the GPU's. `matched`/`exact` are DFlashModel runs (same mode / F32)
/// over the same context and block. The output projection is recovered as
/// ffn_inp - h.
inline void CheckAttentionEndToEnd(const std::string& label, const BlockSpec& b,
                                   const Taps& taps, const OracleTaps& matched,
                                   const OracleTaps& exact,
                                   AttnCheckReport* rep) {
  const std::string sfx = "-" + std::to_string(b.layer);
  const auto o_of = [&](const OracleTaps& o) {
    const auto& ffn_inp = o.F("ffn_inp" + sfx);
    const auto& h = b.layer == 0 ? o.F("inp_noise_embd")
                                 : o.F("l_out-" + std::to_string(b.layer - 1));
    std::vector<float> d(ffn_inp.size());
    for (std::size_t i = 0; i < d.size(); ++i) {
      d[i] = ffn_inp[i] - h[i];
    }
    return d;
  };
  rep->has_e2e = true;
  rep->gated =
      MeasureE2e(taps.F("attn_gated" + sfx), matched.F("attn_gated" + sfx),
                 exact.F("attn_gated" + sfx));
  rep->o = MeasureE2e(taps.F("attn_o_proj" + sfx), o_of(matched), o_of(exact));
  RequireE2e(label + " attn_gated" + sfx, rep->gated);
  RequireE2e(label + " attn_o_proj" + sfx, rep->o);
}

inline void PrintAttnCheck(const AttnCheckReport& r) {
  std::cout << std::setprecision(3) << std::scientific << "  " << r.label
            << ": proj " << r.proj.over_rms << " (" << r.proj.over_bound
            << " of bound), rope " << r.rope_over_amax << ", K/V F16 excess "
            << r.k_excess << "/" << r.v_excess << ", attn " << r.attn_over_rms
            << " (plain float " << r.attn_float_over_rms << ")";
  if (r.window_ratio >= 0) {
    std::cout << ", window probe x" << r.window_ratio;
  }
  if (r.causal_ratio >= 0) {
    std::cout << ", causal probe x" << r.causal_ratio;
  }
  if (r.has_e2e) {
    std::cout << "\n    gated " << E2eText(r.gated) << "\n    o-proj "
              << E2eText(r.o);
  }
  std::cout << std::defaultfloat << "\n";
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_DFLASH_GPU_HPP_
