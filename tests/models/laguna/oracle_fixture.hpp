#ifndef GUFO_TESTS_MODELS_LAGUNA_ORACLE_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_ORACLE_FIXTURE_HPP_

// Tiny synthetic Laguna models for the oracle tests, and an independent naive
// forward written from ARCH.md and the Hugging Face formulation (rotate_half
// RoPE with the inv_freq blend, a j > p - window sliding mask, double
// precision throughout). It shares no code with src/models/laguna/reference.
//
// Every weight is random (default init hid bugs before), including norms, the
// router, its selection bias and the attention gate, and the dimensions are
// all different from each other so a transposed or swapped operand cannot
// cancel out.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "src/models/laguna/reference.hpp"
#include "src/models/laguna/weights.hpp"
#include "tests/models/laguna/support.hpp"

namespace gufo::testing::laguna {

namespace ml = gufo::models::laguna;
namespace ref = gufo::models::laguna::reference;
using core::GgmlType;

/// One tensor: its logical float values (what a perfect decoder returns) and
/// its stored bytes. Row-major [experts][rows][cols], like the GGUF.
struct Mat {
  GgmlType type{GgmlType::kF32};
  std::size_t cols{0};
  std::size_t rows{1};
  std::size_t experts{1};
  std::vector<float> values;
  std::vector<std::uint8_t> bytes;
  [[nodiscard]] double At(std::size_t e, std::size_t r, std::size_t c) const {
    return values[(e * rows + r) * cols + c];
  }
};

struct TinySpec {
  std::uint32_t layers{5};
  std::uint32_t hidden{64};
  std::uint32_t head_dim{32};
  std::uint32_t kv_heads{2};
  std::uint32_t heads_full{12};  // GQA group 6
  std::uint32_t heads_swa{18};   // GQA group 9
  std::uint32_t rotary_full{16};
  std::uint32_t rotary_swa{32};
  std::uint32_t window{4};
  std::uint32_t experts{14};
  std::uint32_t used{3};
  std::uint32_t expert_ff{32};
  std::uint32_t shared_ff{64};
  std::uint32_t dense_ff{96};
  std::uint32_t vocab{50};
  float yarn_factor{32.0F};
  float rope_attn_factor{1.0F};
  bool q8_0_matrices{false};  // projections/experts/embd/output stored Q8_0
  std::uint64_t seed{1};
};

struct LayerMats {
  Mat attn_norm, q, k, v, o, q_norm, k_norm, gate, ffn_norm;
  Mat dense_gate, dense_up, dense_down;
  Mat router, bias, gate_exps, up_exps, down_exps;
  Mat shared_gate, shared_up, shared_down;
};

inline std::uint16_t HalfBits(float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}
inline float HalfValue(std::uint16_t bits) {
  _Float16 half{};
  std::memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
}

/// Stores `m.values` in `m.type` (F32 or Q8_0), rewriting `values` to what
/// that storage decodes to.
inline void Encode(Mat& m) {
  if (m.type == GgmlType::kF32) {
    m.bytes.resize(m.values.size() * sizeof(float));
    std::memcpy(m.bytes.data(), m.values.data(), m.bytes.size());
    return;
  }
  Require(m.type == GgmlType::kQ8_0 && m.cols % 32 == 0, "Encode: Q8_0 shape");
  const std::size_t blocks = m.values.size() / 32;
  m.bytes.assign(blocks * 34, 0);
  for (std::size_t b = 0; b < blocks; ++b) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(m.values[b * 32 + j]));
    }
    const float d = amax / 127.0F;
    const std::uint16_t dh = HalfBits(d);
    std::memcpy(&m.bytes[b * 34], &dh, 2);
    for (std::size_t j = 0; j < 32; ++j) {
      const int q =
          d != 0.0F ? static_cast<int>(std::lround(m.values[b * 32 + j] / d))
                    : 0;
      m.bytes[b * 34 + 2 + j] =
          static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
      m.values[b * 32 + j] = static_cast<float>(q) * HalfValue(dh);
    }
  }
}

class Fixture {
public:
  explicit Fixture(const TinySpec& spec) : spec_(spec), rng_(spec.seed) {
    Build();
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  [[nodiscard]] const ml::ModelWeights& weights() const { return weights_; }
  [[nodiscard]] const ml::Config& config() const { return weights_.config; }
  [[nodiscard]] const TinySpec& spec() const { return spec_; }
  [[nodiscard]] const LayerMats& layer(std::uint32_t il) const {
    return layers_[il];
  }
  [[nodiscard]] const Mat& embd() const { return embd_; }
  [[nodiscard]] const Mat& output() const { return output_; }
  [[nodiscard]] const Mat& output_norm() const { return output_norm_; }

  /// Replaces an F32 tensor's contents and refreshes the weight views.
  void SetF32(Mat& m, std::vector<float> values) {
    Require(m.type == GgmlType::kF32 && values.size() == m.values.size(),
            "SetF32 shape");
    m.values = std::move(values);
    Encode(m);
    Rebind();
  }
  [[nodiscard]] Mat& MutableLayer(std::uint32_t il, Mat LayerMats::* field) {
    return layers_[il].*field;
  }

private:
  float Gain(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng_);
  }
  Mat Random(GgmlType type, std::size_t cols, std::size_t rows,
             std::size_t experts, float gain) {
    Mat m;
    m.type = type;
    m.cols = cols;
    m.rows = rows;
    m.experts = experts;
    m.values.resize(cols * rows * experts);
    const float std_dev = gain / std::sqrt(static_cast<float>(cols));
    std::normal_distribution<float> dist(0.0F, std_dev);
    for (float& v : m.values) {
      v = dist(rng_);
    }
    Encode(m);
    return m;
  }
  Mat Norm(std::size_t n) {
    Mat m;
    m.cols = n;
    m.values.resize(n);
    for (float& v : m.values) {
      v = Gain(0.5F, 1.5F);
    }
    Encode(m);
    return m;
  }
  GgmlType Quant() const {
    return spec_.q8_0_matrices ? GgmlType::kQ8_0 : GgmlType::kF32;
  }
  void Build() {
    const TinySpec& s = spec_;
    ml::Config& c = weights_.config;
    c.num_layers = s.layers;
    c.hidden_size = s.hidden;
    c.ffn_size = s.dense_ff;
    c.vocab_size = s.vocab;
    c.context_length = 262144;
    c.rms_eps = 1e-6F;
    c.num_kv_heads = s.kv_heads;
    c.head_dim = s.head_dim;
    c.sliding_window = s.window;
    c.rope_theta_full = 500000.0F;
    c.rope_theta_swa = 10000.0F;
    c.rotary_dim_full = s.rotary_full;
    c.rotary_dim_swa = s.rotary_swa;
    c.yarn.factor = s.yarn_factor;
    c.yarn.original_context = 8192;
    c.yarn.beta_fast = 32.0F;
    c.yarn.beta_slow = 1.0F;
    c.rope_attn_factor = s.rope_attn_factor;
    c.num_experts = s.experts;
    c.num_experts_used = s.used;
    c.expert_ff = s.expert_ff;
    c.shared_expert_ff = s.shared_ff;
    c.expert_weights_norm = true;
    c.expert_weights_scale = 2.5F;
    c.leading_dense_layers = 1;
    for (std::uint32_t il = 0; il < s.layers; ++il) {
      c.heads_per_layer.push_back(il % 4 == 0 ? s.heads_full : s.heads_swa);
    }
    const std::size_t H = s.hidden;
    embd_ = Random(Quant(), H, s.vocab, 1, 1.0F);
    output_norm_ = Norm(H);
    output_ = Random(Quant(), H, s.vocab, 1, 1.0F);
    layers_.resize(s.layers);
    for (std::uint32_t il = 0; il < s.layers; ++il) {
      LayerMats& l = layers_[il];
      const std::size_t heads = c.n_head(il);
      const std::size_t qd = heads * s.head_dim;
      const std::size_t kd = std::size_t{s.kv_heads} * s.head_dim;
      l.attn_norm = Norm(H);
      l.q = Random(Quant(), H, qd, 1, Gain(0.8F, 1.6F));
      l.k = Random(Quant(), H, kd, 1, Gain(0.8F, 1.6F));
      l.v = Random(Quant(), H, kd, 1, Gain(0.6F, 1.4F));
      l.o = Random(Quant(), qd, H, 1, Gain(0.6F, 1.2F));
      l.q_norm = Norm(s.head_dim);
      l.k_norm = Norm(s.head_dim);
      l.gate = Random(GgmlType::kF32, H, heads, 1, Gain(1.5F, 2.5F));
      l.ffn_norm = Norm(H);
      if (il == 0) {
        l.dense_gate = Random(Quant(), H, s.dense_ff, 1, Gain(0.8F, 1.4F));
        l.dense_up = Random(Quant(), H, s.dense_ff, 1, Gain(0.8F, 1.4F));
        l.dense_down = Random(Quant(), s.dense_ff, H, 1, Gain(0.6F, 1.2F));
      } else {
        l.router = Random(GgmlType::kF32, H, s.experts, 1, Gain(2.0F, 4.0F));
        l.bias = Norm(s.experts);
        for (float& v : l.bias.values) {
          v -= 1.0F;  // in [-0.5, 0.5]
        }
        Encode(l.bias);
        l.gate_exps =
            Random(Quant(), H, s.expert_ff, s.experts, Gain(0.8F, 1.4F));
        l.up_exps =
            Random(Quant(), H, s.expert_ff, s.experts, Gain(0.8F, 1.4F));
        l.down_exps =
            Random(Quant(), s.expert_ff, H, s.experts, Gain(0.6F, 1.2F));
        l.shared_gate = Random(Quant(), H, s.shared_ff, 1, Gain(0.8F, 1.4F));
        l.shared_up = Random(Quant(), H, s.shared_ff, 1, Gain(0.8F, 1.4F));
        l.shared_down = Random(Quant(), s.shared_ff, H, 1, Gain(0.6F, 1.2F));
      }
    }
    weights_.layers.resize(s.layers);
    Rebind();
  }
  ml::TensorRef Ref(const Mat& m, const std::string& name) {
    names_.push_back(name);
    ml::TensorRef t;
    t.data = m.bytes.data();
    t.type = m.type;
    t.cols = m.cols;
    t.rows = m.rows;
    t.experts = m.experts;
    t.name = names_.back();
    return t;
  }
  void Rebind() {
    names_.clear();
    weights_.token_embd = Ref(embd_, "token_embd.weight");
    weights_.output_norm = Ref(output_norm_, "output_norm.weight");
    weights_.output = Ref(output_, "output.weight");
    for (std::uint32_t il = 0; il < spec_.layers; ++il) {
      LayerMats& m = layers_[il];
      ml::LayerWeights& w = weights_.layers[il];
      w.full_attention = il % 4 == 0;
      w.moe = il != 0;
      const std::string p = "blk." + std::to_string(il) + ".";
      w.attn_norm = Ref(m.attn_norm, p + "attn_norm.weight");
      w.attn_q = Ref(m.q, p + "attn_q.weight");
      w.attn_k = Ref(m.k, p + "attn_k.weight");
      w.attn_v = Ref(m.v, p + "attn_v.weight");
      w.attn_output = Ref(m.o, p + "attn_output.weight");
      w.attn_q_norm = Ref(m.q_norm, p + "attn_q_norm.weight");
      w.attn_k_norm = Ref(m.k_norm, p + "attn_k_norm.weight");
      w.attn_gate = Ref(m.gate, p + "attn_gate.weight");
      w.ffn_norm = Ref(m.ffn_norm, p + "ffn_norm.weight");
      if (il == 0) {
        w.ffn_gate = Ref(m.dense_gate, p + "ffn_gate.weight");
        w.ffn_up = Ref(m.dense_up, p + "ffn_up.weight");
        w.ffn_down = Ref(m.dense_down, p + "ffn_down.weight");
      } else {
        w.ffn_gate_inp = Ref(m.router, p + "ffn_gate_inp.weight");
        w.ffn_exp_probs_b = Ref(m.bias, p + "exp_probs_b.bias");
        w.ffn_gate_exps = Ref(m.gate_exps, p + "ffn_gate_exps.weight");
        w.ffn_up_exps = Ref(m.up_exps, p + "ffn_up_exps.weight");
        w.ffn_down_exps = Ref(m.down_exps, p + "ffn_down_exps.weight");
        w.ffn_gate_shexp = Ref(m.shared_gate, p + "ffn_gate_shexp.weight");
        w.ffn_up_shexp = Ref(m.shared_up, p + "ffn_up_shexp.weight");
        w.ffn_down_shexp = Ref(m.shared_down, p + "ffn_down_shexp.weight");
      }
    }
  }

  TinySpec spec_;
  std::mt19937_64 rng_;
  ml::ModelWeights weights_;
  Mat embd_, output_norm_, output_;
  std::vector<LayerMats> layers_;
  std::deque<std::string> names_;
};

// ---------------------------------------------------------------------------
// Independent naive forward (double).
// ---------------------------------------------------------------------------

using Vec = std::vector<double>;

inline Vec MatVec(const Mat& m, std::size_t expert, const Vec& x) {
  Vec y(m.rows, 0.0);
  for (std::size_t r = 0; r < m.rows; ++r) {
    double acc = 0.0;
    for (std::size_t c = 0; c < m.cols; ++c) {
      acc += m.At(expert, r, c) * x[c];
    }
    y[r] = acc;
  }
  return y;
}

inline Vec NaiveRmsNorm(const Vec& x, const Mat& w, double eps) {
  double ms = 0.0;
  for (double v : x) {
    ms += v * v;
  }
  ms /= static_cast<double>(x.size());
  const double inv = 1.0 / std::sqrt(ms + eps);
  Vec y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    y[i] = x[i] * inv * w.values[i];
  }
  return y;
}

inline double NaiveSilu(double x) {
  return x / (1.0 + std::exp(-x));
}

/// Hugging Face rotate_half RoPE with the YaRN inv_freq blend and attention
/// factor, on the leading `rot` channels of one head.
inline Vec NaiveRope(const Vec& head, int pos, std::uint32_t rot, double base,
                     bool yarn, double factor, double orig, double beta_fast,
                     double beta_slow, double attn_factor) {
  const int half = static_cast<int>(rot / 2);
  Vec inv_freq(half);
  for (int i = 0; i < half; ++i) {
    inv_freq[i] = std::pow(base, -2.0 * i / static_cast<double>(rot));
  }
  double mscale = attn_factor;
  if (yarn) {
    const auto correction = [&](double rotations) {
      return rot * std::log(orig / (rotations * 2.0 * M_PI)) /
             (2.0 * std::log(base));
    };
    double low = std::floor(correction(beta_fast));
    double high = std::ceil(correction(beta_slow));
    low = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(rot) - 1.0);
    if (low == high) {
      high += 0.001;
    }
    for (int i = 0; i < half; ++i) {
      const double ramp = std::clamp((i - low) / (high - low), 0.0, 1.0);
      const double extrapolation = 1.0 - ramp;
      inv_freq[i] = (inv_freq[i] / factor) * (1.0 - extrapolation) +
                    inv_freq[i] * extrapolation;
    }
    mscale *= 0.1 * std::log(factor) + 1.0;
  }
  Vec out = head;
  for (int i = 0; i < half; ++i) {
    const double angle = pos * inv_freq[i];
    const double cs = std::cos(angle) * mscale;
    const double sn = std::sin(angle) * mscale;
    const double x1 = head[i];
    const double x2 = head[i + half];
    out[i] = x1 * cs - x2 * sn;  // x*cos + rotate_half(x)*sin
    out[i + half] = x2 * cs + x1 * sn;
  }
  return out;
}

struct NaiveResult {
  std::vector<Vec> logits;              // [T][vocab]
  std::vector<std::vector<Vec>> l_out;  // [layer][T][H]
};

inline NaiveResult NaiveForward(const Fixture& f,
                                const std::vector<std::int32_t>& tokens) {
  const ml::Config& c = f.config();
  const TinySpec& s = f.spec();
  const std::size_t T = tokens.size();
  const std::size_t H = s.hidden;
  const std::size_t hd = s.head_dim;
  const double eps = c.rms_eps;
  std::vector<Vec> h(T, Vec(H));
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t i = 0; i < H; ++i) {
      h[t][i] = f.embd().At(0, static_cast<std::size_t>(tokens[t]), i);
    }
  }
  NaiveResult result;
  for (std::uint32_t il = 0; il < s.layers; ++il) {
    const LayerMats& l = f.layer(il);
    const bool full = il % 4 == 0;
    const std::size_t heads = c.n_head(il);
    const std::size_t group = heads / s.kv_heads;
    std::vector<Vec> q(T), k(T), v(T), gate(T);
    for (std::size_t t = 0; t < T; ++t) {
      const Vec n = NaiveRmsNorm(h[t], l.attn_norm, eps);
      q[t] = MatVec(l.q, 0, n);
      k[t] = MatVec(l.k, 0, n);
      v[t] = MatVec(l.v, 0, n);
      gate[t] = MatVec(l.gate, 0, n);
      const std::uint32_t rot = full ? s.rotary_full : s.rotary_swa;
      const double base = full ? 500000.0 : 10000.0;
      const bool yarn = full && s.yarn_factor != 1.0F;
      const auto prep = [&](Vec& flat, std::size_t nh, const Mat& norm_w) {
        for (std::size_t hh = 0; hh < nh; ++hh) {
          Vec head(flat.begin() + hh * hd, flat.begin() + (hh + 1) * hd);
          head = NaiveRmsNorm(head, norm_w, eps);
          head = NaiveRope(head, static_cast<int>(t), rot, base, yarn,
                           s.yarn_factor, 8192, 32, 1,
                           full ? s.rope_attn_factor : 1.0);
          std::copy(head.begin(), head.end(), flat.begin() + hh * hd);
        }
      };
      prep(q[t], heads, l.q_norm);
      prep(k[t], s.kv_heads, l.k_norm);
    }
    std::vector<Vec> ffn_inp(T);
    for (std::size_t p = 0; p < T; ++p) {
      Vec gated(heads * hd, 0.0);
      for (std::size_t hh = 0; hh < heads; ++hh) {
        const std::size_t kvh = hh / group;
        std::vector<std::size_t> keys;
        for (std::size_t j = 0; j <= p; ++j) {
          // Transformers sliding mask: kv_idx > q_idx - sliding_window.
          const bool visible =
              full || static_cast<long>(j) >
                          static_cast<long>(p) - static_cast<long>(s.window);
          if (visible) {
            keys.push_back(j);
          }
        }
        Vec score(keys.size());
        double top = -1e300;
        for (std::size_t a = 0; a < keys.size(); ++a) {
          double dot = 0.0;
          for (std::size_t d = 0; d < hd; ++d) {
            dot += q[p][hh * hd + d] * k[keys[a]][kvh * hd + d];
          }
          score[a] = dot / std::sqrt(static_cast<double>(hd));
          top = std::max(top, score[a]);
        }
        double denom = 0.0;
        for (double& x : score) {
          x = std::exp(x - top);
          denom += x;
        }
        const double g = std::log1p(std::exp(gate[p][hh]));  // softplus
        for (std::size_t d = 0; d < hd; ++d) {
          double acc = 0.0;
          for (std::size_t a = 0; a < keys.size(); ++a) {
            acc += score[a] / denom * v[keys[a]][kvh * hd + d];
          }
          gated[hh * hd + d] = acc * g;
        }
      }
      const Vec o = MatVec(l.o, 0, gated);
      ffn_inp[p].resize(H);
      for (std::size_t i = 0; i < H; ++i) {
        ffn_inp[p][i] = h[p][i] + o[i];
      }
    }
    std::vector<Vec> out(T, Vec(H));
    for (std::size_t t = 0; t < T; ++t) {
      const Vec n = NaiveRmsNorm(ffn_inp[t], l.ffn_norm, eps);
      Vec mlp(H, 0.0);
      const auto swiglu = [&](const Mat& g, const Mat& u, const Mat& d,
                              std::size_t e) {
        Vec gv = MatVec(g, e, n);
        const Vec uv = MatVec(u, e, n);
        for (std::size_t i = 0; i < gv.size(); ++i) {
          gv[i] = NaiveSilu(gv[i]) * uv[i];
        }
        return MatVec(d, e, gv);
      };
      if (il == 0) {
        mlp = swiglu(l.dense_gate, l.dense_up, l.dense_down, 0);
      } else {
        const Vec logits = MatVec(l.router, 0, n);
        Vec prob(logits.size());
        std::vector<std::size_t> order(logits.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        for (std::size_t e = 0; e < logits.size(); ++e) {
          prob[e] = 1.0 / (1.0 + std::exp(-logits[e]));
        }
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
          return prob[a] + l.bias.values[a] > prob[b] + l.bias.values[b];
        });
        order.resize(s.used);
        double sum = 0.0;
        for (auto e : order) {
          sum += prob[e];
        }
        sum = std::max(sum, std::ldexp(1.0, -14));  // llama.cpp contract
        for (auto e : order) {
          const Vec y = swiglu(l.gate_exps, l.up_exps, l.down_exps, e);
          for (std::size_t i = 0; i < H; ++i) {
            mlp[i] += 2.5 * prob[e] / sum * y[i];
          }
        }
        const Vec shared = swiglu(l.shared_gate, l.shared_up, l.shared_down, 0);
        for (std::size_t i = 0; i < H; ++i) {
          mlp[i] += shared[i];
        }
      }
      for (std::size_t i = 0; i < H; ++i) {
        out[t][i] = ffn_inp[t][i] + mlp[i];
      }
    }
    h = out;
    result.l_out.push_back(out);
  }
  for (std::size_t t = 0; t < T; ++t) {
    const Vec n = NaiveRmsNorm(h[t], f.output_norm(), eps);
    result.logits.push_back(MatVec(f.output(), 0, n));
  }
  return result;
}

/// sqrt(sum (a-b)^2 / sum b^2) over flat arrays.
inline double RelRms(const std::vector<float>& a,
                     const std::vector<double>& b) {
  Require(a.size() == b.size(), "RelRms size");
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += b[i] * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-300));
}

inline std::vector<double> Flatten(const std::vector<Vec>& rows) {
  std::vector<double> flat;
  for (const auto& r : rows) {
    flat.insert(flat.end(), r.begin(), r.end());
  }
  return flat;
}

/// Collects every tap by name.
struct TapStore {
  std::map<std::string, std::vector<float>> f32;
  std::map<std::string, std::vector<std::int32_t>> i32;
  std::map<std::string, std::array<std::uint64_t, 3>> ne;
  ref::TapFn Fn() {
    return [this](const ref::Tap& tap) {
      const std::string name(tap.name);
      ne[name] = tap.ne;
      if (tap.type == ref::TapType::kF32) {
        const auto* p = static_cast<const float*>(tap.data);
        f32[name].assign(p, p + tap.Count());
      } else {
        const auto* p = static_cast<const std::int32_t*>(tap.data);
        i32[name].assign(p, p + tap.Count());
      }
    };
  }
  [[nodiscard]] const std::vector<float>& F(const std::string& name) const {
    const auto it = f32.find(name);
    Require(it != f32.end(), "missing tap " + name);
    return it->second;
  }
};

inline std::vector<std::int32_t> RandomTokens(std::size_t n,
                                              std::uint32_t vocab,
                                              std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::int32_t> t(n);
  for (auto& v : t) {
    v = static_cast<std::int32_t>(rng() % vocab);
  }
  return t;
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_ORACLE_FIXTURE_HPP_
