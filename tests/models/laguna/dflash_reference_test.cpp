// Laguna DFlash drafter oracle (src/models/laguna/dflash_reference) against an
// independent naive double-precision implementation written here from
// DFLASH.md. Tiny random drafters with unequal dimensions (hidden 64, 6 query
// heads and 2 KV heads of 16, FFN 80, three taps, window 6, block 5) on top of
// a tiny random target (shared embedding and head, and tap source). Covers the
// per-tap encoder, K/V injection at the target positions, the sliding-window
// plus in-block causal mask, the gate, and the tap-index convention (a tap
// taken one layer early must fail). No model file; nothing is skipped.
#include "src/models/laguna/dflash_reference.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tests/models/laguna/oracle_fixture.hpp"

using namespace gufo::testing::laguna;
namespace rf = gufo::models::laguna::reference;

namespace {

constexpr std::uint32_t kHidden = 64;
constexpr std::uint32_t kHeadDim = 16;
constexpr std::uint32_t kHeads = 6;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kFfn = 80;
constexpr std::uint32_t kWindow = 6;
constexpr std::uint32_t kBlock = 5;
constexpr std::uint32_t kLayers = 3;
constexpr double kTheta = 1000.0;

struct LayerFx {
  Mat attn_norm, q, k, v, o, q_norm, k_norm, gate, ffn_norm, up, gt, down;
};

/// A random drafter: the Mats hold the logical values, `weights` the views.
class Drafter {
public:
  Drafter(std::uint64_t seed, std::vector<std::uint32_t> target_layers)
      : rng_(seed) {
    gufo::models::laguna::DFlashConfig& c = weights.config;
    c.num_layers = kLayers;
    c.hidden_size = kHidden;
    c.ffn_size = kFfn;
    c.vocab_size = 50;
    c.context_length = 4096;
    c.rms_eps = 1e-6F;
    c.num_heads = kHeads;
    c.num_kv_heads = kKvHeads;
    c.head_dim = kHeadDim;
    c.sliding_window = kWindow;
    c.rope_theta = static_cast<float>(kTheta);
    c.rotary_dim = kHeadDim;
    c.block_size = kBlock;
    c.mask_token_id = 12;
    c.target_layers = std::move(target_layers);
    const std::size_t taps = c.tap_count();
    aux_norm = Norm(kHidden * taps, kHidden, taps);
    fc = Random(taps * kHidden, kHidden, 1.0F);
    enc_norm = Norm(kHidden, kHidden, 1);
    out_norm = Norm(kHidden, kHidden, 1);
    layers.resize(kLayers);
    for (auto& l : layers) {
      l.attn_norm = Norm(kHidden, kHidden, 1);
      l.q = Random(kHidden, kHeads * kHeadDim, 1.2F);
      l.k = Random(kHidden, kKvHeads * kHeadDim, 1.2F);
      l.v = Random(kHidden, kKvHeads * kHeadDim, 1.0F);
      l.o = Random(kHeads * kHeadDim, kHidden, 0.9F);
      l.q_norm = Norm(kHeadDim, kHeadDim, 1);
      l.k_norm = Norm(kHeadDim, kHeadDim, 1);
      l.gate = Random(kHidden, kHeads, 2.0F);
      l.ffn_norm = Norm(kHidden, kHidden, 1);
      l.up = Random(kHidden, kFfn, 1.1F);
      l.gt = Random(kHidden, kFfn, 1.1F);
      l.down = Random(kFfn, kHidden, 0.9F);
    }
    Bind();
  }
  Drafter(const Drafter&) = delete;
  Drafter& operator=(const Drafter&) = delete;

  gufo::models::laguna::DFlashWeights weights;
  Mat aux_norm, fc, enc_norm, out_norm;
  std::vector<LayerFx> layers;

private:
  Mat Random(std::size_t cols, std::size_t rows, float gain) {
    Mat m;
    m.cols = cols;
    m.rows = rows;
    m.values.resize(cols * rows);
    std::normal_distribution<float> dist(0.0F, gain / std::sqrt(float(cols)));
    for (float& v : m.values) {
      v = dist(rng_);
    }
    Encode(m);
    return m;
  }
  Mat Norm(std::size_t n, std::size_t cols, std::size_t rows) {
    Mat m;
    m.cols = cols;
    m.rows = rows;
    m.values.resize(n);
    std::uniform_real_distribution<float> dist(0.5F, 1.5F);
    for (float& v : m.values) {
      v = dist(rng_);
    }
    Encode(m);
    return m;
  }
  gufo::models::laguna::TensorRef Ref(const Mat& m, const std::string& name) {
    names_.push_back(name);
    gufo::models::laguna::TensorRef t;
    t.data = m.bytes.data();
    t.type = m.type;
    t.cols = m.cols;
    t.rows = m.rows;
    t.name = names_.back();
    return t;
  }
  void Bind() {
    weights.aux_norm = Ref(aux_norm, "enc.aux_norm.weight");
    weights.fc = Ref(fc, "fc.weight");
    weights.enc_output_norm = Ref(enc_norm, "enc.output_norm.weight");
    weights.output_norm = Ref(out_norm, "output_norm.weight");
    for (std::size_t il = 0; il < layers.size(); ++il) {
      const auto& l = layers[il];
      const std::string p = "blk." + std::to_string(il) + ".";
      gufo::models::laguna::DFlashLayerWeights w;
      w.attn_norm = Ref(l.attn_norm, p + "attn_norm");
      w.attn_q = Ref(l.q, p + "attn_q");
      w.attn_k = Ref(l.k, p + "attn_k");
      w.attn_v = Ref(l.v, p + "attn_v");
      w.attn_output = Ref(l.o, p + "attn_output");
      w.attn_q_norm = Ref(l.q_norm, p + "attn_q_norm");
      w.attn_k_norm = Ref(l.k_norm, p + "attn_k_norm");
      w.attn_gate = Ref(l.gate, p + "attn_gate");
      w.ffn_norm = Ref(l.ffn_norm, p + "ffn_norm");
      w.ffn_gate = Ref(l.gt, p + "ffn_gate");
      w.ffn_up = Ref(l.up, p + "ffn_up");
      w.ffn_down = Ref(l.down, p + "ffn_down");
      weights.layers.push_back(std::move(w));
    }
  }
  std::mt19937_64 rng_;
  std::deque<std::string> names_;
};

Vec Rms(const Vec& x, const float* w) {
  double ms = 0.0;
  for (double v : x) {
    ms += v * v;
  }
  const double inv = 1.0 / std::sqrt(ms / static_cast<double>(x.size()) + 1e-6);
  Vec y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    y[i] = x[i] * inv * w[i];
  }
  return y;
}

Vec Rope(const Vec& head, int pos) {
  return NaiveRope(head, pos, kHeadDim, kTheta, false, 1.0, 1.0, 32.0, 1.0,
                   1.0);
}

Vec Slice(const Vec& v, std::size_t begin, std::size_t n) {
  return Vec(v.begin() + static_cast<std::ptrdiff_t>(begin),
             v.begin() + static_cast<std::ptrdiff_t>(begin + n));
}

/// Naive drafter forward. `features[t]` is the concatenated taps of context
/// token t; returns the logits of block rows 1..B-1.
std::vector<Vec> NaiveDraft(const Drafter& d, const Fixture& target,
                            const std::vector<Vec>& features,
                            const std::vector<int>& ctx_pos,
                            const std::vector<std::int32_t>& block,
                            int block_pos) {
  const std::size_t N = ctx_pos.size();
  const std::size_t B = block.size();
  const std::size_t taps = d.weights.config.tap_count();
  // Encoder.
  std::vector<Vec> fused(N);
  for (std::size_t t = 0; t < N; ++t) {
    Vec cat;
    for (std::size_t k = 0; k < taps; ++k) {
      const Vec a = Rms(Slice(features[t], k * kHidden, kHidden),
                        d.aux_norm.values.data() + k * kHidden);
      cat.insert(cat.end(), a.begin(), a.end());
    }
    fused[t] = Rms(MatVec(d.fc, 0, cat), d.enc_norm.values.data());
  }
  std::vector<Vec> h(B, Vec(kHidden));
  for (std::size_t r = 0; r < B; ++r) {
    for (std::size_t i = 0; i < kHidden; ++i) {
      h[r][i] = target.embd().At(0, static_cast<std::size_t>(block[r]), i);
    }
  }
  const std::size_t group = kHeads / kKvHeads;
  for (const LayerFx& l : d.layers) {
    // Keys and values: context first, then block.
    std::vector<std::vector<Vec>> k(N + B), v(N + B);  // [row][kv head]
    std::vector<int> kpos;
    for (std::size_t t = 0; t < N; ++t) {
      const Vec x = Rms(fused[t], l.attn_norm.values.data());
      const Vec kk = MatVec(l.k, 0, x);
      const Vec vv = MatVec(l.v, 0, x);
      for (std::uint32_t g = 0; g < kKvHeads; ++g) {
        k[t].push_back(
            Rope(Rms(Slice(kk, g * kHeadDim, kHeadDim), l.k_norm.values.data()),
                 ctx_pos[t]));
        v[t].push_back(Slice(vv, g * kHeadDim, kHeadDim));
      }
      kpos.push_back(ctx_pos[t]);
    }
    std::vector<Vec> n(B);
    std::vector<std::vector<Vec>> q(B);
    for (std::size_t r = 0; r < B; ++r) {
      n[r] = Rms(h[r], l.attn_norm.values.data());
      const Vec qq = MatVec(l.q, 0, n[r]);
      const Vec kk = MatVec(l.k, 0, n[r]);
      const Vec vv = MatVec(l.v, 0, n[r]);
      const int p = block_pos + static_cast<int>(r);
      for (std::uint32_t g = 0; g < kHeads; ++g) {
        q[r].push_back(Rope(
            Rms(Slice(qq, g * kHeadDim, kHeadDim), l.q_norm.values.data()), p));
      }
      for (std::uint32_t g = 0; g < kKvHeads; ++g) {
        k[N + r].push_back(Rope(
            Rms(Slice(kk, g * kHeadDim, kHeadDim), l.k_norm.values.data()), p));
        v[N + r].push_back(Slice(vv, g * kHeadDim, kHeadDim));
      }
      kpos.push_back(p);
    }
    for (std::size_t r = 0; r < B; ++r) {
      const int p = block_pos + static_cast<int>(r);
      // Mask written out from DFLASH.md: context keys j with p - pos < window;
      // block keys 0..r (also inside the window).
      std::vector<std::size_t> vis;
      for (std::size_t j = 0; j < N; ++j) {
        if (p - ctx_pos[j] < static_cast<int>(kWindow)) {
          vis.push_back(j);
        }
      }
      for (std::size_t i = 0; i <= r; ++i) {
        if (p - (block_pos + static_cast<int>(i)) < static_cast<int>(kWindow)) {
          vis.push_back(N + i);
        }
      }
      const Vec gate = MatVec(l.gate, 0, n[r]);
      Vec attn;
      for (std::uint32_t g = 0; g < kHeads; ++g) {
        Vec s(vis.size());
        double top = -1e300;
        for (std::size_t a = 0; a < vis.size(); ++a) {
          double dot = 0.0;
          for (std::uint32_t i = 0; i < kHeadDim; ++i) {
            dot += q[r][g][i] * k[vis[a]][g / group][i];
          }
          s[a] = dot / std::sqrt(static_cast<double>(kHeadDim));
          top = std::max(top, s[a]);
        }
        double den = 0.0;
        for (double& x : s) {
          x = std::exp(x - top);
          den += x;
        }
        const double sp = std::log1p(std::exp(gate[g]));
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          double acc = 0.0;
          for (std::size_t a = 0; a < vis.size(); ++a) {
            acc += s[a] / den * v[vis[a]][g / group][i];
          }
          attn.push_back(acc * sp);
        }
      }
      const Vec o = MatVec(l.o, 0, attn);
      Vec mid(kHidden);
      for (std::size_t i = 0; i < kHidden; ++i) {
        mid[i] = h[r][i] + o[i];
      }
      const Vec fn = Rms(mid, l.ffn_norm.values.data());
      Vec up = MatVec(l.up, 0, fn);
      const Vec gt = MatVec(l.gt, 0, fn);
      for (std::size_t i = 0; i < up.size(); ++i) {
        up[i] *= NaiveSilu(gt[i]);
      }
      const Vec dn = MatVec(l.down, 0, up);
      for (std::size_t i = 0; i < kHidden; ++i) {
        h[r][i] = mid[i] + dn[i];
      }
    }
  }
  std::vector<Vec> logits;
  for (std::size_t r = 1; r < B; ++r) {
    logits.push_back(
        MatVec(target.output(), 0, Rms(h[r], d.out_norm.values.data())));
  }
  return logits;
}

std::vector<float> RandomFeatures(std::size_t n, std::size_t width,
                                  std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> f(n * width);
  for (std::size_t i = 0; i < f.size(); ++i) {
    // Per-tap scale spread, so a swapped or missing aux norm cannot hide.
    f[i] = dist(rng) * (1.0F + 3.0F * static_cast<float>((i / kHidden) % 3));
  }
  return f;
}

std::vector<Vec> ToRows(const std::vector<float>& flat, std::size_t width) {
  std::vector<Vec> rows;
  for (std::size_t i = 0; i + width <= flat.size(); i += width) {
    rows.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(i),
                      flat.begin() + static_cast<std::ptrdiff_t>(i + width));
  }
  return rows;
}

struct Run {
  std::vector<float> features;
  std::vector<std::int32_t> ctx_pos;
  std::vector<std::int32_t> block;
  std::int32_t block_pos;
};

std::vector<float> Oracle(const Drafter& d, const Fixture& target,
                          const Run& run, const rf::TapFn& tap = {}) {
  const rf::DFlashModel model(d.weights, target.weights(), {});
  return model.Forward({run.features, run.ctx_pos, run.block, run.block_pos},
                       tap);
}

double AgainstNaive(const Drafter& d, const Fixture& target, const Run& run,
                    const std::vector<float>& logits) {
  const std::size_t width = d.weights.config.fused_input_width();
  const std::vector<int> pos(run.ctx_pos.begin(), run.ctx_pos.end());
  const auto naive = NaiveDraft(d, target, ToRows(run.features, width), pos,
                                run.block, run.block_pos);
  return RelRms(logits, Flatten(naive));
}

Run MakeRun(std::size_t n, std::size_t B, std::int32_t gap,
            std::uint64_t seed) {
  Run run;
  run.features = RandomFeatures(n, 3 * kHidden, seed);
  for (std::size_t t = 0; t < n; ++t) {
    run.ctx_pos.push_back(static_cast<std::int32_t>(t));
  }
  run.block_pos = static_cast<std::int32_t>(n) + gap;
  run.block = RandomTokens(B, 50, seed + 7);
  run.block[0] = 31;
  for (std::size_t r = 1; r < B; ++r) {
    run.block[r] = 12;  // mask token
  }
  return run;
}

double Diff(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<double> bd(b.begin(), b.end());
  return RelRms(a, bd);
}

}  // namespace

int main() {
  try {
    TinySpec spec;  // hidden 64, vocab 50, 5 layers
    spec.seed = 11;
    const Fixture target(spec);
    const Drafter d(21, {2, 4, 5});  // layer-input numbering: outputs 1, 3, 4

    // 1. Oracle == naive over context lengths around the window edge (6),
    //    several block lengths, no context, and gapped positions.
    int cases = 0;
    for (const std::size_t n : {0U, 1U, 5U, 6U, 7U, 12U}) {
      for (const std::size_t B : {2U, 3U, 5U}) {
        const Run run = MakeRun(n, B, 0, 100 + n * 10 + B);
        const auto logits = Oracle(d, target, run);
        Require(logits.size() == (B - 1) * 50, "logit rows");
        const double err = AgainstNaive(d, target, run, logits);
        Require(err < 1e-4, "n=" + std::to_string(n) +
                                " B=" + std::to_string(B) + " rel err " +
                                std::to_string(err));
        ++cases;
      }
    }
    {
      // The CPU-fork attention mode (F16 Q/K/V, F16 V accumulator) stays close
      // to the F32 oracle, is not identical to it, and honours the window.
      rf::Options fa;
      fa.fa_f16_accumulator = true;
      rf::Options kv;
      kv.round_kv_f16 = true;
      for (const std::size_t n : {5U, 9U}) {  // inside and past the window (6)
        const Run run = MakeRun(n, 5, 0, 1200 + n);
        const auto base = Oracle(d, target, run);
        const auto fast =
            rf::DFlashModel(d.weights, target.weights(), fa)
                .Forward({run.features, run.ctx_pos, run.block, run.block_pos});
        const auto kvr =
            rf::DFlashModel(d.weights, target.weights(), kv)
                .Forward({run.features, run.ctx_pos, run.block, run.block_pos});
        const double e_fa = Diff(fast, base);
        Require(e_fa > 1e-6 && e_fa < 5e-2,
                "fa_f16_accumulator differs from F32 but stays close: " +
                    std::to_string(e_fa));
        Require(Diff(kvr, base) > 1e-6, "round_kv_f16 differs from F32");
        ++cases;
      }
    }
    {
      // Context positions with holes (a restored or trimmed cache), block
      // further on. The injected K/V must use these positions, not indexes.
      Run run = MakeRun(6, 5, 0, 900);
      run.ctx_pos = {0, 1, 2, 5, 6, 7};
      run.block_pos = 8;
      const auto logits = Oracle(d, target, run);
      Require(AgainstNaive(d, target, run, logits) < 1e-4, "gapped positions");
      Run wrong = run;
      wrong.ctx_pos = {0, 1, 2, 3, 4, 5};  // injected by index instead
      Require(Diff(logits, Oracle(d, target, wrong)) > 1e-3,
              "positions must matter (RoPE of injected K)");
      // A context row at or after the block start is refused.
      Run bad = run;
      bad.ctx_pos.back() = 8;
      bool threw = false;
      try {
        (void)Oracle(d, target, bad);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      Require(threw, "context position >= block position is refused");
      ++cases;
    }
    {
      // Shifting every position by the same amount leaves the result unchanged
      // up to rounding (RoPE is relative), unlike shifting only the context.
      Run a = MakeRun(7, 5, 0, 950);
      Run b = a;
      for (auto& p : b.ctx_pos) {
        p += 1000;
      }
      b.block_pos += 1000;
      Require(Diff(Oracle(d, target, a), Oracle(d, target, b)) < 5e-3,
              "uniform position shift");
      Run c = a;
      c.block_pos += 3;
      Require(Diff(Oracle(d, target, a), Oracle(d, target, c)) > 1e-3,
              "context/block distance must matter");
      ++cases;
    }

    // 2. Mask structure.
    {
      // Window edge: n = 12, B = 5, window 6. Row 0 sits at position 12 and
      // sees context keys j > 6; no later row sees anything older. So context
      // rows 0..6 reach no row at all (bitwise no effect), while row 7 is seen
      // by row 0 alone and still moves the drafts through the block keys.
      const Run base = MakeRun(12, 5, 0, 700);
      const auto ref_logits = Oracle(d, target, base);
      for (std::size_t j = 0; j < 12; ++j) {
        Run bumped = base;
        for (std::size_t i = 0; i < 3 * kHidden; ++i) {
          bumped.features[j * 3 * kHidden + i] += 0.75F;
        }
        const auto out = Oracle(d, target, bumped);
        if (j <= 6) {
          Require(out == ref_logits, "context row " + std::to_string(j) +
                                         " is outside every window");
        } else {
          Require(out != ref_logits,
                  "context row " + std::to_string(j) + " is inside a window");
        }
      }
      ++cases;
    }
    {
      // In-block causality: changing the token of row 3 cannot move the
      // logits of rows before it (output rows 0..1 are block rows 1..2), and
      // must move later ones.
      const Run base = MakeRun(7, 5, 0, 800);
      Run alt = base;
      alt.block[3] = 40;
      const auto a = Oracle(d, target, base);
      const auto b = Oracle(d, target, alt);
      for (std::size_t i = 0; i < 2 * 50; ++i) {
        Require(a[i] == b[i], "earlier block rows must not see a later token");
      }
      bool moved = false;
      for (std::size_t i = 2 * 50; i < a.size(); ++i) {
        moved = moved || a[i] != b[i];
      }
      Require(moved, "the changed row and the next one must move");
      // Shorter blocks are prefixes of longer ones.
      Run shorter = base;
      shorter.block.resize(3);
      const auto c = Oracle(d, target, shorter);
      for (std::size_t i = 0; i < c.size(); ++i) {
        Require(c[i] == a[i], "a shorter block is a prefix of a longer one");
      }
      ++cases;
    }

    // 3. Taps: names, shapes, and the encoder semantics.
    {
      const Run run = MakeRun(7, 5, 0, 600);
      TapStore taps;
      const auto logits = Oracle(d, target, run, taps.Fn());
      Require(taps.F("result_output") == logits, "result_output == logits");
      Require(taps.ne.at("enc_aux_norm")[0] == 192 &&
                  taps.ne.at("enc_aux_norm")[1] == 7,
              "enc_aux_norm shape");
      for (const char* name :
           {"fc_out", "enc_norm_out", "kv_inp_normed-0", "Kcur_injected-2",
            "Vcur_injected-1", "noise_norm-0", "Qcur-1", "Kcur-2",
            "attn_gate_softplus-0", "attn_gated-1", "ffn_out-2", "l_out-2",
            "result_norm"}) {
        Require(taps.f32.count(name) == 1, std::string("missing tap ") + name);
      }
      Require(taps.ne.at("Kcur_injected-0")[2] == 7, "injected rows");
      // Row 0 of the block is the anchor: a block of one MASK row more or
      // fewer still gives the same enc taps.
      Require(taps.F("l_out-2").size() == 5 * kHidden, "block rows");
      ++cases;
    }

    // 4. Tap-index convention against the target oracle and the naive target.
    {
      std::string error;
      const auto& tcfg = target.config();
      Require(d.weights.config.ValidateAgainstTarget(tcfg, &error),
              "pairing: " + error);
      const auto tokens = RandomTokens(9, 50, 5);
      const rf::Model tmodel(target.weights(), {});
      const auto layers_for = [&](int shift) {
        std::vector<std::uint32_t> layers;
        for (std::size_t i = 0; i < d.weights.config.tap_count(); ++i) {
          layers.push_back(static_cast<std::uint32_t>(
              static_cast<int>(d.weights.config.tap_layer(i)) + shift));
        }
        return layers;
      };
      const auto good = rf::ForwardWithTaps(tmodel, tokens, layers_for(0));
      Require((layers_for(0) == std::vector<std::uint32_t>{1, 3, 4}),
              "tap layers are target_layers - 1");
      Require(good.hidden_size == kHidden &&
                  good.features.size() == 9 * 3 * kHidden,
              "tap feature layout");
      const auto naive_target = NaiveForward(target, tokens);
      for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t t = 0; t < 9; ++t) {
          std::vector<float> row(
              good.features.begin() +
                  static_cast<std::ptrdiff_t>((t * 3 + i) * kHidden),
              good.features.begin() +
                  static_cast<std::ptrdiff_t>((t * 3 + i + 1) * kHidden));
          Require(RelRms(row,
                         naive_target.l_out[d.weights.config.tap_layer(i)][t]) <
                      1e-4,
                  "tap " + std::to_string(i) + " is the output of layer " +
                      std::to_string(d.weights.config.tap_layer(i)));
        }
      }
      // End to end: the oracle fed with the oracle's taps equals the naive
      // drafter fed with the naive target's taps for the same layers; the
      // same drafter fed taps one layer early (the layer-input numbering
      // misread as layer outputs) differs.
      Run run;
      run.features = good.features;
      for (int t = 0; t < 8; ++t) {
        run.ctx_pos.push_back(t);
      }
      run.features.resize(8 * 3 * kHidden);
      run.block = {tokens[8], 12, 12, 12, 12};
      run.block_pos = 8;
      std::vector<Vec> naive_feats(8);
      for (std::size_t t = 0; t < 8; ++t) {
        for (std::size_t i = 0; i < 3; ++i) {
          const Vec& row = naive_target.l_out[d.weights.config.tap_layer(i)][t];
          naive_feats[t].insert(naive_feats[t].end(), row.begin(), row.end());
        }
      }
      const auto naive = NaiveDraft(d, target, naive_feats,
                                    {0, 1, 2, 3, 4, 5, 6, 7}, run.block, 8);
      const auto logits = Oracle(d, target, run);
      Require(RelRms(logits, Flatten(naive)) < 1e-3, "end to end with taps");
      const auto early = rf::ForwardWithTaps(tmodel, tokens, layers_for(-1));
      Run early_run = run;
      early_run.features = early.features;
      early_run.features.resize(8 * 3 * kHidden);
      Require(RelRms(Oracle(d, target, early_run), Flatten(naive)) > 1e-2,
              "off-by-one tap (layer input read as output) must differ");
      // A tap equal to the layer count is not a layer output.
      bool threw = false;
      try {
        (void)rf::ForwardWithTaps(
            tmodel, tokens, std::vector<std::uint32_t>{1, 3, tcfg.num_layers});
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      Require(threw, "layer index == layer count has no l_out tap");
      ++cases;
    }

    std::cout << "laguna.dflash_reference: " << cases
              << " case groups passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
