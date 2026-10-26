// Oracle attention building blocks, each against an independent check:
//  - partial RoPE (leading channels only), with and without YaRN, and the
//    net cos/sin multiplier against Config::YarnMscale()
//  - grouped-query attention with groups 6 (full layers) and 9 (SWA layers)
//  - the sliding-window boundary at positions 511/512/513 (real window) and
//    on a small window, measured by which keys actually contribute
//  - a long sequence past the window against the naive double implementation
#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "tests/models/laguna/oracle_fixture.hpp"

using namespace gufo::testing::laguna;

namespace {

std::vector<float> RandomFloats(std::size_t n, std::uint64_t seed,
                                float scale = 1.0F) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0F, scale);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

double MaxAbsDiff(const std::vector<float>& a, const Vec& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(a[i] - b[i]));
  }
  return worst;
}

void TestRope(bool yarn, std::uint32_t rot, float attn_factor) {
  const std::uint32_t heads = 3;
  const std::uint32_t hd = 32;
  const std::size_t T = 40;
  ml::Config config;
  config.num_layers = 1;
  config.head_dim = hd;
  config.rotary_dim_full = rot;
  config.rotary_dim_swa = rot;
  config.rope_theta_full = 500000.0F;
  config.rope_theta_swa = 10000.0F;
  config.yarn.factor = yarn ? 32.0F : 1.0F;
  config.yarn.original_context = 8192;
  config.rope_attn_factor = attn_factor;
  const auto params = ref::RopeFor(config, 0);

  const std::vector<float> x0 = RandomFloats(T * heads * hd, 11 + rot);
  std::vector<float> x = x0;
  std::vector<std::int32_t> positions(T);
  for (std::size_t t = 0; t < T; ++t) {
    positions[t] = static_cast<std::int32_t>(t * 7 + 1);  // sparse positions
  }
  ref::ApplyRope(x.data(), positions, heads, hd, params);

  const double mscale = attn_factor * (yarn ? 1.0 + 0.1 * std::log(32.0) : 1.0);
  for (std::size_t t = 0; t < T; ++t) {
    for (std::uint32_t h = 0; h < heads; ++h) {
      const std::size_t base = (t * heads + h) * hd;
      const Vec head(x0.begin() + base, x0.begin() + base + hd);
      Vec expect = head;
      const Vec rotated = NaiveRope(head, positions[t], rot, 500000.0, yarn,
                                    32.0, 8192, 32, 1, attn_factor);
      std::copy(rotated.begin(), rotated.begin() + rot, expect.begin());
      // channels >= rot are untouched (partial RoPE)
      for (std::uint32_t d = rot; d < hd; ++d) {
        Require(x[base + d] == x0[base + d], "channels past rotary_dim change");
      }
      for (std::uint32_t d = 0; d < rot; ++d) {
        Require(std::fabs(x[base + d] - expect[d]) <
                    2e-3 * (1.0 + std::fabs(expect[d])),
                "rope differs from the naive HF form");
      }
      // Rotation preserves each pair's norm up to the net multiplier.
      for (std::uint32_t i = 0; i < rot / 2; ++i) {
        const double before = std::hypot(x0[base + i], x0[base + i + rot / 2]);
        const double after = std::hypot(x[base + i], x[base + i + rot / 2]);
        Require(std::fabs(after - mscale * before) < 1e-5 * (1.0 + before),
                "pair norm not scaled by the net mscale");
      }
    }
  }
  if (yarn) {
    ml::Config named = config;
    Require(
        std::fabs(named.YarnMscale() - attn_factor * 1.3466F) < 2e-4F,
        "Config::YarnMscale disagrees with 1 + 0.1 ln 32 times attn factor");
  }
  // At position 0 cos = mscale and sin = 0: the row is just scaled.
  std::vector<float> first(x0.begin(), x0.begin() + heads * hd);
  const std::int32_t zero = 0;
  ref::ApplyRope(first.data(), std::span<const std::int32_t>(&zero, 1), heads,
                 hd, params);
  for (std::uint32_t d = 0; d < rot; ++d) {
    Require(
        std::fabs(first[d] - mscale * x0[d]) < 1e-6 * (1.0 + std::fabs(x0[d])),
        "position 0 is not a pure mscale");
  }
}

void TestGqa(std::uint32_t heads, std::uint32_t window, std::size_t T) {
  const std::uint32_t kv = 2;
  const std::uint32_t hd = 24;
  const auto q = RandomFloats(T * heads * hd, 21 + heads, 1.3F);
  const auto k = RandomFloats(T * kv * hd, 22 + heads, 1.3F);
  const auto v = RandomFloats(T * kv * hd, 23 + heads);
  std::vector<float> out(T * heads * hd);
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  ref::GqaAttention(q.data(), k.data(), v.data(), T, heads, kv, hd, window,
                    scale, out.data());
  const std::uint32_t group = heads / kv;
  Vec expect(out.size());
  for (std::size_t p = 0; p < T; ++p) {
    for (std::uint32_t h = 0; h < heads; ++h) {
      const std::size_t kvh = h / group;
      std::vector<double> s;
      std::vector<std::size_t> keys;
      for (std::size_t j = 0; j <= p; ++j) {
        if (window == 0 ||
            static_cast<long>(j) >
                static_cast<long>(p) - static_cast<long>(window)) {
          keys.push_back(j);
        }
      }
      double top = -1e300;
      for (auto j : keys) {
        double dot = 0.0;
        for (std::uint32_t d = 0; d < hd; ++d) {
          dot +=
              double(q[(p * heads + h) * hd + d]) * k[(j * kv + kvh) * hd + d];
        }
        s.push_back(dot * scale);
        top = std::max(top, s.back());
      }
      double z = 0.0;
      for (double& e : s) {
        e = std::exp(e - top);
        z += e;
      }
      for (std::uint32_t d = 0; d < hd; ++d) {
        double acc = 0.0;
        for (std::size_t a = 0; a < keys.size(); ++a) {
          acc += s[a] / z * v[(keys[a] * kv + kvh) * hd + d];
        }
        expect[(p * heads + h) * hd + d] = acc;
      }
    }
  }
  Require(MaxAbsDiff(out, expect) < 1e-5,
          "GQA attention differs from naive (heads " + std::to_string(heads) +
              ", window " + std::to_string(window) + ")");
}

/// Measures which keys a query at each position reads. q = 0 makes the
/// softmax uniform over the visible keys, v[j][0] = 1 and v[j][1] = j, so
/// out[0] = 1 and out[1] = mean visible index = (first + p) / 2 gives the
/// first visible key exactly.
void TestWindowBoundary(std::uint32_t window, std::size_t T,
                        const std::vector<std::size_t>& positions) {
  const std::uint32_t heads = 9;  // an SWA layer's group of 9
  const std::uint32_t kv = 1;
  const std::uint32_t hd = 4;
  std::vector<float> q(T * heads * hd, 0.0F);
  std::vector<float> k = RandomFloats(T * kv * hd, 31);
  std::vector<float> v(T * kv * hd, 0.0F);
  for (std::size_t j = 0; j < T; ++j) {
    v[j * hd + 0] = 1.0F;
    v[j * hd + 1] = static_cast<float>(j);
  }
  std::vector<float> out(T * heads * hd);
  ref::GqaAttention(q.data(), k.data(), v.data(), T, heads, kv, hd, window,
                    0.1F, out.data());
  for (const std::size_t p : positions) {
    for (std::uint32_t h = 0; h < heads; ++h) {
      const float* o = out.data() + (p * heads + h) * hd;
      const double first = 2.0 * o[1] - static_cast<double>(p);
      // Key j is visible iff p - j < window, so the first visible key is
      // p - (window - 1), or 0 while the window is not yet full.
      const double want =
          p + 1 > window ? static_cast<double>(p - (window - 1)) : 0.0;
      Require(std::fabs(first - want) < 1e-3,
              "window " + std::to_string(window) + " position " +
                  std::to_string(p) + ": first visible key " +
                  std::to_string(first) + ", expected " + std::to_string(want));
    }
  }
}

void TestLongSequence() {
  // Full model, window 512, 700 tokens, checked layer by layer against the
  // naive forward: the SWA layers must keep dropping keys long after 512.
  TinySpec spec;
  spec.window = 512;
  spec.layers = 5;
  spec.seed = 41;
  const Fixture fixture(spec);
  const auto tokens = RandomTokens(700, spec.vocab, 42);
  TapStore taps;
  const ref::Model model(fixture.weights(), {});
  (void)model.Forward(tokens, taps.Fn());
  const NaiveResult naive = NaiveForward(fixture, tokens);
  for (std::uint32_t il = 0; il < spec.layers; ++il) {
    const double err =
        RelRms(taps.F("l_out-" + std::to_string(il)), Flatten(naive.l_out[il]));
    Require(err < 1e-4, "long sequence l_out-" + std::to_string(il) +
                            " rel rms " + std::to_string(err));
  }
}

}  // namespace

int main() {
  try {
    for (const bool yarn : {false, true}) {
      TestRope(yarn, 16, 1.0F);  // full layers: 16 of 32 channels
      TestRope(yarn, 32, 1.0F);  // full width
      TestRope(yarn, 8, 0.8F);   // attn factor folds into the multiplier
    }
    std::cout
        << "ok  rope (partial, YaRN on/off, mscale vs Config::YarnMscale)\n";

    for (const std::uint32_t heads : {12U, 18U}) {  // GQA groups 6 and 9
      TestGqa(heads, 0, 30);
      TestGqa(heads, 5, 30);
    }
    std::cout << "ok  gqa groups 6 and 9, full and windowed\n";

    TestWindowBoundary(512, 520, {0, 1, 510, 511, 512, 513, 519});
    TestWindowBoundary(4, 12, {0, 1, 2, 3, 4, 5, 11});
    TestWindowBoundary(1, 6, {0, 3, 5});  // a window of one sees only itself
    std::cout << "ok  sliding window boundary (p - j < window; 511/512/513)\n";

    TestLongSequence();
    std::cout << "ok  long sequence past the window\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
