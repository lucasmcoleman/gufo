// Oracle MLP checks: router top-k with a selection-only bias, the
// max(sum, 2^-14) clamp on extreme-negative logits (the case that separated
// llama.cpp from Poolside's raw-sum division on GLM), the 2.5 scale, the
// shared expert and dense layer 0. The model-level checks read the oracle's
// taps and recompute each quantity from the tapped input with the
// independent naive code.
#include <cmath>
#include <iostream>
#include <string>

#include "tests/models/laguna/oracle_fixture.hpp"

using namespace gufo::testing::laguna;

namespace {

constexpr double kClamp = 6.103515625e-5;  // 2^-14

double Sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

void TestRouterBias() {
  // Unbiased scores are in (0.2, 0.9); a bias of +-0.5 reorders them.
  const std::vector<float> logits = {2.0F, 1.5F,  1.0F,  0.5F,
                                     0.0F, -0.5F, -1.0F, -1.4F};
  const std::vector<float> bias = {-0.5F, -0.5F, 0.5F, 0.5F,
                                   0.5F,  0.5F,  0.0F, 0.0F};
  const auto unbiased = ref::Route(logits.data(), nullptr, 8, 3, true, 2.5F,
                                   ref::RouterDenominator::kLlamaClamp);
  const auto biased = ref::Route(logits.data(), bias.data(), 8, 3, true, 2.5F,
                                 ref::RouterDenominator::kLlamaClamp);
  Require((unbiased.ids == std::vector<std::uint32_t>{0, 1, 2}),
          "unbiased top-3");
  // sigmoid + bias: e2 1.23, e3 1.12, e4 1.0, e5 0.88, e0 0.38 -> {2, 3, 4}
  Require((biased.ids == std::vector<std::uint32_t>{2, 3, 4}),
          "bias changes selection");
  // Weights come from the unbiased sigmoids of the selected experts.
  double sum = 0.0;
  for (auto e : biased.ids) {
    sum += Sigmoid(logits[e]);
  }
  for (std::size_t i = 0; i < 3; ++i) {
    const double want = 2.5 * Sigmoid(logits[biased.ids[i]]) / sum;
    Require(std::fabs(biased.weights[i] - want) < 1e-6,
            "weights ignore the bias");
  }
  // Equal biased scores pick the lower expert id.
  const std::vector<float> tie(6, 0.25F);
  const auto tied = ref::Route(tie.data(), nullptr, 6, 2, true, 1.0F,
                               ref::RouterDenominator::kLlamaClamp);
  Require((tied.ids == std::vector<std::uint32_t>{0, 1}),
          "ties pick lower ids");
}

void TestExtremeNegative() {
  std::vector<float> logits(16);
  for (std::size_t e = 0; e < logits.size(); ++e) {
    logits[e] =
        -20.0F + 0.25F * static_cast<float>(e % 5) - 0.5F;  // -20.5 .. -19.5
  }
  const auto llama = ref::Route(logits.data(), nullptr, 16, 4, true, 2.5F,
                                ref::RouterDenominator::kLlamaClamp);
  const auto hf = ref::Route(logits.data(), nullptr, 16, 4, true, 2.5F,
                             ref::RouterDenominator::kHfRaw);
  Require(llama.sum < kClamp, "fixture really is below 2^-14");
  Require(llama.denominator == static_cast<float>(kClamp),
          "denominator clamps to 2^-14");
  double llama_total = 0.0;
  double hf_total = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    Require(std::fabs(llama.weights[i] - 2.5 * llama.probs[i] / kClamp) <
                1e-4 * llama.weights[i],
            "llama weights are s / 2^-14 * 2.5");
    Require(std::fabs(hf.weights[i] - 2.5 * hf.probs[i] / hf.sum) < 1e-5,
            "HF weights are s / sum * 2.5");
    llama_total += llama.weights[i];
    hf_total += hf.weights[i];
  }
  Require(std::fabs(hf_total - 2.5) < 1e-5, "HF weights still sum to 2.5");
  Require(llama_total < 0.5 * hf_total,
          "llama weights are much smaller than HF");
}

/// Reads a tap row for token t.
std::vector<double> Row(const std::vector<float>& tap, std::size_t t,
                        std::size_t width) {
  return std::vector<double>(tap.begin() + t * width,
                             tap.begin() + (t + 1) * width);
}

std::vector<double> NaiveSwiglu(const Mat& g, const Mat& u, const Mat& d,
                                std::size_t e, const Vec& x) {
  Vec gv = MatVec(g, e, x);
  const Vec uv = MatVec(u, e, x);
  for (std::size_t i = 0; i < gv.size(); ++i) {
    gv[i] = NaiveSilu(gv[i]) * uv[i];
  }
  return MatVec(d, e, gv);
}

void TestMlpTaps() {
  TinySpec spec;
  spec.seed = 51;
  spec.shared_ff = 48;  // unequal to the routed expert ff (32)
  const Fixture fixture(spec);
  const auto tokens = RandomTokens(9, spec.vocab, 52);
  TapStore taps;
  const ref::Model model(fixture.weights(), {});
  (void)model.Forward(tokens, taps.Fn());
  const std::size_t H = spec.hidden;
  const std::size_t T = tokens.size();

  // Dense layer 0: SwiGLU 96, no router taps.
  Require(taps.f32.count("ffn_moe_logits-0") == 0 &&
              taps.i32.count("ffn_moe_topk-0") == 0,
          "layer 0 is dense: it has no router taps");
  for (std::size_t t = 0; t < T; ++t) {
    const Vec n = Row(taps.F("ffn_norm-0"), t, H);
    const Vec want =
        NaiveSwiglu(fixture.layer(0).dense_gate, fixture.layer(0).dense_up,
                    fixture.layer(0).dense_down, 0, n);
    const auto got = Row(taps.F("ffn_out-0"), t, H);
    double err = 0.0, ref_norm = 0.0;
    for (std::size_t i = 0; i < H; ++i) {
      err += (got[i] - want[i]) * (got[i] - want[i]);
      ref_norm += want[i] * want[i];
    }
    Require(std::sqrt(err / ref_norm) < 1e-5,
            "dense layer 0 differs from naive SwiGLU");
  }

  // MoE layers: scale, selection, shared expert.
  for (std::uint32_t il = 1; il < spec.layers; ++il) {
    const std::string tag = "-" + std::to_string(il);
    const LayerMats& l = fixture.layer(il);
    const auto& topk = taps.i32.at("ffn_moe_topk" + tag);
    const auto& scaled = taps.F("ffn_moe_weights_scaled" + tag);
    for (std::size_t t = 0; t < T; ++t) {
      const Vec n = Row(taps.F("ffn_norm" + tag), t, H);
      const Vec logits = MatVec(l.router, 0, n);
      // Selection from sigmoid + bias; weights from the bare sigmoid.
      std::vector<std::size_t> order(spec.experts);
      std::iota(order.begin(), order.end(), std::size_t{0});
      std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return Sigmoid(logits[a]) + l.bias.values[a] >
               Sigmoid(logits[b]) + l.bias.values[b];
      });
      double sum = 0.0;
      for (std::uint32_t s = 0; s < spec.used; ++s) {
        Require(static_cast<std::size_t>(topk[t * spec.used + s]) == order[s],
                "expert selection differs from sigmoid + bias top-k");
        sum += Sigmoid(logits[order[s]]);
      }
      double weight_total = 0.0;
      Vec expect(H, 0.0);
      for (std::uint32_t s = 0; s < spec.used; ++s) {
        const double w = scaled[t * spec.used + s];
        Require(std::fabs(w - 2.5 * Sigmoid(logits[order[s]]) / sum) < 1e-5,
                "weight is not 2.5 * s / sum");
        weight_total += w;
        const Vec y =
            NaiveSwiglu(l.gate_exps, l.up_exps, l.down_exps, order[s], n);
        for (std::size_t i = 0; i < H; ++i) {
          expect[i] += w * y[i];
        }
      }
      Require(std::fabs(weight_total - 2.5) < 1e-5,
              "normal-range weights sum to 2.5");
      const Vec shared =
          NaiveSwiglu(l.shared_gate, l.shared_up, l.shared_down, 0, n);
      const auto moe_out = Row(taps.F("ffn_moe_out" + tag), t, H);
      const auto shexp = Row(taps.F("ffn_shexp" + tag), t, H);
      const auto ffn_out = Row(taps.F("ffn_out" + tag), t, H);
      double e1 = 0, e2 = 0, e3 = 0, n1 = 0, n2 = 0, n3 = 0;
      for (std::size_t i = 0; i < H; ++i) {
        e1 += (moe_out[i] - expect[i]) * (moe_out[i] - expect[i]);
        n1 += expect[i] * expect[i];
        e2 += (shexp[i] - shared[i]) * (shexp[i] - shared[i]);
        n2 += shared[i] * shared[i];
        const double total =
            expect[i] + shared[i];  // no gate on the shared expert
        e3 += (ffn_out[i] - total) * (ffn_out[i] - total);
        n3 += total * total;
      }
      Require(std::sqrt(e1 / n1) < 1e-5, "routed sum differs (2.5 scale)");
      Require(std::sqrt(e2 / n2) < 1e-5, "shared expert differs");
      Require(std::sqrt(e3 / n3) < 1e-5, "ffn_out != routed + shared");
    }
  }
}

/// All router logits near -20 through the model: with one token the router
/// weight can be aimed at the tapped ffn_norm output (which it does not
/// influence), then every logit is about -20 and the top-k sum falls
/// under 2^-14.
void TestExtremeNegativeModel() {
  TinySpec spec;
  spec.seed = 61;
  Fixture fixture(spec);
  const std::vector<std::int32_t> tokens = {7};
  const std::size_t H = spec.hidden;
  const std::uint32_t il = 2;
  TapStore first;
  (void)ref::Model(fixture.weights(), {}).Forward(tokens, first.Fn());
  const auto& n = first.F("ffn_norm-2");
  double nn = 0.0;
  for (float v : n) {
    nn += double(v) * v;
  }
  std::mt19937_64 rng(62);
  std::vector<float> router(spec.experts * H);
  for (std::uint32_t e = 0; e < spec.experts; ++e) {
    const double target =
        -20.0 + std::uniform_real_distribution<double>(-0.7, 0.7)(rng);
    for (std::size_t i = 0; i < H; ++i) {
      router[e * H + i] = static_cast<float>(target * n[i] / nn);
    }
  }
  fixture.SetF32(fixture.MutableLayer(il, &LayerMats::router), router);

  TapStore llama;
  (void)ref::Model(fixture.weights(), {}).Forward(tokens, llama.Fn());
  ref::Options hf_options;
  hf_options.router_denominator = ref::RouterDenominator::kHfRaw;
  TapStore hf;
  (void)ref::Model(fixture.weights(), hf_options).Forward(tokens, hf.Fn());

  // Layer 2's input is unchanged (the router only acts after ffn_norm).
  Require(llama.F("ffn_norm-2") == first.F("ffn_norm-2"),
          "router edit moved ffn_norm");
  const auto& sum = llama.F("ffn_moe_weights_sum-2");
  const auto& clamped = llama.F("ffn_moe_weights_sum_clamped-2");
  Require(sum[0] < kClamp, "model-level sum is below 2^-14");
  Require(clamped[0] == static_cast<float>(kClamp), "clamped sum is 2^-14");
  const auto& weights = llama.F("ffn_moe_weights-2");
  const auto& scaled = llama.F("ffn_moe_weights_scaled-2");
  const auto& hf_scaled = hf.F("ffn_moe_weights_scaled-2");
  double hf_total = 0.0;
  for (std::uint32_t s = 0; s < spec.used; ++s) {
    Require(std::fabs(scaled[s] - 2.5 * weights[s] / kClamp) < 1e-4 * scaled[s],
            "llama weight is 2.5 * s / 2^-14");
    hf_total += hf_scaled[s];
  }
  Require(std::fabs(hf_total - 2.5) < 1e-5,
          "HF convention renormalizes to 2.5");
  Require(llama.F("ffn_moe_out-2") != hf.F("ffn_moe_out-2"),
          "the two conventions give different MoE outputs here");

  // And the full forward still matches the naive llama.cpp-contract model.
  const NaiveResult naive = NaiveForward(fixture, tokens);
  const double err = RelRms(llama.F("l_out-4"), Flatten(naive.l_out[4]));
  Require(err < 1e-5, "extreme-negative model differs from naive (rel rms " +
                          std::to_string(err) + ")");
}

}  // namespace

int main() {
  try {
    TestRouterBias();
    std::cout << "ok  router: selection-only bias, tie order\n";
    TestExtremeNegative();
    std::cout << "ok  router: logits near -20 hit the 2^-14 clamp (both "
                 "conventions)\n";
    TestMlpTaps();
    std::cout << "ok  dense layer 0, 2.5 scale, shared expert, MoE taps\n";
    TestExtremeNegativeModel();
    std::cout << "ok  extreme-negative router through the whole model\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
