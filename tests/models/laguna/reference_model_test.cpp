// Whole-model oracle check against the independent naive forward in
// oracle_fixture.hpp, on tiny synthetic models with every weight random.
// Final logits for every position and every layer's output are compared, so a
// fault is localized to the first layer that diverges. Covers: dense layer 0,
// full layers (GQA group 6, partial RoPE 16 of 32 with YaRN) and sliding
// layers (group 9, full-width RoPE, no YaRN), the softplus gate, the MoE with
// selection-only bias, the 2.5 scale and the shared expert, with a small
// window, the real 512 window and positions past it.
#include <iostream>
#include <string>

#include "tests/models/laguna/oracle_fixture.hpp"

using namespace gufo::testing::laguna;

namespace {

struct Case {
  const char* name;
  TinySpec spec;
  std::size_t tokens;
  double tolerance;  // relative RMS error bound
};

void RunCase(const Case& c) {
  const Fixture fixture(c.spec);
  const auto tokens = RandomTokens(c.tokens, c.spec.vocab, c.spec.seed + 100);
  const ref::Model model(fixture.weights(), {});
  TapStore taps;
  const std::vector<float> logits = model.Forward(tokens, taps.Fn());
  const NaiveResult naive = NaiveForward(fixture, tokens);

  // Layer outputs first: the earliest divergence names the broken layer.
  for (std::uint32_t il = 0; il < c.spec.layers; ++il) {
    const double err =
        RelRms(taps.F("l_out-" + std::to_string(il)), Flatten(naive.l_out[il]));
    Require(err < c.tolerance, std::string(c.name) + ": l_out-" +
                                   std::to_string(il) + " rel rms " +
                                   std::to_string(err));
  }
  const double err = RelRms(logits, Flatten(naive.logits));
  Require(err < c.tolerance,
          std::string(c.name) + ": logits rel rms " + std::to_string(err));
  Require(taps.F("result_output") == logits, "result_output tap == logits");
  std::cout << "ok  " << c.name << " (layers " << c.spec.layers << ", T "
            << c.tokens << ", rel rms " << err << ")\n";
}

}  // namespace

int main() {
  try {
    TinySpec base;
    RunCase({"default: window 4, YaRN on full layers", base, 14, 1e-5});

    TinySpec no_yarn = base;
    no_yarn.yarn_factor = 1.0F;
    no_yarn.seed = 2;
    RunCase({"no YaRN", no_yarn, 12, 1e-5});

    TinySpec attn_factor = base;
    attn_factor.rope_attn_factor = 0.75F;
    attn_factor.seed = 3;
    RunCase({"YaRN with rope attn_factor 0.75", attn_factor, 12, 1e-5});

    TinySpec wide_rope = base;
    wide_rope.rotary_full = 32;  // full rotation instead of partial
    wide_rope.rotary_swa = 8;    // and a partial SWA rotation
    wide_rope.seed = 4;
    RunCase({"rotary dims swapped (full 32, swa 8)", wide_rope, 12, 1e-5});

    TinySpec odd = base;
    odd.hidden = 96;
    odd.head_dim = 16;
    odd.rotary_full = 8;
    odd.rotary_swa = 16;
    odd.kv_heads = 3;
    odd.heads_full = 12;  // group 4
    odd.heads_swa = 21;   // group 7
    odd.experts = 9;
    odd.used = 4;
    odd.expert_ff = 40;
    odd.shared_ff = 24;
    odd.dense_ff = 56;
    odd.vocab = 37;
    odd.layers = 6;
    odd.seed = 5;
    RunCase({"unequal everything (hidden 96, kv 3, 9 experts top 4)", odd, 13,
             1e-5});

    // The real window with the sequence running past it: positions 511, 512
    // and 513 all lie inside. float RoPE angles at positions near 600 can
    // differ from double by up to ~4e-5 rad, so the bound is looser (measured
    // 1e-6).
    TinySpec window = base;
    window.window = 512;
    window.seed = 6;
    RunCase({"window 512, 600 tokens", window, 600, 1e-4});

    TinySpec q8 = base;
    q8.q8_0_matrices = true;
    q8.seed = 7;
    RunCase({"Q8_0 weights, exact F32 activations", q8, 12, 1e-5});

    std::cout << "laguna.reference_model: all cases passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
