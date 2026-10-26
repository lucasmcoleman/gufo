// F32 versus Q8-aware oracle. The Q8-aware mode rounds every matmul input the
// way llama.cpp's vec-dot path quantizes it (ggml-cpu vec_dot_type: Q8_0 for
// Q8_0 weights, Q8_K for the K-quants and IQ4_XS; the HIP mul_mat_vec_q path:
// Q8_1). Checked here: the rounding itself against independent hand-written
// quantizers, its use inside the matmul, and that the two modes differ only by
// the expected quantization error.
#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "tests/models/laguna/oracle_fixture.hpp"

using namespace gufo::testing::laguna;

namespace {

std::vector<float> RandomRow(std::size_t n, std::uint64_t seed, float scale) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0F, scale);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

/// quantize_row_q8_0_ref written out: d = amax/127 (fp16 stored), q =
/// round(x/d).
std::vector<float> HandQ8_0(const std::vector<float>& x) {
  std::vector<float> out(x.size());
  for (std::size_t b = 0; b < x.size(); b += 32) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(x[b + j]));
    }
    const float d = amax / 127.0F;
    const float stored = HalfValue(HalfBits(d));
    for (std::size_t j = 0; j < 32; ++j) {
      out[b + j] = std::round(x[b + j] / d) * stored;
    }
  }
  return out;
}

/// quantize_row_q8_K_ref written out.
std::vector<float> HandQ8_K(const std::vector<float>& x) {
  std::vector<float> out(x.size());
  for (std::size_t b = 0; b < x.size(); b += 256) {
    std::size_t at = b;
    for (std::size_t j = 0; j < 256; ++j) {
      if (std::fabs(x[b + j]) > std::fabs(x[at])) {
        at = b + j;
      }
    }
    const float iscale = -127.0F / x[at];
    for (std::size_t j = 0; j < 256; ++j) {
      const float q = std::min(127.0F, std::nearbyint(iscale * x[b + j]));
      out[b + j] = q / iscale;
    }
  }
  return out;
}

double MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, double(std::fabs(a[i] - b[i])));
  }
  return worst;
}

void TestRounding() {
  const std::vector<float> x = RandomRow(2 * 256, 71, 1.7F);
  const auto round = [&](GgmlType type, ref::ActivationQuant mode) {
    std::vector<float> y = x;
    ref::RoundActivations(type, mode, y.data(), 2, 256);
    return y;
  };
  const auto q80 = HandQ8_0(x);
  const auto q8k = HandQ8_K(x);
  Require(
      MaxDiff(round(GgmlType::kQ8_0, ref::ActivationQuant::kCpu), q80) < 1e-6,
      "Q8_0 weight: CPU input is Q8_0-rounded");
  Require(
      MaxDiff(round(GgmlType::kQ6_K, ref::ActivationQuant::kCpu), q8k) < 1e-6,
      "Q6_K weight: CPU input is Q8_K-rounded");
  Require(
      MaxDiff(round(GgmlType::kQ5_K, ref::ActivationQuant::kCpu), q8k) < 1e-6,
      "Q5_K weight: Q8_K");
  Require(
      MaxDiff(round(GgmlType::kIQ4_XS, ref::ActivationQuant::kCpu), q8k) < 1e-6,
      "IQ4_XS weight: Q8_K");
  Require(MaxDiff(round(GgmlType::kQ6_K, ref::ActivationQuant::kGpuQ8_1), q80) <
              1e-6,
          "GPU path rounds every quantized type like Q8_0/Q8_1");
  Require(MaxDiff(round(GgmlType::kIQ4_XS, ref::ActivationQuant::kGpuQ8_1),
                  q80) < 1e-6,
          "GPU path, IQ4_XS");
  Require(round(GgmlType::kF32, ref::ActivationQuant::kCpu) == x,
          "F32 weights keep exact inputs (CPU)");
  Require(round(GgmlType::kF32, ref::ActivationQuant::kGpuQ8_1) == x,
          "F32 weights keep exact inputs (GPU)");
  Require(round(GgmlType::kQ6_K, ref::ActivationQuant::kNone) == x,
          "F32 mode is exact");
  // Rounding is real (not a no-op) and bounded by half a quantization step.
  const auto k_cpu = round(GgmlType::kQ6_K, ref::ActivationQuant::kCpu);
  Require(k_cpu != x, "Q8_K rounding changes the values");
  for (std::size_t b = 0; b < 2; ++b) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 256; ++j) {
      amax = std::max(amax, std::fabs(x[b * 256 + j]));
    }
    for (std::size_t j = 0; j < 256; ++j) {
      Require(std::fabs(k_cpu[b * 256 + j] - x[b * 256 + j]) <=
                  0.51F * amax / 127.0F,
              "Q8_K error exceeds half a step");
    }
  }
  const auto q_cpu = round(GgmlType::kQ8_0, ref::ActivationQuant::kCpu);
  for (std::size_t b = 0; b < 16; ++b) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(x[b * 32 + j]));
    }
    for (std::size_t j = 0; j < 32; ++j) {
      // half a step plus the fp16 error of the stored scale across |q| <= 127
      Require(
          std::fabs(q_cpu[b * 32 + j] - x[b * 32 + j]) <= 0.6F * amax / 127.0F,
          "Q8_0 error exceeds the expected bound");
    }
  }
  // All-zero blocks stay zero.
  std::vector<float> zeros(256, 0.0F);
  ref::RoundActivations(GgmlType::kQ6_K, ref::ActivationQuant::kCpu,
                        zeros.data(), 1, 256);
  for (float v : zeros) {
    Require(v == 0.0F, "zero block");
  }
  // F16 round trip: 1 + 2^-12 is not representable and rounds to 1.
  Require(ref::RoundToF16(1.0F + 0x1p-12F) == 1.0F, "fp16 rounds to nearest");
  Require(ref::RoundToF16(65520.0F) > 1e30F, "fp16 overflow goes to inf");
}

void TestModes() {
  TinySpec spec;
  spec.q8_0_matrices = true;
  spec.layers = 2;
  spec.seed = 1;
  const Fixture fixture(spec);
  const auto tokens = RandomTokens(6, spec.vocab, 82);
  const auto run = [&](ref::ActivationQuant mode, TapStore* taps) {
    ref::Options options;
    options.activation_quant = mode;
    return ref::Model(fixture.weights(), options)
        .Forward(tokens, taps ? taps->Fn() : ref::TapFn{});
  };
  TapStore exact_taps;
  TapStore cpu_taps;
  const auto exact = run(ref::ActivationQuant::kNone, &exact_taps);
  const auto cpu = run(ref::ActivationQuant::kCpu, &cpu_taps);
  const auto gpu = run(ref::ActivationQuant::kGpuQ8_1, nullptr);

  Require(cpu == gpu,
          "all-Q8_0 weights: CPU (Q8_0) and GPU (Q8_1) rounding agree exactly");
  for (float v : cpu) {
    Require(std::isfinite(v), "finite Q8-aware logits");
  }
  std::vector<double> exact_d(exact.begin(), exact.end());
  const double diff = RelRms(cpu, exact_d);
  // Precondition: the rounding flipped no expert choice. A flipped top-k
  // swaps a whole expert's contribution (an error of the router's margin, not
  // of the quantizer: 10 to 20% on these tiny models), so the bound below is
  // only meaningful for a fixture whose margins survive the rounding noise.
  Require(
      exact_taps.i32.at("ffn_moe_topk-1") == cpu_taps.i32.at("ffn_moe_topk-1"),
      "fixture precondition: Q8 rounding must not flip an expert choice");
  // Measured about 2.5% on this fixture (two layers, a dozen rounded matmul
  // inputs on the path to every logit); 5% leaves 2x for a different libm.
  Require(diff > 5e-3, "Q8-aware mode must differ from F32 (rel rms " +
                           std::to_string(diff) + ")");
  Require(diff < 5e-2,
          "Q8-aware mode drifted beyond quantization error (rel rms " +
              std::to_string(diff) + ")");

  // The rounding inside the matmul: recompute layer 1's Q projection from the
  // tapped input with an independent quantizer and double accumulation.
  const std::size_t H = spec.hidden;
  const auto& n = cpu_taps.F("attn_norm-1");
  const LayerMats& l = fixture.layer(1);
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    const std::vector<float> row(n.begin() + t * H, n.begin() + (t + 1) * H);
    const auto rounded = HandQ8_0(row);
    const Vec x(rounded.begin(), rounded.end());
    const Vec want = MatVec(l.q, 0, x);
    const auto& got = cpu_taps.F("Qcur-1");
    double err = 0.0, den = 0.0;
    for (std::size_t r = 0; r < l.q.rows; ++r) {
      const double g = got[t * l.q.rows + r];
      err += (g - want[r]) * (g - want[r]);
      den += want[r] * want[r];
    }
    Require(std::sqrt(err / den) < 1e-5, "Qcur uses Q8_0-rounded input");
  }
  // F32-weight matmuls (router, gate) keep exact inputs in the Q8-aware mode:
  // their taps from identical inputs would match the exact run's math. The
  // gate input is attn_norm of the cpu run, so recompute it exactly.
  {
    const auto& gate = cpu_taps.F("attn_gate_proj-1");
    const std::size_t heads = fixture.config().n_head(1);
    const std::vector<float> row(n.begin(), n.begin() + H);
    const Vec x(row.begin(), row.end());
    const Vec want = MatVec(l.gate, 0, x);
    for (std::size_t r = 0; r < heads; ++r) {
      Require(std::fabs(gate[r] - want[r]) < 1e-4 * (1.0 + std::fabs(want[r])),
              "F32 gate matmul keeps exact input");
    }
  }
  // Single-token: the exact-mode taps are untouched by the other mode.
  Require(exact_taps.F("result_output") == exact,
          "exact run is self-consistent");
  std::cout << "    q8-aware vs f32 logits rel rms " << diff << "\n";
}

}  // namespace

int main() {
  try {
    TestRounding();
    std::cout << "ok  activation rounding (Q8_0, Q8_K, Q8_1, fp16) vs hand "
                 "quantizers\n";
    TestModes();
    std::cout << "ok  F32 vs Q8-aware differ only within quantization error\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
