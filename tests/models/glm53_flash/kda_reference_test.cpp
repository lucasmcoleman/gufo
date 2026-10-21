// CPU oracle checks for the GLM-5.3-Flash KDA mixer (reference_kda.hpp)
// against the HF transformers 09281b8 fixture (fixtures/kda_fixture.bin):
// (a) rc.*: recurrent_kimi_delta_attention (HF:463-513) at Dk 12 / Dv 8,
//     unequal per-key decay, nonzero and zero initial state; output and
//     final state vs HF recurrent and the chunk form (HF:517-614);
//     prefill-then-decode == one call; the 2x2 key-row decay orientation case.
// (b) layer.*: the whole layer-0 mixer (HF:664-769) at hidden 32, 4 heads of
//     8: conv+SiLU (and a split history run), gate g, beta, core output,
//     final state, output gate, gated RMSNorm and o_proj.
// (c) Mutation sanity: a recurrence that decays the value axis must fail.
// Every float comparison is rel-L2 <= 1e-5.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/glm53_flash/reference_kda.hpp"
#include "tests/models/glm53_flash/kda_fixture.hpp"

namespace ref = gufo::models::glm53_flash::reference;
namespace fx = glm53_kda_fixture;

namespace {

constexpr double kTol = 1e-5;
int failures = 0;

using Fixture = std::map<std::string, fx::Array>;

void Check(const std::string& what, const float* got, const float* want,
           std::size_t n) {
  const double rel = fx::RelL2(got, want, n);
  const bool ok = std::isfinite(rel) && rel <= kTol;
  std::cout << (ok ? "ok   " : "FAIL ") << what << " rel_l2 " << rel << "\n";
  if (!ok)
    ++failures;
}

void Check(const std::string& what, const std::vector<float>& got,
           const std::vector<float>& want) {
  if (got.size() != want.size()) {
    std::cout << "FAIL " << what << " size " << got.size() << " vs "
              << want.size() << "\n";
    ++failures;
    return;
  }
  Check(what, got.data(), want.data(), want.size());
}

/// Expects the mutated result to MISS the reference by more than kTol.
void CheckMutantFails(const std::string& what, const std::vector<float>& got,
                      const std::vector<float>& want) {
  const double rel = fx::RelL2(got.data(), want.data(), want.size());
  const bool ok = rel > kTol;
  std::cout << (ok ? "ok   " : "FAIL ") << what << " mutant rel_l2 " << rel
            << " (must exceed " << kTol << ")\n";
  if (!ok)
    ++failures;
}

const fx::Array& Get(const Fixture& f, const char* name) {
  const auto it = f.find(name);
  if (it == f.end())
    throw std::runtime_error(std::string("fixture array missing: ") + name);
  return it->second;
}

/// Slice b of a [B, ...] array.
std::vector<float> Batch(const fx::Array& a, std::uint32_t b) {
  const std::size_t per = a.data.size() / a.shape[0];
  return {a.data.begin() + static_cast<std::ptrdiff_t>(b * per),
          a.data.begin() + static_cast<std::ptrdiff_t>((b + 1) * per)};
}

/// y[t] = W x[t] with W in torch layout [out][in]; F64 accumulation.
std::vector<float> Linear(const std::vector<float>& x, const fx::Array& w,
                          std::uint32_t n_tokens) {
  const std::size_t out = w.shape[0];
  const std::size_t in = w.shape[1];
  std::vector<float> y(n_tokens * out);
  for (std::size_t t = 0; t < n_tokens; ++t) {
    for (std::size_t o = 0; o < out; ++o) {
      double acc = 0.0;
      for (std::size_t i = 0; i < in; ++i)
        acc += static_cast<double>(w.data[(o * in) + i]) * x[(t * in) + i];
      y[(t * out) + o] = static_cast<float>(acc);
    }
  }
  return y;
}

/// The bug the orientation case pins: decay applied along the value axis
/// (S[i][j] *= exp(g[j])); g has Dk entries, the first Dv are used.
void MutantRecurrent(const float* q, const float* k, const float* v,
                     const float* g, const float* beta, float* state, float* o,
                     std::uint32_t n_tokens, std::uint32_t heads,
                     std::uint32_t Dk, std::uint32_t Dv) {
  for (std::size_t t = 0; t < n_tokens; ++t) {
    for (std::size_t h = 0; h < heads; ++h) {
      const std::size_t row = (t * heads) + h;
      float* S = state + (h * Dk * Dv);
      for (std::size_t i = 0; i < Dk; ++i)
        for (std::size_t j = 0; j < Dv; ++j)
          S[(i * Dv) + j] *= std::exp(g[(row * Dk) + j]);
      std::vector<double> u(Dv, 0.0);
      for (std::size_t i = 0; i < Dk; ++i)
        for (std::size_t j = 0; j < Dv; ++j)
          u[j] += static_cast<double>(S[(i * Dv) + j]) * k[(row * Dk) + i];
      for (std::size_t j = 0; j < Dv; ++j)
        u[j] = (v[(row * Dv) + j] - u[j]) * beta[row];
      for (std::size_t i = 0; i < Dk; ++i)
        for (std::size_t j = 0; j < Dv; ++j)
          S[(i * Dv) + j] += static_cast<float>(k[(row * Dk) + i] * u[j]);
      for (std::size_t j = 0; j < Dv; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < Dk; ++i)
          acc += static_cast<double>(S[(i * Dv) + j]) * q[(row * Dk) + i];
        o[(row * Dv) + j] = static_cast<float>(acc);
      }
    }
  }
}

void TestRecurrentFixture(const Fixture& f) {
  const auto& q_all = Get(f, "rc.q");
  const std::uint32_t B = q_all.shape[0];
  const std::uint32_t T = q_all.shape[1];
  const std::uint32_t H = q_all.shape[2];
  const std::uint32_t Dk = q_all.shape[3];
  const std::uint32_t Dv = Get(f, "rc.v").shape[3];
  std::vector<float> out_all;
  std::vector<float> state_all;
  std::vector<float> out_zero_all;
  std::vector<float> state_zero_all;
  std::vector<float> mutant_all;
  for (std::uint32_t b = 0; b < B; ++b) {
    auto q = Batch(q_all, b);
    auto k = Batch(Get(f, "rc.k"), b);
    const auto v = Batch(Get(f, "rc.v"), b);
    const auto g = Batch(Get(f, "rc.g"), b);
    const auto beta = Batch(Get(f, "rc.beta"), b);  // already sigmoided
    ref::KdaPrep(q.data(), k.data(), nullptr, T, H, Dk, 1e-6F);

    auto state = Batch(Get(f, "rc.initial_state"), b);
    std::vector<float> out(std::size_t{T} * H * Dv);
    ref::KdaRecurrent(q.data(), k.data(), v.data(), g.data(), beta.data(),
                      state.data(), out.data(), T, H, Dk, Dv);
    out_all.insert(out_all.end(), out.begin(), out.end());
    state_all.insert(state_all.end(), state.begin(), state.end());

    // Prefill 13 tokens, then decode the rest one at a time.
    auto split_state = Batch(Get(f, "rc.initial_state"), b);
    std::vector<float> split_out(out.size());
    const std::uint32_t pre = 13;
    ref::KdaRecurrent(q.data(), k.data(), v.data(), g.data(), beta.data(),
                      split_state.data(), split_out.data(), pre, H, Dk, Dv);
    for (std::uint32_t t = pre; t < T; ++t) {
      ref::KdaRecurrent(q.data() + (std::size_t{t} * H * Dk),
                        k.data() + (std::size_t{t} * H * Dk),
                        v.data() + (std::size_t{t} * H * Dv),
                        g.data() + (std::size_t{t} * H * Dk),
                        beta.data() + t * H, split_state.data(),
                        split_out.data() + (std::size_t{t} * H * Dv), 1, H, Dk,
                        Dv);
    }
    Check("rc b" + std::to_string(b) + " prefill(13)+decode(7) out vs full",
          split_out, out);
    Check("rc b" + std::to_string(b) + " prefill(13)+decode(7) state vs full",
          split_state, state);

    std::vector<float> zero_state(std::size_t{H} * Dk * Dv, 0.0F);
    std::vector<float> zero_out(out.size());
    ref::KdaRecurrent(q.data(), k.data(), v.data(), g.data(), beta.data(),
                      zero_state.data(), zero_out.data(), T, H, Dk, Dv);
    out_zero_all.insert(out_zero_all.end(), zero_out.begin(), zero_out.end());
    state_zero_all.insert(state_zero_all.end(), zero_state.begin(),
                          zero_state.end());

    auto m_state = Batch(Get(f, "rc.initial_state"), b);
    std::vector<float> m_out(out.size());
    MutantRecurrent(q.data(), k.data(), v.data(), g.data(), beta.data(),
                    m_state.data(), m_out.data(), T, H, Dk, Dv);
    mutant_all.insert(mutant_all.end(), m_out.begin(), m_out.end());
  }
  Check("rc out vs HF out_recurrent", out_all, Get(f, "rc.out_recurrent").data);
  Check("rc state vs HF state_recurrent", state_all,
        Get(f, "rc.state_recurrent").data);
  Check("rc out vs HF out_chunk64", out_all, Get(f, "rc.out_chunk64").data);
  Check("rc state vs HF state_chunk64", state_all,
        Get(f, "rc.state_chunk64").data);
  Check("rc out vs HF out_chunk8", out_all, Get(f, "rc.out_chunk8").data);
  Check("rc state vs HF state_chunk8", state_all,
        Get(f, "rc.state_chunk8").data);
  Check("rc zero-init out vs HF", out_zero_all,
        Get(f, "rc.out_recurrent_zero_init").data);
  Check("rc zero-init state vs HF", state_zero_all,
        Get(f, "rc.state_recurrent_zero_init").data);
  CheckMutantFails("rc value-axis decay out vs HF", mutant_all,
                   Get(f, "rc.out_recurrent").data);
}

void TestOrientation(const Fixture& f) {
  // S = [[1,2],[3,4]] (key rows), decay [0.5, 0.25], beta 0, one step.
  auto state = Get(f, "rc.orient_initial_state").data;
  const auto& g = Get(f, "rc.orient_g").data;
  const std::vector<float> zeros(2, 0.0F);
  const std::vector<float> beta(1, 0.0F);
  std::vector<float> out(2);
  ref::KdaRecurrent(zeros.data(), zeros.data(), zeros.data(), g.data(),
                    beta.data(), state.data(), out.data(), 1, 1, 2, 2);
  Check("orientation 2x2 key-row decay", state,
        Get(f, "rc.orient_final_state").data);
  auto m_state = Get(f, "rc.orient_initial_state").data;
  MutantRecurrent(zeros.data(), zeros.data(), zeros.data(), g.data(),
                  beta.data(), m_state.data(), out.data(), 1, 1, 2, 2);
  CheckMutantFails("orientation 2x2 value-axis decay", m_state,
                   Get(f, "rc.orient_final_state").data);
}

void TestLayerFixture(const Fixture& f) {
  const auto& x_all = Get(f, "layer.x");
  const std::uint32_t B = x_all.shape[0];
  const std::uint32_t T = x_all.shape[1];
  const auto& g_shape = Get(f, "layer.g").shape;
  const std::uint32_t H = g_shape[2];
  const std::uint32_t D = g_shape[3];
  const std::uint32_t width = H * D;
  const std::uint32_t channels = 3 * width;
  const std::uint32_t kernel = Get(f, "layer.w.conv1d.weight").shape[2];
  constexpr float kLowerBound = -5.0F;  // linear_lower_bound
  constexpr float kRmsEps = 1e-5F;      // o_norm, rms_norm_eps
  constexpr float kL2Eps = 1e-6F;       // HF:458

  // GGUF ssm_a = -exp(A_log), computed in F32 as the converter does.
  std::vector<float> a;
  for (const float a_log : Get(f, "layer.w.forget_gate.A_log").data)
    a.push_back(-std::exp(a_log));
  const auto& conv_w = Get(f, "layer.w.conv1d.weight").data;  // [C][1][K]
  const auto& dt_bias = Get(f, "layer.w.forget_gate.dt_bias").data;
  const auto& norm_w = Get(f, "layer.w.o_norm.weight").data;

  std::vector<float> lin_all[3];
  std::vector<float> conv_all, g_all, beta_all, core_all, state_all, gate_all,
      onorm_all, y_all;
  for (std::uint32_t b = 0; b < B; ++b) {
    const auto x = Batch(x_all, b);
    const std::vector<float> lin[3] = {
        Linear(x, Get(f, "layer.w.q_proj.weight"), T),
        Linear(x, Get(f, "layer.w.k_proj.weight"), T),
        Linear(x, Get(f, "layer.w.v_proj.weight"), T)};
    for (int p = 0; p < 3; ++p)
      lin_all[p].insert(lin_all[p].end(), lin[p].begin(), lin[p].end());
    std::vector<float> qkv(std::size_t{T} * channels);
    for (std::uint32_t t = 0; t < T; ++t)
      for (int p = 0; p < 3; ++p)
        std::copy_n(lin[p].begin() + (std::size_t{t} * width), width,
                    qkv.begin() + (std::size_t{t} * channels) + (p * width));

    // Conv + SiLU from a fresh (zero) history, and split 6 + 1 + 1 + ...
    std::vector<float> hist(std::size_t{kernel - 1} * channels, 0.0F);
    std::vector<float> conv(qkv.size());
    ref::KdaConv(qkv.data(), conv_w.data(), hist.data(), conv.data(), T,
                 channels, kernel);
    conv_all.insert(conv_all.end(), conv.begin(), conv.end());
    std::vector<float> split_hist(hist.size(), 0.0F);
    std::vector<float> split_conv(conv.size());
    ref::KdaConv(qkv.data(), conv_w.data(), split_hist.data(),
                 split_conv.data(), 6, channels, kernel);
    for (std::uint32_t t = 6; t < T; ++t) {
      ref::KdaConv(qkv.data() + (std::size_t{t} * channels), conv_w.data(),
                   split_hist.data(),
                   split_conv.data() + (std::size_t{t} * channels), 1, channels,
                   kernel);
    }
    Check("layer b" + std::to_string(b) + " conv prefill(6)+decode vs full",
          split_conv, conv);
    Check("layer b" + std::to_string(b) + " conv history vs full", split_hist,
          hist);

    // Forget gate: f_b(f_a(x)), then KdaGate.
    const auto f_lin =
        Linear(Linear(x, Get(f, "layer.w.forget_gate.f_a_proj.weight"), T),
               Get(f, "layer.w.forget_gate.f_b_proj.weight"), T);
    std::vector<float> g(std::size_t{T} * width);
    ref::KdaGate(f_lin.data(), dt_bias.data(), a.data(), kLowerBound, g.data(),
                 T, H, D);
    g_all.insert(g_all.end(), g.begin(), g.end());

    // q, k, v from the conv output; beta from b_proj.
    std::vector<float> q(std::size_t{T} * width), k(q.size()), v(q.size());
    for (std::uint32_t t = 0; t < T; ++t) {
      const float* row = conv.data() + (std::size_t{t} * channels);
      std::copy_n(row, width, q.begin() + (std::size_t{t} * width));
      std::copy_n(row + width, width, k.begin() + (std::size_t{t} * width));
      std::copy_n(row + (2 * width), width,
                  v.begin() + (std::size_t{t} * width));
    }
    auto beta = Linear(x, Get(f, "layer.w.b_proj.weight"), T);
    ref::KdaPrep(q.data(), k.data(), beta.data(), T, H, D, kL2Eps);
    beta_all.insert(beta_all.end(), beta.begin(), beta.end());

    std::vector<float> state(std::size_t{H} * D * D, 0.0F);
    std::vector<float> core(std::size_t{T} * width);
    ref::KdaRecurrent(q.data(), k.data(), v.data(), g.data(), beta.data(),
                      state.data(), core.data(), T, H, D, D);
    core_all.insert(core_all.end(), core.begin(), core.end());
    state_all.insert(state_all.end(), state.begin(), state.end());

    const auto gate = Linear(Linear(x, Get(f, "layer.w.g_a_proj.weight"), T),
                             Get(f, "layer.w.g_b_proj.weight"), T);
    gate_all.insert(gate_all.end(), gate.begin(), gate.end());
    std::vector<float> onorm(core.size());
    ref::KdaOutput(core.data(), gate.data(), norm_w.data(), kRmsEps,
                   onorm.data(), T, H, D);
    onorm_all.insert(onorm_all.end(), onorm.begin(), onorm.end());
    const auto y = Linear(onorm, Get(f, "layer.w.o_proj.weight"), T);
    y_all.insert(y_all.end(), y.begin(), y.end());
  }
  Check("layer q_lin", lin_all[0], Get(f, "layer.q_lin").data);
  Check("layer k_lin", lin_all[1], Get(f, "layer.k_lin").data);
  Check("layer v_lin", lin_all[2], Get(f, "layer.v_lin").data);
  Check("layer conv_out_silu", conv_all, Get(f, "layer.conv_out_silu").data);
  Check("layer g", g_all, Get(f, "layer.g").data);
  Check("layer beta", beta_all, Get(f, "layer.beta").data);
  Check("layer core_out_recurrent", core_all,
        Get(f, "layer.core_out_recurrent").data);
  Check("layer core_out (chunk)", core_all, Get(f, "layer.core_out").data);
  Check("layer final_state_recurrent", state_all,
        Get(f, "layer.final_state_recurrent").data);
  Check("layer out_gate_logits", gate_all,
        Get(f, "layer.out_gate_logits").data);
  Check("layer o_norm_out", onorm_all, Get(f, "layer.o_norm_out").data);
  Check("layer y", y_all, Get(f, "layer.y").data);
}

void TestGateForms() {
  // Unbounded form a * softplus(z), including the z > 20 guard.
  const float f[3] = {-1.5F, 0.25F, 30.0F};
  const float dt[3] = {0.5F, -0.75F, 1.0F};
  const float a[1] = {-0.8F};
  float g[3];
  ref::KdaGate(f, dt, a, -std::numeric_limits<float>::infinity(), g, 1, 1, 3);
  const float want[3] = {static_cast<float>(-0.8 * std::log1p(std::exp(-1.0))),
                         static_cast<float>(-0.8 * std::log1p(std::exp(-0.5))),
                         -0.8F * 31.0F};
  Check("gate softplus form", g, want, 3);
  // Bounded form lb * sigmoid(-a z) = -5 / (1 + exp(a z)).
  ref::KdaGate(f, dt, a, -5.0F, g, 1, 1, 3);
  const float want_b[3] = {
      static_cast<float>(-5.0 / (1.0 + std::exp(-0.8 * -1.0))),
      static_cast<float>(-5.0 / (1.0 + std::exp(-0.8 * -0.5))),
      static_cast<float>(-5.0 / (1.0 + std::exp(-0.8 * 31.0)))};
  Check("gate bounded form", g, want_b, 3);
}

}  // namespace

int main() {
  try {
    const auto f = fx::Load(GLM53_KDA_FIXTURE);
    TestRecurrentFixture(f);
    TestOrientation(f);
    TestLayerFixture(f);
    TestGateForms();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "glm53_flash kda_reference: PASS\n";
  return 0;
}
