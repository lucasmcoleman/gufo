// CPU oracle checks for GLM-5.3-Flash hyper-connections (reference_hc.hpp).
// (a) Properties on random inputs: comb is doubly stochastic after Sinkhorn
//     (to 1e-4 for flat logits; see TestProperties for the peaked bound),
//     pre in [eps, 1 + eps], post in [0, 2], the head mean of copies is exact.
// (b) A hidden=8 case against golden values produced by running HF
//     transformers 09281b8 modeling_glm5_next.py (Glm5NextTextHyperConnection,
//     HF:271-329; decoder expand, HF:1336-1338; Glm5NextTextHyperHead,
//     HF:332-336) verbatim under torch 2.14 CPU, float32. The inputs below are
//     closed forms evaluated in double and rounded to float in both programs.
// (c) The HF fixture (fixtures/hc_fixture.bin, hidden 32, 10 tokens): every
//     intermediate at rel-L2 <= 1e-5.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/reference_hc.hpp"
#include "tests/models/glm53_flash/hc_fixture.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;

namespace {

// Golden values from hc_golden.py (HF code, torch CPU float32).
constexpr float kLogits[24] = {
    0.509314656F,   -0.47121644F,  0.956145585F,   -1.08724117F,  -0.816751122F,
    0.556103289F,   -0.913061857F, 0.375590414F,   1.42483163F,   -1.07386434F,
    0.149631828F,   1.43197429F,   -1.6979183F,    -1.091061F,    1.62274051F,
    0.291096449F,   -0.662238479F, 0.100694314F,   0.0897672772F, 0.0209058672F,
    -0.0438085049F, 0.0588494986F, -0.0227771848F, -0.415273815F};
constexpr float kPre[4] = {0.612410486F, 0.405738831F, 0.658377171F,
                           0.265312731F};
constexpr float kPost[4] = {0.514299989F, 1.33112872F, 0.518200159F,
                            1.20507336F};
constexpr float kComb[16] = {
    0.544867635F,  0.00839575008F, 0.0180779584F, 0.428657889F,
    0.0119424993F, 0.0330298766F,  0.75710094F,   0.197925225F,
    0.122362241F,  0.520029783F,   0.12058647F,   0.237020567F,
    0.32082662F,   0.438543618F,   0.104233645F,  0.136395261F};
constexpr float kX[8] = {1.69173586F, 2.04832745F, 2.22194815F, 2.19708872F,
                         1.97597003F, 1.57834363F, 1.03972864F, 0.408237696F};
constexpr float kHNew[32] = {
    1.29165566F,   1.44009304F,  1.42533803F,   1.2541641F,   0.951910675F,
    0.558633566F,  0.123465568F, -0.301983207F, 2.79105115F,  2.83513975F,
    2.53654242F,   1.93605137F,  1.11331606F,   0.175621808F, -0.757027924F,
    -1.56835389F,  1.38757205F,  1.60709393F,   1.64824378F,  1.5128423F,
    1.22311008F,   0.818083227F, 0.348050505F,  -0.13216731F, 1.91898441F,
    2.11801124F,   2.04688025F,  1.72472739F,   1.20387614F,  0.561445177F,
    -0.112368941F, -0.727530241F};
constexpr float kHead[8] = {1.84731579F,    2.0000844F,   1.91425109F,
                            1.60694623F,    1.12305319F,  0.528445959F,
                            -0.0994701907F, -0.682508647F};

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

void ExpectNear(const float* got, const float* want, std::size_t n, double tol,
                const std::string& what) {
  double worst = 0.0;
  std::size_t at = 0;
  for (std::size_t i = 0; i < n; ++i) {
    // NaN compares false against any bound, so a non-finite value must be
    // rejected explicitly or it would pass silently.
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
      std::cerr << "FAIL: " << what << " [" << i << "] non-finite got "
                << got[i] << " want " << want[i] << "\n";
      ++failures;
      return;
    }
    const double err = std::abs(static_cast<double>(got[i]) - want[i]) /
                       std::max(1.0, std::abs(static_cast<double>(want[i])));
    if (err > worst) {
      worst = err;
      at = i;
    }
  }
  if (worst > tol) {
    std::cerr << "FAIL: " << what << " [" << at << "] got " << got[at]
              << " want " << want[at] << " (err " << worst << ")\n";
    ++failures;
  }
}

glm::Config MakeConfig(std::uint32_t hidden) {
  glm::Config c;
  c.hidden_size = hidden;
  c.hc_count = 4;
  c.hc_sinkhorn_iters = 20;
  c.hc_eps = 1e-6F;
  c.rms_eps = 1e-5F;
  return c;
}

void TestGolden() {
  constexpr std::uint32_t kHc = 4;
  constexpr std::uint32_t kD = 8;
  const glm::Config c = MakeConfig(kD);
  std::vector<float> H(kHc * kD);
  std::vector<float> fn(24 * kHc * kD);
  std::vector<float> base(24);
  const float scale[3] = {0.8F, 1.2F, 1.7F};
  std::vector<float> a(kD);
  for (std::uint32_t s = 0; s < kHc; ++s) {
    for (std::uint32_t d = 0; d < kD; ++d) {
      H[(s * kD) + d] = static_cast<float>(
          std::sin((0.5 * s) + (0.3 * d) + 0.1) * (1.0 + (0.25 * s)));
    }
  }
  for (std::uint32_t j = 0; j < 24; ++j) {
    for (std::uint32_t k = 0; k < kHc * kD; ++k) {
      fn[(j * kHc * kD) + k] = static_cast<float>(
          0.3 * std::sin((0.7 * j) + (0.19 * k) + (0.05 * j * k)));
    }
    base[j] = static_cast<float>((0.1 * std::cos(1.1 * j)) - 0.05);
  }
  for (std::uint32_t d = 0; d < kD; ++d) {
    a[d] = static_cast<float>(std::cos((0.4 * d) - 0.2));
  }

  ref::HcMix mix;
  std::vector<float> x(kD);
  ref::HcPre(c, fn.data(), base.data(), scale, H.data(), x.data(), &mix);
  std::vector<float> h_new(kHc * kD);
  ref::HcPost(mix, a.data(), H.data(), h_new.data());
  std::vector<float> head(kD);
  ref::HcHeadMean(c, h_new.data(), head.data());

  constexpr double kTol = 1e-5;
  ExpectNear(mix.logits.data(), kLogits, 24, kTol, "golden logits");
  ExpectNear(mix.pre.data(), kPre, 4, kTol, "golden pre");
  ExpectNear(mix.post.data(), kPost, 4, kTol, "golden post");
  ExpectNear(mix.comb.data(), kComb, 16, kTol, "golden comb");
  ExpectNear(x.data(), kX, kD, kTol, "golden x");
  ExpectNear(h_new.data(), kHNew, kHc * kD, kTol, "golden H_new");
  ExpectNear(head.data(), kHead, kD, kTol, "golden head");
  // The golden comb is asymmetric, so a transposed [dst][src] reading of it
  // (or of H_new) cannot pass the checks above.
  Expect(std::abs(kComb[1] - kComb[4]) > 1e-3F, "golden comb is asymmetric");
}

void TestFixture() {
  const auto fx = glm53_hc_fixture::Load(GLM53_HC_FIXTURE);
  const auto& H = fx.at("H");
  const std::uint32_t hidden = H.shape[3];
  const std::uint32_t tokens = H.shape[0] * H.shape[1];
  const glm::Config c = MakeConfig(hidden);
  const std::uint32_t hc_dim = 4 * hidden;
  Expect(fx.at("w.fn").shape[1] == hc_dim, "fixture fn shape");

  std::vector<float> mix(tokens * 24), pre(tokens * 4), post(tokens * 4),
      comb(tokens * 16), x(tokens * hidden), h_new(tokens * hc_dim),
      head(tokens * hidden);
  for (std::uint32_t t = 0; t < tokens; ++t) {
    ref::HcMix m;
    const float* h_old = H.data.data() + (t * hc_dim);
    ref::HcPre(c, fx.at("w.fn").data.data(), fx.at("w.base").data.data(),
               fx.at("w.scale").data.data(), h_old, x.data() + (t * hidden),
               &m);
    std::copy(m.logits.begin(), m.logits.end(), mix.begin() + (t * 24));
    std::copy(m.pre.begin(), m.pre.end(), pre.begin() + (t * 4));
    std::copy(m.post.begin(), m.post.end(), post.begin() + (t * 4));
    std::copy(m.comb.begin(), m.comb.end(), comb.begin() + (t * 16));
    ref::HcPost(m, fx.at("block_out").data.data() + (t * hidden), h_old,
                h_new.data() + (t * hc_dim));
    ref::HcHeadMean(c, h_new.data() + (t * hc_dim), head.data() + (t * hidden));
  }
  const std::pair<const char*, const std::vector<float>*> checks[] = {
      {"mix", &mix},       {"pre", &pre},     {"post", &post},
      {"comb", &comb},     {"collapsed", &x}, {"hc_out", &h_new},
      {"mean_head", &head}};
  for (const auto& [name, got] : checks) {
    const auto& want = fx.at(name).data;
    Expect(want.size() == got->size(), std::string("fixture size ") + name);
    const double rel =
        glm53_hc_fixture::RelL2(got->data(), want.data(), want.size());
    std::cout << "fixture " << name << " rel_l2 " << rel << "\n";
    Expect(rel <= 1e-5,
           std::string("fixture ") + name + " rel_l2 " + std::to_string(rel));
  }
}

void TestProperties() {
  constexpr std::uint32_t kD = 64;
  const glm::Config c = MakeConfig(kD);
  const std::uint32_t hc_dim = 4 * kD;
  std::mt19937 rng(1234);
  std::normal_distribution<float> n01(0.0F, 1.0F);
  for (int trial = 0; trial < 200; ++trial) {
    // HF's 20-iteration Sinkhorn (HF:321-324) converges to 1e-4 only for
    // flat comb logits: measured with the HF lines in torch, a comb-logit
    // std of 0.5 leaves rows within 1e-6, std 1 within 1e-3 and std 2-8
    // within 0.03-0.08 of 1. Columns are normalised last, so they always sum
    // to s / (s + eps). Trials < 100 are flat (logit std < ~0.4) and must be
    // doubly stochastic to 1e-4; the rest are peaked and bound rows by 0.15.
    const bool flat = trial < 100;
    const float w_std =
        flat ? 0.005F : 0.02F * static_cast<float>(1 + (trial % 50));
    const float base_std = flat ? 0.1F : 1.0F;
    std::vector<float> H(hc_dim);
    std::vector<float> fn(24 * hc_dim);
    std::vector<float> base(24);
    for (auto& v : H)
      v = n01(rng) * (trial % 2 == 0 ? 1.0F : 30.0F);
    for (auto& v : fn)
      v = n01(rng) * w_std;
    for (auto& v : base)
      v = n01(rng) * base_std;
    const float scale[3] = {std::min(std::abs(n01(rng)) + 0.1F, 2.0F),
                            std::min(std::abs(n01(rng)) + 0.1F, 2.0F),
                            std::min(std::abs(n01(rng)) + 0.1F, 2.0F)};
    ref::HcMix mix;
    std::vector<float> x(kD);
    ref::HcPre(c, fn.data(), base.data(), scale, H.data(), x.data(), &mix);
    const std::string tag = " (trial " + std::to_string(trial) + ")";
    for (std::uint32_t i = 0; i < 4; ++i) {
      Expect(mix.pre[i] >= c.hc_eps && mix.pre[i] <= 1.0F + c.hc_eps,
             "pre range" + tag);
      Expect(mix.post[i] >= 0.0F && mix.post[i] <= 2.0F, "post range" + tag);
    }
    for (std::uint32_t r = 0; r < 4; ++r) {
      double row = 0.0;
      double col = 0.0;
      for (std::uint32_t k = 0; k < 4; ++k) {
        row += mix.comb[(r * 4) + k];
        col += mix.comb[(k * 4) + r];
        Expect(mix.comb[(r * 4) + k] > 0.0F, "comb positive" + tag);
      }
      Expect(std::abs(row - 1.0) <= (flat ? 1e-4 : 0.15), "comb row sum" + tag);
      Expect(std::abs(col - 1.0) <= 1e-4, "comb col sum" + tag);
    }
    for (std::uint32_t d = 0; d < kD; ++d) {
      double want = 0.0;
      for (std::uint32_t s = 0; s < 4; ++s) {
        want += static_cast<double>(mix.pre[s]) * H[(s * kD) + d];
      }
      Expect(std::abs(x[d] - want) <= 1e-5 * std::max(1.0, std::abs(want)),
             "collapse" + tag);
    }
  }

  // Stream init then head mean round-trips the embedding.
  std::vector<float> emb(kD);
  for (auto& v : emb)
    v = n01(rng);
  std::vector<float> H(hc_dim);
  ref::HcInit(c, emb.data(), H.data());
  std::vector<float> h(kD);
  ref::HcHeadMean(c, H.data(), h.data());
  ExpectNear(h.data(), emb.data(), kD, 1e-7, "init/head round trip");
}

}  // namespace

int main() {
  try {
    TestGolden();
    TestFixture();
    TestProperties();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "glm53_flash hc_reference: PASS\n";
  return 0;
}
