// GPU FFN operator checks against the phase 2 CPU oracle, on random tiny
// layers: no model file, nothing skipped. ffn_real_test.cpp repeats the checks
// on real GGUF slices.
//
// Every case runs the operator tapped, compares each stage with the oracle's
// own building blocks (router logits, top-k ids, weights, gate/up/down per
// expert, shared expert, combine) in the activation mode of the path taken,
// compares the final output with ref::FfnForward, then reruns untapped (two
// queued calls, no synchronization in between) and requires a bit-identical
// result. Checks and tolerances:
//
//  * Projection stages consume the GPU's own quantized activations: qfn_mmq
//    hands the test the packed buffer each projection kernel reads
//    (qfn_mmq_set_activation_capture). Each block is first validated against
//    the float input independently (ValidateBlock): every code within the
//    derived reciprocal error of round(x * 127 / amax), so a code other than
//    the nearest one is allowed only within that error (a few 1e-7 relative)
//    of a .5 boundary; equal inputs with equal codes and one shared
//    multiplier reproducing the whole block; the stored scale and sum the
//    fp16 (or, MMQ D4, float) rounding of a value within the derived error of
//    the unrounded amax / 127 and amax / 127 * sum(q). The bounds are derived
//    per layout (Q8_1 for MMVQ decode, MMQ D4 and DS4 for prefill tiles) in
//    ffn_gpu.hpp (BoundsFor). Projections sharing one input (gate/up, routed
//    and shared expert) must have read bitwise-identical quantizations.
//    The validated codes and scales then feed the oracle's product, which
//    must match every output within the float accumulation bound
//    (AccumulationUnits: (K / 16 + 18) u times the sum of the terms'
//    magnitudes). Nothing is searched: there are no flip subsets or scale
//    nudges, and raw errors are reported for every output.
//  * CPU-only checker tests (TestCheckerRejects) feed fabricated captures:
//    an off-boundary scale, inconsistent codes for equal inputs, a wrong code
//    far from any boundary and a jointly unrealizable block must be
//    rejected; a genuine near-boundary code must pass.
//  * End to end (5e-3 of rms): inputs are moved off Q8 rounding boundaries
//    (AvoidTies), but the oracle's intermediate activations still differ from
//    the GPU's by float rounding, so a near-tie element of an intermediate row
//    can round differently; one flip moves one expert output by about one Q8
//    step times a weight. Observed worst: 2.8e-3 (real layer 0, 37 rows),
//    about 1e-3 on the tiny layers, against Q8 noise of 3e-2 to 1e-1 (printed
//    as "Q8 noise"); a wiring or weighting defect is far larger. The stage
//    checks are the exactness claim.
//  * Scratch growth survives allocation failure (TestReserveFailure): a
//    mocked failing allocator leaves every buffer and capacity untouched.
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <set>

#include "tests/models/laguna/ffn_gpu.hpp"

using namespace gufo::testing::laguna;

namespace {

struct Totals {
  std::map<std::string, double> worst_e2e;  // per quant type, / rms
  std::map<std::string, double> worst_abs;
  // per quant type and stage: worst raw error / rms, / accumulation bound
  std::map<std::string, std::map<std::string, StageReport>> stages;
  std::size_t cases{0};
  std::size_t boundary_codes{0};
  std::size_t blocks{0};
};
Totals g_totals;

void Record(const std::string& type, const CaseReport& r) {
  g_totals.worst_e2e[type] = std::max(g_totals.worst_e2e[type], r.e2e_over_rms);
  g_totals.worst_abs[type] = std::max(g_totals.worst_abs[type], r.e2e_abs);
  for (const auto& [name, st] : r.stages) {
    StageReport& w = g_totals.stages[type][name];
    w.over_rms = std::max(w.over_rms, st.over_rms);
    w.over_bound = std::max(w.over_bound, st.over_bound);
    w.max_scale_rel = std::max(w.max_scale_rel, st.max_scale_rel);
    w.boundary_codes += st.boundary_codes;
    w.blocks += st.blocks;
    g_totals.blocks += st.blocks;
  }
  g_totals.boundary_codes += r.boundary_codes;
  ++g_totals.cases;
  PrintReport(r);
}

// ---- Checker tests on fabricated captures (CPU only) ----------------------

/// A Q8_0 matrix of `sel.size()` rows and 32 columns: row r has weight 1 at
/// column sel[r] and 0 elsewhere (scale 1).
struct OneHot {
  std::vector<std::uint8_t> bytes;
  ml::TensorRef w;
};
OneHot MakeOneHot(const std::vector<int>& sel) {
  OneHot m;
  m.bytes.assign(34 * sel.size(), 0);
  const std::uint16_t one = HalfBitsOf(1.0F);
  for (std::size_t r = 0; r < sel.size(); ++r) {
    std::memcpy(m.bytes.data() + 34 * r, &one, 2);
    m.bytes[34 * r + 2 + sel[r]] = 1;
  }
  m.w.type = GgmlType::kQ8_0;
  m.w.cols = 32;
  m.w.rows = sel.size();
  m.w.experts = 1;
  m.w.data = m.bytes.data();
  m.w.name = "checker.weight";
  return m;
}

/// One 32-element row as the given layout stores it.
QuantRows Fabricate(int layout, const std::vector<std::int8_t>& q, float d,
                    float s) {
  Capture c;
  c.layout = layout;
  c.rows = 1;
  c.k = 32;
  const auto put_half = [](std::uint8_t* p, float v) {
    const std::uint16_t bits = HalfBitsOf(v);
    std::memcpy(p, &bits, 2);
  };
  if (layout == QFN_MMQ_ACT_Q8_1) {
    c.k_padded = 32;
    c.bytes.assign(36, 0);
    put_half(c.bytes.data(), d);
    put_half(c.bytes.data() + 2, s);
    std::memcpy(c.bytes.data() + 4, q.data(), 32);
  } else {
    c.k_padded = 128;
    c.bytes.assign(144, 0);
    if (layout == QFN_MMQ_ACT_MMQ_D4) {
      std::memcpy(c.bytes.data(), &d, 4);
    } else {
      put_half(c.bytes.data(), d);
      put_half(c.bytes.data() + 2, s);
    }
    std::memcpy(c.bytes.data() + 16, q.data(), 32);
  }
  return DecodeCapture(c);
}

/// Checks one fabricated activation row: `x` quantized to `q` with stored
/// `d` and `s`, and `gpu`, the GPU's outputs of the one-hot matrix `sel`.
StageStats CheckFabricated(const std::vector<float>& x,
                           const std::vector<std::int8_t>& q, float d, float s,
                           const std::vector<int>& sel,
                           const std::vector<float>& gpu) {
  const OneHot m = MakeOneHot(sel);
  const QuantRows qr = Fabricate(QFN_MMQ_ACT_Q8_1, q, d, s);
  // One activation row; output m reads column sel[m].
  return CheckCapturedProjection(m.w, {0}, x, 1, gpu, qr);
}

void ExpectRejected(const std::string& what, const StageStats& s,
                    bool by_validation) {
  const bool rejected =
      by_validation ? s.quant.violations != 0 : s.unexplained != 0;
  Require(rejected, "checker accepted " + what);
  std::cout << "  rejects " << what << ": "
            << (by_validation ? s.quant.first : s.first) << "\n";
}

StageStats Rejection(const QuantStats& quant) {
  StageStats s;
  s.quant = quant;
  return s;
}

void TestCheckerRejects() {
  // Block with amax 127, so amax / 127 = 1 exactly and x * 127 / amax = x.
  std::vector<float> x(32, 0.0F);
  x[0] = 127.0F;
  std::vector<std::int8_t> q(32, 0);
  q[0] = 127;
  // codexms #970: the exactly representable scale 1.0 stored as the next
  // fp16 value, with no near-tie input anywhere; the output follows it.
  {
    const float wrong = 1.0009765625F;
    ExpectRejected(
        "an off-boundary fp16 scale (1.0 stored as 1.0009765625)",
        CheckFabricated(x, q, wrong, wrong * 127.0F, {0}, {127.0F * wrong}),
        true);
    // Correct capture, but the product used the wrong scale.
    ExpectRejected("an output computed with an off-boundary scale",
                   CheckFabricated(x, q, 1.0F, 127.0F, {0}, {127.0F * wrong}),
                   false);
  }
  // codexms #970: equal inputs 0.5 and 0.5 in one block (each exactly on a
  // .5 boundary) with codes 0 and 1; each code alone is legitimate.
  {
    auto xe = x;
    xe[1] = xe[2] = 0.5F;
    auto qe = q;
    qe[2] = 1;
    ExpectRejected("equal inputs with different codes (0.5, 0.5 -> 0, 1)",
                   CheckFabricated(xe, qe, 1.0F, 128.0F, {1, 2}, {0.0F, 1.0F}),
                   true);
    // Equal codes either way are fine.
    for (const std::int8_t c : {0, 1}) {
      auto qc = q;
      qc[1] = qc[2] = c;
      const auto ok = CheckFabricated(xe, qc, 1.0F, 127.0F + 2 * c, {1, 2},
                                      {float(c), float(c)});
      Require(ok.quant.violations == 0 && ok.unexplained == 0,
              "checker rejected equal codes for equal inputs: " +
                  ok.quant.first + ok.first);
    }
  }
  // codexms #971: equal magnitudes of opposite sign (+0.5, -0.5) with codes
  // +1 and 0. Each code alone is a legitimate rounding of a .5 boundary, but
  // roundf with one positive multiplier gives q(-x) = -q(x). Every layout.
  {
    auto xs = x;
    xs[1] = 0.5F;
    xs[2] = -0.5F;
    auto qs = q;
    qs[1] = 1;
    ExpectRejected(
        "opposite-sign equal magnitudes with asymmetric codes (+0.5, -0.5 -> "
        "+1, 0)",
        CheckFabricated(xs, qs, 1.0F, 128.0F, {1, 2}, {1.0F, 0.0F}), true);
    for (const int layout : {QFN_MMQ_ACT_MMQ_D4, QFN_MMQ_ACT_MMQ_DS4}) {
      const QuantStats st =
          ValidateQuantRows(Fabricate(layout, qs, 1.0F, 128.0F), xs, 1);
      ExpectRejected(std::string("asymmetric opposite-sign codes in ") +
                         LayoutName(layout),
                     Rejection(st), true);
    }
    // A code with the opposite sign of its input.
    auto qn = q;
    qn[1] = -1;
    ExpectRejected("a code with the opposite sign of its input (+0.5 -> -1)",
                   CheckFabricated(xs, qn, 1.0F, 126.0F, {1}, {-1.0F}), true);
    // Symmetric codes either way are fine: (+1, -1) and (0, 0).
    for (const std::int8_t c : {0, 1}) {
      auto qc = q;
      qc[1] = c;
      qc[2] = static_cast<std::int8_t>(-c);
      const auto ok =
          CheckFabricated(xs, qc, 1.0F, 127.0F, {1, 2}, {float(c), float(-c)});
      Require(ok.quant.violations == 0 && ok.unexplained == 0,
              "checker rejected symmetric codes for opposite-sign inputs: " +
                  ok.quant.first + ok.first);
    }
  }
  // A single wrong code far from any boundary: 40.2 -> 41.
  {
    auto xw = x;
    xw[3] = 40.2F;
    auto qw = q;
    qw[3] = 41;
    ExpectRejected("a wrong code far from a boundary (40.2 -> 41)",
                   CheckFabricated(xw, qw, 1.0F, 168.0F, {3}, {41.0F}), true);
  }
  // Each code alone is a legitimate boundary rounding, but 10.5 + 1 ulp
  // rounding down needs a multiplier below 1 and 20.5 - 1 ulp rounding up
  // one above 1: no single reciprocal gives both.
  {
    auto xj = x;
    xj[5] = std::nextafter(10.5F, 11.0F);
    xj[6] = std::nextafter(20.5F, 20.0F);
    auto qj = q;
    qj[5] = 10;
    qj[6] = 21;
    ExpectRejected(
        "a jointly unrealizable block (10.5+ulp -> 10, "
        "20.5-ulp -> 21)",
        CheckFabricated(xj, qj, 1.0F, 158.0F, {5, 6}, {10.0F, 21.0F}), true);
  }
  // Genuine near-boundary case: x * 127 / amax = 10.5 + 9.5e-7, within the
  // Q8_1 code bound (7u * 10.5 = 4.4e-6) of the boundary, so an approximate
  // reciprocal may round it down. Accepted, and counted.
  {
    auto xn = x;
    xn[4] = std::nextafter(10.5F, 11.0F);
    for (const std::int8_t c : {10, 11}) {
      auto qn = q;
      qn[4] = c;
      const auto ok =
          CheckFabricated(xn, qn, 1.0F, 127.0F + c, {4, 0}, {float(c), 127.0F});
      Require(ok.quant.violations == 0 && ok.unexplained == 0 &&
                  ok.quant.boundary_codes == (c == 10 ? 1U : 0U),
              "checker rejected a genuine near-boundary code: " +
                  ok.quant.first + ok.first);
    }
    std::cout << "  accepts a genuine near-boundary code (10.5+ulp -> 10 or "
                 "11)\n";
  }
  // MMQ layouts: D4 stores the float scale (bound 6u), DS4 fp16 scale and
  // fp16 sum (bound 7u on the unrounded sum).
  {
    QuantStats st = ValidateQuantRows(
        Fabricate(QFN_MMQ_ACT_MMQ_D4, q, 1.0F + 0x1p-20F, 0.0F), x, 1);
    ExpectRejected("an MMQ D4 float scale 16u from amax/127", Rejection(st),
                   true);
    st = ValidateQuantRows(
        Fabricate(QFN_MMQ_ACT_MMQ_D4, q, 1.0F + 0x1p-23F, 0.0F), x, 1);
    Require(st.violations == 0,
            "checker rejected a D4 scale 2u off: " + st.first);
    st = ValidateQuantRows(Fabricate(QFN_MMQ_ACT_MMQ_DS4, q, 1.0F, 127.0625F),
                           x, 1);
    ExpectRejected("an MMQ DS4 block sum one fp16 step off (127.0625)",
                   Rejection(st), true);
    st = ValidateQuantRows(Fabricate(QFN_MMQ_ACT_MMQ_DS4, q, 1.0F, 127.0F), x,
                           1);
    Require(st.violations == 0,
            "checker rejected a correct DS4 block: " + st.first);
  }
}

// ---- Scratch allocation failure ------------------------------------------

std::set<void*> g_live;       // allocations made through the mock
std::size_t g_bad_frees = 0;  // frees of pointers not live (double frees)
std::size_t g_alloc_calls = 0;
bool g_fail_alloc = false;

hipError_t MockAlloc(void** ptr, std::size_t bytes) {
  ++g_alloc_calls;
  if (g_fail_alloc) {
    *ptr = nullptr;
    return hipErrorOutOfMemory;
  }
  const hipError_t e = hipMalloc(ptr, bytes);
  if (e == hipSuccess) {
    g_live.insert(*ptr);
  }
  return e;
}

hipError_t MockFree(void* ptr) {
  if (g_live.erase(ptr) == 0) {
    ++g_bad_frees;
    return hipErrorInvalidValue;
  }
  return hipFree(ptr);
}

/// Scratch growth with a failing allocator: the call fails cleanly, every
/// buffer and capacity is left as it was (a later call that fits reuses
/// them, bit-identically), nothing is freed twice, and the destructor frees
/// exactly what was allocated.
void TestReserveFailure() {
  for (const bool moe : {false, true}) {
    TinyLayer layer(GgmlType::kQ6_K, moe, 77);
    std::string error;
    auto device = gpu::LayerFfn::Upload(layer.config(), layer.layer(), &error);
    Require(device.has_value(), "upload: " + error);
    const std::size_t H = layer.config().hidden_size;
    const auto x1 = RandomRows(1, H, layer.rng());
    const auto x37 = RandomRows(37, H, layer.rng());
    float* dx1 = ToDevice(x1);
    float* dx37 = ToDevice(x37);
    float* dout = nullptr;
    CheckHip(hipMalloc(&dout, x37.size() * sizeof(float)), "alloc");
    g_live.clear();
    g_bad_frees = 0;
    g_fail_alloc = false;
    {
      GpuContext ctx;
      ctx.ops().SetAllocatorForTesting(MockAlloc, MockFree);
      const auto run = [&](const float* dx, std::uint32_t tokens,
                           gpu::Path path) {
        error.clear();
        const bool ok = moe ? ctx.ops().Moe(device->routed(), dx, dout,
                                            static_cast<std::uint32_t>(H),
                                            tokens, path, &error)
                            : ctx.ops().Dense(device->dense(), dx, dout, tokens,
                                              path, &error);
        CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
        return ok;
      };
      Require(run(dx1, 1, gpu::Path::kVector), "first call failed: " + error);
      const auto first = ToHost(dout, H);
      const std::size_t held = ctx.ops().scratch_bytes();
      const std::size_t live = g_live.size();
      const std::int32_t* ids = ctx.ops().ids();
      g_fail_alloc = true;
      Require(!run(dx37, 37, gpu::Path::kBatched),
              "growth succeeded with a failing allocator");
      Require(error.find("hipMalloc scratch") != std::string::npos,
              "unexpected failure message: " + error);
      Require(ctx.ops().scratch_bytes() == held && g_live.size() == live &&
                  ctx.ops().ids() == ids && g_bad_frees == 0,
              "failed growth changed the scratch");
      const std::size_t calls = g_alloc_calls;
      Require(run(dx1, 1, gpu::Path::kVector),
              "call after a failed growth failed: " + error);
      Require(g_alloc_calls == calls, "a fitting call allocated");
      Require(std::memcmp(ToHost(dout, H).data(), first.data(),
                          H * sizeof(float)) == 0,
              "result after a failed growth differs");
      g_fail_alloc = false;
      Require(run(dx37, 37, gpu::Path::kBatched),
              "growth after recovery failed: " + error);
    }
    Require(g_live.empty() && g_bad_frees == 0,
            "destructor leaked or double-freed scratch");
    (void)hipFree(dx1);
    (void)hipFree(dx37);
    (void)hipFree(dout);
    std::cout << "  " << (moe ? "moe" : "dense")
              << ": failed scratch growth leaves buffers and capacity intact, "
                 "no double free\n";
  }
}

using Cases = std::vector<std::pair<std::uint32_t, gpu::Path>>;
const Cases kCases = {{1, gpu::Path::kVector},  {1, gpu::Path::kBatched},
                      {1, gpu::Path::kAuto},    {5, gpu::Path::kVector},
                      {5, gpu::Path::kBatched}, {8, gpu::Path::kVector},
                      {32, gpu::Path::kVector}, {37, gpu::Path::kBatched},
                      {37, gpu::Path::kAuto},   {129, gpu::Path::kBatched}};

void TestRoutedTypes() {
  for (const GgmlType type :
       {GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kIQ4_XS}) {
    TinyLayer layer(type, true, 100 + static_cast<std::uint64_t>(type));
    std::string error;
    auto device = gpu::LayerFfn::Upload(layer.config(), layer.layer(), &error);
    Require(device.has_value(), "upload: " + error);
    GpuContext ctx;
    for (const auto& [tokens, path] : kCases) {
      const auto x =
          RandomRows(tokens, layer.config().hidden_size, layer.rng());
      Record(std::string(gufo::core::ToString(type)),
             RunOperatorCase(CaseLabel(std::string(gufo::core::ToString(type)),
                                       tokens, path),
                             layer.config(), layer.layer(), *device, x, tokens,
                             path, ctx, {}));
    }
  }
}

void TestDenseLayerZero() {
  TinyLayer layer(GgmlType::kQ8_0, false, 7);
  std::string error;
  auto device = gpu::LayerFfn::Upload(layer.config(), layer.layer(), &error);
  Require(device.has_value(), "upload: " + error);
  GpuContext ctx;
  for (const auto& [tokens, path] : kCases) {
    const auto x = RandomRows(tokens, layer.config().hidden_size, layer.rng());
    Record("Q8_0 dense",
           RunOperatorCase(CaseLabel("dense", tokens, path), layer.config(),
                           layer.layer(), *device, x, tokens, path, ctx, {}));
  }
}

/// Router and selection edge cases, on every routed type and both paths.
void TestRouterEdges() {
  for (const GgmlType type :
       {GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kIQ4_XS}) {
    const std::string tname(gufo::core::ToString(type));
    for (const gpu::Path path : {gpu::Path::kVector, gpu::Path::kBatched}) {
      const std::uint32_t tokens = 7;
      // 1. Extreme-negative logits: every logit is about -20, the ten
      //    selected sigmoids sum to about 2e-8 (< 2^-14), so the clamp
      //    decides the weights.
      {
        TinyLayer layer(type, true, 300 + static_cast<std::uint64_t>(type));
        const std::size_t H = layer.config().hidden_size;
        const std::size_t E = layer.config().num_experts;
        std::normal_distribution<float> noise(0.0F, 0.02F);
        std::vector<float> router(E * H);
        for (float& v : router) {
          v = -(20.0F / (3.0F * static_cast<float>(H))) + noise(layer.rng());
        }
        SetF32Values(&layer.router(), router);
        layer.BindMoe();
        auto x = RandomRows(tokens, H, layer.rng());
        for (float& v : x) {
          v += 3.0F;
        }
        std::string error;
        auto device =
            gpu::LayerFfn::Upload(layer.config(), layer.layer(), &error);
        Require(device.has_value(), "upload: " + error);
        GpuContext ctx;
        CaseOptions opt;
        opt.expect_clamp = true;
        opt.min_margin =
            0.0;  // selection runs on bias alone; sums are bitwise equal
        Record(tname, RunOperatorCase(
                          CaseLabel(tname + " extreme-negative", tokens, path),
                          layer.config(), layer.layer(), *device, x, tokens,
                          path, ctx, opt));
      }
      // 2. Exact ties: experts 10..17 have identical router rows and equal
      //    bias, and rank right after the eight experts 0..7 (large bias), so
      //    the last two picks are decided by the tie rule (lower id first).
      {
        TinyLayer layer(type, true, 400 + static_cast<std::uint64_t>(type));
        const std::size_t H = layer.config().hidden_size;
        auto router = F32Values(layer.router());
        auto bias = F32Values(layer.bias());
        for (std::size_t e = 11; e < 18; ++e) {
          std::copy_n(router.begin() + 10 * H, H, router.begin() + e * H);
        }
        for (std::size_t e = 0; e < bias.size(); ++e) {
          bias[e] = e < 8 ? 1.5F : (e >= 10 && e < 18 ? 0.5F : -1.0F);
        }
        SetF32Values(&layer.router(), router);
        SetF32Values(&layer.bias(), bias);
        layer.BindMoe();
        const auto x = RandomRows(tokens, H, layer.rng());
        std::string error;
        auto device =
            gpu::LayerFfn::Upload(layer.config(), layer.layer(), &error);
        Require(device.has_value(), "upload: " + error);
        GpuContext ctx;
        CaseOptions opt;
        opt.expect_tie = true;
        Taps taps;
        Record(tname, RunOperatorCase(CaseLabel(tname + " ties", tokens, path),
                                      layer.config(), layer.layer(), *device, x,
                                      tokens, path, ctx, opt, &taps));
        const auto& ids = taps.I("topk");
        for (std::size_t t = 0; t < tokens; ++t) {
          std::vector<std::int32_t> picked(ids.begin() + t * 10,
                                           ids.begin() + (t + 1) * 10);
          std::sort(picked.begin(), picked.end());
          const std::vector<std::int32_t> want = {0, 1, 2, 3,  4,
                                                  5, 6, 7, 10, 11};
          Require(picked == want, "tie rule: lower expert ids must win");
        }
      }
      // 3. Selection-only bias: the biased top-10 differs from the unbiased
      //    one, the GPU selects by s + bias, and the weights come from the
      //    unbiased s of the selected experts (checked here from the GPU's own
      //    logits in double precision, independent of the oracle).
      {
        TinyLayer layer(type, true, 500 + static_cast<std::uint64_t>(type));
        const std::size_t H = layer.config().hidden_size;
        const std::size_t E = layer.config().num_experts;
        const std::size_t U = layer.config().num_experts_used;
        std::normal_distribution<float> big(0.0F, 0.6F);
        auto bias = F32Values(layer.bias());
        for (float& b : bias) {
          b = big(layer.rng());
        }
        SetF32Values(&layer.bias(), bias);
        layer.BindMoe();
        const auto x = RandomRows(tokens, H, layer.rng());
        std::string error;
        auto device =
            gpu::LayerFfn::Upload(layer.config(), layer.layer(), &error);
        Require(device.has_value(), "upload: " + error);
        GpuContext ctx;
        Taps taps;
        Record(tname, RunOperatorCase(
                          CaseLabel(tname + " bias-selection", tokens, path),
                          layer.config(), layer.layer(), *device, x, tokens,
                          path, ctx, {}, &taps));
        const auto& logits = taps.F("router_logits");
        const auto& ids = taps.I("topk");
        const auto& weights = taps.F("weights");
        bool differs = false;
        for (std::size_t t = 0; t < tokens; ++t) {
          std::vector<double> s(E);
          for (std::size_t e = 0; e < E; ++e) {
            s[e] =
                1.0 / (1.0 + std::exp(-static_cast<double>(logits[t * E + e])));
          }
          const auto top = [&](const auto& score) {
            std::vector<std::size_t> order(E);
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::partial_sort(order.begin(), order.begin() + U, order.end(),
                              [&](std::size_t a, std::size_t b) {
                                return score(a) > score(b);
                              });
            order.resize(U);
            std::sort(order.begin(), order.end());
            return order;
          };
          const auto pb = top([&](std::size_t e) { return s[e] + bias[e]; });
          const auto pp = top([&](std::size_t e) { return s[e]; });
          differs = differs || pb != pp;
          double sum = 0;
          for (std::size_t j = 0; j < U; ++j) {
            sum += s[static_cast<std::size_t>(ids[t * U + j])];
          }
          for (std::size_t j = 0; j < U; ++j) {
            const std::size_t e = static_cast<std::size_t>(ids[t * U + j]);
            Require(std::find(pb.begin(), pb.end(), e) != pb.end(),
                    "selected expert is not in the biased top-k");
            const double want = 2.5 * s[e] / std::max(sum, 6.103515625e-5);
            Require(std::fabs(weights[t * U + j] - want) <= 5e-6 * want,
                    "weights must use the unbiased sigmoid");
          }
        }
        Require(differs, "fixture: bias did not change the selection");
      }
    }
  }
}

/// Two streams run different layers and row counts concurrently with no
/// synchronization between calls; each result must equal the same call made
/// alone on one stream, bit for bit (per-stream scratch isolation).
void TestConcurrentStreams() {
  struct Job {
    std::unique_ptr<TinyLayer> layer;
    std::optional<gpu::LayerFfn> device;
    std::vector<float> x;
    std::uint32_t tokens;
    gpu::Path path;
    float* dx{nullptr};
    float* dout{nullptr};
    std::vector<float> alone;
  };
  std::vector<Job> jobs(2);
  const GgmlType types[2] = {GgmlType::kQ6_K, GgmlType::kIQ4_XS};
  const std::uint32_t tokens[2] = {37, 5};
  const gpu::Path paths[2] = {gpu::Path::kBatched, gpu::Path::kVector};
  GpuContext a;
  GpuContext b;
  GpuContext* ctx[2] = {&a, &b};
  for (int i = 0; i < 2; ++i) {
    Job& j = jobs[i];
    j.layer = std::make_unique<TinyLayer>(types[i], true, 900 + i);
    std::string error;
    j.device =
        gpu::LayerFfn::Upload(j.layer->config(), j.layer->layer(), &error);
    Require(j.device.has_value(), "upload: " + error);
    j.tokens = tokens[i];
    j.path = paths[i];
    j.x = RandomRows(j.tokens, j.layer->config().hidden_size, j.layer->rng());
    j.dx = ToDevice(j.x);
    CheckHip(hipMalloc(&j.dout, j.x.size() * sizeof(float)), "alloc");
  }
  const auto run = [&](int i, int stream) {
    Job& j = jobs[i];
    std::string error;
    Require(ctx[stream]->ops().Moe(j.device->routed(), j.dx, j.dout,
                                   j.layer->config().hidden_size, j.tokens,
                                   j.path, &error),
            "operator: " + error);
  };
  for (int i = 0; i < 2; ++i) {  // alone, on stream 0
    run(i, 0);
    CheckHip(hipStreamSynchronize(ctx[0]->stream()), "sync");
    jobs[i].alone = ToHost(jobs[i].dout, jobs[i].x.size());
    CheckHip(hipMemset(jobs[i].dout, 0xFF, jobs[i].x.size() * sizeof(float)),
             "clear");
  }
  for (int round = 0; round < 6; ++round) {  // concurrently, no syncs
    run(0, 0);
    run(1, 1);
    run(0, 0);
    run(1, 1);
  }
  for (auto& c : ctx) {
    CheckHip(hipStreamSynchronize(c->stream()), "sync");
  }
  for (int i = 0; i < 2; ++i) {
    const auto got = ToHost(jobs[i].dout, jobs[i].x.size());
    Require(std::memcmp(got.data(), jobs[i].alone.data(),
                        got.size() * sizeof(float)) == 0,
            "concurrent stream result differs from the solo result");
    (void)hipFree(jobs[i].dx);
    (void)hipFree(jobs[i].dout);
  }
  std::cout << "  concurrent streams: bit-identical to solo runs\n";
}

}  // namespace

int main() {
  try {
    std::cout << "stage checker on fabricated captures\n";
    TestCheckerRejects();
    std::cout << "scratch allocation failure\n";
    TestReserveFailure();
    std::cout << "routed experts, random tiny layers\n";
    TestRoutedTypes();
    std::cout << "dense layer 0\n";
    TestDenseLayerZero();
    std::cout << "router edge cases\n";
    TestRouterEdges();
    std::cout << "streams\n";
    TestConcurrentStreams();
    std::cout << "SUMMARY " << g_totals.cases << " operator cases passed; "
              << g_totals.blocks << " captured activation blocks validated, "
              << g_totals.boundary_codes
              << " codes legitimately not the nearest (within the derived "
                 "reciprocal error of a .5 boundary)\n";
    for (const auto& [type, worst] : g_totals.worst_e2e) {
      std::cout << "  " << type << ": worst end-to-end " << worst
                << " of rms (abs " << g_totals.worst_abs[type] << ")\n";
      for (const auto& [name, st] : g_totals.stages[type]) {
        std::cout << "    " << name << ": worst " << st.over_rms << " of rms, "
                  << st.over_bound << " of bound; " << st.boundary_codes
                  << " boundary codes in " << st.blocks
                  << " blocks, scale deviation " << st.max_scale_rel << "\n";
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
