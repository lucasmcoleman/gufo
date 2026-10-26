// GPU FFN operator checks against the phase 2 CPU oracle, on random tiny
// layers: no model file, nothing skipped. ffn_real_test.cpp repeats the checks
// on real GGUF slices.
//
// Every case runs the operator tapped, compares each stage with the oracle's
// own building blocks (router logits, top-k ids, weights, gate/up/down per
// expert, shared expert, combine) in the activation mode of the path taken,
// compares the final output with ref::FfnForward, then reruns untapped (two
// queued calls, no synchronization in between) and requires a bit-identical
// result. Tolerances:
//
//  * Stage checks (5e-5 of the oracle's rms): the integer dot products are
//    exact and both sides apply the same Q8 rounding, so only float summation
//    order differs: about 1e-6 observed, up to 2e-5 on real Q5_K weights,
//    whose tiles add d*sc*dot and an opposing min term in float32 and so
//    amplify the rounding by their cancellation. The oracle models each path's
//    activation rounding exactly: MMVQ Q8_1 scales are fp16, MMQ's D4 layout
//    (Q8_0, Q6_K, IQ4_XS) keeps a float scale, and MMQ Q4_K/Q5_K tiles also
//    round d*sc, dmin*m and the block sum to fp16 (ref::MatMulRows).
//  * A stage row may differ more only when an input element lies within 4e-5
//    (kTieMargin, in Q8 codes) of a Q8 rounding boundary: the GPU's
//    quantization uses a reciprocal that is not correctly rounded, so it can
//    round that element the other way (a one-LSB change of one activation).
//    CheckProjection accepts such a row only if re-rounding some of those
//    near-tie elements reproduces the GPU row to the same 5e-5. The other
//    source is a stored fp16 scale (or Q5_K block sum) within an ulp of a
//    rounding boundary: the GPU forms scales with approximate reciprocals, so
//    it can round to the neighbouring fp16 value (relative change 2^-11 of one
//    block's contribution). A row passes that way only if moving exactly one
//    such scale by one fp16 step reproduces the GPU row to 5e-5. Anything else
//    is unexplained and fails.
//  * End to end (5e-3 of rms): inputs are moved off Q8 rounding boundaries
//    (AvoidTies), but the oracle's intermediate activations still differ from
//    the GPU's by float rounding, so a near-tie element of an intermediate row
//    can round differently; one flip moves one expert output by about one Q8
//    step times a weight. Observed worst: 2.8e-3 (real layer 0, 37 rows),
//    about 1e-3 on the tiny layers, against Q8 noise of 3e-2 to 1e-1 (printed
//    as "Q8 noise"); a wiring or weighting defect is far larger. The stage
//    checks are the exactness claim.
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>

#include "tests/models/laguna/ffn_gpu.hpp"

using namespace gufo::testing::laguna;

namespace {

struct Totals {
  std::map<std::string, double> worst_e2e;  // per quant type, / rms
  std::map<std::string, double> worst_abs;
  std::map<std::string, double> worst_stage;
  std::size_t cases{0};
  std::size_t flip_rows{0};
  std::size_t scale_rows{0};
};
Totals g_totals;

void Record(const std::string& type, const CaseReport& r) {
  g_totals.worst_e2e[type] = std::max(g_totals.worst_e2e[type], r.e2e_over_rms);
  g_totals.worst_abs[type] = std::max(g_totals.worst_abs[type], r.e2e_abs);
  g_totals.worst_stage[type] =
      std::max(g_totals.worst_stage[type], r.stage_over_rms);
  g_totals.flip_rows += r.flip_rows;
  g_totals.scale_rows += r.scale_rows;
  ++g_totals.cases;
  PrintReport(r);
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
    std::cout << "routed experts, random tiny layers\n";
    TestRoutedTypes();
    std::cout << "dense layer 0\n";
    TestDenseLayerZero();
    std::cout << "router edge cases\n";
    TestRouterEdges();
    std::cout << "streams\n";
    TestConcurrentStreams();
    std::cout << "SUMMARY " << g_totals.cases << " operator cases passed, "
              << g_totals.flip_rows
              << " stage rows explained as Q8 tie flips and "
              << g_totals.scale_rows << " as one fp16 scale boundary\n";
    for (const auto& [type, worst] : g_totals.worst_e2e) {
      std::cout << "  " << type << ": worst end-to-end " << worst
                << " of rms (abs " << g_totals.worst_abs[type]
                << "), worst unexplained stage " << g_totals.worst_stage[type]
                << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
