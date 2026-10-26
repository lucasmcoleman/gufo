// GPU FFN operators on real Laguna-S-2.1 weights, against the CPU oracle: the
// dense layer 0 (Q8_0), and one MoE layer per routed expert type: layer 2
// (Q6_K), layer 6 (Q5_K) and layer 20 (IQ4_XS), all with the Q8_0 shared
// expert. Each layer's tensors (about 2 GiB for the routed ones) are uploaded
// alone and freed before the next. Without the GGUF the test exits 77 and
// CTest reports a skip, which verifies nothing. The checks and tolerances are
// those of ffn_ops_test.cpp (captured, validated activations per stage),
// which also describes them.
#include <iostream>
#include <memory>

#include "tests/models/laguna/ffn_gpu.hpp"

using namespace gufo::testing::laguna;

namespace {

std::vector<float> LoadNorm(const ml::TensorRef& t) {
  std::vector<float> out(t.ElementCount());
  ref::DequantizeRow(t.type, t.data, out.data(), out.size());
  return out;
}

/// Inputs like the model's: RMS-normalized random rows times ffn_norm.weight.
std::vector<float> ModelLikeRows(std::size_t tokens, const ml::TensorRef& norm,
                                 std::mt19937_64& rng) {
  const auto gain = LoadNorm(norm);
  const std::size_t H = gain.size();
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> x(tokens * H);
  for (std::size_t t = 0; t < tokens; ++t) {
    double sq = 0;
    for (std::size_t i = 0; i < H; ++i) {
      x[t * H + i] = dist(rng);
      sq += static_cast<double>(x[t * H + i]) * x[t * H + i];
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(sq / H) + 1e-6F);
    for (std::size_t i = 0; i < H; ++i) {
      x[t * H + i] *= scale * gain[i];
    }
  }
  return x;
}

struct Case {
  std::uint32_t tokens;
  gpu::Path path;
};

struct Worst {
  double e2e{0}, abs{0};
  std::size_t cases{0}, boundary{0};
  std::map<std::string, StageReport> stages;
};
std::map<std::string, Worst> g_worst;

void Run(const std::string& type, const ml::ModelWeights& weights,
         std::uint32_t layer_index, const std::vector<Case>& cases,
         std::uint64_t seed) {
  const ml::LayerWeights& lw = weights.layers.at(layer_index);
  std::string error;
  auto device = gpu::LayerFfn::Upload(weights.config, lw, &error);
  Require(device.has_value(),
          "upload layer " + std::to_string(layer_index) + ": " + error);
  std::cout << "layer " << layer_index << " (" << type << "), "
            << device->device_bytes() / (1024.0 * 1024.0) << " MiB on device\n";
  GpuContext ctx;
  std::mt19937_64 rng(seed);
  for (const auto& [tokens, path] : cases) {
    const auto x = ModelLikeRows(tokens, lw.ffn_norm, rng);
    CaseOptions opt;
    opt.f32_reference = tokens <= 5;  // the F32 oracle at 37 rows is slow
    const auto report = RunOperatorCase(
        CaseLabel("L" + std::to_string(layer_index) + " " + type, tokens, path),
        weights.config, lw, *device, x, tokens, path, ctx, opt);
    PrintReport(report);
    Worst& w = g_worst[type];
    w.e2e = std::max(w.e2e, report.e2e_over_rms);
    w.abs = std::max(w.abs, report.e2e_abs);
    for (const auto& [name, st] : report.stages) {
      StageReport& ws = w.stages[name];
      ws.over_rms = std::max(ws.over_rms, st.over_rms);
      ws.over_bound = std::max(ws.over_bound, st.over_bound);
      ws.boundary_codes += st.boundary_codes;
      ws.blocks += st.blocks;
    }
    w.boundary += report.boundary_codes;
    ++w.cases;
  }
}

/// Real experts with a synthetic router whose logits are all about -20, so the
/// ten selected sigmoids sum to far below 2^-14 and the clamp decides.
void RunExtremeNegative(const ml::ModelWeights& weights,
                        std::uint32_t layer_index, const std::string& type) {
  ml::LayerWeights lw = weights.layers.at(layer_index);
  const std::size_t H = weights.config.hidden_size;
  const std::size_t E = weights.config.num_experts;
  std::mt19937_64 rng(4242);
  std::normal_distribution<float> noise(0.0F, 0.02F);
  std::vector<float> router(E * H);
  for (float& v : router) {
    v = -(20.0F / (3.0F * static_cast<float>(H))) + noise(rng);
  }
  lw.ffn_gate_inp.data = router.data();
  std::string error;
  auto device = gpu::LayerFfn::Upload(weights.config, lw, &error);
  Require(device.has_value(), "upload: " + error);
  GpuContext ctx;
  for (const gpu::Path path : {gpu::Path::kVector, gpu::Path::kBatched}) {
    auto x = ModelLikeRows(5, lw.ffn_norm, rng);
    for (float& v : x) {
      v = std::fabs(v) + 3.0F;
    }
    CaseOptions opt;
    opt.expect_clamp = true;
    opt.min_margin =
        0.0;  // selection runs on bias alone; sums are bitwise equal
    opt.f32_reference = false;
    const auto report =
        RunOperatorCase(CaseLabel("L" + std::to_string(layer_index) + " " +
                                      type + " extreme-negative router",
                                  5, path),
                        weights.config, lw, *device, x, 5, path, ctx, opt);
    PrintReport(report);
    ++g_worst[type].cases;
  }
}

}  // namespace

int main() {
  try {
    auto reader = OpenModelOrSkip("laguna ffn_real_test");
    std::string error;
    auto weights = ml::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), "weights bind: " + error);
    const std::vector<Case> cases = {{1, gpu::Path::kVector},
                                     {1, gpu::Path::kBatched},
                                     {5, gpu::Path::kVector},
                                     {5, gpu::Path::kBatched},
                                     {37, gpu::Path::kBatched}};
    // Layer types by the header manifest: Q6_K L1-4/43-47, Q5_K L5-9/38-42,
    // IQ4_XS L10-37.
    Require(weights->layers[2].ffn_gate_exps.type == GgmlType::kQ6_K &&
                weights->layers[6].ffn_gate_exps.type == GgmlType::kQ5_K &&
                weights->layers[20].ffn_gate_exps.type == GgmlType::kIQ4_XS,
            "unexpected routed expert types in the GGUF");
    Run("Q8_0 dense", *weights, 0, cases, 1);
    Run("Q6_K", *weights, 2, cases, 2);
    Run("Q5_K", *weights, 6, cases, 3);
    Run("IQ4_XS", *weights, 20, cases, 4);
    RunExtremeNegative(*weights, 20, "IQ4_XS");
    std::cout << "SUMMARY real-weight operator cases passed\n";
    for (const auto& [type, w] : g_worst) {
      std::cout << "  " << type << ": " << w.cases
                << " cases, worst end-to-end " << w.e2e << " of rms (abs "
                << w.abs << "), boundary codes " << w.boundary << "\n";
      for (const auto& [name, st] : w.stages) {
        std::cout << "    " << name << ": worst " << st.over_rms << " of rms, "
                  << st.over_bound << " of bound; " << st.boundary_codes
                  << " boundary codes in " << st.blocks << " blocks\n";
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
