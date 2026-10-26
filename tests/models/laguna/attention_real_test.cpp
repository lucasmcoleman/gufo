// GPU attention operator on real Laguna-S-2.1 layers, against the CPU oracle:
// layer 0 and layer 4 (full attention, 48 heads, GQA 6, rotary 64 with YaRN)
// and layer 1 (sliding window 512, 72 heads, GQA 9, rotary 128), all Q6_K
// projections with the F32 gate. Each layer's tensors (about 100 MB) are
// uploaded alone. Without the GGUF the test exits 77 and CTest reports a skip,
// which verifies nothing. The checks and tolerances are those of
// attention_ops_test.cpp, which also describes them.
#include <iostream>
#include <memory>

#include "tests/models/laguna/attention_gpu.hpp"

using namespace gufo::testing::laguna;

namespace {

std::vector<float> LoadVec(const ml::TensorRef& t) {
  std::vector<float> out(t.ElementCount());
  ref::DequantizeRow(t.type, t.data, out.data(), out.size());
  return out;
}

/// Inputs like the model's: RMS-normalized random rows times attn_norm.weight,
/// with quantizer ties moved off the boundary.
std::vector<float> ModelLikeRows(std::size_t tokens, const ml::TensorRef& norm,
                                 std::mt19937_64& rng) {
  const auto gain = LoadVec(norm);
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
  AvoidTies(&x, tokens, H);
  return x;
}

std::map<std::string, GroupWorst> g_groups;
std::size_t g_cases = 0;

void RunLayer(const ml::ModelWeights& weights, std::uint32_t il,
              std::uint32_t T,
              const std::vector<std::pair<std::string, Plan>>& plans,
              const std::vector<std::uint32_t>& boundary, std::uint64_t seed) {
  const ml::Config& config = weights.config;
  const ml::LayerWeights& lw = weights.layers.at(il);
  std::string error;
  auto device = gpu::LayerAttn::Upload(config, lw, il, &error);
  Require(device.has_value(),
          "upload layer " + std::to_string(il) + ": " + error);
  const std::string kind = config.is_full_attention(il) ? "full" : "swa";
  std::cout << "layer " << il << " (" << kind << ", " << config.n_head(il)
            << " heads, GQA " << config.gqa_group(il) << "), "
            << device->device_bytes() / (1024.0 * 1024.0) << " MiB on device\n";
  AttnContext ctx;
  std::mt19937_64 rng(seed);
  const auto n = ModelLikeRows(T, lw.attn_norm, rng);
  std::map<int, OracleOut> oracle;
  CheckOptions opt;
  opt.oracle_cache = &oracle;
  opt.boundary_rows = boundary;
  std::vector<float> first;
  for (const auto& [what, plan] : plans) {
    const std::string label = "L" + std::to_string(il) + " " + kind + " " +
                              what + " T=" + std::to_string(T);
    const GpuRun tapped =
        RunPlan(ctx, device->weights(), config, il, n, plan, true);
    const AttnReport rep =
        CheckRun(label, config, lw, il, n, plan, tapped, opt);
    PrintAttnReport(rep);
    g_groups["L" + std::to_string(il) + " " + kind].Add(rep);
    ++g_cases;
    for (int rerun = 0; rerun < 2; ++rerun) {
      const GpuRun again =
          RunPlan(ctx, device->weights(), config, il, n, plan, false);
      Require(BitIdentical(again.out, tapped.out),
              label + ": untapped rerun differs from the tapped run");
    }
    if (plan.paths[0] == gpu::Path::kBatched) {
      if (first.empty()) {
        first = tapped.out;
      } else {
        Require(BitIdentical(first, tapped.out),
                label + ": output depends on the chunk split");
      }
    }
  }
}

}  // namespace

int main() {
  try {
    auto reader = OpenModelOrSkip("laguna attention_real_test");
    std::string error;
    auto weights = ml::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), "weights bind: " + error);
    const ml::Config& c = weights->config;
    Require(c.is_full_attention(0) && c.is_full_attention(4) &&
                !c.is_full_attention(1) && c.n_head(0) == 48 &&
                c.n_head(1) == 72 && c.sliding_window == 512,
            "unexpected layer layout in the GGUF");
    Require(std::fabs(c.YarnMscale() - 1.3466F) < 1e-3F,
            "YaRN mscale of the GGUF is not about 1.3466");
    // Full layers: prefill in one or two chunks, and the vector path.
    for (const std::uint32_t il : {0U, 4U}) {
      RunLayer(*weights, il, 150,
               {{"one chunk", UniformPlan({150}, gpu::Path::kBatched)},
                {"two chunks", UniformPlan({75, 75}, gpu::Path::kBatched)},
                {"prefill + 6 decode rows",
                 UniformPlan(Cat({144}, Singles(6)), gpu::Path::kBatched)}},
               {}, 100 + il);
      RunLayer(*weights, il, 40,
               {{"vector chunks", UniformPlan({32, 8}, gpu::Path::kVector)},
                {"vector decode singles",
                 UniformPlan(Cat({32}, Singles(8)), gpu::Path::kVector)}},
               {}, 200 + il);
    }
    // Sliding layer: across the 512 window, and the boundary rows.
    RunLayer(*weights, 1, 520,
             {{"one chunk", UniformPlan({520}, gpu::Path::kBatched)},
              {"prefill 511 then decode rows 511..519",
               UniformPlan(Cat({511}, Singles(9)), gpu::Path::kBatched)},
              {"chunks 300 220", UniformPlan({300, 220}, gpu::Path::kBatched)},
              {"chunks 513 7", UniformPlan({513, 7}, gpu::Path::kBatched)}},
             {511, 512, 513}, 300);
    RunLayer(*weights, 1, 70,
             {{"vector chunks", UniformPlan({32, 32, 6}, gpu::Path::kVector)},
              {"vector decode singles",
               UniformPlan(Cat({32}, Singles(38)), gpu::Path::kVector)}},
             {}, 301);
    std::cout << "SUMMARY " << g_cases
              << " real-weight attention cases passed\n";
    for (const auto& [name, g] : g_groups) {
      PrintGroup(name, g);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
