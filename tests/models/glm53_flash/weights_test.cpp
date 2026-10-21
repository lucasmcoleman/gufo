// V1 loader check for GLM-5.3-Flash: the real Maya-S artifact must parse, bind
// every tensor with the expected shape, and describe the expected layer plan.
// Reads headers only; nothing is dequantized or uploaded. Set
// GLM53_FLASH_GGUF to the first shard; without it the test reports a skip
// (which is not a pass for model qualification).

#include "src/models/glm53_flash/weights.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace glm = gufo::models::glm53_flash;

namespace {

constexpr int kSkip = 77;  // CTest SKIP_RETURN_CODE convention.

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::size_t CountBound(const glm::LayerWeights& l) {
  std::size_t n = 0;
  auto add = [&](const glm::TensorRef& t) { n += t.empty() ? 0 : 1; };
  for (const auto* m : {&l.hc_attn, &l.hc_ffn}) {
    add(m->fn);
    add(m->base);
    add(m->scale);
  }
  for (const auto* t : {&l.attn_norm,       &l.ffn_norm,    &l.kda_q,
                        &l.kda_k,           &l.kda_v,       &l.kda_conv_q,
                        &l.kda_conv_k,      &l.kda_conv_v,  &l.kda_f_a,
                        &l.kda_f_b,         &l.kda_dt_bias, &l.kda_a,
                        &l.kda_beta,        &l.kda_g_a,     &l.kda_g_b,
                        &l.kda_norm,        &l.attn_out,    &l.mla_q_a,
                        &l.mla_q_a_norm,    &l.mla_q_b,     &l.mla_kv_a,
                        &l.mla_kv_a_norm,   &l.mla_k_b,     &l.mla_v_b,
                        &l.idx_q_b,         &l.idx_k,       &l.idx_k_norm,
                        &l.idx_k_norm_bias, &l.idx_proj,    &l.idx_gate,
                        &l.idx_ape,         &l.ffn_gate,    &l.ffn_up,
                        &l.ffn_down,        &l.router,      &l.router_bias,
                        &l.ffn_gate_exps,   &l.ffn_up_exps, &l.ffn_down_exps,
                        &l.shexp_gate,      &l.shexp_up,    &l.shexp_down,
                        &l.nextn_eh_proj,   &l.nextn_enorm, &l.nextn_hnorm,
                        &l.nextn_head_norm}) {
    add(*t);
  }
  return n;
}

}  // namespace

int main() {
  const char* path = std::getenv("GLM53_FLASH_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GLM53_FLASH_GGUF not set\n";
    return kSkip;
  }
  try {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, "open failed: " + error);

    auto weights =
        glm::ModelWeights::Bind(*reader, /*bind_nextn=*/true, &error);
    Require(weights.has_value(), "bind failed: " + error);
    const auto& c = weights->config;

    Require(c.num_layers == 45 && c.num_layers_all == 46, "layer count");
    Require(c.hidden_size == 4096 && c.vocab_size == 154880, "embedding");
    Require(c.CountLayers(glm::Config::Mixer::kKda) == 34, "34 KDA layers");
    Require(c.CountLayers(glm::Config::Mixer::kMla) == 11, "11 MLA layers");
    for (std::uint32_t i = 0; i < c.num_layers; ++i) {
      Require(c.IsMlaLayer(i) == ((i % 4) == 3), "MLA every 4th layer");
    }
    Require(c.IsMlaLayer(45), "NextN block is MLA");
    Require(c.num_experts == 288 && c.num_experts_used == 8, "experts");
    Require(c.expert_weights_norm && c.expert_weights_scale == 2.5F, "router");
    Require(c.indexer_kpool == 4 && c.IndexerPoolBudget() == 512, "indexer");
    Require(c.leading_dense_layers == 3, "dense layers");

    std::size_t bound = 3;  // token_embd, output, output_norm
    std::map<std::string, std::size_t> down_types;
    for (const auto& l : weights->layers) {
      bound += CountBound(l);
      if (!l.ffn_down_exps.empty())
        ++down_types[std::string(gufo::core::ToString(l.ffn_down_exps.type))];
    }
    Require(bound == reader->GetTensorCount(),
            "bound " + std::to_string(bound) + " of " +
                std::to_string(reader->GetTensorCount()) + " tensors");

    std::cout << "bound " << bound << " tensors;";
    for (const auto& [type, n] : down_types)
      std::cout << " down " << type << "=" << n;
    std::cout << "\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
