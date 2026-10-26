#include "src/models/laguna/weights.hpp"

#include <string>
#include <utility>

#include "src/models/laguna/tensor_binder.hpp"

namespace gufo::models::laguna {
namespace {

using detail::kF32Only;
using detail::kMatrix;
using detail::TensorBinder;

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const detail::Format format = detail::FormatOf(type);
  return format.block == 0 ? 0 : (cols / format.block) * format.bytes;
}

namespace {

LayerWeights BindLayer(TensorBinder& b, const Config& c, std::uint32_t il) {
  LayerWeights l;
  l.full_attention = c.is_full_attention(il);
  l.moe = c.is_moe(il);
  const std::string p = "blk." + std::to_string(il) + ".";
  const std::uint64_t hidden = c.hidden_size;
  const std::uint64_t q_dim = c.q_dim(il);
  const std::uint64_t kv_dim = c.kv_dim();

  l.attn_norm = b.Get(p + "attn_norm.weight", hidden, 1, 1, kF32Only);
  l.attn_q = b.Get(p + "attn_q.weight", hidden, q_dim, 1, kMatrix);
  l.attn_k = b.Get(p + "attn_k.weight", hidden, kv_dim, 1, kMatrix);
  l.attn_v = b.Get(p + "attn_v.weight", hidden, kv_dim, 1, kMatrix);
  l.attn_output = b.Get(p + "attn_output.weight", q_dim, hidden, 1, kMatrix);
  l.attn_q_norm = b.Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, kF32Only);
  l.attn_k_norm = b.Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, kF32Only);
  // One scalar per head. llama.cpp also accepts a per-element gate
  // (heads * head_dim rows, Laguna-M.1); this runtime does not implement it.
  l.attn_gate =
      b.Get(p + "attn_gate.weight", hidden, c.n_head(il), 1, kF32Only);
  l.ffn_norm = b.Get(p + "ffn_norm.weight", hidden, 1, 1, kF32Only);

  if (l.moe) {
    l.ffn_gate_inp =
        b.Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1, kF32Only);
    l.ffn_exp_probs_b =
        b.Get(p + "exp_probs_b.bias", c.num_experts, 1, 1, kF32Only);
    l.ffn_gate_exps = b.Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                            c.num_experts, kMatrix);
    l.ffn_up_exps = b.Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                          c.num_experts, kMatrix);
    l.ffn_down_exps = b.Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                            c.num_experts, kMatrix);
    l.ffn_gate_shexp = b.Get(p + "ffn_gate_shexp.weight", hidden,
                             c.shared_expert_ff, 1, kMatrix);
    l.ffn_up_shexp = b.Get(p + "ffn_up_shexp.weight", hidden,
                           c.shared_expert_ff, 1, kMatrix);
    l.ffn_down_shexp = b.Get(p + "ffn_down_shexp.weight", c.shared_expert_ff,
                             hidden, 1, kMatrix);
  } else {
    l.ffn_gate = b.Get(p + "ffn_gate.weight", hidden, c.ffn_size, 1, kMatrix);
    l.ffn_up = b.Get(p + "ffn_up.weight", hidden, c.ffn_size, 1, kMatrix);
    l.ffn_down = b.Get(p + "ffn_down.weight", c.ffn_size, hidden, 1, kMatrix);
  }
  return l;
}

}  // namespace

std::size_t ModelWeights::ExpectedTensorCount(const Config& config) {
  const std::size_t dense = config.leading_dense_layers;
  const std::size_t moe = config.num_layers - dense;
  return 3 + 9 * static_cast<std::size_t>(config.num_layers) + 3 * dense +
         8 * moe;
}

std::optional<ModelWeights> ModelWeights::Bind(
    const Config& config, std::span<const core::GgufTensorInfo> tensors,
    std::span<const core::GgufMappedRegion> regions, std::string* error_msg) {
  TensorBinder b(tensors, regions, error_msg);
  ModelWeights w;
  w.config = config;
  const std::uint64_t hidden = config.hidden_size;
  w.token_embd =
      b.Get("token_embd.weight", hidden, config.vocab_size, 1, kMatrix);
  w.output_norm = b.Get("output_norm.weight", hidden, 1, 1, kF32Only);
  w.output = b.Get("output.weight", hidden, config.vocab_size, 1, kMatrix);
  w.layers.reserve(config.num_layers);
  for (std::uint32_t il = 0; il < config.num_layers && b.ok(); ++il) {
    w.layers.push_back(BindLayer(b, config, il));
  }
  b.RequireNoLeftovers();
  if (!b.ok()) {
    return std::nullopt;
  }
  w.tensor_count = b.claimed();
  w.total_bytes = b.claimed_bytes();
  if (w.tensor_count != ExpectedTensorCount(config)) {
    if (error_msg != nullptr) {
      *error_msg = "bound " + std::to_string(w.tensor_count) +
                   " tensors, the layout needs " +
                   std::to_string(ExpectedTensorCount(config));
    }
    return std::nullopt;
  }
  return w;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  const auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  return Bind(*config, reader.GetTensors(), reader.GetMappedRegions(),
              error_msg);
}

}  // namespace gufo::models::laguna
