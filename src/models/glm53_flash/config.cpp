#include "src/models/glm53_flash/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::glm53_flash {
namespace {

constexpr std::string_view kArchitecture = "glm5-next";
/// llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID.
constexpr std::uint32_t kGatingSigmoid = 2;

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return fallback;
  }

  float F32(std::string_view key, float fallback, bool required = true) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value.has_value()) {
      if (required || gguf.FindMetadata(key) != nullptr)
        Fail("missing or invalid GGUF float " + std::string(key));
      return fallback;
    }
    if (!std::isfinite(*value))
      Fail("non-finite GGUF float " + std::string(key));
    return *value;
  }

  bool Bool(std::string_view key, bool fallback) {
    const auto value = gguf.GetMetadataBool(key);
    if (!value.has_value() && gguf.FindMetadata(key) != nullptr)
      Fail("invalid GGUF bool " + std::string(key));
    return value.value_or(fallback);
  }

  /// Integer arrays of any width; GGUF writers pick the narrowest type.
  std::vector<std::uint64_t> U64Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }

  std::vector<float> F32Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<float> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    const auto* d = std::get_if<std::vector<double>>(&meta->value);
    if (d == nullptr) {
      Fail("GGUF array must contain floats: " + std::string(key));
      return out;
    }
    for (double v : *d) {
      if (!std::isfinite(v) || v < 0.0) {
        Fail("invalid GGUF float array value " + std::string(key));
        return {};
      }
      out.push_back(static_cast<float>(v));
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not glm5-next");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "glm5-next.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
  };

  c.num_layers_all = r.U32(key("block_count"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"), false, 0);
  if (c.num_layers_all == 0 || c.nextn_layers >= c.num_layers_all) {
    r.Fail("nextn_predict_layers must be below block_count");
    return std::nullopt;
  }
  c.num_layers = c.num_layers_all - c.nextn_layers;
  c.hidden_size = r.U32(key("embedding_length"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-5F);
  c.norm_eps = r.F32(key("attention.layer_norm_epsilon"), 1e-6F, false);

  c.num_heads = r.U32(key("attention.head_count"));
  {
    const auto kv = r.U64Array(key("attention.head_count_kv"));
    if (kv.size() != c.num_layers_all) {
      r.Fail("attention.head_count_kv must describe every layer");
    } else {
      c.mixers.reserve(kv.size());
      for (auto v : kv) {
        if (v > 1) {
          r.Fail("head_count_kv must be 0 (KDA) or 1 (MLA) per layer");
          break;
        }
        c.mixers.push_back(v == 0 ? Mixer::kKda : Mixer::kMla);
      }
    }
  }

  c.hc_count = r.U32(key("hyper_connection.count"));
  c.hc_sinkhorn_iters = r.U32(key("hyper_connection.sinkhorn_iterations"));
  c.hc_eps = r.F32(key("hyper_connection.epsilon"), 1e-6F);

  c.kda_head_dim = r.U32(key("kda.head_dim"));
  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.kda_gate_lower_bound = r.F32(key("kda.gate_lower_bound"), -5.0F);

  c.q_lora_rank = r.U32(key("attention.q_lora_rank"));
  c.kv_lora_rank = r.U32(key("attention.kv_lora_rank"));
  c.mla_head_dim = r.U32(key("attention.key_length_mla"));
  c.mla_value_dim = r.U32(key("attention.value_length_mla"));
  // Absorbed MLA: the cached K and V are both the kv_lora_rank latent.
  if (r.U32(key("attention.key_length")) != c.kv_lora_rank ||
      r.U32(key("attention.value_length")) != c.kv_lora_rank)
    r.Fail("absorbed MLA requires key/value_length == kv_lora_rank");
  if (r.U32(key("rope.dimension_count"), false, 0) != 0)
    r.Fail("glm5-next MLA with RoPE is not supported (expected NoPE)");

  c.indexer_heads = r.U32(key("attention.indexer.head_count"));
  c.indexer_head_dim = r.U32(key("attention.indexer.key_length"));
  c.indexer_top_k = r.U32(key("attention.indexer.top_k"));
  c.indexer_kpool = r.U32(key("attention.indexer.kpool"));
  // Variants this runtime does not implement. Absent keys take llama.cpp's
  // defaults (tail selected, a full indexer on every layer), which is what
  // the Maya-S artifact relies on; an explicit other value is rejected
  // instead of silently computed as the default.
  if (gguf.FindMetadata(key("attention.indexer.kpool_select_tail")) !=
          nullptr &&
      !r.Bool(key("attention.indexer.kpool_select_tail"), true))
    r.Fail("indexer kpool_select_tail=false is not supported");
  if (const auto* types = gguf.FindMetadata(key("attention.indexer.types"))) {
    bool all_full = false;
    if (const auto* u = std::get_if<std::uint64_t>(&types->value)) {
      all_full = *u == 1;
    } else if (const auto* b = std::get_if<bool>(&types->value)) {
      all_full = *b;
    } else if (const auto* a =
                   std::get_if<std::vector<std::uint64_t>>(&types->value)) {
      all_full = !a->empty();
      for (auto v : *a)
        all_full = all_full && v == 1;
    } else if (const auto* s = std::get_if<std::int64_t>(&types->value)) {
      all_full = *s == 1;
    } else if (const auto* sa =
                   std::get_if<std::vector<std::int64_t>>(&types->value)) {
      all_full = !sa->empty();
      for (auto v : *sa)
        all_full = all_full && v == 1;
    }
    if (!all_full)
      r.Fail("only a full (non-shared) indexer on every layer is supported");
  }

  c.leading_dense_layers = r.U32(key("leading_dense_block_count"));
  c.dense_ff = r.U32(key("feed_forward_length"));
  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_expert_ff = r.U32(key("expert_shared_feed_forward_length"));
  c.num_shared_experts = r.U32(key("expert_shared_count"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"), 1.0F);
  c.expert_weights_norm = r.Bool(key("expert_weights_norm"), false);
  if (r.U32(key("expert_gating_func")) != kGatingSigmoid)
    r.Fail("only the sigmoid expert gating function is supported");
  if (r.U32(key("expert_group_count"), false, 1) != 1 ||
      r.U32(key("expert_group_used_count"), false, 1) != 1)
    r.Fail("grouped expert routing is not supported");
  c.swiglu_clamp_exp = r.F32Array(key("swiglu_clamp_exp"));
  c.swiglu_clamp_shexp = r.F32Array(key("swiglu_clamp_shexp"));
  if (c.swiglu_clamp_exp.size() != c.num_layers_all ||
      c.swiglu_clamp_shexp.size() != c.num_layers_all)
    r.Fail("swiglu clamp arrays must describe every layer");

  if (!r.ok) {
    return std::nullopt;
  }

  // Shapes this runtime implements. They are checks, not assumptions: a file
  // that differs is rejected instead of computed wrong.
  if (c.hidden_size == 0 || c.vocab_size == 0 || c.num_heads == 0)
    r.Fail("embedding, vocabulary and head counts must be non-zero");
  if (!(c.rms_eps > 0.0F) || !(c.norm_eps > 0.0F) || !(c.hc_eps > 0.0F))
    r.Fail("normalization epsilons must be positive");
  if (c.hc_count != 4)
    r.Fail("hyper-connections are implemented for exactly 4 streams");
  if (c.hc_sinkhorn_iters == 0)
    r.Fail("hyper_connection.sinkhorn_iterations must be positive");
  if (c.kda_head_dim != 128)
    r.Fail("KDA is implemented for a head dimension of 128");
  if (c.ssm_conv_kernel != 4)
    r.Fail("KDA short convolution is implemented for kernel 4");
  if (!(c.kda_gate_lower_bound < 0.0F))
    r.Fail("kda.gate_lower_bound must be negative");
  if (c.indexer_kpool == 0 || c.indexer_top_k == 0 ||
      (c.indexer_top_k % c.indexer_kpool) != 0)
    r.Fail("indexer top_k must be a positive multiple of kpool");
  if (c.num_shared_experts != 1)
    r.Fail("exactly one shared expert is supported");
  if (c.num_experts_used == 0 || c.num_experts_used > c.num_experts)
    r.Fail("expert_used_count must be within expert_count");
  if (c.leading_dense_layers >= c.num_layers)
    r.Fail("leading_dense_block_count must leave MoE layers");
  if (c.CountLayers(Mixer::kMla) == 0 || c.CountLayers(Mixer::kKda) == 0)
    r.Fail("glm5-next expects both KDA and MLA layers");
  for (std::uint32_t i = c.num_layers; i < c.num_layers_all; ++i) {
    if (c.mixers[i] != Mixer::kMla)
      r.Fail("NextN layers must be MLA layers");
  }

  if (!r.ok) {
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::glm53_flash
