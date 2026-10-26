#include "src/models/laguna/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::laguna {
namespace {

constexpr std::string_view kArchitecture = "laguna";

/// Reads GGUF metadata and records the first failure. Required keys that are
/// absent or of the wrong type fail loudly; nothing falls back silently except
/// the keys documented as optional at their call site.
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

  static std::string Key(std::string_view suffix) {
    return "laguna." + std::string(suffix);
  }

  std::uint32_t U32(std::string_view suffix) {
    const auto key = Key(suffix);
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + key);
      return 0;
    }
    return *value;
  }

  std::uint32_t U32Or(std::string_view suffix, std::uint32_t fallback) {
    const auto key = Key(suffix);
    if (gguf.FindMetadata(key) == nullptr) {
      return fallback;
    }
    return U32(suffix);
  }

  float F32(std::string_view suffix) {
    const auto key = Key(suffix);
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value.has_value() || !std::isfinite(*value)) {
      Fail("missing or invalid GGUF float " + key);
      return 0.0F;
    }
    return *value;
  }

  float F32Or(std::string_view suffix, float fallback) {
    if (gguf.FindMetadata(Key(suffix)) == nullptr) {
      return fallback;
    }
    return F32(suffix);
  }

  bool Bool(std::string_view suffix) {
    const auto key = Key(suffix);
    const auto value = gguf.GetMetadataBool(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF bool " + key);
      return false;
    }
    return *value;
  }

  /// Per-layer integers. The GGUF stores an array when the value varies by
  /// layer and may store a scalar when it does not; both are accepted and the
  /// result always has exactly `layers` entries.
  std::vector<std::uint32_t> PerLayerU32(std::string_view suffix,
                                         std::uint32_t layers) {
    const auto key = Key(suffix);
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint32_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF value " + key);
      return out;
    }
    const auto push = [&](std::uint64_t v) {
      if (v == 0 || v > UINT32_MAX) {
        Fail("GGUF per-layer value out of range in " + key);
        return;
      }
      out.push_back(static_cast<std::uint32_t>(v));
    };
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (const auto v : *u) {
        push(v);
      }
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF per-layer value in " + key);
          return {};
        }
        push(static_cast<std::uint64_t>(v));
      }
    } else if (const auto scalar = gguf.GetMetadataUint32(key)) {
      if (*scalar == 0) {
        Fail("GGUF per-layer value out of range in " + key);
        return {};
      }
      return std::vector<std::uint32_t>(layers, *scalar);
    } else {
      Fail("GGUF value must be an integer or integer array: " + key);
      return {};
    }
    if (out.size() != layers) {
      Fail(key + " has " + std::to_string(out.size()) +
           " entries but block_count is " + std::to_string(layers));
      return {};
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not laguna");
    return std::nullopt;
  }

  Config c;
  c.num_layers = r.U32("block_count");
  c.hidden_size = r.U32("embedding_length");
  c.ffn_size = r.U32("feed_forward_length");
  c.vocab_size = r.U32("vocab_size");
  c.context_length = r.U32("context_length");
  c.rms_eps = r.F32("attention.layer_norm_rms_epsilon");

  if (!r.ok || c.num_layers == 0 || c.num_layers > 4096) {
    r.Fail("laguna.block_count is out of range");
    return std::nullopt;
  }
  c.heads_per_layer = r.PerLayerU32("attention.head_count", c.num_layers);
  c.num_kv_heads = r.U32("attention.head_count_kv");
  c.head_dim = r.U32("attention.key_length");
  if (r.U32("attention.value_length") != c.head_dim) {
    r.Fail("attention.value_length must equal attention.key_length");
  }
  c.sliding_window = r.U32("attention.sliding_window");

  c.rope_theta_full = r.F32("rope.freq_base");
  c.rope_theta_swa = r.F32("rope.freq_base_swa");
  c.rotary_dim_full = r.U32("rope.dimension_count");
  c.rotary_dim_swa = r.U32("rope.dimension_count_swa");
  c.rope_attn_factor = r.F32Or("rope.scaling.attn_factor", 1.0F);
  if (gguf.GetMetadataString(Reader::Key("rope.scaling.type")) != "yarn") {
    r.Fail("laguna.rope.scaling.type must be yarn");
  }
  c.yarn.factor = r.F32("rope.scaling.factor");
  c.yarn.original_context = r.U32("rope.scaling.original_context_length");
  c.yarn.beta_fast = r.F32("rope.scaling.yarn_beta_fast");
  c.yarn.beta_slow = r.F32("rope.scaling.yarn_beta_slow");
  c.yarn.yarn_attn_factor = r.F32("rope.scaling.yarn_attn_factor");

  c.num_experts = r.U32("expert_count");
  c.num_experts_used = r.U32("expert_used_count");
  c.expert_ff = r.U32("expert_feed_forward_length");
  c.shared_expert_ff = r.U32("expert_shared_feed_forward_length");
  c.shared_expert_count = r.U32Or("expert_shared_count", 1);
  c.expert_weights_norm = r.Bool("expert_weights_norm");
  c.expert_weights_scale = r.F32("expert_weights_scale");
  if (r.U32("expert_gating_func") != kGatingSigmoid) {
    r.Fail("laguna.expert_gating_func must be 2 (sigmoid)");
  }
  c.leading_dense_layers = r.U32("leading_dense_block_count");

  if (!r.ok) {
    return std::nullopt;
  }

  // Structural validity, independent of the profile below.
  if (c.hidden_size == 0 || c.ffn_size == 0 || c.vocab_size == 0 ||
      c.context_length == 0 || c.context_length > INT32_MAX || c.rms_eps <= 0 ||
      c.num_kv_heads == 0 || c.head_dim == 0 || c.sliding_window == 0 ||
      c.rope_theta_full <= 0 || c.rope_theta_swa <= 0 ||
      c.rotary_dim_full == 0 || c.rotary_dim_swa == 0 ||
      c.rotary_dim_full % 2 != 0 || c.rotary_dim_swa % 2 != 0 ||
      c.rotary_dim_full > c.head_dim || c.rotary_dim_swa > c.head_dim ||
      c.yarn.factor < 1.0F || c.yarn.original_context == 0 ||
      c.yarn.beta_fast <= c.yarn.beta_slow || c.yarn.beta_slow <= 0 ||
      c.rope_attn_factor <= 0 || c.num_experts == 0 ||
      c.num_experts_used == 0 || c.num_experts_used > c.num_experts ||
      c.expert_ff == 0 || c.shared_expert_ff == 0 ||
      c.shared_expert_count != 1 || c.expert_weights_scale <= 0 ||
      c.leading_dense_layers >= c.num_layers) {
    r.Fail("laguna metadata describes an invalid shape");
    return std::nullopt;
  }
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.n_head(il) % c.num_kv_heads != 0) {
      r.Fail("laguna layer " + std::to_string(il) +
             " query heads are not a multiple of the KV heads");
      return std::nullopt;
    }
  }

  // The GGUF vocabulary, when it travels with the metadata, must agree.
  if (const auto* tokens = gguf.FindMetadata("tokenizer.ggml.tokens");
      tokens != nullptr) {
    const auto* list =
        std::get_if<std::vector<std::string_view>>(&tokens->value);
    if (list == nullptr || list->size() != c.vocab_size) {
      r.Fail("tokenizer.ggml.tokens does not match laguna.vocab_size");
      return std::nullopt;
    }
  }

  // Profile of this runtime: Laguna-S-2.1. Other Laguna sizes (different
  // layer counts, 128-wide gate or SWA-free variants) are not implemented.
  std::uint32_t full_heads = 0;
  std::uint32_t swa_heads = 0;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    auto& slot = c.is_full_attention(il) ? full_heads : swa_heads;
    if (slot == 0) {
      slot = c.n_head(il);
    } else if (slot != c.n_head(il)) {
      r.Fail(
          "laguna head_count must be constant within full and within "
          "sliding layers; layer " +
          std::to_string(il) + " differs");
      return std::nullopt;
    }
  }
  if (c.num_layers != 48 || c.hidden_size != 3072 || c.ffn_size != 12288 ||
      c.vocab_size != 100352 || c.num_kv_heads != 8 || c.head_dim != 128 ||
      full_heads != 48 || swa_heads != 72 || c.sliding_window != 512 ||
      c.rotary_dim_full != 64 || c.rotary_dim_swa != 128 ||
      c.num_experts != 256 || c.num_experts_used != 10 || c.expert_ff != 1024 ||
      c.shared_expert_ff != 1024 || !c.expert_weights_norm ||
      c.leading_dense_layers != 1) {
    r.Fail("unsupported Laguna geometry (only Laguna-S-2.1 is implemented)");
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::laguna
