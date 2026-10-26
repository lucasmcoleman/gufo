#ifndef GUFO_MODELS_LAGUNA_CONFIG_HPP_
#define GUFO_MODELS_LAGUNA_CONFIG_HPP_

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::laguna {

/// Static YaRN parameters of the full-attention layers, as the GGUF records
/// them. Sliding-window layers never use them (plain RoPE).
struct YarnParams {
  float factor{1.0F};                 ///< rope.scaling.factor, 32.
  std::uint32_t original_context{0};  ///< rope.scaling.original_context_length.
  float beta_fast{32.0F};
  float beta_slow{1.0F};
  /// `rope.scaling.yarn_attn_factor`. llama.cpp d854d98 does not read this key
  /// for Laguna (only `rope.scaling.attn_factor`, see
  /// Config::rope_attn_factor), so it is kept for provenance and takes no part
  /// in YarnMscale().
  float yarn_attn_factor{1.0F};
};

/// Architecture parameters of a Laguna GGUF artifact. Every value is read from
/// the file (the GGUF wins over any Hugging Face config); the profile checks in
/// FromGguf only bound what this runtime was written and validated for
/// (Laguna-S-2.1: 48 layers, 12 full and 36 sliding-window attention layers).
struct Config {
  /// Layers with `il % kFullAttentionInterval == 0` use full attention; the
  /// rest are sliding-window. llama.cpp: `load_swa_pattern(ml, 4, true)`.
  static constexpr std::uint32_t kFullAttentionInterval = 4;
  /// `expert_gating_func` value for sigmoid routing (llama.cpp enum).
  static constexpr std::uint32_t kGatingSigmoid = 2;

  std::uint32_t num_layers{0};      ///< block_count, 48.
  std::uint32_t hidden_size{0};     ///< embedding_length, 3072.
  std::uint32_t ffn_size{0};        ///< feed_forward_length, 12288 (dense).
  std::uint32_t vocab_size{0};      ///< 100352.
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  /// Query heads per layer: 48 on full layers, 72 on sliding layers.
  std::vector<std::uint32_t> heads_per_layer;
  std::uint32_t num_kv_heads{0};    ///< 8 on every layer.
  std::uint32_t head_dim{0};        ///< key_length == value_length, 128.
  std::uint32_t sliding_window{0};  ///< 512; key j visible from p iff p-j < n.

  float rope_theta_full{0.0F};       ///< Full layers, 500000.
  float rope_theta_swa{0.0F};        ///< Sliding layers, 10000.
  std::uint32_t rotary_dim_full{0};  ///< Full layers, 64 (leading channels).
  std::uint32_t rotary_dim_swa{0};   ///< Sliding layers, 128.
  YarnParams yarn;
  /// `rope.scaling.attn_factor` (absent in this GGUF: 1.0), the key llama.cpp
  /// actually multiplies into the YaRN magnitude.
  float rope_attn_factor{1.0F};

  std::uint32_t num_experts{0};           ///< 256.
  std::uint32_t num_experts_used{0};      ///< 10.
  std::uint32_t expert_ff{0};             ///< 1024.
  std::uint32_t shared_expert_ff{0};      ///< 1024.
  std::uint32_t shared_expert_count{1};   ///< Absent in the GGUF; llama.cpp: 1.
  bool expert_weights_norm{false};        ///< true.
  float expert_weights_scale{1.0F};       ///< 2.5.
  std::uint32_t leading_dense_layers{0};  ///< 1; those layers use dense SwiGLU.

  [[nodiscard]] bool is_full_attention(std::uint32_t il) const noexcept {
    return il % kFullAttentionInterval == 0;
  }
  [[nodiscard]] std::uint32_t n_head(std::uint32_t il) const {
    return heads_per_layer.at(il);
  }
  /// Leading channels of each head that RoPE rotates (NeoX layout).
  [[nodiscard]] std::uint32_t rotary_dim(std::uint32_t il) const noexcept {
    return is_full_attention(il) ? rotary_dim_full : rotary_dim_swa;
  }
  [[nodiscard]] float rope_theta(std::uint32_t il) const noexcept {
    return is_full_attention(il) ? rope_theta_full : rope_theta_swa;
  }
  [[nodiscard]] bool is_moe(std::uint32_t il) const noexcept {
    return il >= leading_dense_layers;
  }
  /// Query heads per KV head: 6 on full layers, 9 on sliding layers.
  [[nodiscard]] std::uint32_t gqa_group(std::uint32_t il) const {
    return n_head(il) / num_kv_heads;
  }
  [[nodiscard]] std::uint32_t q_dim(std::uint32_t il) const {
    return n_head(il) * head_dim;
  }
  [[nodiscard]] std::uint32_t kv_dim() const noexcept {
    return num_kv_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t full_layer_count() const noexcept {
    return (num_layers + kFullAttentionInterval - 1) / kFullAttentionInterval;
  }
  /// YaRN applies to full layers only; sliding layers run plain RoPE.
  [[nodiscard]] bool uses_yarn(std::uint32_t il) const noexcept {
    return is_full_attention(il) && yarn.factor != 1.0F;
  }
  /// Net multiplier on cos and sin of a YaRN layer as llama.cpp d854d98
  /// computes it (llama-context.cpp:269-306 cancels ggml's internal mscale,
  /// ggml re-applies 1 + 0.1 ln(factor)): rope_attn_factor * (1 + 0.1 ln f).
  /// With factor 32 this is about 1.3466, not the GGUF's yarn_attn_factor 1.0.
  /// From reading the source; confirm against a tap before relying on it.
  [[nodiscard]] float YarnMscale() const noexcept {
    return rope_attn_factor * (1.0F + 0.1F * std::log(yarn.factor));
  }

  /// Reads and validates the `laguna.*` metadata. Fails (nullopt and a message
  /// in `error_msg`) on any missing key, malformed value or shape outside the
  /// supported profile.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_CONFIG_HPP_
