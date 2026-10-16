#ifndef GUFO_MODELS_GLM53_FLASH_CONFIG_HPP_
#define GUFO_MODELS_GLM53_FLASH_CONFIG_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm53_flash {

/// Architecture parameters of a `glm5-next` GGUF artifact (GLM-5.3-Flash).
/// Every value is read from the file. The hard checks in FromGguf only bound
/// what this runtime implements: KDA linear attention on the layers whose
/// `head_count_kv` is 0, absorbed MLA with a pooled DSA indexer on the layers
/// whose `head_count_kv` is 1, 4-stream hyper-connections around every
/// sub-layer, and sigmoid-routed MoE after the leading dense layers.
struct Config {
  enum class Mixer : std::uint8_t { kKda, kMla };

  std::uint32_t num_layers{0};  ///< Trunk layers (NextN block excluded), 45.
  std::uint32_t num_layers_all{0};  ///< block_count, includes NextN, 46.
  std::uint32_t nextn_layers{0};    ///< 1
  std::uint32_t hidden_size{0};     ///< n_embd, 4096.
  std::uint32_t vocab_size{0};      ///< 154880.
  std::uint32_t context_length{0};  ///< 1048576.
  float rms_eps{1e-5F};             ///< attention.layer_norm_rms_epsilon
  float norm_eps{1e-6F};  ///< attention.layer_norm_epsilon (indexer LayerNorm)

  /// Per-layer mixer, derived from attention.head_count_kv (0 = KDA, 1 = MLA).
  std::vector<Mixer> mixers;

  // Hyper-connections: hc_count parallel residual streams of hidden_size,
  // mixed by a Sinkhorn-normalised hc_count x hc_count matrix.
  std::uint32_t hc_count{0};           ///< 4
  std::uint32_t hc_sinkhorn_iters{0};  ///< 20
  float hc_eps{1e-6F};

  std::uint32_t num_heads{0};  ///< 64, shared by KDA and MLA query heads.

  // Kimi Delta Attention.
  std::uint32_t kda_head_dim{0};     ///< 128 (key and value dim per head)
  std::uint32_t ssm_conv_kernel{0};  ///< 4
  float kda_gate_lower_bound{0.0F};  ///< -5

  // Absorbed multi-head latent attention, no RoPE.
  std::uint32_t q_lora_rank{0};    ///< 1536
  std::uint32_t kv_lora_rank{0};   ///< 512 (latent width = cache row)
  std::uint32_t mla_head_dim{0};   ///< key_length_mla, 256 (q/k nope dim)
  std::uint32_t mla_value_dim{0};  ///< value_length_mla, 256

  // DSA indexer over pools of `indexer_kpool` tokens.
  std::uint32_t indexer_heads{0};     ///< 32
  std::uint32_t indexer_head_dim{0};  ///< 128
  std::uint32_t indexer_top_k{0};     ///< 2048 raw tokens = top_k / kpool pools
  std::uint32_t indexer_kpool{0};     ///< 4

  // FFN: leading dense layers, then MoE with one always-on shared expert.
  std::uint32_t leading_dense_layers{0};  ///< 3
  std::uint32_t dense_ff{0};              ///< 12288
  std::uint32_t num_experts{0};           ///< 288
  std::uint32_t num_experts_used{0};      ///< 8
  std::uint32_t expert_ff{0};             ///< 2048
  std::uint32_t shared_expert_ff{0};      ///< 2048
  std::uint32_t num_shared_experts{0};    ///< 1
  float expert_weights_scale{1.0F};       ///< 2.5
  bool expert_weights_norm{false};        ///< true
  /// SwiGLU clamp per layer for routed and shared experts (and the dense FFN,
  /// which llama.cpp clamps with the routed value). 0 disables the clamp.
  std::vector<float> swiglu_clamp_exp;
  std::vector<float> swiglu_clamp_shexp;

  [[nodiscard]] std::uint32_t HcDim() const noexcept {
    return hc_count * hidden_size;
  }
  /// Mixing logits per token: hc_count pre, hc_count post, hc_count^2 comb.
  [[nodiscard]] std::uint32_t HcMixDim() const noexcept {
    return (2 * hc_count) + (hc_count * hc_count);
  }
  [[nodiscard]] std::uint32_t KdaDim() const noexcept {
    return num_heads * kda_head_dim;
  }
  [[nodiscard]] std::uint32_t MlaQDim() const noexcept {
    return num_heads * mla_head_dim;
  }
  [[nodiscard]] std::uint32_t MlaOutDim() const noexcept {
    return num_heads * mla_value_dim;
  }
  [[nodiscard]] std::uint32_t IndexerPoolBudget() const noexcept {
    return indexer_kpool == 0 ? 0 : indexer_top_k / indexer_kpool;
  }
  [[nodiscard]] bool IsKdaLayer(std::uint32_t layer) const noexcept {
    return layer < mixers.size() && mixers[layer] == Mixer::kKda;
  }
  [[nodiscard]] bool IsMlaLayer(std::uint32_t layer) const noexcept {
    return layer < mixers.size() && mixers[layer] == Mixer::kMla;
  }
  [[nodiscard]] bool IsDenseFfnLayer(std::uint32_t layer) const noexcept {
    return layer < leading_dense_layers;
  }
  [[nodiscard]] std::uint32_t CountLayers(Mixer mixer) const noexcept {
    std::uint32_t n = 0;
    for (std::uint32_t i = 0; i < num_layers; ++i) {
      n += mixers[i] == mixer ? 1U : 0U;
    }
    return n;
  }

  /// Reads and validates the `glm5-next.*` metadata.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_CONFIG_HPP_
