#ifndef GUFO_MODELS_GLM53_FLASH_WEIGHTS_HPP_
#define GUFO_MODELS_GLM53_FLASH_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/config.hpp"

namespace gufo::models::glm53_flash {

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices
/// (routed experts, or the 64 per-head MLA absorption matrices).
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// One hyper-connection mixer (`hc_attn_*` or `hc_ffn_*`): F32 weights that
/// map the RMS-normalised flattened streams to pre/post/comb logits.
struct HcMixer {
  TensorRef fn;     ///< [hc_dim -> 2*hc + hc^2], F32
  TensorRef base;   ///< [2*hc + hc^2] bias, F32
  TensorRef scale;  ///< [3] pre/post/comb logit scales, F32
};

struct LayerWeights {
  Config::Mixer mixer{Config::Mixer::kKda};
  bool dense_ffn{false};
  bool nextn{false};

  HcMixer hc_attn;  ///< Empty on the NextN block (plain residual).
  HcMixer hc_ffn;
  TensorRef attn_norm;  ///< [hidden]
  TensorRef ffn_norm;   ///< [hidden]

  // Kimi Delta Attention (KDA layers).
  TensorRef kda_q;       ///< [hidden -> kda_dim]
  TensorRef kda_k;       ///< [hidden -> kda_dim]
  TensorRef kda_v;       ///< [hidden -> kda_dim]
  TensorRef kda_conv_q;  ///< [conv_kernel, 1, kda_dim] depthwise, F32
  TensorRef kda_conv_k;
  TensorRef kda_conv_v;
  TensorRef kda_f_a;      ///< [hidden -> head_dim] forget-gate low rank
  TensorRef kda_f_b;      ///< [head_dim -> kda_dim]
  TensorRef kda_dt_bias;  ///< [kda_dim]
  TensorRef kda_a;        ///< [heads] = -exp(A_log)
  TensorRef kda_beta;     ///< [hidden -> heads]
  TensorRef kda_g_a;      ///< [hidden -> head_dim] output-gate low rank
  TensorRef kda_g_b;      ///< [head_dim -> kda_dim]
  TensorRef kda_norm;     ///< [head_dim] per-head output RMSNorm
  TensorRef
      attn_out;  ///< KDA: [kda_dim -> hidden]; MLA: [heads*v_dim -> hidden]

  // Absorbed MLA + DSA indexer (MLA layers, and the NextN block).
  TensorRef mla_q_a;          ///< [hidden -> q_lora]
  TensorRef mla_q_a_norm;     ///< [q_lora]
  TensorRef mla_q_b;          ///< [q_lora -> heads * mla_head_dim]
  TensorRef mla_kv_a;         ///< [hidden -> kv_lora] (the cached latent)
  TensorRef mla_kv_a_norm;    ///< [kv_lora]
  TensorRef mla_k_b;          ///< [mla_head_dim -> kv_lora] x heads
  TensorRef mla_v_b;          ///< [kv_lora -> mla_value_dim] x heads
  TensorRef idx_q_b;          ///< [q_lora -> indexer_heads * indexer_dim]
  TensorRef idx_k;            ///< [hidden -> indexer_dim]
  TensorRef idx_k_norm;       ///< [indexer_dim] LayerNorm weight
  TensorRef idx_k_norm_bias;  ///< [indexer_dim] LayerNorm bias
  TensorRef idx_proj;         ///< [hidden -> indexer_heads] head weights
  TensorRef idx_gate;         ///< [hidden -> indexer_dim] pooling gate
  TensorRef idx_ape;          ///< [indexer_dim, kpool] position bias, F32

  // Dense SwiGLU FFN (leading layers).
  TensorRef ffn_gate;  ///< [hidden -> dense_ff]
  TensorRef ffn_up;
  TensorRef ffn_down;  ///< [dense_ff -> hidden]

  // Mixture of experts.
  TensorRef router;         ///< [hidden -> num_experts], F32
  TensorRef router_bias;    ///< [num_experts] selection-only bias, F32
  TensorRef ffn_gate_exps;  ///< [hidden -> expert_ff] x experts
  TensorRef ffn_up_exps;
  TensorRef ffn_down_exps;  ///< [expert_ff -> hidden] x experts
  TensorRef shexp_gate;     ///< [hidden -> shared_ff]
  TensorRef shexp_up;
  TensorRef shexp_down;  ///< [shared_ff -> hidden]

  // NextN draft block only.
  TensorRef nextn_eh_proj;    ///< [2*hidden -> hidden], [enorm(e) | hnorm(h)]
  TensorRef nextn_enorm;      ///< [hidden]
  TensorRef nextn_hnorm;      ///< [hidden]
  TensorRef nextn_head_norm;  ///< [hidden] shared_head_norm
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;              ///< [hidden -> vocab]
  TensorRef output;                  ///< [hidden -> vocab]
  TensorRef output_norm;             ///< [hidden]
  std::vector<LayerWeights> layers;  ///< Trunk layers, then NextN if bound.

  /// Binds and validates every tensor of the artifact (trunk and, when
  /// `bind_nextn`, the NextN block). Payloads stay mapped and untouched; only
  /// headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, bool bind_nextn,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_WEIGHTS_HPP_
