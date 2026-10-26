#ifndef GUFO_MODELS_LAGUNA_WEIGHTS_HPP_
#define GUFO_MODELS_LAGUNA_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/config.hpp"

namespace gufo::models::laguna {

/// Non-owning view of one GGUF tensor. `cols` x `rows` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows (ne[1])
/// the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning mapping.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return name.empty(); }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// Every tensor of one transformer layer, by its llama.cpp name. Dense layers
/// (the leading `leading_dense_layers`) fill `ffn_*`; the others fill the
/// router, routed-expert and shared-expert tensors.
struct LayerWeights {
  bool full_attention{false};
  bool moe{false};

  TensorRef attn_norm;    ///< blk.N.attn_norm.weight   [hidden] F32
  TensorRef attn_q;       ///< attn_q.weight  [hidden -> heads * head_dim]
  TensorRef attn_k;       ///< attn_k.weight  [hidden -> kv_heads * head_dim]
  TensorRef attn_v;       ///< attn_v.weight  [hidden -> kv_heads * head_dim]
  TensorRef attn_output;  ///< attn_output.weight [heads * head_dim -> hidden]
  TensorRef attn_q_norm;  ///< attn_q_norm.weight [head_dim] F32
  TensorRef attn_k_norm;  ///< attn_k_norm.weight [head_dim] F32
  TensorRef attn_gate;    ///< attn_gate.weight [hidden -> heads] F32, per head
  TensorRef ffn_norm;     ///< ffn_norm.weight [hidden] F32

  // Dense SwiGLU (layer 0).
  TensorRef ffn_gate;  ///< [hidden -> ffn]
  TensorRef ffn_up;    ///< [hidden -> ffn]
  TensorRef ffn_down;  ///< [ffn -> hidden]

  // Mixture of experts (layers 1 to 47).
  TensorRef ffn_gate_inp;     ///< Router [hidden -> experts] F32
  TensorRef ffn_exp_probs_b;  ///< blk.N.exp_probs_b.bias [experts] F32
  TensorRef ffn_gate_exps;    ///< [hidden -> expert_ff] x experts
  TensorRef ffn_up_exps;      ///< [hidden -> expert_ff] x experts
  TensorRef ffn_down_exps;    ///< [expert_ff -> hidden] x experts
  TensorRef ffn_gate_shexp;   ///< [hidden -> shared_ff]
  TensorRef ffn_up_shexp;     ///< [hidden -> shared_ff]
  TensorRef ffn_down_shexp;   ///< [shared_ff -> hidden]
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< token_embd.weight [hidden -> vocab]
  TensorRef output_norm;  ///< output_norm.weight [hidden] F32
  TensorRef output;       ///< output.weight [hidden -> vocab], untied
  std::vector<LayerWeights> layers;
  std::size_t tensor_count{0};   ///< Tensors bound, equal to the file's.
  std::uint64_t total_bytes{0};  ///< Sum of bound tensor payload bytes.

  /// Tensors a model with this configuration must carry: 3 global, 9 per layer
  /// (norms, attention, gate), 3 per dense layer, 8 per MoE layer.
  [[nodiscard]] static std::size_t ExpectedTensorCount(const Config& config);

  /// Binds and validates every tensor of the artifact. Payloads stay mapped
  /// and untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Same binding over an explicit tensor list. Fails on a missing tensor, a
  /// wrong shape or ggml type, a duplicate name, or any tensor the model does
  /// not account for. `regions` locates payloads for `file_offset` and the
  /// truncation check; pass an empty span for header-only lists.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const Config& config, std::span<const core::GgufTensorInfo> tensors,
      std::span<const core::GgufMappedRegion> regions,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_WEIGHTS_HPP_
